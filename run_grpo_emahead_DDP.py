# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import torch
import torch.distributed
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from coconut_grpo_emahead import Coconut
from dataset import (
    get_dataset,
    get_grpo_dataset,
    MyCollator,
)

import gc
import time
import copy
import json
import yaml
import random
import datetime
import argparse
import bitsandbytes as bnb
import matplotlib.pyplot as plt
from tqdm import tqdm
# from copy import copy
from utils import Config, set_seed, visualize_latent_pca, save_latent_histogram

import torch

global_time = 0

def start_timer():
    global global_time
    global_time = time.time()

def measure_timer(out_str):
    global global_time
    cur_time = time.time()
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{cur_time - global_time:.2f}s] {out_str}")
    global_time = cur_time

def print_memory_breakdown(model, optimizer):
    """
    Prints a breakdown of memory usage by Model Weights, Grads, and Optimizer States.
    """
    # 1. Model Weights
    param_mem = 0
    grad_mem = 0
    for param in model.parameters():
        param_mem += param.numel() * param.element_size()
        if param.grad is not None:
            grad_mem += param.grad.numel() * param.grad.element_size()
            
    # 2. Optimizer States
    opt_mem = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                opt_mem += v.numel() * v.element_size()
                
    # 3. Totals
    total_allocated = torch.cuda.memory_allocated()
    # Activations is roughly the difference, though it includes temp buffers
    activations_and_buffers = total_allocated - (param_mem + grad_mem + opt_mem)

    # Convert to GB
    to_gb = 1024**3
    
    print(f"--- VRAM Breakdown ---")
    print(f"Model Weights:    {param_mem / to_gb:.2f} GB")
    print(f"Gradients:        {grad_mem / to_gb:.2f} GB")
    print(f"Optimizer States: {opt_mem / to_gb:.2f} GB")
    print(f"Activations/Misc: {activations_and_buffers / to_gb:.2f} GB")
    print(f"----------------------")
    print(f"Total Allocated:  {total_allocated / to_gb:.2f} GB")
    print(f"Max Reserved:     {torch.cuda.max_memory_reserved() / to_gb:.2f} GB")
    print(f"Max Allocated:    {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Usage: Call this inside your training loop after optimizer.step()
# print_memory_breakdown(model, optimizer)


def capture_rng_state(device):
    state = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(device=device).clone().cpu()
    return state


def restore_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device=device)


def assert_rollout_replay_embeddings_match(
    replay_outputs,
    rollout_embeddings,
    atol=1e-5,
    rtol=1e-4,
):
    # The replay may be longer than the rollout due to global right-padding of token_ids to max_len
    # across all chunks. Only compare up to the rollout's actual length (causal attention ensures
    # the prefix is unaffected by the extra padded steps that the replay runs beyond it).
    rollout_len = rollout_embeddings.shape[1]
    replay_len  = replay_outputs.inputs_embeds.shape[1]
    if replay_outputs.inputs_embeds.shape[0] != rollout_embeddings.shape[0] or replay_outputs.inputs_embeds.shape[2] != rollout_embeddings.shape[2]:
        raise AssertionError(
            f"Replay embedding batch/hidden mismatch: {replay_outputs.inputs_embeds.shape} vs {rollout_embeddings.shape}"
        )
    if replay_len < rollout_len:
        raise AssertionError(
            f"Replay is shorter than rollout ({replay_len} < {rollout_len}) — unexpected"
        )
    replay_prefix = replay_outputs.inputs_embeds[:, :rollout_len, :]
    if not torch.allclose(replay_prefix, rollout_embeddings, atol=atol, rtol=rtol):
        diff = (replay_prefix - rollout_embeddings).abs()           # (B, L, H)
        max_diff = diff.max().item()
        # Find the worst-offending position and batch item
        flat_idx = diff.reshape(-1).argmax().item()
        B, L, H = diff.shape
        b_idx = flat_idx // (L * H)
        l_idx = (flat_idx % (L * H)) // H
        h_idx = flat_idx % H
        # Check whether divergence is in the prompt portion or generated portion
        prompt_len_guess = rollout_embeddings.shape[1] - (replay_outputs.inputs_embeds.shape[1] - rollout_len)
        prompt_diff  = diff[:, :prompt_len_guess, :].max().item() if prompt_len_guess > 0 else 0.0
        gen_diff     = diff[:, prompt_len_guess:, :].max().item()  if prompt_len_guess < L else 0.0
        # Check per-position: at which generated step does divergence first appear?
        per_pos_max = diff.max(dim=0).values.max(dim=-1).values   # (L,)
        first_bad = (per_pos_max > atol).nonzero(as_tuple=False)
        first_bad_pos = first_bad[0].item() if first_bad.numel() > 0 else -1
        raise AssertionError(
            f"Replay embeddings diverged (max abs diff {max_diff:.6e})\n"
            f"  Worst at: batch={b_idx}, pos={l_idx}, hidden={h_idx}\n"
            f"  replay value={replay_prefix[b_idx, l_idx, h_idx].item():.4f}  "
            f"rollout value={rollout_embeddings[b_idx, l_idx, h_idx].item():.4f}\n"
            f"  Max diff in prompt portion (pos<{prompt_len_guess}): {prompt_diff:.6e}\n"
            f"  Max diff in generated portion (pos>={prompt_len_guess}): {gen_diff:.6e}\n"
            f"  First position with diff>{atol}: pos={first_bad_pos} "
            f"({'prompt' if first_bad_pos < prompt_len_guess else 'generated'})"
        )


def test_rollout_replay_embeddings(
    parallel_model,
    prompt_ids,
    attention_mask,
    generated_ids,
    rollout_embeddings,
    rng_state,
    device,
    atol=1e-5,
    rtol=1e-4,
):
    was_training = parallel_model.module.training
    parallel_model.module.train()
    restore_rng_state(rng_state, device)
    with torch.no_grad():
        replay_outputs = parallel_model(
            input_ids=prompt_ids.to(device),
            attention_mask=attention_mask.to(device),
            replay_generated_ids=generated_ids.to(device),
        )
    assert_rollout_replay_embeddings_match(
        replay_outputs,
        rollout_embeddings.to(device),
        atol=atol,
        rtol=rtol,
    )
    if not was_training:
        parallel_model.module.eval()
    return replay_outputs

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def extract_answer(text, eot_token):
    """Extract the numerical answer following '###' from generated text."""
    return text.split("#")[-1].replace(",", "").replace(eot_token, "").strip()


def compute_reward(generated_ids, tokenizer, ground_truth,
                            latent_id, length_penalty=0.01, format_penalty=-0.5,
                            latent_step_reward=0.0, coconut_mode=True, is_print=False):
    """
    Combined reward balancing correctness, format compliance, and latent usage.

    Reward scale:
        +1.0 + latent_step_reward * n_latent   correct answer, clean format
        +0.0 + latent_step_reward * n_latent   wrong answer, clean format
        format_penalty (default -0.5)           format violation

    Args:
        generated_ids: (B, L) tensor — full sequence including question tokens
        tokenizer:     tokenizer for decoding
        ground_truth:  expected answer string
        latent_id:     token id used for latent positions
        format_penalty: negative reward for format violations
        latent_step_reward: small positive reward per latent token (encourages exploration)
        coconut_mode:  whether to keep special tokens when decoding

    Returns:
        total_reward (list[float]), n_latent (list[int])
    """
    total_reward, base_reward, accuracies, extracted_answers = [], [], [], []
    n_latents, n_latents_correct = [], []
    for i in range(generated_ids.shape[0]):     # iterate across batch
        text = tokenizer.decode(generated_ids[i, :], skip_special_tokens=not coconut_mode)      # list of strings
        answer = extract_answer(text, "<|endoftext|>")
        extracted_answers.append(answer)

        num_latent = (generated_ids[i, :] == latent_id).sum().item()
        n_latents.append(num_latent)

        # Check for format violations (negative reward)
        format_violation = False

        # 1. Latent markers leaked into text answer
        if "<|start-latent|>" in answer or "<|end-latent|>" in answer:
            format_violation = True
            answer = answer.replace("<|start-latent|>", "").replace("<|end-latent|>", "")

        # 2. Empty or invalid answer (no parseable content after ###)
        cleaned = answer.strip()
        if not cleaned or cleaned == "#":
            format_violation = True

        # 3. No latent tokens at all (model skipped reasoning entirely)
        if num_latent == 0:
            format_violation = True

        # 4. CoT steps found during reasoning
        if "<<" in answer or ">>" in answer:
            format_violation = True

        # Compute correctness (tracked separately for accuracy metric)
        correctness = 1.0 if answer == ground_truth[i] else 0.0
        accuracies.append(correctness)

        # Compute reward
        if format_violation:
            reward_base = format_penalty
        else:
            reward_base = correctness - length_penalty * max(0, num_latent - 1)
        reward = reward_base + (latent_step_reward * num_latent if not format_violation else 0.0)

        if answer == ground_truth[i]:
            n_latents_correct.append(num_latent)

        base_reward.append(reward_base)
        total_reward.append(reward)

        if is_print:
            print(text)
            print(f"Predicted answer: '{answer}'")
            print(f"Ground truth: {ground_truth[i]}")
            print(reward, num_latent)

    accuracy = sum(accuracies) / len(accuracies)

    return total_reward, base_reward, accuracy, extracted_answers, accuracies, n_latents, n_latents_correct


# ---------------------------------------------------------------------------
# Log-probability (termination head + lm head)
# ---------------------------------------------------------------------------

def compute_full_log_probs(outputs, input_ids, question_len, latent_id):
    """Sum of log-probs for all generated tokens (question excluded).

    For each generated position pos ∈ [question_len, full_len):

      • token == latent_id  →  log p = log_softmax(term_logits[pos-1])[1]
                                        (termination head chose "latent")

      • token == text_token →  log p = log_softmax(term_logits[pos-1])[0]
                                       + log_softmax(lm_logits[pos-1])[token]
                                        (termination head chose "text", then
                                         lm head chose the specific token)

    By including termination decisions, the GRPO gradient flows through both
    heads so the model learns jointly WHEN to use latents and WHAT to say.

    Args:
        outputs:      Coconut forward output (has .logits and .termination_logits)
        input_ids:    (1, L) full token ids
        question_len: int — number of question tokens at the start
        latent_id:    int — token id for latent positions

    Returns:
        Scalar tensor: sum of log-probs over generated tokens.
    """
    full_len = input_ids.shape[1]
    if full_len <= question_len:
        return torch.tensor(0.0, device=input_ids.device)

    # fp32 for numerical stability
    lm_log_p   = F.log_softmax(outputs.logits.float(), dim=-1).to(input_ids.device)            # (1, L, vocab)
    term_log_p = F.log_softmax(outputs.termination_logits.float(), dim=-1).to(input_ids.device) # (1, L, 2)

    # logits at position t predict the token at position t+1
    # generated tokens: positions question_len … full_len-1
    # predicted by:     positions question_len-1 … full_len-2
    gen_tokens   = input_ids[0, question_len:]                                     # (gen_len,)
    pred_lm      = lm_log_p[0, question_len - 1 : full_len - 1, :]               # (gen_len, vocab)
    pred_term    = term_log_p[0, question_len - 1 : full_len - 1, :]             # (gen_len, 2)

    is_latent    = (gen_tokens == latent_id)                                       # (gen_len,) bool

    term_lp_lat  = pred_term[:, 1]                                                 # log p(latent)
    term_lp_txt  = pred_term[:, 0]                                                 # log p(text)
    token_lp     = pred_lm.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)       # log p(token|text)

    per_pos_lp   = torch.where(is_latent, term_lp_lat, term_lp_txt + token_lp)   # (gen_len,)
    return per_pos_lp.sum()


def compute_log_probs(token_ids, model_outputs, prompt_len, latent_id, pad_id):
    """Compute log-probs for the next token being latent vs. text.

    This is used during generation to get the log-prob ratio for GRPO.

    Args:
        output_embeds: (B, L, hidden_size) output of the last decoder layer
        token_ids:     (B, L) tensor of the next token id (either latent_id or text token)
        model_outputs: Outputs object returned by Coconut forward pass
        prompt_lens:   (B) length of the prompt(question) for each example in the batch
        latent_id:     int — token id for latent positions
    Returns:
        pos_log_probs.sum(dim=-1): (B,) tensor of summed log-probs per batch 
    """
    gen_tokens  = token_ids[:, prompt_len:]   # (B, gen_len)
    mask_latent = (gen_tokens == latent_id)

    # Slice to generated positions BEFORE log_softmax to avoid materialising the question
    # portion of logits. Keep lm log-probs in bf16 until after the gather to halve peak
    # memory compared to a full (B, L, vocab) float32 tensor.
    gen_len = gen_tokens.shape[1]
    needed_steps = prompt_len - 1 + gen_len
    if model_outputs.logits.shape[1] < needed_steps or model_outputs.termination_logits.shape[1] < needed_steps:
        raise ValueError(
            f"Replay outputs are too short for log-prob computation: "
            f"logits={model_outputs.logits.shape[1]}, term={model_outputs.termination_logits.shape[1]}, "
            f"needed={needed_steps}"
        )
    lm_logits_gen  = model_outputs.logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]    # (B, gen_len, vocab) bf16
    lm_log_p_gen   = F.log_softmax(lm_logits_gen, dim=-1)                                      # (B, gen_len, vocab) bf16
    pred_token     = lm_log_p_gen.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1).float()     # (B, gen_len) fp32

    # Termination logits are 2-class; float32 is cheap here.
    term_logits_gen = model_outputs.termination_logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]  # (B, gen_len, 2)
    pred_term       = F.log_softmax(term_logits_gen.float(), dim=-1)                           # (B, gen_len, 2) fp32

    pos_log_probs = torch.where(mask_latent, pred_term[:, :, 1], pred_token + pred_term[:, :, 0])
    loss_mask     = (gen_tokens != pad_id)

    return pos_log_probs, loss_mask


# def compute_log_probs(token_ids, model_outputs, prompt_len, latent_id, pad_id):
#     """Compute log-probs for the next token being latent vs. text.

#     This is used during generation to get the log-prob ratio for GRPO.

#     Args:
#         output_embeds: (B, L, hidden_size) output of the last decoder layer
#         token_ids:     (B, L) tensor of the next token id (either latent_id or text token)
#         model_outputs: Outputs object returned by Coconut forward pass
#         prompt_lens:   (B) length of the prompt(question) for each example in the batch
#         latent_id:     int — token id for latent positions
#     Returns:
#         pos_log_probs.sum(dim=-1): (B,) tensor of summed log-probs per batch 
#     """
#     # If training is unstable, consider using output_embeds instead of model_outputs.logits

#     # term_logits:   (B, L, 2) tensor of termination head logits for the current position

#     lm_log_p = F.log_softmax(model_outputs.logits.float(), dim=-1).to(token_ids.device)                 # (B, L, vocab)
#     term_log_p = F.log_softmax(model_outputs.termination_logits.float(), dim=-1).to(token_ids.device)   # (B, L, 2)

#     gen_tokens          = token_ids[:, prompt_len:]           # (B, gen_len)
#     mask_latent         = (gen_tokens == latent_id)           # (B, gen_len) — only over generated portion

#     # logit at position t predicts token at t+1, so logits [prompt_len-1 .. L-2] predict gen_tokens
#     pred_term_text      = lm_log_p[:, prompt_len-1:-1, :]    # (B, gen_len, vocab)
#     pred_term_latent    = term_log_p[:, prompt_len-1:-1, :]  # (B, gen_len, 2)

#     pred_token = pred_term_text.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)    # (B, gen_len)

#     # If token is latent, log-prob(latent) from term_head
#     # If token is text, log-prob(text) from term_head + log-prob(token) from lm_head
#     pos_log_probs = torch.where(mask_latent, pred_term_latent[:, :, 1], pred_token + pred_term_latent[:, :, 0])     # (B, gen_len)
#     loss_mask = (gen_tokens != pad_id)

#     return pos_log_probs, loss_mask


def aggregate_outputs(all_outputs, question_lens):
    """
    Aggregate a list of model output embeddings for use in log prob computation.

    Args:
        all_outputs: list of (B, L_i, hidden_size) tensors from different batches
        question_lens: list of lengths of the question portion for each example in all_outputs
    Returns:
        (B, max_L, hidden_size) tensor, where all outputs are left-padded to max_L so that prompts align at the same index.
    """
    max_len = max(out.shape[1] for out in all_outputs)
    padded_outputs = []
    for out in all_outputs:
        pad_len = max_len - out.shape[1]
        if pad_len > 0:
            padding = torch.zeros(out.shape[0], pad_len, out.shape[2], device=out.device)
            padded_out = torch.cat([out, padding], dim=1)
        else:
            padded_out = out
        padded_outputs.append(padded_out)
    
    return torch.cat(padded_outputs, dim=0)  # (N, max_len, hidden_size)


def save_model(parallel_model, save_path):
    state_dict = {k: v.detach().cpu().clone()
                  for k, v in parallel_model.module.state_dict().items()}
    torch.save({"model_state_dict": state_dict}, save_path)
    print(f"Saved model checkpoint at {save_path}!")


def ddp_reduce_scalar(value, device, op=dist.ReduceOp.SUM, dtype=torch.float64):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype)
    dist.all_reduce(tensor, op=op)
    return tensor.item()


def ddp_global_mean(local_sum, local_count, device):
    total_sum = ddp_reduce_scalar(local_sum, device=device, op=dist.ReduceOp.SUM, dtype=torch.float64)
    total_count = ddp_reduce_scalar(local_count, device=device, op=dist.ReduceOp.SUM, dtype=torch.long)
    return total_sum / max(total_count, 1)


def ddp_gather_list(local_items):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, list(local_items))
    merged = []
    for items in gathered:
        merged.extend(items)
    return merged


def main():
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Init ####################################################################################################################
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = rank == 0

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if is_main_process:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    configs.__dict__["start_time"] = current_time
    save_dir = os.path.join(configs.save_path, configs.name, current_time)

    if not os.path.exists(save_dir) and is_main_process:
        os.makedirs(save_dir)

    torch.distributed.barrier(device_ids=[local_rank])

    if is_main_process:
        import shutil
        for _src in [__file__, os.path.join(os.path.dirname(__file__), "coconut_grpo_emahead.py"), os.path.join(os.path.dirname(__file__), "args", "gsm_vllr_grpo_emahead.yaml")]:
            shutil.copy2(_src, os.path.join(save_dir, os.path.basename(_src)))

    cur_ckpts = os.listdir(save_dir)


    # Check if resume ##########################################################################################################
    # # check if the job is preempted and resumed.  - NOT used!
    # if len([f for f in cur_ckpts if not f.endswith("txt") and not f.endswith("pt")]) > 0 and not configs.only_eval and configs.resume == 0:      # configured to ignore txt logs
    #     # if there are previous checkpoints, and only_eval is False
    #     # it means the previous run was preempted and the program is restarted.
    #     # need to find the latest checkpoint and resume from that.

    #     if rank == 0:
    #         print(
    #             f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
    #         )

    #     checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
    #     checkpoints.sort(key=lambda x: int(x.split("_")[1]))

    #     # Get the last item in the sorted list
    #     latest_checkpoint = checkpoints[-1] if checkpoints else None
    #     configs.resume = int(latest_checkpoint.split("_")[1])
    #     load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

    #     configs.load_model_path = load_dir
    #     print(f"Loading from previous run epoch_{configs.resume}!")

    if configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    # Define model & things ####################################################################################################
    model_config = AutoConfig.from_pretrained(configs.model_id)
    model_config.attention_dropout = configs.dropout
    if configs.sdpa_attention:
        if configs.bf16:
            model = AutoModelForCausalLM.from_pretrained(configs.model_id, config=model_config, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(configs.model_id, config=model_config, attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(configs.model_id, config=model_config)
    if configs.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if is_main_process:
        print("Tokenizer IDs for special tokens:")
        print(f"BOS token ID: {tokenizer.bos_token_id}")
        print(f"EOS token ID: {tokenizer.eos_token_id}")
        print(f"latent_id: {latent_id}")
        print(f"start_id: {start_id}")
        print(f"end_id: {end_id}")

    loaded = False

    previous_savepoint = None
    if configs.load_model_path != "None":
        saved_checkpoint = torch.load(
            configs.load_model_path, map_location='cpu'
        )
        saved_weights = saved_checkpoint["model_state_dict"]
        previous_savepoint = configs.load_model_path
        # saved_weights = torch.load(configs.load_model_path, map_location=torch.device(rank))

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id] 
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        ref_base_model = copy.deepcopy(model)
        ref_model = Coconut(ref_base_model, latent_id, start_id, end_id, tokenizer.eos_token_id,
                            configs.termination_gamma, ema_decay=configs.ema_decay, bottleneck_ratio=configs.bottleneck_ratio)
        if configs.bf16:
            ref_model = ref_model.to(dtype=torch.bfloat16)
        ref_model = ref_model.to(device=device)
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id,
                        configs.termination_gamma, ema_decay=configs.ema_decay, bottleneck_ratio=configs.bottleneck_ratio)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    if is_main_process:
        print(f"Running DDP on rank = {rank}, local_rank = {local_rank}, world size = {world_size}")
    model = model.to(device)

    if configs.bf16:
        model.to(torch.bfloat16)

    assert ref_model is not None, "ref_model is not defined"

    parallel_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    del model

    if is_main_process:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        total_train_samples = 3200 if configs.debug else int(configs.num_steps * configs.train_batch_size * 1.5)

        # Mixed dataset loading: primary (e.g. MATH) + secondary (e.g. GSM8K)
        train_path_secondary = getattr(configs, 'train_path_secondary', 'None')
        dataset_mix_ratio = getattr(configs, 'dataset_mix_ratio', 1.0)  # fraction from primary

        if train_path_secondary != 'None' and dataset_mix_ratio < 1.0:
            n_primary = int(total_train_samples * dataset_mix_ratio)
            n_secondary = total_train_samples - n_primary

            base_dataset_primary = get_dataset(configs.train_path, tokenizer, max_size=n_primary)
            base_dataset_secondary = get_dataset(train_path_secondary, tokenizer, max_size=n_secondary)

            from datasets import concatenate_datasets
            base_dataset_train = concatenate_datasets([base_dataset_primary, base_dataset_secondary])

            if is_main_process:
                print(f"Mixed dataset: {len(base_dataset_primary)} from {configs.train_path} "
                      f"+ {len(base_dataset_secondary)} from {train_path_secondary} "
                      f"= {len(base_dataset_train)} total")
            del base_dataset_primary, base_dataset_secondary
        else:
            base_dataset_train = get_dataset(
                configs.train_path, tokenizer, max_size=total_train_samples
            )

    # if "gsm" in configs.val_path:
    #     # max_new_tokens = 64                   # change
    #     max_new_tokens = 128
    # else:
    #     max_new_tokens = 128
    max_new_tokens = configs.max_new_tokens


    if not configs.debug and not configs.only_eval and is_main_process:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        # if configs.resume != 0:
        #     optimizer = bnb.optim.Adam8bit.load_state_dict(saved_checkpoint["optimizer_state_dict"])
        # else:
        optimizer = bnb.optim.Adam8bit(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    # Cosine LR scheduler with linear warmup.
    # Warmup: linearly ramp from 0 → lr over warmup_steps.
    # Then cosine decay to eta_min = 10% of peak lr over the remaining steps.
    warmup_steps = getattr(configs, 'warmup_steps', 100)
    total_steps = getattr(configs, 'num_steps', 2000)
    eta_min = configs.lr * 0.1

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / configs.lr, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)


    for epoch in range(configs.resume, configs.num_epochs):
        # Remove old datasets
        if 'train_dataloader' in locals():
            del train_dataloader
        if 'dataset_train' in locals():
            del dataset_train
        if 'valid_gen_dataloader' in locals():
            del valid_gen_dataloader
        if 'dataset_gen_val' in locals():
            del dataset_gen_val
        gc.collect()

        # Load validation dataset
        dataset_gen_val = get_grpo_dataset(
            base_dataset_valid,
            start_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts or configs.no_bot_tokens,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            # batch_size=1,
            batch_size=configs.eval_batch_size,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )
        if isinstance(valid_gen_dataloader.sampler, DistributedSampler):
            valid_gen_dataloader.sampler.set_epoch(epoch)

        # Load training dataset
        if not configs.only_eval:
            dataset_train = get_grpo_dataset(
                base_dataset_train,
                start_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts or configs.no_bot_tokens,
                shuffle=True,
                max_question_len=128,
                num_samples=configs.num_steps*configs.train_batch_size
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                # batch_size=configs.batch_size_training,
                batch_size=configs.train_batch_size,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if configs.reset_optimizer:
                del optimizer

                # optimizer = optim.AdamW(
                #     parallel_model.parameters(),
                #     lr=configs.lr,
                #     weight_decay=configs.weight_decay,
                # )
                optimizer = bnb.optim.Adam8bit(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            # Begin training #####################################################################################
            total_length = len(train_dataloader)
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
                disable=not is_main_process,
            )
            total_loss = 0.0

            ref_model_path = configs.load_model_path
            ref_save_dir = os.path.join(save_dir, "ref_models")
            os.makedirs(ref_save_dir, exist_ok=True)

            best_accuracy = 0.0

            # Log to file for debugging
            log_path = os.path.join(save_dir, f"checkpoint_{epoch+1}_rank{rank}.log")

            # Running average of n_latent from the previous step, used to set the
            # forced-rollout minimum (= 2x this value).  Seeded from configs.forced_latent_init.
            avg_n_latent_last_batch = float(configs.forced_latent_init)

            for step, batch in enumerate(train_dataloader):
                # batch = dict(["input_ids", "attention_mask", "position_ids", "idx", "answer"])
                # answer is list[int]

                if (step + 1) % 100 == 0 and is_main_process:
                    print(f"Max Reserved:     {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                    print(f"Max Allocated:    {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

                if step == 25:
                    best_accuracy = 0.0     # reset best accuracy to ensure model saves occur after the initial epochs


                # Replace reference model every N steps
                if (step + 1) % configs.ref_model_renewal_steps == 0:
                    # Save current model
                    ref_model_path = os.path.join(ref_save_dir, f"checkpoint_{epoch}_{step}.pt")
                    print("Replacing reference model...")
                    save_model(parallel_model, ref_model_path)

                    # Replace ref model weights
                    state_dict = {k: v.detach().clone()
                                  for k, v in parallel_model.module.state_dict().items()}
                    ref_model.load_state_dict(state_dict, strict=False)
                    del state_dict
                    print("Reference model loaded")

                all_rewards = []
                all_base_rewards = []
                all_token_ids = []
                all_prompt_ids = []
                all_prompt_masks = []
                all_rollout_rng_states = []
                all_correct = 0
                all_total = 0

                # Generate Rollouts ######################################################################################################################################################
                # for i in range(configs.num_rollouts):
                # Changed from generating batch_size * num_rollout to setting batch_size=1 and generating 16 rollouts per step. This is primarily due to GPU restrictions.
                start_timer()   #timer

                # Phase 1: Rollout
                parallel_model.module.train()       # enables dropout for diverse reasoning trajectories
                
                # modify input to batch * num_rollouts, by repeating each input
                batch_len, len_question = batch["input_ids"].shape
                batch["input_ids"] = torch.repeat_interleave(batch["input_ids"], configs.num_rollouts, dim=0)
                batch["attention_mask"] = torch.repeat_interleave(batch["attention_mask"], configs.num_rollouts, dim=0)
                answers = [ans for ans in batch["answer"] for _ in range(configs.num_rollouts)]     # answers = batch["answer"]       # [ans1, ans2, ...]

                blacklist = {"idx", "answer"}
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key not in blacklist
                }

                # Build forced-rollout minimum latent tensor.
                # The last forced_rollout_fraction of rollouts per question are forced to generate at least 2x the previous batch's average n_latent.
                num_forced_per_q = max(1, int(configs.num_rollouts * configs.forced_rollout_fraction))
                forced_min = min(int(2 * avg_n_latent_last_batch), max_new_tokens - 8)              # cap at max_new_tokens - 8 to prevent context explosion
                forced_min_latents_vec = torch.zeros(batch_len * configs.num_rollouts, dtype=torch.long, device=device)
                for q_idx in range(batch_len):
                    start_forced = q_idx * configs.num_rollouts + (configs.num_rollouts - num_forced_per_q)
                    forced_min_latents_vec[start_forced : (q_idx + 1) * configs.num_rollouts] = forced_min

                # policy_minibatch_size controls both rollout generation chunk size AND the
                # policy backward minibatch size.  They must be equal: the RNG state is
                # captured once per chunk, so the replay batch must be identical to the
                # rollout batch (batch size affects per-step RNG consumption via dropout).
                # Smaller values reduce backward VRAM at the cost of rollout throughput.
                rollout_chunk_size = getattr(configs, 'policy_minibatch_size', configs.train_minibatch_size * 8)
                n_latents_correct_sum = 0
                n_latents_correct_num = 0

                measure_timer("Batch prep") #timer

                with torch.no_grad():       # Start generation
                    for rollout_start in range(0, batch["input_ids"].shape[0], rollout_chunk_size):         # generate in mini-batches
                        rollout_end = min(rollout_start + rollout_chunk_size, batch["input_ids"].shape[0])
                        prompt_chunk = batch["input_ids"][rollout_start:rollout_end]
                        attn_chunk = batch["attention_mask"][rollout_start:rollout_end]
                        answers_chunk = answers[rollout_start:rollout_end]
                        forced_min_chunk = forced_min_latents_vec[rollout_start:rollout_end]
                        rng_state = capture_rng_state(local_rank)
                        generated = parallel_model.module.generate_batched(
                            prompt_chunk,
                            attention_mask=attn_chunk,
                            max_new_tokens=max_new_tokens,
                            output_embedding=False,
                            synced_gpus=True,
                            term_temperature=configs.term_temperature,
                            forced_min_latents=forced_min_chunk,
                        )

                        rewards, base_rewards, step_accuracy, ans_chunk, accuracies, n_latents, n_latents_correct = compute_reward(
                            generated,
                            tokenizer,
                            answers_chunk,
                            latent_id,
                            length_penalty=configs.length_penalty_per_token,
                            format_penalty=getattr(configs, 'format_penalty', -0.5),
                            latent_step_reward=getattr(configs, 'latent_step_reward', 0.0),
                            coconut_mode=configs.coconut,
                        )

                        n_latents_correct_sum += sum(n_latents_correct)
                        n_latents_correct_num += len(n_latents_correct)

                        all_rewards.extend(rewards)
                        all_base_rewards.extend(base_rewards)
                        all_token_ids.append(generated)
                        all_prompt_ids.append(prompt_chunk)
                        all_prompt_masks.append(attn_chunk)
                        all_rollout_rng_states.append(rng_state)
                        all_correct += sum(accuracies)
                        all_total += len(accuracies)
                    
                    measure_timer("Rollout generation") #timer

                    avg_reward = sum(all_rewards) / len(all_rewards)
                    avg_n_latent_last_batch = sum(
                        (token_ids == latent_id).sum().item() for token_ids in all_token_ids
                    ) / max(len(all_rewards), 1)
                    global_avg_reward = ddp_global_mean(sum(all_rewards), len(all_rewards), device)
                    global_avg_base_reward = ddp_global_mean(sum(all_base_rewards), len(all_base_rewards), device)
                    global_accuracy = ddp_global_mean(all_correct, all_total, device)
                    global_avg_n_latent = ddp_global_mean(
                        sum((token_ids == latent_id).sum().item() for token_ids in all_token_ids),
                        len(all_rewards),
                        device,
                    )
                    global_n_latent_correct = ddp_global_mean(
                        n_latents_correct_sum,
                        n_latents_correct_num,
                        device,
                    )

                    if wandb_run and is_main_process:
                        log_dict = {
                            "train/batch_avg_reward": global_avg_reward,
                            "train/batch_avg_base_reward": global_avg_base_reward,
                            "train/batch_accuracy": global_accuracy,
                            "train/batch_n_latent_avg": global_avg_n_latent,
                            "train/batcn_n_latent_correct_avg": global_n_latent_correct,
                            "train/forced_min_latents": forced_min,
                        }
                        wandb_run.log(log_dict)

                    # Right-pad all_token_ids to the longest length before concatenating
                    max_len = max(token_ids.shape[1] for token_ids in all_token_ids)
                    padded_token_ids = []
                    for token_ids in all_token_ids:
                        pad_len = max_len - token_ids.shape[1]
                        if pad_len > 0:
                            padded = torch.nn.functional.pad(token_ids, (0, pad_len), value=tokenizer.eos_token_id)
                        else:
                            padded = token_ids
                        padded_token_ids.append(padded)
                    
                    measure_timer("Pad and concat for logging") #timer
                    
                    generated_all = torch.cat(padded_token_ids, dim=0)
                    with open(log_path, "a") as f:
                        f.write(f"Step {step}, Average Reward: {avg_reward:.3f}\n")

                        for i in range(batch_len):
                            for j in range(configs.num_rollouts):
                                idx_ans = i * configs.num_rollouts + j
                                decoded_question = tokenizer.decode(batch["input_ids"][idx_ans].cpu())
                                decoded_generated = tokenizer.decode(generated_all[idx_ans, len_question:])
                                f.write(f"Question RAW: {decoded_question}\n")
                                f.write(f"Generated RAW: {decoded_generated}\n")
                                f.write(f"Question: {decoded_question.replace('<|endoftext|>', '')}\n")
                                f.write(f"Generated: {decoded_generated.replace('<|endoftext|>', '')}\n")
                                f.write(f"Predicted answer: '{extract_answer(decoded_generated, tokenizer.eos_token)}'\n")
                                f.write(f"Ground Truth: {answers[idx_ans]}\n")
                                f.write(f"Reward: {all_rewards[idx_ans]:.2f}\n")
                                f.write("\n")
                            f.write("\n\n\n")

                        f.write("\n" * 5 + "-" * 100 + "\n" * 5)
                    del generated_all

                    measure_timer("Finish logging") #timer

                # Calculate overall accuracy, save model if better than previous best
                overall_accuracy = global_accuracy

                # if overall_accuracy > best_accuracy:
                #     best_accuracy = overall_accuracy
                #     if rank == 0:
                #         save_model(parallel_model, os.path.join(ref_save_dir, f"best_model_step{step}_{overall_accuracy:.2f}"))

                # Policy ####################################################################################################################
                all_losses = []
                all_rewards_raw = list(all_rewards)
                prompt_ids_cpu = torch.cat(all_prompt_ids, dim=0).to(device)
                prompt_masks_cpu = torch.cat(all_prompt_masks, dim=0).to(device)

                # Reuse padded_token_ids already built for logging — no second pass needed.
                token_ids_cpu = torch.cat(padded_token_ids, dim=0).to(device)
                token_ids = token_ids_cpu
                all_rewards = torch.tensor(all_rewards, device=device, dtype=torch.float32)

                advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

                with torch.no_grad():       # Reference model rollout
                    ref_model.eval()
                    ref_lp_chunks = []
                    ref_mask_chunks = []
                    for mb_start in range(0, token_ids_cpu.shape[0], rollout_chunk_size):
                        mb_end = min(mb_start + rollout_chunk_size, token_ids_cpu.shape[0])
                        ref_outputs = ref_model(
                            input_ids=prompt_ids_cpu[mb_start:mb_end],
                            attention_mask=prompt_masks_cpu[mb_start:mb_end],
                            replay_generated_ids=token_ids_cpu[mb_start:mb_end],
                        )
                        ref_lp_chunk, ref_mask_chunk = compute_log_probs(
                            token_ids[mb_start:mb_end],
                            ref_outputs,
                            len_question,
                            latent_id,
                            tokenizer.eos_token_id,
                        )
                        ref_lp_chunks.append(ref_lp_chunk)
                        ref_mask_chunks.append(ref_mask_chunk)
                        del ref_outputs

                    ref_lp_per_token = torch.cat(ref_lp_chunks, dim=0)
                    ref_loss_mask = torch.cat(ref_mask_chunks, dim=0)
                
                measure_timer("Reference rollout")  #timer

                total_rollouts  = token_ids.shape[0]
                train_mb_size   = rollout_chunk_size
                total_nonmasked = ref_loss_mask.sum().clamp(min=1)
                max_log_ratio   = torch.tensor(0.0, device=device)
                step_loss       = 0.0

                # Diagnostic accumulators: decompose log-ratio by latent vs text positions
                diag_latent_lr_sum  = torch.tensor(0.0, device=device)
                diag_latent_lr_max  = torch.tensor(0.0, device=device)
                diag_latent_count   = torch.tensor(0, dtype=torch.long, device=device)
                diag_text_lr_sum    = torch.tensor(0.0, device=device)
                diag_text_lr_max    = torch.tensor(0.0, device=device)
                diag_text_count     = torch.tensor(0, dtype=torch.long, device=device)
                diag_kl_loss_sum    = torch.tensor(0.0, device=device)
                diag_policy_loss_sum = torch.tensor(0.0, device=device)

                optimizer.zero_grad()

                parallel_model.module.train()
                for mb_start in range(0, total_rollouts, train_mb_size):
                    mb_end      = min(mb_start + train_mb_size, total_rollouts)
                    restore_rng_state(all_rollout_rng_states[mb_start // train_mb_size], local_rank)
                    mb_prompt   = prompt_ids_cpu[mb_start:mb_end].to(device)
                    mb_mask     = prompt_masks_cpu[mb_start:mb_end].to(device)
                    mb_tok      = token_ids[mb_start:mb_end]
                    mb_adv      = advantages[mb_start:mb_end]

                    mb_ref_lp   = ref_lp_per_token[mb_start:mb_end]
                    mb_ref_mask = ref_loss_mask[mb_start:mb_end]

                    mb_new_outputs = parallel_model(
                        input_ids=mb_prompt,
                        attention_mask=mb_mask,
                        replay_generated_ids=mb_tok,
                        term_temperature=configs.term_temperature,
                    )
                    mb_new_lp, _   = compute_log_probs(
                        mb_tok, mb_new_outputs, len_question, latent_id, tokenizer.eos_token_id
                    )
                    del mb_new_outputs

                    mb_log_ratio = mb_new_lp - mb_ref_lp
                    mb_log_ratio_clamped = mb_log_ratio.clamp(-5, 5)        # prevent overflow
                    mb_ratio     = torch.exp(mb_log_ratio_clamped)
                    # mb_clipped   = mb_ratio
                    mb_clipped   = torch.clamp(mb_ratio, 1.0 - configs.clip_ratio_lower_bound, 1.0 + configs.clip_ratio_upper_bound)

                    t1 = mb_ratio   * mb_adv.unsqueeze(-1)
                    t2 = mb_clipped * mb_adv.unsqueeze(-1)
                    mb_policy_loss = -torch.min(t1, t2)

                    mb_kl         = mb_ratio - mb_log_ratio_clamped - 1.0
                    mb_total_loss = mb_policy_loss + configs.kl_beta * mb_kl
                    # Divide by global token count so gradients accumulate to the same total as a single-batch pass
                    mb_loss       = (mb_total_loss * mb_ref_mask).sum() / total_nonmasked

                    mb_loss.backward()
                    step_loss    += mb_loss.item()
                    max_log_ratio = torch.max(max_log_ratio, mb_log_ratio.detach().abs().max())

                    # Diagnostic: separate KL vs policy loss contributions
                    with torch.no_grad():
                        diag_policy_loss_sum += (mb_policy_loss * mb_ref_mask).sum() / total_nonmasked
                        diag_kl_loss_sum     += (configs.kl_beta * mb_kl * mb_ref_mask).sum() / total_nonmasked

                    # Diagnostic: decompose log-ratios by latent vs text positions
                    with torch.no_grad():
                        mb_gen = mb_tok[:, len_question:]                         # (mb, gen_len)
                        mb_is_latent = (mb_gen == latent_id) & mb_ref_mask        # (mb, gen_len)
                        mb_is_text   = (~(mb_gen == latent_id)) & mb_ref_mask
                        mb_lr_abs    = mb_log_ratio.detach().abs()

                        if mb_is_latent.any():
                            diag_latent_lr_sum  += mb_lr_abs[mb_is_latent].sum()
                            diag_latent_lr_max   = torch.max(diag_latent_lr_max, mb_lr_abs[mb_is_latent].max())
                            diag_latent_count   += mb_is_latent.sum()
                        if mb_is_text.any():
                            diag_text_lr_sum    += mb_lr_abs[mb_is_text].sum()
                            diag_text_lr_max     = torch.max(diag_text_lr_max, mb_lr_abs[mb_is_text].max())
                            diag_text_count     += mb_is_text.sum()
                
                measure_timer("Policy Minibatch generation")

                all_losses.append(step_loss)
                torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                total_loss = sum(all_losses) / len(all_losses)
                global_total_loss = ddp_reduce_scalar(total_loss, device=device, op=dist.ReduceOp.SUM, dtype=torch.float64) / world_size
                global_max_log_ratio = ddp_reduce_scalar(max_log_ratio, device=device, op=dist.ReduceOp.MAX, dtype=torch.float64)
                global_diag_latent_lr_max = ddp_reduce_scalar(diag_latent_lr_max, device=device, op=dist.ReduceOp.MAX, dtype=torch.float64)
                global_diag_text_lr_max = ddp_reduce_scalar(diag_text_lr_max, device=device, op=dist.ReduceOp.MAX, dtype=torch.float64)
                global_diag_latent_lr_sum = ddp_reduce_scalar(diag_latent_lr_sum, device=device, op=dist.ReduceOp.SUM, dtype=torch.float64)
                global_diag_text_lr_sum = ddp_reduce_scalar(diag_text_lr_sum, device=device, op=dist.ReduceOp.SUM, dtype=torch.float64)
                global_diag_latent_count = ddp_reduce_scalar(diag_latent_count, device=device, op=dist.ReduceOp.SUM, dtype=torch.long)
                global_diag_text_count = ddp_reduce_scalar(diag_text_count, device=device, op=dist.ReduceOp.SUM, dtype=torch.long)
                global_diag_policy_loss = ddp_reduce_scalar(diag_policy_loss_sum, device=device, op=dist.ReduceOp.SUM, dtype=torch.float64) / world_size
                global_diag_kl_loss = ddp_reduce_scalar(diag_kl_loss_sum, device=device, op=dist.ReduceOp.SUM, dtype=torch.float64) / world_size
                global_zero_reward_frac = ddp_global_mean(
                    sum(1 for r in all_rewards_raw if r <= 0),
                    len(all_rewards_raw),
                    device,
                )
                global_zero_base_reward_frac = ddp_global_mean(
                    sum(1 for r in all_base_rewards if r <= 0),
                    len(all_base_rewards),
                    device,
                )
                global_step_avg_base_reward = ddp_global_mean(
                    sum(all_base_rewards),
                    len(all_base_rewards),
                    device,
                )

                # Free large rollout tensors accumulated this step
                del token_ids_cpu, token_ids, prompt_ids_cpu, prompt_masks_cpu, all_token_ids, all_prompt_ids, all_prompt_masks, all_rewards
                gc.collect()
                torch.cuda.empty_cache()

                pbar.update(1)


                ###########################################################################################################################################
                if step == 1 and is_main_process:
                    print_memory_breakdown(parallel_model, optimizer)     # use to check how much VRAM is being used

                if wandb_run and is_main_process:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": step + 1,
                        "train/loss": global_total_loss,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/max_log_ratio": global_max_log_ratio,
                        # Diagnostic: log-ratio decomposition
                        "diag/latent_max_log_ratio": global_diag_latent_lr_max,
                        "diag/latent_mean_log_ratio": global_diag_latent_lr_sum / max(global_diag_latent_count, 1),
                        "diag/text_max_log_ratio": global_diag_text_lr_max,
                        "diag/text_mean_log_ratio": global_diag_text_lr_sum / max(global_diag_text_count, 1),
                        # Diagnostic: KL vs policy loss decomposition
                        "diag/policy_loss": global_diag_policy_loss,
                        "diag/kl_loss": global_diag_kl_loss,
                        "diag/kl_to_policy_ratio": global_diag_kl_loss / max(abs(global_diag_policy_loss), 1e-8),
                        # Diagnostic: reward sparsity (fraction of rollouts with zero or negative reward)
                        "diag/zero_reward_frac": global_zero_reward_frac,
                        "diag/zero_base_reward_frac": global_zero_base_reward_frac,
                        "diag/step_avg_base_reward": global_step_avg_base_reward,
                    }
                    wandb_run.log(log_dict)

                if is_main_process:
                    pbar.set_description(
                        f"GRPO Epoch: {epoch+1}/{configs.num_epochs}, "
                        f"step {step}/{len(train_dataloader)} "
                        f"(loss: {global_total_loss:.4f})"
                    )

                # Evaluation step #########################################################################################################################
                if (step + 1) % configs.eval_per_steps == 0:
                    # val generation accuracy
                    total_length = len(valid_gen_dataloader)

                    # pbar = tqdm(
                    #     colour="blue", desc="Test Accuracy",
                    #     total=len(valid_gen_dataloader), dynamic_ncols=True,
                    # )
                    # cor, cor_cot, total = (
                    #     torch.tensor(0, device=rank),
                    #     torch.tensor(0, device=rank),
                    #     torch.tensor(0, device=rank),
                    # )
                    log_path = os.path.join(save_dir, f"checkpoint_{epoch+1}_{step}_eval_rank{rank}.log")

                    with torch.no_grad():
                        parallel_model.module.eval()
                        all_correct = 0
                        all_total = 0
                        all_rewards = []
                        all_n_latents = []

                        for idx, batch in enumerate(valid_gen_dataloader):
                            test_idx = batch["idx"]

                            batch_len, len_question = batch["input_ids"].shape
                            answers = batch["answer"]

                            blacklist = {"idx", "answer"}
                            batch = {
                                key: batch[key].to(device) for key in batch.keys() if key not in blacklist
                            }

                            generated, embeddings, model_outputs = parallel_model.module.generate_batched(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                max_new_tokens=max_new_tokens,
                                output_embedding=True,
                                synced_gpus=True,
                                term_temperature=0.0
                            )
                            del model_outputs

                            # Compute rewards
                            rewards, base_rewards, step_accuracy, ans, accuracies, n_latents, n_latents_correct = compute_reward(
                                generated,
                                tokenizer,
                                answers,
                                latent_id,
                                length_penalty=configs.length_penalty_per_token,
                                format_penalty=getattr(configs, 'format_penalty', -0.5),
                                latent_step_reward=getattr(configs, 'latent_step_reward', 0.0),
                                coconut_mode=configs.coconut,
                            )
                            all_correct += sum(accuracies)
                            all_total += len(accuracies)
                            all_rewards.extend(rewards)
                            all_n_latents.extend(n_latents)

                            avg_reward = sum(rewards) / len(rewards)
                            avg_latent = sum(n_latents) / len(n_latents)

                            # if wandb_run and rank == 0:
                            #     log_dict = {
                            #         "eval/batch_avg_reward": avg_reward,
                            #         "eval/batch_accuracy": step_accuracy,
                            #         "eval/batch_n_latent": sum(n_latents)/len(n_latents),
                            #     }
                            #     wandb_run.log(log_dict)


                            # Log to file for debugging
                            with open(log_path, "a") as f:
                                f.write(f"Step {step}, Average Reward: {avg_reward:.3f}, Average n_latents: {avg_latent:.2f}\n")

                                for i in range(batch_len):
                                    f.write(f"Question RAW: {tokenizer.decode(batch['input_ids'][i])}\n")
                                    f.write(f"Generated RAW: {tokenizer.decode(generated[i, len_question:])}\n")
                                    f.write(f"Question: {tokenizer.decode(batch['input_ids'][i]).replace("<|endoftext|>", "")}\n")
                                    f.write(f"Generated: {tokenizer.decode(generated[i, len_question:]).replace("<|endoftext|>", "")}\n")
                                    f.write(f"Predicted answer: '{ans[i]}'\n")
                                    f.write(f"Ground Truth: {answers[i]}\n")
                                    f.write(f"Reward: {rewards[i]:.2f}\n")
                                    f.write("\n")


                            # pbar.update(1)
                            # pbar.set_description(
                            #     f"Test accuracy: {step_accuracy}"
                            # )
                        
                        eval_accuracy = ddp_global_mean(all_correct, all_total, device)
                        eval_reward = ddp_global_mean(sum(all_rewards), len(all_rewards), device)
                        eval_avg_latent = ddp_global_mean(sum(all_n_latents), len(all_n_latents), device)

                        if is_main_process and eval_accuracy > best_accuracy:
                            best_accuracy = eval_accuracy
                            
                            eval_savepath = os.path.join(
                                save_dir, "ref_models", f"eval_best_model_{epoch+1}_{step+1}_{best_accuracy:.2f}.pt"
                            )
                            save_model(parallel_model, eval_savepath)

                    if wandb_run and is_main_process:
                        wandb_run.log({
                            "eval/accuracy": eval_accuracy,
                            "eval/reward": eval_reward,
                            "eval/avg_n_latent": eval_avg_latent
                            })

                    log_path = os.path.join(save_dir, f"checkpoint_{epoch+1}.log")

            pbar.close()
            dist.barrier()

            # Save model after checkpoint
            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):

                if is_main_process:
                    model_save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}_{current_time}.pt")
                    print(f"Saving model after epoch {epoch+1}...")
                    save_model(parallel_model, model_save_path)

                # checkpoint = {
                #     "epoch": epoch,
                #     "model_state_dict": parallel_model.state_dict(),
                #     # "optimimzer_state_dict": optimizer.state_dict()
                # }
                # del checkpoint

                dist.barrier()
                gc.collect()
                torch.cuda.empty_cache()


        # Test step ######################################################################################################################################################################
        total_length = len(valid_gen_dataloader)
        question_idx = 0

        pbar = tqdm(
            colour="blue", desc="Test Accuracy",
            total=len(valid_gen_dataloader), dynamic_ncols=True, disable=not is_main_process,
        )

        log_path = os.path.join(save_dir, f"Test_Evaluation_rank{rank}.log")

        with torch.no_grad():
            parallel_model.module.eval()
            all_correct = 0
            all_total = 0
            all_rewards = []
            all_n_latents = []
            all_n_latents_correct = []
            all_n_latents_incorrect = []
            all_n_latents_noformat = []

            if configs.save_pca_figures:
                pca_variances_correct = []
                pca_variances_incorrect = []
                pca_variances_noformat = []

            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"]

                batch_len, len_question = batch["input_ids"].shape
                answers = batch["answer"]

                blacklist = {"idx", "answer"}
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key not in blacklist
                }

                if configs.forced_n_latent == 0:
                    generated, embeddings, model_outputs = parallel_model.module.generate_batched(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        output_embedding=True,
                        synced_gpus=True,
                        term_temperature=0.0,
                    )
                else:
                    generated, embeddings, model_outputs = parallel_model.module.generate_batched_n(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        num_latent=configs.forced_n_latent,
                        output_embedding=True,
                        synced_gpus=True,
                    )

                del model_outputs

                # Compute rewards
                rewards, base_rewards, step_accuracy, ans, correctness, n_latents, n_latents_correct = compute_reward(
                    generated,
                    tokenizer,
                    answers,
                    latent_id,
                    length_penalty=configs.length_penalty_per_token,
                    format_penalty=getattr(configs, 'format_penalty', -0.5),
                    latent_step_reward=getattr(configs, 'latent_step_reward', 0.0),
                    coconut_mode=configs.coconut,
                )
                all_correct += sum(correctness)
                all_total += len(correctness)
                all_rewards.extend(rewards)
                all_n_latents.extend(n_latents)
                for i in range(len(n_latents)):
                    if correctness[i] > 0.5:
                        all_n_latents_correct.append(n_latents[i])
                    elif len(ans[i]) > 20:
                        all_n_latents_noformat.append(n_latents[i])
                    else:
                        all_n_latents_incorrect.append(n_latents[i])

                avg_reward = sum(rewards) / len(rewards)
                avg_latent = sum(n_latents) / len(n_latents)

                # Log to file for debugging
                with open(log_path, "a") as f:
                    f.write(f"Average Reward: {avg_reward:.3f}, Average n_latents: {avg_latent:.2f}\n")

                    for i in range(batch_len):
                        f.write(f"Question RAW: {tokenizer.decode(batch['input_ids'][i])}\n")
                        f.write(f"Generated RAW: {tokenizer.decode(generated[i, len_question:])}\n")
                        f.write(f"Question: {tokenizer.decode(batch['input_ids'][i]).replace("<|endoftext|>", "")}\n")
                        f.write(f"Generated: {tokenizer.decode(generated[i, len_question:]).replace("<|endoftext|>", "")}\n")
                        f.write(f"Predicted answer: '{ans[i]}'\n")
                        f.write(f"Ground Truth: {answers[i]}\n")
                        f.write(f"Reward: {rewards[i]:.2f}\n")
                        f.write("\n")


                eval_accuracy = ddp_global_mean(all_correct, all_total, device)
                if is_main_process:
                    pbar.update(1)
                    pbar.set_description(
                        f"Test accuracy: {eval_accuracy:.2%}"
                    )

                # Plot to PCA
                if configs.save_pca_figures:
                    dir_plot_correct = os.path.join(save_dir, "plots", f"rank_{rank}", "correct")
                    dir_plot_incorrect = os.path.join(save_dir, "plots", f"rank_{rank}", "incorrect")
                    dir_plot_noformat = os.path.join(save_dir, "plots", f"rank_{rank}", "wrong format")
                    os.makedirs(dir_plot_correct, exist_ok=True)
                    os.makedirs(dir_plot_incorrect, exist_ok=True)
                    os.makedirs(dir_plot_noformat, exist_ok=True)

                    for i in range(batch_len):
                        dir_plot_save = dir_plot_correct if correctness[i] > 0.5 else dir_plot_incorrect
                        dir_plot_save = dir_plot_noformat if len(ans[i]) > 20 else dir_plot_save

                        plot_question = tokenizer.decode(batch['input_ids'][i]).replace("<|endoftext|>", "")
                        plot_answer = answers[i]
                        plot_generated_answer = ans[i] if ans[i].find("latent") == -1 else "Wrong format"
                        plot_reward = rewards[i]

                        latent_mask = generated[i] == latent_id  # (batch_size, seq_len) boolean mask
                        latent_hidden_states = embeddings[i][latent_mask].to(float).cpu().numpy()  # (num_latent_tokens, hidden_dim)

                        pca, pca_var = visualize_latent_pca(latent_hidden_states, output_dir=dir_plot_save, question=plot_question, answer=plot_answer, predicted=plot_generated_answer, reward=plot_reward, idx=question_idx+i)

                        if correctness[i] > 0.5:
                            pca_variances_correct.append(pca_var)
                        elif len(ans[i]) > 20:
                            pca_variances_noformat.append(pca_var)
                        else:
                            pca_variances_incorrect.append(pca_var)
                        
                question_idx += batch_len

            eval_accuracy = ddp_global_mean(all_correct, all_total, device)
            eval_reward = ddp_global_mean(sum(all_rewards), len(all_rewards), device)
            eval_avg_latent = ddp_global_mean(sum(all_n_latents), len(all_n_latents), device)

            gathered_n_latents = ddp_gather_list(all_n_latents)
            gathered_n_latents_correct = ddp_gather_list(all_n_latents_correct)
            gathered_n_latents_incorrect = ddp_gather_list(all_n_latents_incorrect)
            gathered_n_latents_noformat = ddp_gather_list(all_n_latents_noformat)

            if configs.save_pca_figures:
                gathered_pca_variances_correct = ddp_gather_list(pca_variances_correct)
                gathered_pca_variances_incorrect = ddp_gather_list(pca_variances_incorrect)
                gathered_pca_variances_noformat = ddp_gather_list(pca_variances_noformat)

            if is_main_process:
                print(f"Evaluation Accuracy: {eval_accuracy:.2%}")
                print(f"Evaluation Rewards: {eval_reward:.3f}")
                print(f"Evaluation Average Latent Steps: {eval_avg_latent:.2f}")

                save_latent_histogram(
                    gathered_n_latents,
                    gathered_n_latents_correct,
                    gathered_n_latents_incorrect,
                    gathered_n_latents_noformat,
                    output_dir=os.path.join(save_dir, "plots"),
                )

                if configs.save_pca_figures:
                    if gathered_pca_variances_correct:
                        print(f"Correct Var: {sum(gathered_pca_variances_correct)/len(gathered_pca_variances_correct):.3f}")
                    if gathered_pca_variances_incorrect:
                        print(f"Incorrect Var: {sum(gathered_pca_variances_incorrect)/len(gathered_pca_variances_incorrect):.3f}")
                    if gathered_pca_variances_noformat:
                        print(f"Noformat Var: {sum(gathered_pca_variances_noformat)/len(gathered_pca_variances_noformat):.3f}")

        if wandb_run and is_main_process:
            wandb_run.log({
                "eval/accuracy": eval_accuracy,
                "eval/reward": eval_reward,
                "eval/avg_n_latent": eval_avg_latent
                })


        pbar.close()
        
        
        # print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        # dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        # dist.all_reduce(cor,     op=dist.ReduceOp.SUM)
        # dist.all_reduce(total,   op=dist.ReduceOp.SUM)

        # cor_cot, cor, total = cor_cot.item(), cor.item(), total.item()
        # if rank == 0:
        #     print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
        #     print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        # sys.stdout.flush()

        if wandb_run and is_main_process:
            wandb_run.log({
                # "eval/acc": cor / total,
                "eval/acc": eval_accuracy,
                "eval/reward": eval_reward
                })

        if configs.only_eval:
            break

        dist.barrier()

        # if (cor / total > best_acc
        #         and configs.save_only_improve
        #         and not configs.debug
        #         and not configs.only_eval):
        #     checkpoint = {
        #         "epoch":            epoch,
        #         "model_state_dict": parallel_model.state_dict(),
        #     }
        #     if rank == 0:
        #         # ckpt_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}_{current_time}.pt")
        #         # torch.save(checkpoint, ckpt_path)
        #         # print("saving model.")
        #         # previous_savepoint = ckpt_path

        #         model_save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}_{current_time}.pt")

        #         with FSDP.summon_full_params(parallel_model, writeback=False, rank0_only=False):
        #             state_dict = {k: v.detach().cpu().clone()
        #             for k, v in parallel_model.module.state_dict().items()}
        #         torch.save(
        #             {
        #                 "model_state_dict": state_dict,
        #                 # "optimizer_state_dict": optimizer.state_dict(),
        #             },
        #             model_save_path,
        #         )
        #         print(f"Saved reference model at {model_save_path}!")

        #     best_acc = cor / total

        #     dist.barrier()
        #     del checkpoint
        #     gc.collect()
        #     torch.cuda.empty_cache()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
