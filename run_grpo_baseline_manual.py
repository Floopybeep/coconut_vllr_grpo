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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from coconut_grpo_baseline_manual import Coconut
from dataset import (
    get_dataset,
    get_grpo_dataset,
    MyCollator,
)

import gc
import copy
import json
import yaml
import random
import datetime
import argparse
import itertools
import functools
import bitsandbytes as bnb
import matplotlib.pyplot as plt
from tqdm import tqdm
# from copy import copy
from utils import Config, set_seed, visualize_latent_pca, save_latent_histogram

import torch

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

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def extract_answer(text, eot_token):
    """Extract the numerical answer following '###' from generated text."""
    return text.split("#")[-1].replace(",", "").replace(eot_token, "").strip()


def compute_reward(generated_ids, tokenizer, ground_truth,
                            latent_id, length_penalty=0.01, format_score_ratio=0.2, coconut_mode=True, is_print=False):
    """
    Combined reward balancing correctness and latent-chain efficiency.
    80% correctness, 20% format score

    Args:
        generated_ids: (B, L) tensor — full sequence including question tokens
        tokenizer:     tokenizer for decoding
        ground_truth:  expected answer string
        latent_id:     token id used for latent positions
        coconut_mode:  whether to keep special tokens when decoding

    Returns:
        total_reward (list[float]), n_latent (list[int])
    """
    total_reward, accuracies, extracted_answers = [], [], []
    n_latents, n_latents_correct = [], []
    for i in range(generated_ids.shape[0]):     # iterate across batch
        text = tokenizer.decode(generated_ids[i, :], skip_special_tokens=not coconut_mode)      # list of strings
        answer = extract_answer(text, "<|endoftext|>")
        extracted_answers.append(answer)

        # Check for output format penalty
        format_score = 1.0
        if answer.find("<|start-latent|>") != -1 or answer.find("<|end-latent|>") != -1:
            format_score = 0.0
            answer = answer.replace("<|start-latent|>", "").replace("<|end-latent|>", "")

        # Compute answer score
        correctness = 1.0 if answer == ground_truth[i] else 0.0               # for list of answers, given as [ans1, ans2, ...]
        # correctness = 1.0 if answer == ground_truth else 0.0                    # batch["answer"] is a list of strings, so use [i] if not indexed before function
        accuracies.append(correctness)
        num_latent = (generated_ids[i, :] == latent_id).sum().item()
        correctness = max(0, correctness - length_penalty * max(0, num_latent - 1))
        if num_latent == 0:
            correctness = 0.0

        n_latents.append(num_latent)
        if answer == ground_truth[i]:
            n_latents_correct.append(num_latent)

        # Count latent_id tokens in generated_ids
        reward = max(0, (1 - format_score_ratio) * correctness + format_score_ratio * format_score)
        total_reward.append(reward)

        if is_print:
            print(text)
            print(f"Predicted answer: '{answer}'")
            print(f"Ground truth: {ground_truth[i]}")
            print(reward, num_latent)
    
    accuracy = sum(accuracies) / len(accuracies)

    return total_reward, accuracy, extracted_answers, accuracies, n_latents, n_latents_correct


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
    lm_logits_gen  = model_outputs.logits[:, prompt_len-1:-1, :]                               # (B, gen_len, vocab) bf16
    lm_log_p_gen   = F.log_softmax(lm_logits_gen, dim=-1)                                      # (B, gen_len, vocab) bf16
    pred_token     = lm_log_p_gen.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1).float()     # (B, gen_len) fp32

    # Termination logits are 2-class; float32 is cheap here.
    term_logits_gen = model_outputs.termination_logits[:, prompt_len-1:-1, :]                  # (B, gen_len, 2)
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
    with FSDP.summon_full_params(parallel_model, writeback=False, rank0_only=False):
        state_dict = {k: v.detach().cpu().clone()
        for k, v in parallel_model.module.state_dict().items()}

    torch.save({"model_state_dict": state_dict}, save_path)
    print(f"Saved model checkpoint at {save_path}!")


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

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    configs.__dict__["start_time"] = current_time
    save_dir = os.path.join(configs.save_path, configs.name, current_time)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier(device_ids=[local_rank])

    if rank == 0:
        import shutil
        for _src in [__file__, os.path.join(os.path.dirname(__file__), "coconut_grpo_baseline_manual.py"), os.path.join(os.path.dirname(__file__), "args", "gsm_vllr_grpo_coconut_baseline.yaml")]:
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
        ref_model = Coconut(ref_base_model, latent_id, start_id, end_id, tokenizer.eos_token_id, configs.termination_gamma)
        if configs.bf16:
            ref_model = ref_model.to(dtype=torch.bfloat16)
        ref_model = ref_model.to(device="cpu")
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, configs.termination_gamma)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # GPT2Block,       # for GPT2, we don't need to shard layers (it becomes DDP)
            LlamaDecoderLayer  # only shard llama's layers.
        },
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    assert ref_model is not None, "ref_model is not defined"

    # if only eval, use ddp (to avoid bugs in fsdp)
    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])

    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank, use_orig_params=True
        )

    del model

    if rank == 0:
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
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=3200 if configs.debug else 100000000
        )

    # if "gsm" in configs.val_path:
    #     # max_new_tokens = 64                   # change
    #     max_new_tokens = 128
    # else:
    #     max_new_tokens = 128
    max_new_tokens = configs.max_new_tokens


    if not configs.debug and not configs.only_eval and rank == 0:
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

        # Load training dataset
        if not configs.only_eval:
            dataset_train = get_grpo_dataset(
                base_dataset_train,
                start_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts or configs.no_bot_tokens,
                shuffle=True,
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
            )
            total_loss = 0.0

            ref_model_path = configs.load_model_path
            ref_save_dir = os.path.join(save_dir, "ref_models")
            os.makedirs(ref_save_dir, exist_ok=True)

            best_accuracy = 0.0

            # Log to file for debugging
            log_path = os.path.join(save_dir, f"checkpoint_{epoch+1}.log")

            for step, batch in enumerate(train_dataloader):     
                # batch = dict(["input_ids", "attention_mask", "position_ids", "idx", "answer"])
                # answer is list[int]

                if (step + 1) % 100 == 0:
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
                    with FSDP.summon_full_params(parallel_model, writeback=False, rank0_only=False):
                        state_dict = {k: v.detach().cpu().clone()
                                        for k, v in parallel_model.module.state_dict().items()}
                    ref_model.load_state_dict(state_dict, strict=False)
                    del state_dict
                    print("Reference model loaded")

                all_rewards = []
                all_outputs = []
                all_token_ids = []
                all_accuracies = []

                # Generate Rollouts ######################################################################################################################################################
                # for i in range(configs.num_rollouts):
                # Changed from generating batch_size * num_rollout to setting batch_size=1 and generating 16 rollouts per step. This is primarily due to GPU restrictions.

                # Phase 1: Rollout
                parallel_model.module.train()
                # modify input to batch * num_rollouts, by repeating each input. Dropout will ensure different outputs.
                # answers = batch["answer"]       # [ans1, ans2, ...]

                batch_len, len_question = batch["input_ids"].shape
                batch["input_ids"] = torch.repeat_interleave(batch["input_ids"], configs.num_rollouts, dim=0)
                batch["attention_mask"] = torch.repeat_interleave(batch["attention_mask"], configs.num_rollouts, dim=0)
                answers = [ans for ans in batch["answer"] for _ in range(configs.num_rollouts)]

                blacklist = {"idx", "answer"}
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key not in blacklist        # set tokenized text to device(the GPU that current code runs in - multi-GPU setting)
                }

                # Feed input to model
                with torch.no_grad():
                    with FSDP.summon_full_params(parallel_model):
                        generated, embeddings, model_outputs = parallel_model.module.generate_batched(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            output_embedding=True,
                            synced_gpus=True,
                        )
                    del model_outputs

                    # Compute rewards
                    rewards, step_accuracy, ans, accuracies, n_latents, n_latents_correct = compute_reward(
                        generated,
                        tokenizer,
                        answers,
                        latent_id,
                        length_penalty=configs.length_penalty_per_token,
                        coconut_mode=configs.coconut,
                    )

                    # Append for batched processing
                    all_rewards.extend(rewards)
                    all_outputs.append(embeddings.cpu())  # offload to CPU to free GPU VRAM between rollouts
                    all_token_ids.append(generated)
                    all_accuracies.append(step_accuracy)

                    avg_reward = sum(rewards) / len(rewards)

                    if wandb_run and rank == 0:
                        log_dict = {
                            "train/batch_avg_reward": avg_reward,
                            "train/batch_accuracy": step_accuracy,
                            "train/batch_n_latent_avg": sum(n_latents)/len(n_latents),
                            "train/batcn_n_latent_correct_avg": sum(n_latents_correct)/len(n_latents_correct) if len(n_latents_correct) > 0 else 0
                        }
                        wandb_run.log(log_dict)


                    # Log to file for debugging
                    with open(log_path, "a") as f:
                        f.write(f"Step {step}, Average Reward: {avg_reward:.3f}\n")

                        for i in range(batch_len):
                            for j in range(configs.num_rollouts):
                                idx_ans = i * configs.num_rollouts + j
                                f.write(f"Question RAW: {tokenizer.decode(batch['input_ids'][idx_ans])}\n")
                                f.write(f"Generated RAW: {tokenizer.decode(generated[idx_ans, len_question:])}\n")
                                f.write(f"Question: {tokenizer.decode(batch['input_ids'][idx_ans]).replace("<|endoftext|>", "")}\n")
                                f.write(f"Generated: {tokenizer.decode(generated[idx_ans, len_question:]).replace("<|endoftext|>", "")}\n")
                                f.write(f"Predicted answer: '{ans[idx_ans]}'\n")
                                f.write(f"Ground Truth: {answers[idx_ans]}\n")
                                f.write(f"Reward: {rewards[idx_ans]:.2f}\n")
                                f.write("\n")
                            f.write("\n\n\n")
                        
                        f.write("\n" *5 + "-" * 100 + "\n" *5)

                # Calculate overall accuracy, save model if better than previous best
                overall_accuracy = sum(all_accuracies) / len(all_accuracies)

                # if overall_accuracy > best_accuracy:
                #     best_accuracy = overall_accuracy
                #     if rank == 0:
                #         save_model(parallel_model, os.path.join(ref_save_dir, f"best_model_step{step}_{overall_accuracy:.2f}"))

                # Policy ####################################################################################################################
                # Start training step after accumulating rewards and log-probs (Deepseek-R1 style, N=16), Mini-batches
                all_losses = []

                # Append pad tokens to outputs/token_ids to match max length
                max_len = max(out.shape[1] for out in all_outputs)
                for i in range(len(all_outputs)):
                    out = all_outputs[i]
                    tid = all_token_ids[i]
                    pad_len = max_len - out.shape[1]
                    if pad_len > 0:
                        emb_padding         = torch.zeros(out.shape[0], pad_len, out.shape[2], device=out.device, dtype=out.dtype)
                        all_outputs[i]      = torch.cat([out, emb_padding], dim=1)
                        id_padding          = torch.full((tid.shape[0], pad_len), tokenizer.eos_token_id, device=tid.device, dtype=tid.dtype)
                        all_token_ids[i]    = torch.cat([tid, id_padding], dim=1)
                    else:
                        all_outputs[i]      = out
                        all_token_ids[i]    = tid

                # pick 32 rollouts at a time, randomly sampling from the full set of rollouts
                # Currently, for question A, B, C, D, the output layout is: (A1, B1, C1, D1), (A2, B2, C2, D2), ...
                # total_indices = list(range(len(all_rewards)))
                # random.shuffle(total_indices)
                

                # aggregate into tensors for easier indexing
                with torch.no_grad():
                    all_outputs     = torch.cat(all_outputs)            # (N * num_rollouts, L, hidden)
                    all_token_ids   = torch.cat(all_token_ids)        # (N * num_rollouts, L)
                    all_rewards     = torch.tensor(all_rewards, device=rank, dtype=torch.float32)

                # Phase 2: Group-Normalised Advantages
                advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

                # Phase 3: GRPO Policy Loss (mini-batch gradient accumulation)
                # all_outputs is already on CPU (offloaded during rollout); all_token_ids on GPU.
                token_ids = all_token_ids.to(rank)

                # Ref model log-probs — all at once on CPU, no gradient needed.
                with torch.no_grad():
                    ref_model.eval()
                    ref_outputs = ref_model(input_embeds=all_outputs)  # all_outputs already on CPU
                    ref_lp_per_token, ref_loss_mask = compute_log_probs(
                        token_ids.cpu(), ref_outputs, len_question, latent_id, tokenizer.eos_token_id
                    )
                    del ref_outputs
                    gc.collect()
                    ref_lp_per_token = ref_lp_per_token.to(device=rank)
                    ref_loss_mask    = ref_loss_mask.to(device=rank)

                # Training forward in mini-batches; loss is normalised over ALL tokens globally.
                total_rollouts  = all_outputs.shape[0]
                train_mb_size   = configs.train_minibatch_size
                total_nonmasked = ref_loss_mask.sum().clamp(min=1)
                max_log_ratio   = torch.tensor(0.0, device=rank)
                step_loss       = 0.0

                optimizer.zero_grad()
                # Switch to eval mode before computing log-probs.  This disables dropout
                # in the policy model's forward pass, making log-probs deterministic given
                # the stored embeddings.  The generation-phase dropout noise is already baked
                # into the stored latent hidden states; adding fresh dropout here would create
                # a different distribution than what generated the tokens, corrupting the log-ratio.
                parallel_model.module.eval()
                # Compute in mini-batches to reduce peak VRAM usage, by deleting outputs frequently (only need log_probs for reward)
                for mb_start in range(0, total_rollouts, train_mb_size):
                    mb_end      = min(mb_start + train_mb_size, total_rollouts)
                    mb_embeds   = all_outputs[mb_start:mb_end].to(rank)   # CPU → GPU for this mini-batch

                    mb_tok      = token_ids[mb_start:mb_end]
                    mb_adv      = advantages[mb_start:mb_end]

                    mb_ref_lp   = ref_lp_per_token[mb_start:mb_end]
                    mb_ref_mask = ref_loss_mask[mb_start:mb_end]

                    mb_new_outputs = parallel_model(input_embeds=mb_embeds)
                    mb_new_lp, _   = compute_log_probs(
                        mb_tok, mb_new_outputs, len_question, latent_id, tokenizer.eos_token_id
                    )
                    del mb_new_outputs

                    mb_log_ratio = mb_new_lp - mb_ref_lp
                    # Clamp log_ratio before exp() — at ±5 the effective ratio is ~148,
                    # far outside the PPO clip window, so clamping does not distort the
                    # gradient under normal training; it only prevents exp() overflow.
                    mb_log_ratio_clamped = mb_log_ratio.clamp(-5.0, 5.0)
                    mb_ratio     = torch.exp(mb_log_ratio_clamped)
                    mb_clipped   = torch.clamp(mb_ratio, 1.0 - configs.clip_ratio, 1.0 + configs.clip_ratio)

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

                all_losses.append(step_loss)
                parallel_model.clip_grad_norm_(max_norm=1.0)
                optimizer.step()

                total_loss = sum(all_losses) / len(all_losses)

                # Free large rollout tensors accumulated this step
                del all_outputs, all_token_ids, all_rewards
                gc.collect()
                torch.cuda.empty_cache()

                pbar.update(1)


                ###########################################################################################################################################
                if step == 1:
                    print_memory_breakdown(parallel_model, optimizer)     # use to check how much VRAM is being used

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": step + 1,
                        "train/loss": total_loss,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/max_log_ratio": max_log_ratio.item(),
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"GRPO Epoch: {epoch+1}/{configs.num_epochs}, "
                    f"step {step}/{len(train_dataloader)} "
                    f"(loss: {total_loss:.4f})"
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
                    log_path = os.path.join(save_dir, f"checkpoint_{epoch+1}_{step}_eval.log")

                    with torch.no_grad():
                        parallel_model.module.eval()
                        all_accuracies = []
                        all_rewards = []
                        all_n_latents = []

                        for idx, batch in enumerate(valid_gen_dataloader):
                            test_idx = batch["idx"]

                            batch_len, len_question = batch["input_ids"].shape
                            answers = batch["answer"]

                            blacklist = {"idx", "answer"}
                            batch = {
                                key: batch[key].to(rank) for key in batch.keys() if key not in blacklist        # set tokenized text to device(the GPU that current code runs in - multi-GPU setting)
                            }

                            with FSDP.summon_full_params(parallel_model):
                                generated, embeddings, model_outputs = parallel_model.module.generate_batched(
                                    batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    max_new_tokens=max_new_tokens,
                                    output_embedding=True,
                                    synced_gpus=True,
                                )
                            del model_outputs

                            # Compute rewards
                            rewards, step_accuracy, ans, accuracies, n_latents, n_latents_correct = compute_reward(
                                generated,
                                tokenizer,
                                answers,
                                latent_id,
                                length_penalty=configs.length_penalty_per_token,
                                coconut_mode=configs.coconut,
                            )
                            all_accuracies.append(step_accuracy)
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
                        
                        eval_accuracy = sum(all_accuracies) / len(all_accuracies)
                        eval_reward = sum(all_rewards) / len(all_rewards)
                        eval_avg_latent = sum(all_n_latents) / len(all_n_latents)

                        if eval_accuracy > best_accuracy:
                            best_accuracy = eval_accuracy
                            
                            eval_savepath = os.path.join(
                                save_dir, "ref_models", f"eval_best_model_{epoch+1}_{step+1}_{best_accuracy:.2f}.pt"
                            )
                            save_model(parallel_model, eval_savepath)

                    if wandb_run and rank == 0:
                        wandb_run.log({
                            "eval/accuracy": eval_accuracy,
                            "eval/reward": eval_reward,
                            "eval/avg_n_latent": eval_avg_latent
                            })

                    log_path = os.path.join(save_dir, f"checkpoint_{epoch+1}.log")

            pbar.close()
            pbar.close()
            dist.barrier()

            # Save model after checkpoint
            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):

                if rank == 0:
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
            total=len(valid_gen_dataloader), dynamic_ncols=True,
        )

        log_path = os.path.join(save_dir, f"Test_Evaluation.log")

        with torch.no_grad():
            parallel_model.module.eval()
            all_accuracies = []
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
                    key: batch[key].to(rank) for key in batch.keys() if key not in blacklist        # set tokenized text to device(the GPU that current code runs in - multi-GPU setting)
                }

                with FSDP.summon_full_params(parallel_model):
                    generated, embeddings, model_outputs = parallel_model.module.generate_batched(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        output_embedding=True,
                        synced_gpus=True,
                    )
                del model_outputs

                # Compute rewards
                rewards, step_accuracy, ans, correctness, n_latents, n_latents_correct = compute_reward(
                    generated,
                    tokenizer,
                    answers,
                    latent_id,
                    length_penalty=configs.length_penalty_per_token,
                    coconut_mode=configs.coconut,
                )
                all_accuracies.append(step_accuracy)
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


                eval_accuracy = sum(all_accuracies) / len(all_accuracies)

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {eval_accuracy:.2%}"
                )

                # Plot to PCA
                if configs.save_pca_figures:
                    dir_plot_correct = os.path.join(save_dir, "plots", "correct")
                    dir_plot_incorrect = os.path.join(save_dir, "plots", "incorrect")
                    dir_plot_noformat = os.path.join(save_dir, "plots", "wrong format")
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
            
            eval_accuracy = sum(all_accuracies) / len(all_accuracies)
            eval_reward = sum(all_rewards) / len(all_rewards)
            eval_avg_latent = sum(all_n_latents) / len(all_n_latents)

            print(f"Evaluation Accuracy: {eval_accuracy:.2%}")
            print(f"Evaluation Rewards: {eval_reward:.3f}")
            print(f"Evaluation Average Latent Steps: {eval_avg_latent:.2f}")

            # Save latent token count histograms (total, correct, incorrect, wrong-format)
            save_latent_histogram(
                all_n_latents, all_n_latents_correct, all_n_latents_incorrect, all_n_latents_noformat,
                output_dir=os.path.join(save_dir, "plots"),
            )

            print(f"Correct Var: {sum(pca_variances_correct)/len(pca_variances_correct):.3f}")
            print(f"Incorrect Var: {sum(pca_variances_incorrect)/len(pca_variances_incorrect):.3f}")
            print(f"Noformat Var: {sum(pca_variances_noformat)/len(pca_variances_noformat):.3f}")

        if wandb_run and rank == 0:
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

        if wandb_run:
            wandb_run.log({
                # "eval/acc": cor / total, 
                "eval/acc": sum(all_accuracies) / len(all_accuracies), 
                "eval/reward": sum(all_rewards) / len(all_rewards)
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
