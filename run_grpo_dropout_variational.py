# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Variational-dropout variant of run_grpo_dropout_proof.py.
# Inherits all proof-faithful changes (D1, D2, D3) and adds:
#   V1: uses CoconutVariational instead of Coconut. Each chain's latent
#       recurrence steps share a single dropout mask (one xi per chain,
#       reused at every t). Maps to a single posterior sample
#       theta_tilde(xi) of the perturbed parameters --- the Bayesian
#       interpretation of dropout (Gal & Ghahramani 2016) holds end to
#       end. Updated draft eq 2.3:
#           xi ~ p(xi),  tilde_theta_t(xi) == tilde_theta(xi) for all t.
#   V2: configure_variational_dropout zeros attention dropout explicitly.
#       Attention-prob mask shape grows with kv-cache length, so the
#       RNG-restore trick used by CoconutVariational does not produce
#       a coherent variational pattern there. Setting it to 0 keeps the
#       Bayesian story exact for residual / MLP / embedding dropouts,
#       which are the main sources of regularization in modern decoders
#       (LLaMA, Mistral, Qwen all ship attention_dropout = 0 already).
#   V3: ref_model_renewal_steps = 0 in the YAML -> pi_ref is frozen at
#       SFT init for the entire run. Matches Theorem 4.4 corollary's
#       requirement that the KL anchor be a fixed reference distribution.
#       Strongly recommended; refresh > 0 is left as an option for staged
#       training but is not proof-faithful.
#
# All D1/D2/D3 properties from research_draft.tex are preserved.

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import copy
import datetime
import gc
import json
import time

import bitsandbytes as bnb
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from coconut_variational import CoconutVariational
from dataset import get_dataset, get_grpo_dataset, MyCollator
from utils import Config, set_seed


global_time = 0
is_print_time = True

def start_timer():
    global global_time
    global_time = time.time()


def measure_timer(out_str):
    global global_time, is_print_time
    cur_time = time.time()

    if is_print_time:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_alloc = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"[{cur_time - global_time:.2f}s] {out_str} "
                f"(alloc={alloc:.2f}GB, reserved={reserved:.2f}GB, max_alloc={max_alloc:.2f}GB)"
            )
        else:
            print(f"[{cur_time - global_time:.2f}s] {out_str}")
    global_time = cur_time


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
    rollout_len = rollout_embeddings.shape[1]
    replay_len = replay_outputs.inputs_embeds.shape[1]
    if replay_outputs.inputs_embeds.shape[0] != rollout_embeddings.shape[0] or replay_outputs.inputs_embeds.shape[2] != rollout_embeddings.shape[2]:
        raise AssertionError(
            f"Replay embedding batch/hidden mismatch: {replay_outputs.inputs_embeds.shape} vs {rollout_embeddings.shape}"
        )
    if replay_len < rollout_len:
        raise AssertionError(
            f"Replay is shorter than rollout ({replay_len} < {rollout_len})"
        )

    replay_prefix = replay_outputs.inputs_embeds[:, :rollout_len, :]
    if not torch.allclose(replay_prefix, rollout_embeddings, atol=atol, rtol=rtol):
        diff = (replay_prefix - rollout_embeddings).abs()
        max_diff = diff.max().item()
        flat_idx = diff.reshape(-1).argmax().item()
        batch_size, seq_len, hidden = diff.shape
        b_idx = flat_idx // (seq_len * hidden)
        l_idx = (flat_idx % (seq_len * hidden)) // hidden
        h_idx = flat_idx % hidden
        raise AssertionError(
            f"Replay embeddings diverged (max abs diff {max_diff:.6e}) at "
            f"batch={b_idx}, pos={l_idx}, hidden={h_idx}"
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
            return_replay_inputs_embeds=True,
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


def extract_answer(text, eot_token):
    return text.split("#")[-1].replace(",", "").replace(eot_token, "").strip()


def clean_console_text(text):
    return text.replace("<|endoftext|>", "").strip()


def build_first_batch_outputs_table(
    tokenizer,
    repeated_input_ids,
    generated_all,
    repeated_answers,
    all_rewards,
    prompt_len,
    num_rollouts,
    all_base_rewards,
):
    question_text = clean_console_text(
        tokenizer.decode(repeated_input_ids[0].detach().cpu())
    )
    table = wandb.Table(
        columns=[
            "rollout_index",
            "question",
            "generated",
            "predicted_answer",
            "ground_truth",
            "reward",
            "base_reward",
        ]
    )
    for rollout_idx in range(min(num_rollouts, generated_all.shape[0])):
        decoded_generated = tokenizer.decode(
            generated_all[rollout_idx, prompt_len:].detach().cpu()
        )
        cleaned_generated = clean_console_text(decoded_generated)
        predicted_answer = extract_answer(decoded_generated, tokenizer.eos_token)
        table.add_data(
            rollout_idx + 1,
            question_text,
            cleaned_generated,
            predicted_answer,
            repeated_answers[rollout_idx],
            float(all_rewards[rollout_idx]),
            float(all_base_rewards[rollout_idx]),
        )
    return table


def compute_reward(
    generated_ids,
    tokenizer,
    ground_truth,
    latent_id,
    length_penalty=0.01,
    format_penalty=-0.5,
    latent_step_reward=0.0,
    coconut_mode=True,
    is_print=False,
):
    total_reward, base_reward, accuracies, extracted_answers = [], [], [], []
    n_latents, n_latents_correct = [], []
    for i in range(generated_ids.shape[0]):
        text = tokenizer.decode(
            generated_ids[i, :],
            skip_special_tokens=not coconut_mode,
        )
        answer = extract_answer(text, tokenizer.eos_token)
        extracted_answers.append(answer)

        num_latent = (generated_ids[i, :] == latent_id).sum().item()
        n_latents.append(num_latent)

        format_violation = False
        if "<|start-latent|>" in answer or "<|end-latent|>" in answer:
            format_violation = True
            answer = answer.replace("<|start-latent|>", "").replace("<|end-latent|>", "")

        cleaned = answer.strip()
        if not cleaned or cleaned == "#":
            format_violation = True

        if num_latent == 0:
            format_violation = True

        if "<<" in answer or ">>" in answer:
            format_violation = True

        correctness = 1.0 if answer == ground_truth[i] else 0.0
        accuracies.append(correctness)

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

    accuracy = sum(accuracies) / max(len(accuracies), 1)
    return total_reward, base_reward, accuracy, extracted_answers, accuracies, n_latents, n_latents_correct


def build_response_mask(token_ids, response_start, eos_id):
    if token_ids.shape[1] <= response_start:
        return torch.zeros(
            token_ids.shape[0],
            0,
            dtype=torch.bool,
            device=token_ids.device,
        )

    response_tokens = token_ids[:, response_start:]
    gen_len = response_tokens.shape[1]
    eos_mask = response_tokens == eos_id
    first_eos = torch.full(
        (response_tokens.shape[0],),
        gen_len,
        dtype=torch.long,
        device=token_ids.device,
    )
    for batch_idx in range(response_tokens.shape[0]):
        eos_pos = eos_mask[batch_idx].nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            first_eos[batch_idx] = eos_pos[0].item() + 1

    positions = torch.arange(gen_len, device=token_ids.device).unsqueeze(0)
    return positions < first_eos.unsqueeze(1)


def compute_log_probs(token_ids, model_outputs, prompt_len, num_latents, eos_id):
    response_start = prompt_len + num_latents
    if token_ids.shape[1] <= response_start:
        return (
            torch.zeros(token_ids.shape[0], 0, device=token_ids.device),
            torch.zeros(token_ids.shape[0], 0, dtype=torch.bool, device=token_ids.device),
        )

    response_tokens = token_ids[:, response_start:]
    gen_len = response_tokens.shape[1]
    needed_steps = response_start - 1 + gen_len
    if model_outputs.logits.shape[1] < needed_steps:
        raise ValueError(
            f"Replay outputs are too short for log-prob computation: "
            f"logits={model_outputs.logits.shape[1]}, needed={needed_steps}"
        )

    lm_logits_gen = model_outputs.logits[
        :,
        response_start - 1 : response_start - 1 + gen_len,
        :,
    ]
    lm_log_p_gen = F.log_softmax(lm_logits_gen, dim=-1)
    pred_token = lm_log_p_gen.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1).float()
    loss_mask = build_response_mask(token_ids, response_start, eos_id)
    return pred_token, loss_mask


def save_model(parallel_model, save_path):
    state_dict = {
        k: v.detach().cpu().clone()
        for k, v in parallel_model.module.state_dict().items()
    }
    torch.save({"model_state_dict": state_dict}, save_path)
    print(f"Saved model checkpoint at {save_path}!")


def run_validation(
    parallel_model,
    valid_gen_dataloader,
    tokenizer,
    configs,
    latent_id,
    eval_num_latents,
    max_new_tokens,
    device,
    eval_savepath,
    question_val,
    rank,
    desc="Test Accuracy",
):
    pbar = tqdm(
        colour="blue",
        desc=desc,
        total=len(valid_gen_dataloader),
        dynamic_ncols=True,
    )
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    was_training = parallel_model.module.training
    parallel_model.module.eval()
    with torch.no_grad():
        for _, batch in enumerate(valid_gen_dataloader):
            answers = batch["answer"]
            idxs = batch["idx"]
            eval_input_ids = batch["input_ids"].to(device)
            eval_attention_mask = batch["attention_mask"].to(device)

            outputs = parallel_model.module.generate_batched(
                eval_input_ids,
                attention_mask=eval_attention_mask,
                num_latents=eval_num_latents,
                max_new_tokens=max_new_tokens,
                output_embedding=False,
                synced_gpus=not configs.only_eval,
            )

            _, _, _, extracted, accuracies, _, _ = compute_reward(
                outputs,
                tokenizer,
                answers,
                latent_id,
                length_penalty=configs.length_penalty_per_token,
                format_penalty=getattr(configs, "format_penalty", -0.5),
                latent_step_reward=getattr(configs, "latent_step_reward", 0.0),
                coconut_mode=configs.coconut,
            )

            correct += int(sum(accuracies))
            total += len(accuracies)
            pbar.update(1)
            pbar.set_description(
                f"{desc}: {float(correct.detach().float() / total.detach().float()):.2f}"
            )

            if rank == 0:
                with open(eval_savepath, "a+") as fp:
                    for row_idx, pred in enumerate(extracted):
                        question = question_val[idxs[row_idx].cpu().item()]
                        fp.write(f"\nQuestion {idxs[row_idx].cpu().item() + 1}:\n")
                        fp.write(f"Question = {question}\n")
                        fp.write(f"Full output:\n{tokenizer.decode(outputs[row_idx])}\n")
                        fp.write(f"Extracted Output:\n{pred}\n")
                        fp.write(f"Answer = {answers[row_idx]}\n")
                        fp.write("-" * 30 + "\n")

    pbar.close()

    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    if was_training:
        parallel_model.module.train()

    return correct.item(), total.item()


def compute_group_advantages(rewards, num_rollouts):
    if rewards.numel() == 0:
        return rewards
    grouped = rewards.view(-1, num_rollouts)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - group_mean) / (group_std + 1e-8)
    return advantages.reshape(-1)


def configure_dropout(model_config, dropout):
    for attr in [
        "attention_dropout",
        "attention_probs_dropout_prob",
        "hidden_dropout",
        "hidden_dropout_prob",
        "dropout",
        "embd_pdrop",
        "resid_pdrop",
        "summary_first_dropout",
    ]:
        if hasattr(model_config, attr):
            setattr(model_config, attr, dropout)


# V2: variational variant. Attention dropout cannot be made variational by
# RNG restore alone (mask shape grows with kv-cache length), so we zero it
# and set every other dropout to the requested rate. The remaining
# dropouts all act on tensors of shape (B, 1, H) at each latent step, where
# CoconutVariational's RNG-restore yields a true locked mask across t.
ATTENTION_DROPOUT_ATTRS = {
    "attention_dropout",
    "attention_probs_dropout_prob",
}


def configure_variational_dropout(model_config, dropout):
    for attr in [
        "attention_dropout",
        "attention_probs_dropout_prob",
        "hidden_dropout",
        "hidden_dropout_prob",
        "dropout",
        "embd_pdrop",
        "resid_pdrop",
        "summary_first_dropout",
    ]:
        if hasattr(model_config, attr):
            value = 0.0 if attr in ATTENTION_DROPOUT_ATTRS else dropout
            setattr(model_config, attr, value)


def main():
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Coconut GRPO with dropout replay")
    parser.add_argument("config_file")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = local_rank

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    configs.__dict__["start_time"] = current_time
    save_dir = os.path.join(configs.save_path, configs.name, current_time)
    # D2: default to "dropout_matched" so KL anchor is evaluated under the
    # same dropout mask as rollout (CRN coupling assumed in Prop 4.5).
    reference_model_mode = getattr(configs, "reference_model_mode", "dropout_matched")
    ref_model_device_name = getattr(configs, "ref_model_device", "cuda").lower()

    if ref_model_device_name not in {"cuda", "cpu"}:
        raise ValueError(
            f"Unsupported ref_model_device={ref_model_device_name!r}; expected 'cuda' or 'cpu'"
        )
    if ref_model_device_name == "cpu" and reference_model_mode == "dropout_matched":
        raise ValueError(
            "reference_model_mode=dropout_matched requires the reference model on CUDA"
        )

    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    global is_print_time
    is_print_time = configs.track_timing

    if is_print_time:
        print("Tracking timing...")

    dist.barrier(device_ids=[local_rank])

    if rank == 0:
        import shutil

        for src in [
            __file__,
            os.path.join(os.path.dirname(__file__), "coconut.py"),
            args.config_file,
        ]:
            shutil.copy2(src, os.path.join(save_dir, os.path.basename(src)))

    if configs.resume != 0 and configs.load_model_path == "None":
        print(
            f"Warning: resume={configs.resume} but load_model_path is None"
        )

    model_config = AutoConfig.from_pretrained(configs.model_id)
    # V2: zero attention dropout so the variational mask interpretation
    # holds for every dropout layer that actually fires.
    configure_variational_dropout(model_config, configs.dropout)
    if configs.sdpa_attention:
        if configs.bf16:
            model = AutoModelForCausalLM.from_pretrained(
                configs.model_id,
                config=model_config,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                configs.model_id,
                config=model_config,
                attn_implementation="sdpa",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id,
            config=model_config,
        )

    if getattr(configs, "grad_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

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
    saved_weights = None
    if configs.load_model_path != "None":
        saved_checkpoint = torch.load(configs.load_model_path, map_location="cpu")
        saved_weights = saved_checkpoint["model_state_dict"]

        if configs.coconut and not any(k.startswith("base_causallm") for k in saved_weights.keys()):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))
        elif not configs.coconut and any(k.startswith("base_causallm") for k in saved_weights.keys()):
            raise ValueError("Cannot load coconut model weights into a base causal LM")
        elif not configs.coconut:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    ref_model = None
    ref_model_device = torch.device("cpu" if ref_model_device_name == "cpu" else f"cuda:{local_rank}")
    if configs.coconut:
        ref_base_model = copy.deepcopy(model)
        ref_model = CoconutVariational(
            ref_base_model,
            latent_id,
            start_id,
            end_id,
            tokenizer.eos_token_id,
        )
        if configs.bf16 and ref_model_device_name != "cpu":
            ref_model = ref_model.to(dtype=torch.bfloat16)
        ref_model = ref_model.to(device=ref_model_device)

        model = CoconutVariational(
            model,
            latent_id,
            start_id,
            end_id,
            tokenizer.eos_token_id,
        )

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))
        if ref_model is not None:
            print(ref_model.load_state_dict(saved_weights, strict=False))

    print(f"Running DDP on rank = {rank}, world size = {world_size}")
    model = model.to(device)
    if configs.bf16:
        model.to(torch.bfloat16)

    assert ref_model is not None, "ref_model is not defined"
    for param in ref_model.parameters():
        param.requires_grad_(False)

    parallel_model = DDP(model, device_ids=[local_rank])
    del model

    if rank == 0:
        print(parallel_model)

    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip()
        for d in json.load(open(configs.val_path))
    ]

    base_dataset_valid = get_dataset(
        configs.val_path,
        tokenizer,
        max_size=32 if configs.debug else 100000000,
    )

    if not configs.only_eval:
        total_train_samples = (
            3200 if configs.debug else int(configs.num_steps * configs.train_batch_size * 1.5)
        )
        train_path_secondary = getattr(configs, "train_path_secondary", "None")
        dataset_mix_ratio = getattr(configs, "dataset_mix_ratio", 1.0)

        if train_path_secondary != "None" and dataset_mix_ratio < 1.0:
            n_primary = int(total_train_samples * dataset_mix_ratio)
            n_secondary = total_train_samples - n_primary
            base_dataset_primary = get_dataset(configs.train_path, tokenizer, max_size=n_primary)
            base_dataset_secondary = get_dataset(train_path_secondary, tokenizer, max_size=n_secondary)
            from datasets import concatenate_datasets

            base_dataset_train = concatenate_datasets(
                [base_dataset_primary, base_dataset_secondary]
            )
            del base_dataset_primary, base_dataset_secondary
        else:
            base_dataset_train = get_dataset(
                configs.train_path,
                tokenizer,
                max_size=total_train_samples,
            )

    max_new_tokens = configs.max_new_tokens
    num_latents = configs.num_latents
    eval_num_latents = getattr(configs, "eval_num_latents", num_latents)

    wandb.login(key="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")

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
    else:
        wandb_run = None

    optimizer = bnb.optim.Adam8bit(
        parallel_model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay,
    )

    warmup_steps = getattr(configs, "warmup_steps", 100)
    total_steps = getattr(configs, "num_steps", 2000)
    eta_min = configs.lr * 0.1
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / configs.lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=eta_min,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    restore_policy_rng = getattr(configs, "restore_policy_rng", True)
    verify_replay_embeddings = getattr(configs, "verify_replay_embeddings", True)
    strict_replay_structure = getattr(configs, "strict_replay_structure", True)
    log_first_batch_outputs_table = getattr(
        configs,
        "log_first_batch_outputs_table",
        getattr(configs, "print_first_batch_outputs", False),
    )

    for epoch in range(configs.resume, configs.num_epochs):
        if "train_dataloader" in locals():
            del train_dataloader
        if "dataset_train" in locals():
            del dataset_train
        if "valid_gen_dataloader" in locals():
            del valid_gen_dataloader
        if "dataset_gen_val" in locals():
            del dataset_gen_val
        gc.collect()

        dataset_gen_val = get_grpo_dataset(
            base_dataset_valid,
            start_id,
            no_special_marker=configs.cot
            or configs.no_cot
            or configs.no_thoughts
            or configs.no_bot_tokens,
        )
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=configs.eval_batch_size,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:
            dataset_train = get_grpo_dataset(
                base_dataset_train,
                start_id,
                no_special_marker=configs.cot
                or configs.no_cot
                or configs.no_thoughts
                or configs.no_bot_tokens,
                shuffle=True,
                max_question_len=configs.dataset_max_len,
                num_samples=configs.num_steps * configs.train_batch_size,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.train_batch_size,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            if getattr(configs, "reset_optimizer", False):
                del optimizer
                optimizer = bnb.optim.Adam8bit(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            total_length = len(train_dataloader)
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch + 1}",
                total=total_length,
                dynamic_ncols=True,
            )

            ref_save_dir = os.path.join(save_dir, "ref_models")
            os.makedirs(ref_save_dir, exist_ok=True)
            log_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}.log")

            for step, batch in enumerate(train_dataloader):
                if (step + 1) % 100 == 0:
                    print(
                        f"Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB"
                    )
                    print(
                        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
                    )

                current_step = step + 1
                # ref_model_renewal_steps == 0 -> pi_ref stays frozen at SFT
                # init for the entire run (proof-faithful default; recommended).
                # > 0 -> periodic refresh of pi_ref toward the current policy.
                if configs.ref_model_renewal_steps > 0:
                    ref_refresh_every = (
                        configs.ref_model_renewal_steps // 10
                        if current_step <= warmup_steps
                        else configs.ref_model_renewal_steps
                    )
                    if ref_refresh_every > 0 and current_step % ref_refresh_every == 0:

                        ref_model_path = os.path.join(ref_save_dir, f"checkpoint_{epoch}_{step}.pt")
                        if rank == 0 and current_step % configs.ref_model_renewal_steps == 0:
                            print("Replacing reference model...")
                            save_model(parallel_model, ref_model_path)
                        state_dict = {
                            k: v.detach().cpu().clone()
                            for k, v in parallel_model.module.state_dict().items()
                        }
                        ref_model.load_state_dict(state_dict, strict=False)
                        del state_dict

                all_rewards = []
                all_base_rewards = []
                all_token_ids = []
                all_prompt_ids = []
                all_prompt_masks = []
                all_rollout_rng_states = []
                all_correct = 0
                all_total = 0
                n_latents_correct_sum = 0
                n_latents_correct_num = 0

                start_timer()

                batch_len, prompt_len = batch["input_ids"].shape
                repeated_answers = [
                    answer
                    for answer in batch["answer"]
                    for _ in range(configs.num_rollouts)
                ]
                repeated_input_ids = torch.repeat_interleave(
                    batch["input_ids"], configs.num_rollouts, dim=0
                ).to(device)
                repeated_attention_mask = torch.repeat_interleave(
                    batch["attention_mask"], configs.num_rollouts, dim=0
                ).to(device)

                rollout_chunk_size = getattr(
                    configs,
                    "rollout_chunk_size",
                    getattr(configs, "policy_minibatch_size", configs.train_minibatch_size * 8),
                )
                policy_mb_size = getattr(configs, "policy_minibatch_size", rollout_chunk_size)

                if restore_policy_rng and strict_replay_structure and rollout_chunk_size != policy_mb_size:
                    raise ValueError(
                        "Replay-consistent policy updates require rollout_chunk_size == policy_minibatch_size"
                    )
                if reference_model_mode == "dropout_matched" and strict_replay_structure and rollout_chunk_size != policy_mb_size:
                    raise ValueError(
                        "Dropout-matched reference replay also requires rollout_chunk_size == policy_minibatch_size"
                    )

                measure_timer("Batch prep")

                # Rollout ####################################################################################################################

                parallel_model.module.train()
                with torch.inference_mode():
                    for rollout_start in range(0, repeated_input_ids.shape[0], rollout_chunk_size):
                        rollout_end = min(
                            rollout_start + rollout_chunk_size,
                            repeated_input_ids.shape[0],
                        )
                        prompt_chunk = repeated_input_ids[rollout_start:rollout_end]
                        attn_chunk = repeated_attention_mask[rollout_start:rollout_end]
                        answers_chunk = repeated_answers[rollout_start:rollout_end]
                        rng_state = capture_rng_state(device)

                        if verify_replay_embeddings:
                            generated, rollout_embeddings = parallel_model.module.generate_batched(
                                prompt_chunk,
                                attention_mask=attn_chunk,
                                num_latents=num_latents,
                                max_new_tokens=max_new_tokens,
                                output_embedding=True,
                                synced_gpus=True,
                            )
                            test_rollout_replay_embeddings(
                                parallel_model,
                                prompt_chunk,
                                attn_chunk,
                                generated,
                                rollout_embeddings,
                                rng_state,
                                device,
                            )
                            del rollout_embeddings
                        else:
                            generated = parallel_model.module.generate_batched(
                                prompt_chunk,
                                attention_mask=attn_chunk,
                                num_latents=num_latents,
                                max_new_tokens=max_new_tokens,
                                output_embedding=False,
                                synced_gpus=True,
                            )

                        rewards, base_rewards, _, _, accuracies, _, n_latents_correct = compute_reward(
                            generated,
                            tokenizer,
                            answers_chunk,
                            latent_id,
                            length_penalty=configs.length_penalty_per_token,
                            format_penalty=getattr(configs, "format_penalty", -0.5),
                            latent_step_reward=getattr(configs, "latent_step_reward", 0.0),
                            coconut_mode=configs.coconut,
                        )

                        n_latents_correct_sum += sum(n_latents_correct)
                        n_latents_correct_num += len(n_latents_correct)

                        all_rewards.extend(rewards)
                        all_base_rewards.extend(base_rewards)
                        all_token_ids.append(generated.detach().cpu())
                        all_prompt_ids.append(prompt_chunk.detach().cpu())
                        all_prompt_masks.append(attn_chunk.detach().cpu())
                        all_rollout_rng_states.append(rng_state)
                        all_correct += sum(accuracies)
                        all_total += len(accuracies)
                        del generated

                measure_timer("Rollout generation")

                avg_reward = sum(all_rewards) / max(len(all_rewards), 1)
                global_step = epoch * total_length + step + 1

                max_len = max(token_ids.shape[1] for token_ids in all_token_ids)
                padded_token_ids = []
                for token_ids in all_token_ids:
                    pad_len = max_len - token_ids.shape[1]
                    if pad_len > 0:
                        padded = F.pad(token_ids, (0, pad_len), value=tokenizer.eos_token_id)
                    else:
                        padded = token_ids
                    padded_token_ids.append(padded)

                generated_all = torch.cat(padded_token_ids, dim=0)
                with open(log_path, "a") as f:
                    f.write(f"Step {step}, Average Reward: {avg_reward:.3f}\n")
                    for i in range(batch_len):
                        for j in range(configs.num_rollouts):
                            idx_ans = i * configs.num_rollouts + j
                            decoded_question = tokenizer.decode(repeated_input_ids[idx_ans].cpu())
                            decoded_generated = tokenizer.decode(generated_all[idx_ans, prompt_len:])
                            f.write(f"Question RAW: {decoded_question}\n")
                            f.write(f"Generated RAW: {decoded_generated}\n")
                            f.write(f"Question: {decoded_question.replace('<|endoftext|>', '')}\n")
                            f.write(f"Generated: {decoded_generated.replace('<|endoftext|>', '')}\n")
                            f.write(f"Predicted answer: '{extract_answer(decoded_generated, tokenizer.eos_token)}'\n")
                            f.write(f"Ground Truth: {repeated_answers[idx_ans]}\n")
                            f.write(f"Reward: {all_rewards[idx_ans]:.2f}\n\n")
                        f.write("\n\n")
                    f.write("\n" * 5 + "-" * 100 + "\n" * 5)
                wandb_step_payload = None
                if wandb_run and rank == 0:
                    wandb_step_payload = {
                        "train/epoch": epoch + 1,
                        "train/step": step + 1,
                        "train/global_step": global_step,
                        "train/batch_avg_reward": avg_reward,
                        "train/batch_avg_base_reward": sum(all_base_rewards) / max(len(all_base_rewards), 1),
                        "train/batch_accuracy": all_correct / max(all_total, 1),
                        "train/batch_n_latent_avg": num_latents,
                        "train/batch_n_latent_correct_avg": n_latents_correct_sum / n_latents_correct_num if n_latents_correct_num > 0 else 0,
                    }
                    if log_first_batch_outputs_table:
                        wandb_step_payload["train/first_batch_outputs"] = build_first_batch_outputs_table(
                            tokenizer=tokenizer,
                            repeated_input_ids=repeated_input_ids,
                            generated_all=generated_all,
                            repeated_answers=repeated_answers,
                            all_rewards=all_rewards,
                            prompt_len=prompt_len,
                            num_rollouts=configs.num_rollouts,
                            all_base_rewards=all_base_rewards,
                        )

                # Advantage Calculation ##########################################################################################################

                prompt_ids_cpu = torch.cat(all_prompt_ids, dim=0).to(device)
                prompt_masks_cpu = torch.cat(all_prompt_masks, dim=0).to(device)
                token_ids = generated_all.to(device)
                reward_tensor = torch.tensor(all_rewards, device=device, dtype=torch.float32)
                advantages = compute_group_advantages(reward_tensor, configs.num_rollouts)

                # Theta_old Replay ###############################################################################################################
                # D1: replay generated tokens through current parallel_model
                # (still at theta_old: no optimizer.step yet) under matched
                # dropout RNG. Yields log p_{theta_old}(y | x, xi) for the
                # PPO ratio denominator. Chunk granularity must equal the
                # rollout granularity so rng_state index aligns.

                with torch.no_grad():
                    theta_old_lp_chunks = []
                    theta_old_mask_chunks = []
                    was_training = parallel_model.module.training
                    parallel_model.module.train()
                    for mb_start in range(0, token_ids.shape[0], rollout_chunk_size):
                        mb_end = min(mb_start + rollout_chunk_size, token_ids.shape[0])
                        restore_rng_state(
                            all_rollout_rng_states[mb_start // rollout_chunk_size],
                            device,
                        )
                        old_outputs = parallel_model.module(
                            input_ids=prompt_ids_cpu[mb_start:mb_end],
                            attention_mask=prompt_masks_cpu[mb_start:mb_end],
                            replay_generated_ids=token_ids[mb_start:mb_end],
                        )
                        old_lp_chunk, old_mask_chunk = compute_log_probs(
                            token_ids[mb_start:mb_end],
                            old_outputs,
                            prompt_len,
                            num_latents,
                            tokenizer.eos_token_id,
                        )
                        theta_old_lp_chunks.append(old_lp_chunk.detach())
                        theta_old_mask_chunks.append(old_mask_chunk.detach())
                        del old_outputs, old_lp_chunk, old_mask_chunk

                    theta_old_lp = torch.cat(theta_old_lp_chunks, dim=0)
                    theta_old_loss_mask = torch.cat(theta_old_mask_chunks, dim=0)
                    if not was_training:
                        parallel_model.module.eval()

                measure_timer("Theta_old replay")

                # Reference Replay ###############################################################################################################

                with torch.no_grad():
                    ref_lp_chunks = []
                    ref_mask_chunks = []
                    if reference_model_mode == "dropout_matched":
                        ref_model.train()
                        ref_chunk_size = rollout_chunk_size
                    else:
                        ref_model.eval()
                        ref_chunk_size = policy_mb_size

                    for mb_start in range(0, token_ids.shape[0], ref_chunk_size):
                        mb_end = min(mb_start + ref_chunk_size, token_ids.shape[0])
                        if reference_model_mode == "dropout_matched":
                            restore_rng_state(all_rollout_rng_states[mb_start // rollout_chunk_size], device)

                        if ref_model_device_name == "cpu":
                            ref_prompt = prompt_ids_cpu[mb_start:mb_end].cpu()
                            ref_mask = prompt_masks_cpu[mb_start:mb_end].cpu()
                            ref_tok = token_ids[mb_start:mb_end].cpu()
                        else:
                            ref_prompt = prompt_ids_cpu[mb_start:mb_end]
                            ref_mask = prompt_masks_cpu[mb_start:mb_end]
                            ref_tok = token_ids[mb_start:mb_end]

                        ref_outputs = ref_model(
                            input_ids=ref_prompt,
                            attention_mask=ref_mask,
                            replay_generated_ids=ref_tok,
                        )
                        ref_lp_chunk, ref_mask_chunk = compute_log_probs(
                            ref_tok,
                            ref_outputs,
                            prompt_len,
                            num_latents,
                            tokenizer.eos_token_id,
                        )
                        ref_lp_chunks.append(ref_lp_chunk.cpu() if ref_model_device_name == "cpu" else ref_lp_chunk)
                        ref_mask_chunks.append(ref_mask_chunk.cpu() if ref_model_device_name == "cpu" else ref_mask_chunk)
                        del ref_outputs, ref_lp_chunk, ref_mask_chunk
                        del ref_prompt, ref_mask, ref_tok

                    ref_lp_per_token = torch.cat(ref_lp_chunks, dim=0)
                    ref_loss_mask = torch.cat(ref_mask_chunks, dim=0)
                    if ref_model_device_name == "cpu":
                        ref_lp_per_token = ref_lp_per_token.to(device)
                        ref_loss_mask = ref_loss_mask.to(device)

                measure_timer("Reference replay")

                # Policy Update ############################################################################################################

                total_rollouts = token_ids.shape[0]
                total_nonmasked = ref_loss_mask.sum().clamp(min=1)
                max_log_ratio = torch.tensor(0.0, device=device)
                max_log_ratio_ref = torch.tensor(0.0, device=device)
                step_loss = 0.0

                optimizer.zero_grad()
                parallel_model.module.train()

                for mb_start in range(0, total_rollouts, policy_mb_size):
                    mb_end = min(mb_start + policy_mb_size, total_rollouts)
                    if restore_policy_rng:
                        restore_rng_state(
                            all_rollout_rng_states[mb_start // rollout_chunk_size],
                            device,
                        )

                    mb_prompt = prompt_ids_cpu[mb_start:mb_end]
                    mb_mask = prompt_masks_cpu[mb_start:mb_end]
                    mb_tok = token_ids[mb_start:mb_end]
                    mb_adv = advantages[mb_start:mb_end]
                    # D1: theta_old log-prob = ratio denominator.
                    mb_old_lp = theta_old_lp[mb_start:mb_end]
                    # D3: ref log-prob used only for the KL anchor.
                    mb_ref_lp = ref_lp_per_token[mb_start:mb_end]
                    mb_ref_mask = ref_loss_mask[mb_start:mb_end]

                    mb_new_outputs = parallel_model(
                        input_ids=mb_prompt,
                        attention_mask=mb_mask,
                        replay_generated_ids=mb_tok,
                    )
                    mb_new_lp, _ = compute_log_probs(
                        mb_tok,
                        mb_new_outputs,
                        prompt_len,
                        num_latents,
                        tokenizer.eos_token_id,
                    )

                    # PPO clip on rho = pi_theta / pi_{theta_old} (eq 4.6).
                    mb_log_ratio = mb_new_lp - mb_old_lp
                    mb_log_ratio_clamped = mb_log_ratio.clamp(-5, 5)
                    mb_ratio = torch.exp(mb_log_ratio_clamped)
                    mb_clipped = torch.clamp(
                        mb_ratio,
                        1.0 - configs.clip_ratio_lower_bound,
                        1.0 + configs.clip_ratio_upper_bound,
                    )

                    t1 = mb_ratio * mb_adv.unsqueeze(-1)
                    t2 = mb_clipped * mb_adv.unsqueeze(-1)
                    mb_policy_loss = -torch.min(t1, t2)

                    # k3 KL estimator against pi_ref (eq 4.7 KL term).
                    mb_log_ratio_ref = mb_new_lp - mb_ref_lp
                    mb_log_ratio_ref_clamped = mb_log_ratio_ref.clamp(-5, 5)
                    mb_ratio_ref = torch.exp(mb_log_ratio_ref_clamped)
                    mb_kl = mb_ratio_ref - mb_log_ratio_ref_clamped - 1.0

                    mb_total_loss = mb_policy_loss + configs.kl_beta * mb_kl
                    mb_loss = (mb_total_loss * mb_ref_mask).sum() / total_nonmasked

                    mb_loss.backward()
                    step_loss += mb_loss.item()
                    if mb_log_ratio.numel() > 0:
                        max_log_ratio = torch.max(
                            max_log_ratio,
                            mb_log_ratio.detach().abs().max(),
                        )
                        max_log_ratio_ref = torch.max(
                            max_log_ratio_ref,
                            mb_log_ratio_ref.detach().abs().max(),
                        )
                    del mb_new_outputs, mb_new_lp, mb_log_ratio, mb_log_ratio_clamped
                    del mb_ratio, mb_clipped, t1, t2, mb_policy_loss
                    del mb_log_ratio_ref, mb_log_ratio_ref_clamped, mb_ratio_ref, mb_kl
                    del mb_total_loss, mb_loss
                    del mb_prompt, mb_mask, mb_tok, mb_adv, mb_old_lp, mb_ref_lp, mb_ref_mask

                measure_timer("Policy replay")

                torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                if wandb_step_payload is not None:
                    wandb_step_payload.update(
                        {
                            "train/loss": step_loss,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/max_log_ratio": max_log_ratio.item(),
                            "train/max_log_ratio_ref": max_log_ratio_ref.item(),
                            "train/zero_reward_frac": sum(1 for r in all_rewards if r <= 0) / max(len(all_rewards), 1),
                        }
                    )
                    wandb_run.log(wandb_step_payload, step=global_step)

                pbar.update(1)
                pbar.set_description(
                    f"GRPO Epoch: {epoch + 1}/{configs.num_epochs}, "
                    f"step {step}/{len(train_dataloader)} (loss: {step_loss:.4f})"
                )

                del token_ids, prompt_ids_cpu, prompt_masks_cpu, reward_tensor, advantages
                del generated_all, padded_token_ids, ref_lp_per_token, ref_loss_mask
                del theta_old_lp, theta_old_loss_mask
                del theta_old_lp_chunks, theta_old_mask_chunks
                del all_token_ids, all_prompt_ids, all_prompt_masks, all_rollout_rng_states
                del repeated_input_ids, repeated_attention_mask
                del ref_lp_chunks, ref_mask_chunks
                del wandb_step_payload
                gc.collect()
                torch.cuda.empty_cache()
                measure_timer("Step cleanup")
                torch.cuda.reset_peak_memory_stats()

                if (step + 1) % configs.eval_per_steps == 0:
                    eval_savepath_step = os.path.join(
                        save_dir,
                        f"checkpoint_{epoch + 1}_step_{step + 1}_evaluation_{current_time}.txt",
                    )
                    correct_step, total_step = run_validation(
                        parallel_model,
                        valid_gen_dataloader,
                        tokenizer,
                        configs,
                        latent_id,
                        eval_num_latents,
                        max_new_tokens,
                        device,
                        eval_savepath_step,
                        question_val,
                        rank,
                        desc=f"Eval @ step {step + 1}",
                    )
                    if rank == 0:
                        eval_acc = correct_step / max(total_step, 1)
                        print(
                            f"[Step {step + 1}] Validation accuracy: "
                            f"{correct_step} / {total_step} = {eval_acc:.4f}"
                        )
                        if wandb_run:
                            wandb_run.log(
                                {
                                    "eval/acc": eval_acc,
                                    "eval/correct": correct_step,
                                    "eval/total": total_step,
                                },
                                step=global_step,
                            )
                    measure_timer("Periodic eval")
                    torch.cuda.reset_peak_memory_stats()

            pbar.close()

            if not configs.save_only_improve and not configs.debug and not configs.only_eval:
                if rank == 0:
                    ckpt_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}_{current_time}.pt")
                    save_model(parallel_model, ckpt_path)
                dist.barrier()

        eval_savepath = os.path.join(
            save_dir,
            f"checkpoint_{epoch + 1}_evaluation_{current_time}.txt",
        )
        correct_end, total_end = run_validation(
            parallel_model,
            valid_gen_dataloader,
            tokenizer,
            configs,
            latent_id,
            eval_num_latents,
            max_new_tokens,
            device,
            eval_savepath,
            question_val,
            rank,
            desc="Test Accuracy",
        )

        if rank == 0:
            eval_acc_end = correct_end / max(total_end, 1)
            print(
                f"Accuracy on validation set: {correct_end} / {total_end} = {eval_acc_end}"
            )
            if wandb_run:
                wandb_run.log({"eval/acc": eval_acc_end})

        if configs.only_eval:
            break

        dist.barrier()

    if wandb_run:
        wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
