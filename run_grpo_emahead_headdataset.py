#!/usr/bin/env python3
"""Two-phase head training: dataset construction + balanced MLP training.

Phase 1 — Build dataset (expensive, runs once):
  For each training batch, cycle through N in [min_latent..max_latent]:
    generate_batched_n -> check correctness
    forward() -> extract per-position feature vectors [current, running_mean, diff]
    Keep only correct samples (clean "stop at N" signal)
  Save dataset to file.

Phase 2 — Train head (cheap, many epochs):
  Load dataset, oversample "stop" class to match "continue" count,
  train the head MLP directly on balanced feature vectors.
  No LLM forward needed — just a tiny MLP on pre-extracted features.
  Load trained weights back into model, save, and evaluate.

Usage:
    torchrun --nnodes 1 --nproc_per_node 1 run_grpo_emahead_headdataset.py args/gsm_vllr_grpo_emahead_headdataset.yaml
"""

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import wandb
from coconut_grpo_emahead import Coconut
from dataset import get_dataset, get_grpo_dataset, MyCollator

import gc
import json
import yaml
import random
import datetime
import argparse
import functools
import bitsandbytes as bnb
from tqdm import tqdm
from utils import Config, set_seed


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def extract_answer(text, eot_token):
    return text.split("#")[-1].replace(",", "").replace(eot_token, "").strip()


def compute_reward(generated_ids, tokenizer, ground_truth, latent_id,
                   coconut_mode=True):
    accuracies, extracted_answers, n_latents = [], [], []
    for i in range(generated_ids.shape[0]):
        text = tokenizer.decode(generated_ids[i, :], skip_special_tokens=not coconut_mode)
        answer = extract_answer(text, "<|endoftext|>")
        extracted_answers.append(answer)
        num_latent = (generated_ids[i, :] == latent_id).sum().item()
        n_latents.append(num_latent)
        accuracies.append(1.0 if answer == ground_truth[i] else 0.0)
    accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return accuracy, extracted_answers, accuracies, n_latents


def save_model(parallel_model, save_path):
    with FSDP.summon_full_params(parallel_model, writeback=False, rank0_only=False):
        state_dict = {k: v.detach().cpu().clone()
                      for k, v in parallel_model.module.state_dict().items()}
    torch.save({"model_state_dict": state_dict}, save_path)
    print(f"Saved model checkpoint at {save_path}!")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_hidden(hidden_states, len_question, n_latent, ema_decay):
    """Extract per-position [current, running_mean, last-current] feature vectors.

    Mirrors the feature computation in Coconut.forward() and
    DifferentialTerminationHead.forward(), so that Phase 2 training is on
    the exact same features the head sees at inference time.

    Args:
        hidden_states: (B, L, hidden) from forward().output_embeds
        len_question:  padded question length (includes <start-latent> marker)
        n_latent:      number of latent tokens appended after question
        ema_decay:     EMA decay factor

    Returns:
        features_list: list of (B, hidden*3) tensors, one per position
        labels_list:   list of int labels (same label for all B samples)
    """
    q_last = len_question - 1
    features_list, labels_list = [], []

    # q_last position: label=1 (next is latent), diff=0
    cur_q = hidden_states[:, q_last, :]
    feat_q = torch.cat([cur_q, cur_q, torch.zeros_like(cur_q)], dim=-1)
    features_list.append(feat_q)
    labels_list.append(1)

    # Latent positions: EMA seeded from q_last hidden state
    ema = hidden_states[:, q_last, :].clone()
    for k in range(n_latent):
        pos = len_question + k
        cur = hidden_states[:, pos, :]
        prev = hidden_states[:, pos - 1, :]
        ema = ema_decay * ema + (1.0 - ema_decay) * cur
        diff = prev - cur                       # last_hidden - current_hidden
        feat = torch.cat([cur, ema.clone(), diff], dim=-1)
        features_list.append(feat)
        labels_list.append(0 if k == n_latent - 1 else 1)

    return features_list, labels_list


# ===========================================================================
# Phase 1: Dataset construction
# ===========================================================================

def build_dataset(parallel_model, train_dataloader, tokenizer, latent_id,
                  max_new_tokens, min_latent, max_latent, ema_decay,
                  construction_steps, rank, save_path):
    """Generate with various N, check correctness, extract features from
    correct samples.  Returns and saves the dataset dict."""

    n_values = list(range(min_latent, max_latent + 1))
    all_features = []       # list of (B_correct, hidden*3) tensors
    all_labels = []         # list of (B_correct,) long tensors
    total_correct = 0
    total_samples = 0
    n_stop = 0
    n_continue = 0

    pbar = tqdm(colour="green", desc="Phase 1: Building dataset",
                total=min(construction_steps, len(train_dataloader)),
                dynamic_ncols=True)

    parallel_model.module.eval()

    for step, batch in enumerate(train_dataloader):
        if step >= construction_steps:
            break

        batch_len, len_question = batch["input_ids"].shape
        answers = batch["answer"]
        blacklist = {"idx", "answer"}
        batch = {k: batch[k].to(rank) for k in batch if k not in blacklist}

        # Cycle through N values for even coverage
        N = n_values[step % len(n_values)]

        # ---- Generate with N latent tokens ----
        with torch.no_grad():
            with FSDP.summon_full_params(parallel_model):
                generated = parallel_model.module.generate_batched_n(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_latent=N,
                    synced_gpus=True,
                )

        # ---- Check correctness ----
        _, _, accuracies, _ = compute_reward(
            generated, tokenizer, answers, latent_id, coconut_mode=True)
        del generated

        correct_mask = torch.tensor([a > 0.5 for a in accuracies], dtype=torch.bool)
        n_correct = correct_mask.sum().item()
        total_correct += n_correct
        total_samples += batch_len

        if n_correct == 0:
            pbar.update(1)
            continue

        # ---- Forward on correct samples to get hidden states ----
        correct_ids = batch["input_ids"][correct_mask]
        correct_attn = batch["attention_mask"][correct_mask]

        latent_suffix = torch.full((n_correct, N), latent_id,
                                   device=rank, dtype=torch.long)
        train_ids = torch.cat([correct_ids, latent_suffix], dim=1)
        train_mask = torch.cat([
            correct_attn,
            torch.ones(n_correct, N, device=rank, dtype=correct_attn.dtype)
        ], dim=1)
        train_pos = train_mask.long().cumsum(-1) - 1
        train_pos.clamp_(min=0)

        with torch.no_grad():
            with FSDP.summon_full_params(parallel_model):
                outputs = parallel_model.module.forward(
                    input_ids=train_ids,
                    attention_mask=train_mask,
                    position_ids=train_pos,
                    labels=None,
                )
        h = outputs.output_embeds       # (n_correct, L, hidden)

        # ---- Extract per-position features ----
        feat_list, lab_list = extract_features_from_hidden(
            h, len_question, N, ema_decay)

        for feat, lab in zip(feat_list, lab_list):
            all_features.append(feat.cpu().float())     # store in float32
            all_labels.append(torch.full((n_correct,), lab, dtype=torch.long))
            if lab == 0:
                n_stop += n_correct
            else:
                n_continue += n_correct

        del outputs, h
        gc.collect()
        torch.cuda.empty_cache()

        pbar.update(1)
        pbar.set_description(
            f"Phase 1: {total_correct}/{total_samples} correct, "
            f"stop={n_stop} cont={n_continue}")

    pbar.close()

    if len(all_features) == 0:
        raise RuntimeError("No correct samples found during dataset construction!")

    features = torch.cat(all_features)       # (total, hidden*3)
    labels = torch.cat(all_labels)           # (total,)

    dataset = {
        'features': features,
        'labels': labels,
        'hidden_size': features.shape[1] // 3,
        'ema_decay': ema_decay,
        'n_stop': n_stop,
        'n_continue': n_continue,
        'total_correct': total_correct,
        'total_samples': total_samples,
    }

    torch.save(dataset, save_path)
    if rank == 0:
        print(f"\nDataset saved to {save_path}")
        print(f"  Total feature vectors: {len(features)}")
        print(f"  Stop (label 0): {n_stop}  |  Continue (label 1): {n_continue}")
        print(f"  Imbalance ratio: {n_continue / max(n_stop, 1):.1f}:1")

    return dataset


# ===========================================================================
# Phase 2: Balanced head training
# ===========================================================================

def train_head(parallel_model, dataset, configs, rank, save_dir,
               wandb_run=None):
    """Train the head MLP on a class-balanced feature dataset.
    Runs entirely on pre-extracted features — no LLM forward needed."""

    features = dataset['features']           # (total, hidden*3) float32
    labels = dataset['labels']               # (total,) long
    hidden_size = dataset['hidden_size']
    bottleneck_ratio = getattr(configs, 'bottleneck_ratio', 4)
    bf16 = getattr(configs, 'bf16', True)

    # ---- Balance classes by oversampling "stop" ----
    stop_idx = (labels == 0).nonzero(as_tuple=True)[0]
    cont_idx = (labels == 1).nonzero(as_tuple=True)[0]
    n_stop = len(stop_idx)
    n_cont = len(cont_idx)

    if n_stop == 0:
        raise RuntimeError("No stop samples in dataset — cannot train.")

    # Oversample stop to match continue count
    repeats = n_cont // n_stop
    remainder = n_cont - repeats * n_stop
    stop_expanded = stop_idx.repeat(repeats)
    if remainder > 0:
        extra = stop_idx[torch.randperm(n_stop)[:remainder]]
        stop_expanded = torch.cat([stop_expanded, extra])

    balanced_idx = torch.cat([cont_idx, stop_expanded])
    balanced_features = features[balanced_idx]
    balanced_labels = labels[balanced_idx]

    if rank == 0:
        print(f"Balanced dataset: {len(balanced_features)} samples "
              f"(stop={len(stop_expanded)}, continue={n_cont})")

    # ---- Create standalone head for fast training ----
    bottleneck = hidden_size // bottleneck_ratio
    head = nn.Sequential(
        nn.Linear(hidden_size * 3, bottleneck),
        nn.GELU(),
        nn.Linear(bottleneck, 2),
    ).to(rank).float()

    # Copy current weights from the FSDP model
    with FSDP.summon_full_params(parallel_model, writeback=False, rank0_only=False):
        src_state = parallel_model.module.latent_termination_head.head.state_dict()
        head.load_state_dict({k: v.float().clone() for k, v in src_state.items()})

    # ---- Training loop ----
    head_lr = getattr(configs, 'head_train_lr', 1e-3)
    head_epochs = getattr(configs, 'head_train_epochs', 20)
    head_mb = getattr(configs, 'head_train_mb_size', 512)

    optimizer = torch.optim.Adam(head.parameters(), lr=head_lr, weight_decay=1e-4)

    total_mb_steps = head_epochs * ((len(balanced_features) + head_mb - 1) // head_mb)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_mb_steps, eta_min=head_lr * 0.01)

    criterion = CrossEntropyLoss()

    if rank == 0:
        print(f"Training head for {head_epochs} epochs, "
              f"lr={head_lr}, mb_size={head_mb}, "
              f"total_mb_steps={total_mb_steps}")

    for epoch in range(head_epochs):
        perm = torch.randperm(len(balanced_features))
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_stop_correct = 0
        epoch_stop_total = 0
        n_batches = 0

        for mb_start in range(0, len(balanced_features), head_mb):
            mb_end = min(mb_start + head_mb, len(balanced_features))
            idx = perm[mb_start:mb_end]
            mb_feat = balanced_features[idx].to(rank)
            mb_lab = balanced_labels[idx].to(rank)

            logits = head(mb_feat)
            loss = criterion(logits, mb_lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(dim=-1)
            epoch_loss += loss.item()
            epoch_correct += (preds == mb_lab).sum().item()
            # Track stop-class recall specifically
            stop_mask = (mb_lab == 0)
            epoch_stop_correct += ((preds == 0) & stop_mask).sum().item()
            epoch_stop_total += stop_mask.sum().item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_correct / len(balanced_features)
        stop_recall = epoch_stop_correct / max(epoch_stop_total, 1)

        if rank == 0:
            print(f"  Epoch {epoch + 1:3d}/{head_epochs}: "
                  f"loss={avg_loss:.4f}  acc={avg_acc:.2%}  "
                  f"stop_recall={stop_recall:.2%}")

        if wandb_run and rank == 0:
            wandb_run.log({
                "head_train/loss": avg_loss,
                "head_train/accuracy": avg_acc,
                "head_train/stop_recall": stop_recall,
                "head_train/lr": optimizer.param_groups[0]['lr'],
                "head_train/epoch": epoch + 1,
            })

    # ---- Load trained weights back into the full model ----
    trained_state = head.state_dict()
    if bf16:
        trained_state = {k: v.to(torch.bfloat16) for k, v in trained_state.items()}

    with FSDP.summon_full_params(parallel_model, writeback=True):
        parallel_model.module.latent_termination_head.head.load_state_dict(
            trained_state)

    if rank == 0:
        print("Trained head weights loaded back into full model.")

    del head
    return parallel_model


# ===========================================================================
# Evaluation
# ===========================================================================

def run_evaluation(parallel_model, valid_gen_dataloader, tokenizer,
                   latent_id, max_new_tokens, rank, save_dir, tag):
    """Evaluate using generate_batched (termination-head driven)."""
    eval_log_path = os.path.join(save_dir, f"eval_{tag}.log")

    with torch.no_grad():
        parallel_model.module.eval()
        all_correct, all_total = 0, 0
        all_n_latents = []

        for vbatch in tqdm(valid_gen_dataloader, desc=f"Eval ({tag})",
                           dynamic_ncols=True, colour="cyan"):
            vbatch_len, vlen_q = vbatch["input_ids"].shape
            vanswers = vbatch["answer"]
            vbatch = {k: vbatch[k].to(rank) for k in vbatch
                      if k not in {"idx", "answer"}}

            with FSDP.summon_full_params(parallel_model):
                vgenerated = parallel_model.module.generate_batched(
                    vbatch["input_ids"],
                    attention_mask=vbatch["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    synced_gpus=True,
                    term_temperature=0.0,
                )

            _, vans, vaccs, vn_lats = compute_reward(
                vgenerated, tokenizer, vanswers, latent_id)
            all_correct += sum(vaccs)
            all_total += len(vaccs)
            all_n_latents.extend(vn_lats)

            with open(eval_log_path, "a") as f:
                for i in range(vbatch_len):
                    q = tokenizer.decode(
                        vbatch['input_ids'][i]).replace("<|endoftext|>", "")
                    g = tokenizer.decode(
                        vgenerated[i, vlen_q:]).replace("<|endoftext|>", "")
                    f.write(f"Q: {q}\nGen: {g}\n"
                            f"Pred='{vans[i]}' GT={vanswers[i]} "
                            f"Correct={vaccs[i]:.0f}\n\n")

        acc = all_correct / max(all_total, 1)
        avg_lat = sum(all_n_latents) / max(len(all_n_latents), 1)

        if rank == 0:
            print(f"[Eval {tag}] Accuracy: {acc:.2%}, "
                  f"Avg latent: {avg_lat:.1f}")

    return acc, avg_lat


# ===========================================================================
# Main
# ===========================================================================

def main():
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Head dataset training")
    parser.add_argument("config_file")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

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
    dist.barrier(device_ids=[local_rank])

    if rank == 0:
        import shutil
        for _src in [__file__,
                     os.path.join(os.path.dirname(__file__),
                                  "coconut_grpo_emahead.py"),
                     args.config_file]:
            shutil.copy2(_src, os.path.join(save_dir, os.path.basename(_src)))

    # ---- Model setup (same as headonly script) ----------------------------
    model_config = AutoConfig.from_pretrained(configs.model_id)
    model_config.attention_dropout = getattr(configs, 'dropout', 0.0)

    load_kwargs = {}
    if getattr(configs, 'sdpa_attention', False):
        load_kwargs["attn_implementation"] = "sdpa"
    if getattr(configs, 'bf16', True):
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id, config=model_config, **load_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if rank == 0:
        print(f"latent_id={latent_id}, start_id={start_id}, end_id={end_id}")

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for tid in [latent_id, start_id, end_id]:
        embeddings.weight.data[tid] = embeddings.weight.data[target_id].clone()
        model.lm_head.weight.data[tid] = model.lm_head.weight.data[target_id].clone()

    loaded = False
    saved_weights = None
    if configs.load_model_path != "None":
        ckpt = torch.load(configs.load_model_path, map_location='cpu')
        saved_weights = ckpt["model_state_dict"]
        del ckpt
        if not any(k.startswith("base_causallm") for k in saved_weights):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id,
                    getattr(configs, 'termination_gamma', 1.0),
                    ema_decay=getattr(configs, 'ema_decay', 0.9),
                    bottleneck_ratio=getattr(configs, 'bottleneck_ratio', 4))

    if saved_weights is not None and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))
    del saved_weights

    model = model.to(rank)
    if getattr(configs, 'bf16', True):
        model.to(torch.bfloat16)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer})
    parallel_model = FSDP(model, auto_wrap_policy=llama_auto_wrap_policy,
                          device_id=rank, use_orig_params=True)
    del model

    # Freeze everything except termination head
    with FSDP.summon_full_params(parallel_model, writeback=True):
        for name, p in parallel_model.module.named_parameters():
            p.requires_grad = name.startswith("latent_termination_head")

    if rank == 0:
        n_train = sum(p.numel() for p in parallel_model.parameters()
                      if p.requires_grad)
        n_total = sum(p.numel() for p in parallel_model.parameters())
        print(f"Trainable: {n_train:,} / {n_total:,} "
              f"({100 * n_train / n_total:.4f}%)")

    # ---- Datasets ---------------------------------------------------------
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    max_new_tokens = getattr(configs, 'max_new_tokens', 128)
    min_latent = getattr(configs, 'min_latent', 3)
    max_latent = getattr(configs, 'max_latent', 18)
    ema_decay = getattr(configs, 'ema_decay', 0.9)
    no_special = (getattr(configs, 'cot', False)
                  or getattr(configs, 'no_cot', False)
                  or getattr(configs, 'no_thoughts', False)
                  or getattr(configs, 'no_bot_tokens', False))

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer,
        max_size=32 if getattr(configs, 'debug', False) else 100000000)
    dataset_gen_val = get_grpo_dataset(base_dataset_valid, start_id,
                                       no_special_marker=no_special)
    valid_gen_dataloader = torch.utils.data.DataLoader(
        dataset_gen_val, num_workers=1, pin_memory=True,
        batch_size=getattr(configs, 'eval_batch_size', 1),
        collate_fn=collator,
        sampler=DistributedSampler(dataset_gen_val, shuffle=False))

    # Wandb
    if not getattr(configs, 'debug', False) and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
    else:
        wandb_run = None

    only_eval = getattr(configs, 'only_eval', False)

    # ---- Pre-training evaluation ------------------------------------------
    if rank == 0:
        print("\n" + "=" * 60)
        print("Pre-training evaluation")
        print("=" * 60)
    pre_acc, pre_lat = run_evaluation(
        parallel_model, valid_gen_dataloader, tokenizer,
        latent_id, max_new_tokens, rank, save_dir, "pre_train")

    if only_eval:
        dist.destroy_process_group()
        return

    # ---- Phase 1: Build or load feature dataset ---------------------------
    dataset_load_path = getattr(configs, 'dataset_load_path', 'None')

    if dataset_load_path != 'None' and os.path.exists(dataset_load_path):
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Loading pre-built dataset from {dataset_load_path}")
            print(f"{'=' * 60}")
        feature_dataset = torch.load(dataset_load_path, map_location='cpu')
        if rank == 0:
            print(f"  Loaded {len(feature_dataset['features'])} samples "
                  f"(stop={feature_dataset['n_stop']}, "
                  f"continue={feature_dataset['n_continue']})")
    else:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print("Phase 1: Building feature dataset")
            print(f"{'=' * 60}")

        construction_steps = getattr(configs, 'construction_steps', 200)
        total_train = (3200 if getattr(configs, 'debug', False)
                       else int(construction_steps * configs.train_batch_size * 2))
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=total_train)
        dataset_train = get_grpo_dataset(
            base_dataset_train, start_id, no_special_marker=no_special,
            shuffle=True, max_question_len=128,
            num_samples=construction_steps * configs.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train, num_workers=1, shuffle=False, pin_memory=True,
            batch_size=configs.train_batch_size, collate_fn=collator,
            sampler=DistributedSampler(dataset_train, shuffle=True))

        dataset_save_path = os.path.join(save_dir, "head_features.pt")
        feature_dataset = build_dataset(
            parallel_model, train_dataloader, tokenizer, latent_id,
            max_new_tokens, min_latent, max_latent, ema_decay,
            construction_steps, rank, dataset_save_path)

        del train_dataloader, dataset_train, base_dataset_train
        gc.collect()

    # ---- Phase 2: Train head on balanced dataset --------------------------
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Phase 2: Training head on balanced dataset")
        print(f"{'=' * 60}")

    parallel_model = train_head(
        parallel_model, feature_dataset, configs, rank, save_dir, wandb_run)

    # ---- Save model -------------------------------------------------------
    model_save_path = os.path.join(save_dir, "model_trained_head.pt")
    if rank == 0:
        save_model(parallel_model, model_save_path)

    # ---- Post-training evaluation -----------------------------------------
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Post-training evaluation")
        print(f"{'=' * 60}")
    post_acc, post_lat = run_evaluation(
        parallel_model, valid_gen_dataloader, tokenizer,
        latent_id, max_new_tokens, rank, save_dir, "post_train")

    # ---- Summary ----------------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"  Pre-training:  acc={pre_acc:.2%}  avg_latent={pre_lat:.1f}")
        print(f"  Post-training: acc={post_acc:.2%}  avg_latent={post_lat:.1f}")
        print(f"  Model saved to: {model_save_path}")

    if wandb_run and rank == 0:
        wandb_run.log({
            "summary/pre_accuracy": pre_acc,
            "summary/pre_avg_latent": pre_lat,
            "summary/post_accuracy": post_acc,
            "summary/post_avg_latent": post_lat,
        })

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
