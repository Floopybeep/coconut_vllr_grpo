#!/usr/bin/env python3
"""
Wandb hyperparameter sweep for GRPO + Coconut latent reasoning.

Usage:
  1. Create the sweep (prints sweep ID):
       python run_grpo_tfhead_sweep.py --create-sweep

  2. Start an agent (single-GPU, auto-initialises distributed for FSDP):
       CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>

     For parallel agents on different GPUs:
       CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id> &
       CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id> &

  Each run trains for SWEEP_TRAIN_STEPS steps, evaluating periodically.
  The sweep optimises eval/best_accuracy via Bayesian search.
"""

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import wandb
import gc
import copy
import json
import yaml
import random
import argparse
import functools
import bitsandbytes as bnb
from tqdm import tqdm

from coconut_grpo_tfhead import Coconut
from dataset import get_dataset, get_grpo_dataset, MyCollator
from utils import Config, set_seed
from run_grpo_tfhead import compute_reward, compute_log_probs


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

SWEEP_PROJECT = "coconut-grpo-tfhead"
BASE_CONFIG = "args/gsm_vllr_grpo_coconut_tfhead.yaml"

SWEEP_TRAIN_STEPS = 200
SWEEP_EVAL_STEPS = 200

SWEEP_CONFIG = {
    "method": "bayes",
    "name": "grpo-hparam-sweep",
    "metric": {"name": "eval/best_accuracy", "goal": "maximize"},
    "program": "run_grpo_tfhead_sweep.py",
    "command": ["${env}", "${interpreter}", "${program}"],
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 3e-5,
        },
        "kl_beta": {
            "distribution": "log_uniform_values",
            "min": 0.001,
            "max": 0.1,
        },
        "clip_ratio": {
            "values": [0.2, 0.5, 1.0, 5.0, 10.0],
        },
        "term_temperature": {
            "distribution": "uniform",
            "min": 0.8,
            "max": 2.5,
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.20,
        }
    },
    "run_cap": 50,
}

# Sweep parameters that need to be applied at model construction time
# (before the model is loaded, since dropout is baked into the config).
_MODEL_INIT_PARAMS = {"dropout"}

# Sweep parameters that map to optimizer / training config.
_TRAIN_PARAMS = {"lr", "kl_beta", "clip_ratio", "term_temperature", "num_rollouts"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_distributed():
    """Initialise a single-GPU distributed env so FSDP works unchanged."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(random.randint(29500, 29999)))

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank


def create_sweep():
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=SWEEP_PROJECT)
    print(f"\nSweep created!")
    print(f"  Project:  {SWEEP_PROJECT}")
    print(f"  Sweep ID: {sweep_id}")
    entity = wandb.api.default_entity or "<your-wandb-username>"
    print(f"\nStart an agent:")
    print(f"  CUDA_VISIBLE_DEVICES=0 wandb agent {entity}/{SWEEP_PROJECT}/{sweep_id}")
    return sweep_id


def run_eval(parallel_model, dataloader, configs, rank, latent_id, tokenizer, max_new_tokens):
    """Run eval, return (accuracy, avg_reward, avg_n_latent)."""
    parallel_model.module.eval()
    all_accs, all_rewards, all_n_latents = [], [], []

    for batch in dataloader:
        batch_len, len_q = batch["input_ids"].shape
        answers = batch["answer"]
        batch = {k: batch[k].to(rank) for k in batch if k not in {"idx", "answer"}}

        with FSDP.summon_full_params(parallel_model):
            gen, _, _ = parallel_model.module.generate_batched(
                batch["input_ids"],
                max_new_tokens=max_new_tokens,
                output_embedding=True,
                synced_gpus=True,
            )

        rewards, acc, _, _, n_lats, _ = compute_reward(
            gen, tokenizer, answers, latent_id,
            length_penalty=configs.length_penalty_per_token,
            accuracy_score_ratio=configs.accuracy_reward_ratio,
            format_score_ratio=configs.format_reward_ratio,
            coconut_mode=configs.coconut,
        )
        all_accs.append(acc)
        all_rewards.extend(rewards)
        all_n_latents.extend(n_lats)

    accuracy = sum(all_accs) / len(all_accs)
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_n_lat = sum(all_n_latents) / len(all_n_latents)
    return accuracy, avg_reward, avg_n_lat


# ---------------------------------------------------------------------------
# Single sweep trial
# ---------------------------------------------------------------------------

def run_trial():
    rank = init_distributed()

    # ---- Base config ----
    with open(BASE_CONFIG) as f:
        config_dict = yaml.safe_load(f)

    # ---- Wandb init (connects to sweep agent automatically) ----
    wandb_run = wandb.init(project=SWEEP_PROJECT, reinit=True)
    sweep_overrides = dict(wandb.config)

    print("Sweep overrides:")
    for k, v in sweep_overrides.items():
        config_dict[k] = v
        print(f"  {k}: {v}")

    configs = Config(config_dict)
    set_seed(configs.seed)

    # ---- Model ----
    model_config = AutoConfig.from_pretrained(configs.model_id)
    model_config.attention_dropout = configs.dropout

    if configs.sdpa_attention:
        kwargs = {"attn_implementation": "sdpa"}
        if configs.bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id, config=model_config, **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(configs.model_id, config=model_config)

    if configs.grad_checkpointing:
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

    # ---- Load checkpoint ----
    loaded = False
    saved_weights = None

    if configs.load_model_path != "None":
        saved_checkpoint = torch.load(configs.load_model_path, map_location="cpu")
        saved_weights = saved_checkpoint["model_state_dict"]

        if configs.coconut and not any(
            k.startswith("base_causallm") for k in saved_weights
        ):
            loaded = True
            model.load_state_dict(saved_weights, strict=False)
        elif not configs.coconut and any(
            k.startswith("base_causallm") for k in saved_weights
        ):
            raise ValueError("Cannot load coconut weights into a causallm model")
        elif configs.coconut and any(
            k.startswith("base_causallm") for k in saved_weights
        ):
            pass  # will load after Coconut wrapper

    # ---- Token embeddings ----
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
            model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id].clone()

    # ---- Coconut wrapper + reference model ----
    ref_base = copy.deepcopy(model)
    ref_model = Coconut(
        ref_base, latent_id, start_id, end_id,
        tokenizer.eos_token_id, configs.termination_gamma,
    )
    if configs.bf16:
        ref_model = ref_model.to(dtype=torch.bfloat16)
    ref_model = ref_model.to(device="cpu")

    model = Coconut(
        model, latent_id, start_id, end_id,
        tokenizer.eos_token_id, configs.termination_gamma,
    )

    if configs.load_model_path != "None" and not loaded and saved_weights is not None:
        model.load_state_dict(saved_weights, strict=False)
    del saved_weights, saved_checkpoint
    gc.collect()

    model = model.to(rank)
    if configs.bf16:
        model.to(torch.bfloat16)

    llama_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    parallel_model = FSDP(
        model, auto_wrap_policy=llama_wrap, device_id=rank, use_orig_params=True,
    )
    del model

    # ---- Datasets ----
    base_train = get_dataset(configs.train_path, tokenizer, max_size=100000000)
    base_valid = get_dataset(configs.val_path, tokenizer, max_size=100000000)

    no_special = configs.cot or configs.no_cot or configs.no_thoughts or configs.no_bot_tokens
    dataset_train = get_grpo_dataset(base_train, start_id, no_special_marker=no_special, shuffle=True)
    dataset_valid = get_grpo_dataset(base_valid, start_id, no_special_marker=no_special)

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, num_workers=1, shuffle=False, pin_memory=True,
        batch_size=configs.train_batch_size, collate_fn=collator,
        sampler=DistributedSampler(dataset_train, shuffle=True),
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, num_workers=1, pin_memory=True,
        batch_size=configs.eval_batch_size, collate_fn=collator,
        sampler=DistributedSampler(dataset_valid, shuffle=False),
    )

    # ---- Optimizer ----
    optimizer = bnb.optim.Adam8bit(
        parallel_model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay,
    )

    max_new_tokens = configs.max_new_tokens
    best_accuracy = 0.0

    # ---- Training loop ----
    pbar = tqdm(total=SWEEP_TRAIN_STEPS, desc="Sweep trial", dynamic_ncols=True)

    for step, batch in enumerate(train_loader):
        if step >= SWEEP_TRAIN_STEPS:
            break

        # Refresh reference model
        if step > 0 and step % configs.ref_model_renewal_steps == 0:
            with FSDP.summon_full_params(parallel_model, writeback=False, rank0_only=False):
                sd = {k: v.detach().cpu().clone()
                      for k, v in parallel_model.module.state_dict().items()}
            ref_model.load_state_dict(sd, strict=False)
            del sd
            print(f"  Reference model refreshed at step {step}")

        # ---- Phase 1: Rollout generation ----
        parallel_model.module.train()
        batch_len, len_question = batch["input_ids"].shape
        batch["input_ids"] = torch.repeat_interleave(
            batch["input_ids"], configs.num_rollouts, dim=0,
        )
        answers = [a for a in batch["answer"] for _ in range(configs.num_rollouts)]
        batch = {k: batch[k].to(rank) for k in batch if k not in {"idx", "answer"}}

        with torch.no_grad():
            with FSDP.summon_full_params(parallel_model):
                generated, embeddings, _ = parallel_model.module.generate_batched(
                    batch["input_ids"],
                    max_new_tokens=max_new_tokens,
                    output_embedding=True,
                    synced_gpus=True,
                    term_temperature=configs.term_temperature,
                )

            rewards, step_acc, _, _, n_latents, _ = compute_reward(
                generated, tokenizer, answers, latent_id,
                length_penalty=configs.length_penalty_per_token,
                accuracy_score_ratio=configs.accuracy_reward_ratio,
                format_score_ratio=configs.format_reward_ratio,
                coconut_mode=configs.coconut,
            )

        # ---- Phase 2: Per-group normalised advantages ----
        token_ids = generated.to(rank)
        all_outputs = embeddings.cpu()
        all_rewards = torch.tensor(rewards, device=rank, dtype=torch.float32)

        rewards_grouped = all_rewards.view(batch_len, configs.num_rollouts)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = ((rewards_grouped - group_mean) / (group_std + 1e-8)).view(-1)

        # ---- Reference model log-probs ----
        with torch.no_grad():
            ref_model.eval()
            ref_out = ref_model(input_embeds=all_outputs, token_ids=token_ids.cpu())
            ref_lp, ref_mask = compute_log_probs(
                token_ids.cpu(), ref_out, len_question, latent_id, tokenizer.eos_token_id,
            )
            del ref_out
            gc.collect()
            ref_lp = ref_lp.to(rank)
            ref_mask = ref_mask.to(rank)

        # ---- Phase 3: Policy update (mini-batch) ----
        total_nonmasked = ref_mask.sum().clamp(min=1)
        total_rollouts = all_outputs.shape[0]
        mb_size = configs.train_minibatch_size
        step_loss = 0.0
        max_log_ratio = torch.tensor(0.0, device=rank)

        optimizer.zero_grad()
        parallel_model.module.eval()

        for mb_start in range(0, total_rollouts, mb_size):
            mb_end = min(mb_start + mb_size, total_rollouts)
            mb_embeds = all_outputs[mb_start:mb_end].to(rank)
            mb_tok = token_ids[mb_start:mb_end]
            mb_adv = advantages[mb_start:mb_end]
            mb_ref_lp = ref_lp[mb_start:mb_end]
            mb_ref_mask = ref_mask[mb_start:mb_end]

            mb_out = parallel_model(input_embeds=mb_embeds, token_ids=mb_tok)
            mb_new_lp, _ = compute_log_probs(
                mb_tok, mb_out, len_question, latent_id, tokenizer.eos_token_id,
            )
            del mb_out

            mb_log_ratio = mb_new_lp - mb_ref_lp
            mb_log_ratio_c = mb_log_ratio.clamp(-5.0, 5.0)
            mb_ratio = torch.exp(mb_log_ratio_c)
            mb_clipped = torch.clamp(
                mb_ratio, 1.0 - configs.clip_ratio, 1.0 + configs.clip_ratio,
            )

            t1 = mb_ratio * mb_adv.unsqueeze(-1)
            t2 = mb_clipped * mb_adv.unsqueeze(-1)
            policy_loss = -torch.min(t1, t2)

            kl = mb_ratio - mb_log_ratio_c - 1.0
            total_loss = policy_loss + configs.kl_beta * kl
            loss = (total_loss * mb_ref_mask).sum() / total_nonmasked

            loss.backward()
            step_loss += loss.item()
            max_log_ratio = torch.max(
                max_log_ratio, mb_log_ratio.detach().abs().max(),
            )

        parallel_model.clip_grad_norm_(max_norm=1.0)
        optimizer.step()

        del all_outputs, token_ids, all_rewards
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Logging ----
        avg_n_lat = sum(n_latents) / len(n_latents)
        wandb_run.log({
            "train/loss": step_loss,
            "train/batch_accuracy": step_acc,
            "train/batch_avg_reward": sum(rewards) / len(rewards),
            "train/avg_n_latent": avg_n_lat,
            "train/max_log_ratio": max_log_ratio.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
        })

        pbar.update(1)
        pbar.set_description(
            f"loss={step_loss:.4f} acc={step_acc:.1%} lat={avg_n_lat:.1f}"
        )

        # ---- Periodic evaluation ----
        if (step + 1) % SWEEP_EVAL_STEPS == 0:
            with torch.no_grad():
                eval_acc, eval_rew, eval_lat = run_eval(
                    parallel_model, valid_loader, configs,
                    rank, latent_id, tokenizer, max_new_tokens,
                )

            if eval_acc > best_accuracy:
                best_accuracy = eval_acc

            wandb_run.log({
                "eval/accuracy": eval_acc,
                "eval/reward": eval_rew,
                "eval/avg_n_latent": eval_lat,
                "eval/best_accuracy": best_accuracy,
            })
            print(
                f"  Step {step+1}: eval_acc={eval_acc:.2%} "
                f"best={best_accuracy:.2%} lat={eval_lat:.1f}"
            )

    pbar.close()
    wandb_run.finish()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wandb sweep for GRPO hyperparameter optimisation",
    )
    parser.add_argument(
        "--create-sweep", action="store_true",
        help="Register the sweep on wandb and print the sweep ID",
    )
    args = parser.parse_args()

    if args.create_sweep:
        create_sweep()
    else:
        run_trial()


if __name__ == "__main__":
    main()
