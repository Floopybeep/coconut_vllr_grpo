#!/usr/bin/env python3
"""
Wandb hyperparameter sweep for baseline Coconut SFT training.

Sweeps over c_thought and uniform_prob. Each trial trains for 100 steps
then runs a single evaluation pass.

Usage:
  1. Create the sweep (prints sweep ID):
       python run_baseline_sweep.py --create-sweep

  2. Start an agent (single-GPU, auto-initialises distributed for FSDP):
       CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>

     For parallel agents on different GPUs:
       CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id> &
       CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id> &
"""

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import wandb
import gc
import json
import yaml
import random
import argparse
import functools
import bitsandbytes as bnb
from tqdm import tqdm

from coconut_baseline import Coconut
from dataset import get_dataset, get_question_latent_dataset, get_cot_latent_dataset, MyCollator
from utils import Config, set_seed

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

SWEEP_PROJECT = "coconut-baseline-sweep"
BASE_CONFIG = "args/gsm_vllr_coconut_baseline.yaml"

SWEEP_TRAIN_STEPS = 300

SWEEP_CONFIG = {
    "method": "bayes",
    "name": "baseline-c_thought-uniform_prob",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "program": "run_baseline_sweep.py",
    "command": ["${env}", "${interpreter}", "${program}"],
    "parameters": {
        "c_thought": {
            "values": [1, 2, 3],
        },
        "uniform_prob": {
            "values": [0.0, 0.5, 1.0],
        },
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 3e-5,
        }
    },
}


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

    # Force training from scratch for fair comparison
    config_dict["load_model_path"] = "None"
    config_dict["resume"] = 0
    config_dict["only_eval"] = False
    config_dict["debug"] = False

    configs = Config(config_dict)
    set_seed(configs.seed)

    # ---- Model ----
    if configs.sdpa_attention:
        kwargs = {"attn_implementation": "sdpa"}
        if configs.bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(configs.model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(configs.model_id)

    if configs.grad_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # ---- Token embeddings ----
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
            model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id].clone()

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    # ---- Coconut wrapper ----
    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, configs.termination_gamma)

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
    base_dataset_train = get_dataset(configs.train_path, tokenizer, max_size=100000000)
    base_dataset_valid = get_dataset(configs.val_path, tokenizer, max_size=100000000)

    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))]

    # Use scheduled_stage based on the config (start from stage 1 for coconut training)
    scheduled_stage = 1

    no_special = configs.cot or configs.no_cot or configs.no_thoughts or getattr(configs, 'no_bot_tokens', False)

    dataset_train = get_cot_latent_dataset(
        scheduled_stage, base_dataset_train, configs,
        start_id, latent_id, end_id,
        no_special_marker=no_special,
        shuffle=True,
    )

    dataset_gen_val = get_question_latent_dataset(
        scheduled_stage, base_dataset_valid, configs,
        start_id, latent_id, end_id,
        no_special_marker=no_special,
    )

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, num_workers=1, shuffle=False, pin_memory=True,
        batch_size=configs.batch_size_training, collate_fn=collator,
        sampler=DistributedSampler(dataset_train, shuffle=True),
    )

    valid_gen_loader = torch.utils.data.DataLoader(
        dataset_gen_val, num_workers=1, pin_memory=True,
        batch_size=1, collate_fn=collator,
        sampler=DistributedSampler(dataset_gen_val, shuffle=False),
    )

    max_new_tokens = 128

    # ---- Optimizer ----
    optimizer = bnb.optim.Adam8bit(
        parallel_model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay,
    )

    # ---- Training loop (100 steps) ----
    parallel_model.module.train()
    pbar = tqdm(total=SWEEP_TRAIN_STEPS, desc="Sweep trial", dynamic_ncols=True)

    for step, batch in enumerate(train_loader):
        if step >= SWEEP_TRAIN_STEPS:
            break

        batch = {key: batch[key].to(rank) for key in batch.keys() if key != "idx"}

        outputs = parallel_model(**batch)
        loss = outputs.loss / configs.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % configs.gradient_accumulation_steps == 0 or step == SWEEP_TRAIN_STEPS - 1:
            optimizer.step()
            optimizer.zero_grad()

        wandb_run.log({
            "train/step": step,
            "train/loss": loss.item() * configs.gradient_accumulation_steps,
        })

        pbar.update(1)
        pbar.set_description(
            f"loss={loss.item() * configs.gradient_accumulation_steps:.4f}"
        )

    pbar.close()

    # ---- Evaluation (single pass) ----
    print("Running evaluation...")
    cor = torch.tensor(0, device=rank)
    total = torch.tensor(0, device=rank)

    with torch.no_grad():
        parallel_model.module.eval()
        pbar = tqdm(total=len(valid_gen_loader), desc="Eval", dynamic_ncols=True)

        for idx, batch in enumerate(valid_gen_loader):
            test_idx = batch["idx"][0]
            q = batch["input_ids"][0]

            batch = {
                k: v.to(rank)
                for k, v in batch.items()
                if v is not None and k not in ["idx", "position_ids"]
            }

            assert len(batch["input_ids"]) == 1
            answer = answers_val[test_idx.cpu().item()]
            total += 1

            if not configs.coconut:
                batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v is not None and k != "latent_tokens"
                }

            with FSDP.summon_full_params(parallel_model):
                gen_outputs = parallel_model.module.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=True,
                )

            text_output = tokenizer.decode(gen_outputs[0], skip_special_tokens=not configs.coconut)
            answer_output = text_output.split("#")[-1].replace(",", "").strip()

            cor += answer_output == answer

            if idx == 0:
                print(tokenizer.decode(q, skip_special_tokens=not configs.coconut))
                print(text_output)
                print(answer_output)

            pbar.update(1)
            pbar.set_description(
                f"Eval acc: {float(cor) / float(total):.2%}"
            )

        pbar.close()

    dist.all_reduce(cor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    accuracy = cor.item() / max(total.item(), 1)
    print(f"Eval accuracy: {cor.item()} / {total.item()} = {accuracy:.2%}")

    wandb_run.log({
        "eval/accuracy": accuracy,
        "eval/correct": cor.item(),
        "eval/total": total.item(),
    })

    wandb_run.finish()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wandb sweep for baseline Coconut c_thought and uniform_prob",
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
