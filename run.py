# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

import gc
import yaml
import json
import datetime
import argparse
import itertools
import functools
import bitsandbytes as bnb
from tqdm import tqdm
from copy import copy
from utils import Config, set_seed

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


def main():

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
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Check if resume ##########################################################################################################
    # check if the job is preempted and resumed.
    if len([f for f in cur_ckpts if not f.endswith("txt")]) > 0 and not configs.only_eval and configs.resume == 0:      # configured to ignore txt logs
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
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
    if configs.sdpa_attention:
        if configs.bf16:
            model = AutoModelForCausalLM.from_pretrained(configs.model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(configs.model_id, attn_implementation="sdpa")
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

    loaded = False

    if configs.load_model_path != "None":
        # saved_checkpoint = torch.load(
        #     configs.load_model_path, map_location=torch.device(rank)
        # )
        # saved_weights = saved_checkpoint["model_state_dict"]
        saved_weights = torch.load(configs.load_model_path, map_location=torch.device(rank))

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

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

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

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )

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
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        # Load training dataset
        if not configs.only_eval:

            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer or epoch % configs.epochs_per_stage == 0:
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

            # Begin training #####################################################################################3
            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            current_table = wandb.Table(columns=["step", "text"]) 
            for step, batch in enumerate(train_dataloader):     # batch = batch of dicts(["question", "steps", "answer", "idx"])

                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])        # batch_size
                    text_str = ""

                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    current_table.add_data(total_train_steps, text_str)

                    # ... populate current_table ...

                    # text_table.add_data(total_train_steps, text_str)
                    # # copy the table due to a bug in wandb
                    # # https://github.com/wandb/wandb/issues/2981

                    # wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"        # set tokenized text to device(the GPU that current code runs in - multi-GPU setting)
                }
                if step == 0 and wandb_run and rank == 0:      # Change 5: only log data_table once at step 0, not every step
                    wandb_run.log({"data_table": current_table})

                # print(batch.keys())
                # print(batch["attention_mask"])
                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if step == 100:
                    print_memory_breakdown(parallel_model, optimizer)     # use to check if batch_size can be increased

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        # "train/loss": loss.detach().float() * configs.gradient_accumulation_steps,
                        "train/loss": loss.item() * configs.gradient_accumulation_steps,
                        "train/acc": torch.sum(torch.argmax(outputs.logits, dim=-1) == batch["input_ids"]).item() / (outputs.logits.shape[0] * outputs.logits.shape[1]),
                        "train/term_acc": torch.sum(torch.argmax(outputs.termination_logits, dim=-1) == outputs.termination_labels).item() / (outputs.logits.shape[0] * outputs.logits.shape[1])
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            # Save model after checkpoint
            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                states = parallel_model.state_dict()
                if rank == 0:
                    torch.save(
                        states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                    )
                    print("saving model.")

                dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # Validation step #######################################################################################
            # val loss
            total_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                if wandb_run and rank == 0:
                    log_dict = {
                        "val/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("validation loss", total_loss / len(valid_loss_dataloader))

        # val generation accuracy
        total_length = len(valid_gen_dataloader)

        # Evaluation step #################################################################################################
        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
        )

        with torch.no_grad():
            parallel_model.module.eval()
            eval_savepath = os.path.join(save_dir, f"checkpoint_{epoch + 1}_evaluation_{current_time}.txt")
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }
                # https://github.com/huggingface/transformers/issues/32492

                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                outputs = parallel_model.module.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    synced_gpus=not configs.only_eval,
                )

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                if idx < 5 and rank == 0:
                    # print some examples
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )
                
                if rank == 0:
                    with open(eval_savepath, "a+") as fp:
                        fp.write(f"\nQuestion {test_idx+1}:\nCoT = {answer_cot}\n")
                        fp.write(f"Full output:\n{tokenizer.decode(outputs[0])}\n")
                        fp.write(f"Extracted Output:\n{answer_output}\n")
                        fp.write(f"Answer = {answer}\n")
                        fp.write("-" * 30)

            pbar.close()
            print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        if rank == 0:
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        if configs.only_eval:
            break

        dist.barrier()
        if (
            cor / total > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": parallel_model.state_dict(),
                "optimimzer_state_dict": optimizer.state_dict()
            }
            # states = parallel_model.state_dict()

            if rank == 0:
                torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch + 1}_{current_time}.pt"))
                print("saving model.")

            best_acc = cor / total

            dist.barrier()
            del states
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
