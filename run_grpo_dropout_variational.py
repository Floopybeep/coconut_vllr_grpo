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
#   V2 (revised): configure_variational_dropout sets ALL dropout config
#       attrs to the requested rate, including attention_dropout. The
#       original plan zeroed attention_dropout for a clean Bayesian story,
#       but Qwen2 / LLaMA / Mistral / Gemma have no other dropout layers
#       in their architecture, so zeroing it left the model deterministic
#       and rollouts collapsed to identical generations. With attention
#       dropout active, the variational interpretation is approximate on
#       these models (attention mask shape varies with kv-cache length so
#       per-step masks differ within a chain), but the proof framework
#       (D1 / D2 / D3, Theorem 4.4, Prop 4.5) is unaffected --- those only
#       require xi to be theta-independent and replayable, which holds.
#       For GPT-2-family models that ship with embd_pdrop / resid_pdrop,
#       the residual and MLP dropouts still get a true locked mask across
#       t via CoconutVariational's RNG restore, recovering the clean
#       Bayesian story for those layers.
#   V3: ref_model_renewal_steps = 0 in the YAML -> pi_ref is frozen at
#       SFT init for the entire run. Matches Theorem 4.4 corollary's
#       requirement that the KL anchor be a fixed reference distribution.
#       Strongly recommended; refresh > 0 is left as an option for staged
#       training but is not proof-faithful.
#   V4: Huberized k3 KL. The prior log_ratio.clamp(-5, 5) zeroed the
#       gradient outside the bound, so any token with |log_ratio_ref| > 5
#       escaped KL regularization entirely (a pre-collapse signature on
#       long runs). The new estimator keeps k3 inside |r| <= delta and
#       extends linearly outside, value- and slope-matched at the
#       boundary, so KL stays C^1 with bounded but nonzero tail gradient.
#   V5: kl_beta annealed alongside lr -- current_kl_beta =
#       configs.kl_beta * (lr_now / lr_peak). Without this, lr cosine-
#       annealing leaves a constant KL pull on a shrinking step, dragging
#       theta back toward the frozen ref. With it, the trust region
#       shrinks proportionally and each step retains the same fractional
#       KL pull.
#   V6: Sequence-mean loss aggregation (per draft Algorithm 1, standard
#       GRPO). Per-token loss is averaged within each rollout, then
#       averaged across rollouts. Replaces the prior token-mean
#       (sum / total_tokens) which underweighted short responses and
#       entangled per-sequence weight with sequence length.
#   V8: Dr.GRPO advantage (Liu et al. 2024). compute_group_advantages
#       drops the sigma_r normalization: A^(k) = r^(k) - mu_r instead
#       of (r^(k) - mu_r) / (sigma_r + eps). The std-divided form is
#       a biased estimator of the policy gradient because sigma_r is
#       a nonlinear function of every r^(j) including r^(k), so the
#       control-variate identity that zeros out a baseline in
#       expectation does NOT extend to dividing by it. The bias
#       reweights prompts by inverse difficulty variance and was the
#       gap in the previous Theorem 4.4 proof. With the fix, the
#       gradient is unbiased up to a constant (1 - 1/K) scale that
#       folds into the learning rate. See revised draft Theorem 4.4
#       in research_draft_revised.tex.
#   V7: Group-level filter on rollout accuracy. After rollouts, drop
#       whole groups whose accuracy is outside
#       [filter_min_accuracy, filter_max_accuracy] (defaults 0.05, 0.95)
#       BEFORE theta_old replay / ref replay / policy update. Saves
#       ~25% of step time on filtered groups and concentrates the
#       gradient on "marginal" prompts where the reward signal is
#       strongest. Acts as an online curriculum -- as the policy
#       improves, easy groups drop out automatically.
#       Constraint: rollout_chunk_size MUST equal num_rollouts so each
#       chunk is one group; this preserves the chunk-aligned
#       RNG-restore replay structure. Validated at step start; raises
#       a clear ValueError if violated. Toggle via enable_group_filter
#       (default False so existing YAMLs are unaffected).
#       Skip behavior: an all_reduce on the per-rank kept count makes
#       a global skip decision. If NO rank has kept groups, every rank
#       skips the policy update entirely (no optimizer.step, no
#       lr_scheduler.step) -- the prior fallback of forcing zero-var
#       groups through produced a pure-KL gradient toward ref, which
#       is exactly the dynamic the filter is meant to prevent. If at
#       least one rank has kept groups, locally-empty ranks fall back
#       to using all their chunks so DDP backward stays in lockstep;
#       the resulting pure-KL gradient on those ranks is diluted by
#       1/world_size in the gradient average.
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
import math
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


def compute_log_probs(
    token_ids,
    model_outputs,
    prompt_len,
    num_latents,
    eos_id,
    return_entropy=False,
):
    response_start = prompt_len + num_latents
    if token_ids.shape[1] <= response_start:
        zero_lp = torch.zeros(token_ids.shape[0], 0, device=token_ids.device)
        zero_mask = torch.zeros(
            token_ids.shape[0], 0, dtype=torch.bool, device=token_ids.device
        )
        if return_entropy:
            return zero_lp, zero_mask, zero_lp.clone()
        return zero_lp, zero_mask

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
    if return_entropy:
        # Token-level entropy H[pi(.|s_t)] = -sum_v p log p over the full vocab.
        # Detached: used for diagnostics, not for the loss.
        with torch.no_grad():
            entropy = -(lm_log_p_gen.exp() * lm_log_p_gen).sum(-1).float()
        return pred_token, loss_mask, entropy
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


# Dr.GRPO (Liu et al. 2024, "Understanding R1-Zero-Like Training")
# advantage: A^(k) = r^(k) - mean(r), no std normalization.
#
# The original GRPO advantage A^(k) = (r^(k) - mu_r) / (sigma_r + eps)
# is a biased estimator of the policy gradient: sigma_r is a nonlinear
# function of every r^(j) in the group (including r^(k)), so the
# control-variate identity that justifies subtracting a baseline does
# not extend to dividing by it. The bias upweights low-variance
# (easy / hard) prompts and downweights high-variance ones, distorting
# the per-prompt learning signal.
#
# Dropping sigma_r recovers the standard control-variate proof:
# E[(r^(k) - mu_r) * grad log pi^(k)] = (1 - 1/K) * grad J(theta)
# per prompt, where the (1 - 1/K) factor is a constant scale absorbed
# into the learning rate. This matches Theorem 4.4 of the revised
# draft (research_draft_revised.tex).
def compute_group_advantages(rewards, num_rollouts):
    if rewards.numel() == 0:
        return rewards
    grouped = rewards.view(-1, num_rollouts)
    group_mean = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - group_mean
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


# V2 (revised): variational variant.
#
# Original V2 plan: zero attention_dropout because its mask shape grows
# with kv-cache length, breaking the RNG-restore variational trick.
#
# Reality: Qwen2 / LLaMA / Mistral / Gemma decoders have NO residual,
# MLP, or embedding dropout in their architectures. attention_dropout
# is the only dropout layer that exists. Zeroing it leaves the model
# with zero stochasticity, all rollouts collapse to identical
# generations, and group advantages are 0.
#
# Resolution: leave attention_dropout at the configured rate. On these
# architectures, the variational interpretation degrades to "regular
# per-step dropout on attention probs" (mask shape varies with t), but
# the GRPO proof framework (Theorem 4.4 rho_old==1, Prop 4.5 CRN
# coupling) is unaffected because those only require xi to be
# theta-independent and replayable, not strictly variational.
#
# For models that DO have residual / MLP dropout (GPT-2 family), this
# function still routes the configured rate to those layers, where the
# RNG-restore in CoconutVariational does produce a true locked mask
# across t -- giving the clean Bayesian story for those layers.
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
            setattr(model_config, attr, dropout)


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

    if configs.verify_replay_embeddings:
        model.enable_mask_audit(strict=True)

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
            base_dataset_primary = get_dataset(configs.train_path, tokenizer, max_size=n_primary)

            n_secondary = total_train_samples - len(base_dataset_primary)
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
        start_factor=2e-7 / configs.lr,
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
            best_eval_acc = 0.0

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
                        if current_step <= warmup_steps and configs.enable_faster_warmup_renewal
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
                all_chunk_accuracies = []
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
                    getattr(configs, "policy_minibatch_size", configs.train_minibatch_size * 4),
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
                # V7: group filter requires one chunk == one group so the
                # chunk-aligned RNG-restore replay structure stays intact
                # when whole groups are dropped.
                if getattr(configs, "enable_group_filter", False) and rollout_chunk_size != configs.num_rollouts:
                    raise ValueError(
                        f"enable_group_filter requires rollout_chunk_size == num_rollouts "
                        f"(got rollout_chunk_size={rollout_chunk_size}, num_rollouts={configs.num_rollouts}). "
                        f"Set both to the same value in the YAML."
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
                        # V7: per-chunk accuracy for the group filter. With
                        # rollout_chunk_size == num_rollouts (validated above
                        # when the filter is enabled), one chunk == one group.
                        all_chunk_accuracies.append(
                            sum(accuracies) / max(len(accuracies), 1)
                        )
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
                        "train_stat/epoch": epoch + 1,
                        "train_stat/step": step + 1,
                        # "train/global_step": global_step,
                        "train_rollout/batch_avg_reward": avg_reward,
                        "train_rollout/batch_avg_base_reward": sum(all_base_rewards) / max(len(all_base_rewards), 1),
                        "train_rollout/batch_accuracy": all_correct / max(all_total, 1),
                        "train_rollout/batch_n_latent_avg": num_latents,
                        "train_rollout/batch_n_latent_correct_avg": n_latents_correct_sum / n_latents_correct_num if n_latents_correct_num > 0 else 0,
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

                # Group-Level Filter #############################################################################################################
                # V7: drop whole groups whose accuracy is outside
                # [filter_min_accuracy, filter_max_accuracy] before any
                # downstream replay or policy update. Saves ~25% of step
                # time on filtered groups (theta_old replay + ref replay +
                # policy backward) and concentrates compute on "marginal"
                # prompts where the gradient signal is strongest.
                # Curriculum-friendly: as the policy improves, easy
                # (acc -> 1) and hard (acc -> 0) groups drop out
                # automatically.
                #
                # Constraint: rollout_chunk_size == num_rollouts (validated
                # at step start) so each chunk corresponds to exactly one
                # group. Filtering whole chunks preserves the chunk-aligned
                # RNG-restore replay structure required for variational
                # dropout bit-exact reproduction.
                #
                # Diagnostics: wandb logs train_rollout/filter_kept_frac
                # (fraction of groups passing the filter). Pre-filter
                # rollout stats (batch_avg_reward, batch_accuracy,
                # first_batch_outputs) reflect the full batch; downstream
                # stats (frac_zero_var_groups, adv_abs_mean, etc.) are
                # computed on the filtered subset.
                filter_kept_frac = 1.0
                skip_step = False
                if getattr(configs, "enable_group_filter", False):
                    filter_min = getattr(configs, "filter_min_accuracy", 0.05)
                    filter_max = getattr(configs, "filter_max_accuracy", 0.95)
                    chunk_keep = [
                        filter_min <= acc <= filter_max
                        for acc in all_chunk_accuracies
                    ]
                    n_kept = sum(chunk_keep)
                    n_chunks = len(chunk_keep)

                    # All-reduce the kept count so every rank makes the
                    # same skip decision. DDP requires backward() to run
                    # in lockstep across ranks; if some ranks skip and
                    # others don't, the participating ranks hang on grad
                    # sync. We coordinate globally:
                    #   global == 0 -> all ranks skip the policy update
                    #     (no optimizer.step, no lr_scheduler.step). Avoids
                    #     the pure-KL drag the prior fallback introduced
                    #     when zero-var groups were forced through.
                    #   global > 0 but local == 0 -> this rank participates
                    #     via local fallback (use all its chunks). Its
                    #     pure-KL gradient is diluted by 1/world_size in
                    #     the DDP gradient average.
                    n_kept_tensor = torch.tensor(n_kept, device=device, dtype=torch.long)
                    dist.all_reduce(n_kept_tensor, op=dist.ReduceOp.SUM)
                    n_kept_global = n_kept_tensor.item()

                    if n_kept_global == 0:
                        if rank == 0:
                            print(
                                f"[Step {step + 1}] No groups passed filter "
                                f"[{filter_min}, {filter_max}] on any rank; "
                                f"skipping policy update."
                            )
                        skip_step = True
                        filter_kept_frac = 0.0
                    elif n_kept == 0:
                        chunk_keep = [True] * n_chunks
                        n_kept = n_chunks
                        filter_kept_frac = 0.0
                    else:
                        filter_kept_frac = n_kept / max(n_chunks, 1)

                    if not skip_step and n_kept < n_chunks:
                        # Apply filter to chunk-level lists.
                        all_token_ids = [
                            t for t, k in zip(all_token_ids, chunk_keep) if k
                        ]
                        all_prompt_ids = [
                            t for t, k in zip(all_prompt_ids, chunk_keep) if k
                        ]
                        all_prompt_masks = [
                            t for t, k in zip(all_prompt_masks, chunk_keep) if k
                        ]
                        all_rollout_rng_states = [
                            s for s, k in zip(all_rollout_rng_states, chunk_keep) if k
                        ]
                        # Apply filter to rollout-level rewards (chunks of
                        # num_rollouts each).
                        filtered_rewards = []
                        for chunk_idx, keep in enumerate(chunk_keep):
                            if keep:
                                start = chunk_idx * configs.num_rollouts
                                end = start + configs.num_rollouts
                                filtered_rewards.extend(all_rewards[start:end])
                        all_rewards = filtered_rewards
                        # Slice generated_all to keep only filtered rows.
                        row_keep = []
                        for keep in chunk_keep:
                            row_keep.extend([keep] * configs.num_rollouts)
                        row_keep_tensor = torch.tensor(
                            row_keep, dtype=torch.bool, device=generated_all.device
                        )
                        generated_all = generated_all[row_keep_tensor]

                if wandb_step_payload is not None:
                    wandb_step_payload["train_rollout/filter_kept_frac"] = filter_kept_frac

                measure_timer("Group filter")

                # Skip-step short-circuit: when the global filter dropped
                # every group, log rollout/filter stats only and move to
                # the next batch. lr_scheduler and optimizer are NOT
                # stepped (skipped step = no progress at all). Periodic
                # eval might miss firing if its trigger step coincides
                # with a skip, but with a sane filter this is rare and
                # the next non-skipped step still triggers eval normally.
                if skip_step:
                    if wandb_step_payload is not None:
                        wandb_run.log(wandb_step_payload, step=global_step)
                    pbar.update(1)
                    pbar.set_description(
                        f"GRPO Epoch: {epoch + 1}/{configs.num_epochs}, "
                        f"step {step}/{len(train_dataloader)} (skipped: filter empty)"
                    )
                    del all_token_ids, all_prompt_ids, all_prompt_masks, all_rollout_rng_states
                    del repeated_input_ids, repeated_attention_mask
                    del generated_all, padded_token_ids, wandb_step_payload
                    gc.collect()
                    torch.cuda.empty_cache()
                    measure_timer("Step cleanup (skipped)")
                    torch.cuda.reset_peak_memory_stats()
                    continue

                # Advantage Calculation ##########################################################################################################

                prompt_ids_cpu = torch.cat(all_prompt_ids, dim=0).to(device)
                prompt_masks_cpu = torch.cat(all_prompt_masks, dim=0).to(device)
                token_ids = generated_all.to(device)
                reward_tensor = torch.tensor(all_rewards, device=device, dtype=torch.float32)
                advantages = compute_group_advantages(reward_tensor, configs.num_rollouts)

                adv_clip = getattr(configs, "advantage_clip", 5.0)          # clip against advantage blow-up in near-uniform cases
                advantages = advantages.clamp(min=-adv_clip, max=adv_clip)

                # Advantage Diagnostics ##########################################################################################################
                # Reward / advantage signal health checks. Stagnant loss + high
                # grad_norm is usually caused by zero-variance groups (advantage
                # collapses to 0, gradient is pure KL noise). These metrics
                # surface that case directly.
                #   reward_std_within_group: mean within-group reward std.
                #     Healthy > 0.1, dead < 1e-3 (groups uniform -> no signal).
                #   adv_abs_mean: mean |advantage| over rollouts.
                #     Healthy > 0.3, dead < 1e-2.
                #   adv_zero_frac: fraction of rollouts with |adv| < 1e-6.
                #     Healthy < 0.3, dead > 0.7.
                #   frac_zero_var_groups: fraction of groups with std == 0
                #     (all rollouts identical reward -> contributes nothing
                #     to gradient). Tests bimodal-difficulty hypothesis:
                #     easy prompts get all-correct, hard prompts get all-
                #     wrong, only medium prompts produce signal. Healthy
                #     < 0.3, problematic > 0.5.
                with torch.no_grad():
                    reward_grouped = reward_tensor.view(-1, configs.num_rollouts)
                    group_std = reward_grouped.std(dim=1)
                    reward_std_within_group = group_std.mean().item()
                    frac_zero_var_groups = (group_std < 1e-6).float().mean().item()
                    adv_abs = advantages.abs()
                    adv_abs_mean = adv_abs.mean().item()
                    adv_abs_max = adv_abs.max().item()
                    adv_zero_frac = (adv_abs < 1e-6).float().mean().item()

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

                # Zero-Variance Group Masking ################################################################################################
                # DAPO-style "Dynamic Sampling" (Yu et al. 2024). Groups
                # with std == 0 (all rollouts identical reward) contribute
                # zero policy gradient -- their advantages are 0 by
                # construction. Without this mask, those tokens still
                # incur KL pull toward pi_ref, which uses up the KL
                # budget on uninformative samples and drowns the real
                # policy signal. Multiplying ref_loss_mask by the per-
                # rollout valid mask removes them from BOTH policy_loss
                # and kl_loss aggregation. total_nonmasked recomputed
                # below uses the masked version, so the loss denominator
                # shrinks honestly.
                if getattr(configs, "mask_zero_var_groups", True):
                    group_valid_per_group = (group_std > 1e-6).to(ref_loss_mask.device)
                    group_valid_per_rollout = group_valid_per_group.repeat_interleave(
                        configs.num_rollouts
                    )
                    effective_batch_frac = group_valid_per_group.float().mean().item()
                    ref_loss_mask = ref_loss_mask * group_valid_per_rollout.to(
                        ref_loss_mask.dtype
                    ).unsqueeze(-1)
                else:
                    effective_batch_frac = 1.0

                # Policy Update ############################################################################################################

                total_rollouts = token_ids.shape[0]
                total_nonmasked = ref_loss_mask.sum().clamp(min=1)
                max_log_ratio = torch.tensor(0.0, device=device)
                max_log_ratio_ref = torch.tensor(0.0, device=device)
                # Ref-drift diagnostics (sums weighted by ref_loss_mask;
                # divided by sum_log_ratio_count after the loop to get means).
                sum_log_ratio_ref_abs = torch.tensor(0.0, device=device)
                sum_clamp_count = torch.tensor(0.0, device=device)
                sum_log_ratio_count = torch.tensor(0.0, device=device)
                # Policy entropy diagnostics. Healthy: entropy stable or
                # slowly decreasing. Mode collapse signature: rapid decrease
                # toward 0 prior to reward collapse.
                sum_entropy = torch.tensor(0.0, device=device)
                sum_entropy_count = torch.tensor(0.0, device=device)
                step_loss = 0.0
                # policy_loss diagnostics, mask-weighted across the step.
                # Post-Dr.GRPO advantages sum to zero within each group, so
                # E[mb_policy_loss] is ~0 (sanity-check metric, kept as
                # sum_policy_loss). The actual gradient-flowing magnitude
                # is sum_policy_loss_abs = mean(|mb_policy_loss|), which
                # measures clipped per-token signal strength. Healthy:
                # roughly stable / slowly drifting. Collapse signature:
                # decay toward 0 prior to reward / accuracy plateau.
                sum_policy_loss = torch.tensor(0.0, device=device)
                sum_policy_loss_abs = torch.tensor(0.0, device=device)
                sum_policy_signal = torch.tensor(0.0, device=device)
                sum_kl_loss = 0.0

                optimizer.zero_grad()
                parallel_model.module.train()

                # KL annealing: scale kl_beta by current_lr / peak_lr so the
                # KL trust-region pull shrinks proportionally with the step
                # size. Without this, late-training (lr cosine-annealed to
                # eta_min) leaves a constant KL pull on a shrinking step,
                # which drags theta back toward the frozen ref ("trust-region
                # collapse"). With annealing, each step has the same fraction
                # of trust region. During warmup lr ramps from ~0 to peak,
                # so kl_beta also ramps up, letting the policy update freely
                # while still warming.
                current_kl_beta = configs.kl_beta * (
                    optimizer.param_groups[0]["lr"] / configs.lr
                )

                # Sequence-mean denominator: count rollouts that have at
                # least one unmasked response token. Zero-var groups are
                # already zeroed by ref_loss_mask (DAPO masking above), so
                # they don't contribute and aren't counted.
                total_valid_seqs = (
                    (ref_loss_mask.sum(dim=-1) > 0).sum().clamp(min=1).float()
                )

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
                    mb_new_lp, _, mb_entropy = compute_log_probs(
                        mb_tok,
                        mb_new_outputs,
                        prompt_len,
                        num_latents,
                        tokenizer.eos_token_id,
                        return_entropy=True,
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
                    # Mask-weighted policy_loss diagnostics. Aggregate sums
                    # here; divide once at log time by sum_log_ratio_count.
                    mask_f_loss = mb_ref_mask.float()
                    mb_pl_det = mb_policy_loss.detach()
                    sum_policy_loss = sum_policy_loss + (mb_pl_det * mask_f_loss).sum()
                    sum_policy_loss_abs = sum_policy_loss_abs + (mb_pl_det.abs() * mask_f_loss).sum()
                    sum_policy_signal = sum_policy_signal + (t1.detach().abs() * mask_f_loss).sum()

                    # Huberized k3 KL estimator against pi_ref (eq 4.7 KL term).
                    # Inside |r| <= delta: standard k3, exp(r) - r - 1.
                    # Outside: linear extension matched in value AND slope at
                    # +/- delta, so the KL stays continuously differentiable
                    # and the gradient stays nonzero in the tails. The prior
                    # log_ratio.clamp(-5, 5) zeroed the gradient outside the
                    # bound (autograd of clamp is 0 there), letting drifted
                    # tokens escape KL regularization entirely -- the
                    # pre-collapse signature on long runs.
                    mb_log_ratio_ref = mb_new_lp - mb_ref_lp
                    kl_huber_delta = getattr(configs, "kl_huber_delta", 5.0)
                    e_pos = math.exp(kl_huber_delta)
                    e_neg = math.exp(-kl_huber_delta)
                    # exp on clamped r is finite everywhere; gradient through
                    # this branch only flows where torch.where selects it
                    # (|r| <= delta), where r_clamped == r so the gradient
                    # matches the unclamped k3 gradient exp(r) - 1.
                    r_clamped = mb_log_ratio_ref.clamp(-kl_huber_delta, kl_huber_delta)
                    kl_inside = torch.exp(r_clamped) - mb_log_ratio_ref - 1.0
                    kl_pos = (e_pos - 1.0) * (mb_log_ratio_ref - kl_huber_delta) + (
                        e_pos - kl_huber_delta - 1.0
                    )
                    kl_neg = (e_neg - 1.0) * (mb_log_ratio_ref + kl_huber_delta) + (
                        e_neg + kl_huber_delta - 1.0
                    )
                    mb_kl = torch.where(
                        mb_log_ratio_ref > kl_huber_delta,
                        kl_pos,
                        torch.where(
                            mb_log_ratio_ref < -kl_huber_delta,
                            kl_neg,
                            kl_inside,
                        ),
                    )
                    sum_kl_loss += mb_kl.detach().mean().item()

                    mb_total_loss = mb_policy_loss + current_kl_beta * mb_kl
                    # Sequence-mean (per draft Algorithm 1 / standard GRPO):
                    # average per-token loss within each rollout, then average
                    # across rollouts. Token-mean (sum / total_tokens) under-
                    # weights short responses and entangles per-sequence weight
                    # with sequence length. Zero-var rollouts are zeroed in
                    # mb_ref_mask so they contribute 0 / max(0,1) = 0 here, and
                    # are excluded from total_valid_seqs.
                    mb_per_seq_token_count = mb_ref_mask.sum(dim=-1).clamp(min=1)
                    mb_per_seq_loss = (
                        mb_total_loss * mb_ref_mask
                    ).sum(dim=-1) / mb_per_seq_token_count
                    mb_loss = mb_per_seq_loss.sum() / total_valid_seqs

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
                        # Mask-weighted accumulators for mean and clamp fraction
                        # of |log_ratio_ref|. Aggregated across minibatches and
                        # divided once at log time below.
                        log_ratio_ref_abs = mb_log_ratio_ref.detach().abs()
                        mask_f = mb_ref_mask.float()
                        sum_log_ratio_ref_abs = sum_log_ratio_ref_abs + (log_ratio_ref_abs * mask_f).sum()
                        sum_clamp_count = sum_clamp_count + ((log_ratio_ref_abs > 5.0).float() * mask_f).sum()
                        sum_log_ratio_count = sum_log_ratio_count + mask_f.sum()
                        # Mask-weighted policy entropy.
                        if mb_entropy.numel() > 0:
                            sum_entropy = sum_entropy + (mb_entropy * mask_f).sum()
                            sum_entropy_count = sum_entropy_count + mask_f.sum()
                    del mb_entropy
                    del mb_new_outputs, mb_new_lp, mb_log_ratio, mb_log_ratio_clamped
                    del mb_ratio, mb_clipped, t1, t2, mb_policy_loss
                    del mb_log_ratio_ref, r_clamped, kl_inside, kl_pos, kl_neg, mb_kl
                    del mb_total_loss, mb_loss, mb_per_seq_loss, mb_per_seq_token_count
                    del mb_prompt, mb_mask, mb_tok, mb_adv, mb_old_lp, mb_ref_lp, mb_ref_mask

                measure_timer("Policy replay")

                # clip_grad_norm_ returns the total pre-clip L2 norm across
                # all parameters. Spike at end of LR warmup or pre-collapse
                # is the diagnostic signature of an LR that is too high.
                grad_clip_norm = getattr(configs, "grad_clip_norm", 0.5)
                pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parallel_model.parameters(), max_norm=grad_clip_norm
                )
                optimizer.step()
                lr_scheduler.step()

                if wandb_step_payload is not None:
                    log_ratio_count = sum_log_ratio_count.clamp(min=1)
                    mean_log_ratio_ref = (sum_log_ratio_ref_abs / log_ratio_count).item()
                    clamp_frac_ref = (sum_clamp_count / log_ratio_count).item()
                    entropy_count = sum_entropy_count.clamp(min=1)
                    mean_entropy = (sum_entropy / entropy_count).item()
                    # policy_loss_signed is the mask-weighted mean of
                    # -min(rho*A, clip*A). Post-Dr.GRPO it sits near 0
                    # because A^(k) sums to zero per group; deviation from
                    # 0 measures clip-induced asymmetry. policy_loss_abs
                    # is the mask-weighted mean of |min(rho*A, clip*A)| --
                    # the actual gradient-flowing magnitude. policy_signal
                    # is the pre-clip mean of |rho*A|; ratio
                    # policy_loss_abs / policy_signal measures clip
                    # bite-fraction (1.0 = no clipping, < 1.0 = clip
                    # actively dampening the update).
                    mean_policy_loss = (sum_policy_loss / log_ratio_count).item()
                    mean_policy_loss_abs = (sum_policy_loss_abs / log_ratio_count).item()
                    mean_policy_signal = (sum_policy_signal / log_ratio_count).item()
                    clip_bite_frac = (
                        mean_policy_loss_abs / mean_policy_signal
                        if mean_policy_signal > 1e-12
                        else 0.0
                    )
                    wandb_step_payload.update(
                        {
                            "train_stat/loss": step_loss,
                            "train_stat/policy_loss": mean_policy_loss,
                            "train_stat/policy_loss_abs": mean_policy_loss_abs,
                            "train_stat/policy_signal": mean_policy_signal,
                            "train_stat/clip_bite_frac": clip_bite_frac,
                            "train_stat/kl_loss": sum_kl_loss / log_ratio_count,
                            "train_stat/learning_rate": optimizer.param_groups[0]["lr"],
                            # Annealed kl_beta = configs.kl_beta * (lr / peak_lr).
                            # Tracks lr schedule so trust region scales with step
                            # size. Equals configs.kl_beta at lr peak (post-warmup),
                            # eta_min/lr * configs.kl_beta at the end of cosine.
                            "train_stat/kl_beta": current_kl_beta,
                            "train_policy/max_log_ratio": max_log_ratio.item(),
                            "train_policy/max_log_ratio_ref": max_log_ratio_ref.item(),
                            # Mean |log_ratio_ref| across non-masked tokens.
                            # Healthy: < 1. Concerning: > 3.
                            "train_policy/mean_log_ratio_ref": mean_log_ratio_ref,
                            # Fraction of non-masked tokens with
                            # |log_ratio_ref| > kl_huber_delta (default 5).
                            # Post-Huberization, KL still has nonzero
                            # (constant) gradient in this region, so these
                            # tokens are still regularized -- the fraction
                            # just measures how many tokens have entered
                            # the linear-tail regime. Persistent high values
                            # still indicate large drift from pi_ref.
                            # Healthy: < 5%. Concerning: > 30%.
                            "train_policy/clamp_frac_ref": clamp_frac_ref,
                            "train_adv/zero_reward_frac": sum(1 for r in all_rewards if r <= 0) / max(len(all_rewards), 1),
                            # Mean within-group reward std. Group-relative
                            # advantages depend on within-group variance.
                            # Healthy > 0.1, dead < 1e-3 (groups uniform).
                            "train_adv/reward_std_within_group": reward_std_within_group,
                            # Fraction of groups with std == 0 (all rollouts
                            # identical reward -> zero gradient contribution
                            # from this group). High value = bimodal prompt
                            # difficulty (easy or hard, no medium). Healthy
                            # < 0.3, problematic > 0.5.
                            "train_adv/frac_zero_var_groups": frac_zero_var_groups,
                            # Fraction of groups kept after DAPO-style zero-
                            # variance group masking (Yu et al. 2024). Equals
                            # 1 - frac_zero_var_groups when mask is enabled.
                            # Effective batch size = num_rollouts *
                            # effective_batch_frac.
                            "train_adv/effective_batch_frac": effective_batch_frac,
                            # Mean |advantage| across rollouts. Healthy > 0.3,
                            # dead < 1e-2 -> zero policy gradient signal.
                            "train_adv/adv_abs_mean": adv_abs_mean,
                            # Max |advantage| post-clip. If consistently at
                            # advantage_clip ceiling, near-uniform groups are
                            # blowing up the normalization (eps in
                            # compute_group_advantages too small) -- spike
                            # source for grad_norm.
                            "train_adv/adv_abs_max": adv_abs_max,
                            # Fraction of rollouts with |adv| < 1e-6. Healthy
                            # < 0.3, dead > 0.7. High value = groups uniform
                            # (all-correct or all-wrong), no learning signal.
                            "train_adv/adv_zero_frac": adv_zero_frac,
                            # Mean policy entropy over response tokens.
                            # Mode collapse signature: rapid decrease toward 0.
                            "train_policy/policy_entropy": mean_entropy,
                            # Total pre-clip gradient L2 norm. Healthy:
                            # roughly stable. Spike at end of warmup ->
                            # peak LR is too high.
                            "train_policy/grad_norm": pre_clip_grad_norm.item()
                            if torch.is_tensor(pre_clip_grad_norm)
                            else float(pre_clip_grad_norm),
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
                del sum_log_ratio_ref_abs, sum_clamp_count, sum_log_ratio_count
                del sum_entropy, sum_entropy_count
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

                        if not configs.save_only_improve or eval_acc >= best_eval_acc or (step + 1) % configs.mandatory_save_per_epochs == 0:
                            ckpt_path_eval = os.path.join(save_dir, "ref_models", f"checkpoint_{epoch + 1}_step_{step + 1}_{int(eval_acc * 10000)}.pt")
                            save_model(parallel_model, ckpt_path_eval)
                            if eval_acc > best_eval_acc:
                                best_eval_acc = eval_acc

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
