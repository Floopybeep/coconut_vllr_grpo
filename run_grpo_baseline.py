# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import torch
import torch.nn.functional as F
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

from coconut_baseline import Coconut
from dataset import get_dataset, MyCollator

import gc
import yaml
import json
import random
import datetime
import argparse
import functools
import bitsandbytes as bnb
from tqdm import tqdm
from utils import Config, set_seed


# ---------------------------------------------------------------------------
# GRPO with Dropout-based Diversity and Variable-Length Latent Chains
#
# The model is given ONLY the question as the prompt.  It generates its own
# latent chain (via the termination head) followed by the text answer.
# This enables end-to-end learning of "when and how long to think in latent
# space".
#
# Rollout diversity:  Dropout is kept ACTIVE during generation
# (model.train() + torch.no_grad()).  Greedy decoding (argmax) is used
# throughout — probabilistic token sampling is not meaningful in Coconut's
# latent space.  Different dropout masks across rollouts produce the
# necessary response variety.
#
# Reward (combined, range ≈ [0, 1]):
#   correctness_reward    = 1.0 if extracted answer == ground truth else 0.0
#   latent_efficiency     = max(0, 1 − 0.01 × max(0, n_latent − 1))
#   total_reward          = 0.5 × correctness + 0.5 × latent_efficiency
#
# Log-probability (for GRPO ratio):
#   Includes BOTH the termination-head decisions (latent vs. text) and the
#   lm-head token choices, so the policy gradient optimises the full
#   generative process end-to-end.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_question_only_dataset(base_dataset):
    """Return a dataset whose prompt is just the tokenised question.

    The model will generate both the latent chain and the text answer from
    this starting point, with the termination head controlling chain length.
    """
    def process(sample):
        tokens = sample["question_tokenized"]
        return {
            "input_ids":      tokens,
            "attention_mask": [1] * len(tokens),
            "position_ids":   list(range(len(tokens))),
            "idx":            sample["idx"],
        }
    return base_dataset.map(
        process, remove_columns=list(base_dataset.features), num_proc=32
    )


def load_gsm_symbolic_by_id(path):
    """Load a GSM-symbolic JSONL file and group records by question id.

    Returns:
        dict mapping question_id (int) → list of variant dicts.
    """
    by_id = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            by_id.setdefault(d["id"], []).append(d)
    return by_id


def get_gsm_symbolic_mixture_counts(epoch, num_epochs, total=50):
    """Piecewise-linear mixture schedule for GSM-symbolic augmentation.

    Three anchor points (fractions of `total` questions):

      epoch 0            : sym=75 %, p1=25 %,  p2= 0 %
      epoch middle*      : sym=25 %, p1=50 %,  p2=25 %
      epoch num_epochs-1 : sym= 0 %, p1=25 %,  p2=75 %

    *middle = floor((num_epochs − 1) / 2), so for num_epochs=9 this is
    epoch 4 (the 5th epoch).

    n_sym and n_p1 are independently rounded; n_p2 = total − n_sym − n_p1
    so the three counts always sum exactly to `total`.

    Args:
        epoch:      current epoch index (0-based, absolute within the run)
        num_epochs: total number of training epochs
        total:      total symbolic questions to add per epoch (default 50)

    Returns:
        (n_sym, n_p1, n_p2) — integer counts summing to `total`.
    """
    if num_epochs <= 1:
        # Edge case: single epoch → use start mixture
        n_sym = int(round(0.75 * total))
        n_p1  = total - n_sym
        return n_sym, n_p1, 0

    mid = (num_epochs - 1) / 2.0   # 4.0 for num_epochs=9

    if epoch <= mid:
        t   = epoch / mid
        sym = 0.75 + t * (0.25 - 0.75)   # 0.75 → 0.25
        p1  = 0.25 + t * (0.50 - 0.25)   # 0.25 → 0.50
    else:
        t   = (epoch - mid) / (num_epochs - 1 - mid)
        sym = 0.25 + t * (0.00 - 0.25)   # 0.25 → 0.00
        p1  = 0.50 + t * (0.25 - 0.50)   # 0.50 → 0.25

    n_sym = int(round(sym * total))
    n_p1  = int(round(p1  * total))
    n_p2  = total - n_sym - n_p1

    # Clamp to ≥ 0 in case rounding pushes n_p2 negative; absorb into n_p1
    if n_p2 < 0:
        n_p1 += n_p2
        n_p2  = 0

    return n_sym, n_p1, n_p2


def sample_gsm_symbolic_mixture(sym_by_id, p1_by_id, p2_by_id,
                                n_sym, n_p1, n_p2, rng):
    """Sample one variant per question from each GSM-symbolic dataset.

    For each dataset we:
      1. Randomly choose `n` distinct question IDs (without replacement).
      2. From each chosen ID pick one variant record at random.
      3. Extract the numeric answer from the "#### N" suffix.

    Args:
        sym_by_id, p1_by_id, p2_by_id:
            dicts mapping question_id → list of variant dicts,
            as returned by load_gsm_symbolic_by_id().
        n_sym, n_p1, n_p2: number of questions to draw from each source.
        rng: random.Random instance (seeded per-epoch for reproducibility).

    Returns:
        List of dicts, each with keys 'question' (str) and 'answer' (str).
    """
    result = []
    for by_id, count in [(sym_by_id, n_sym), (p1_by_id, n_p1), (p2_by_id, n_p2)]:
        if count == 0 or not by_id:
            continue
        ids        = list(by_id.keys())
        chosen_ids = rng.sample(ids, min(count, len(ids)))
        for qid in chosen_ids:
            variant    = rng.choice(by_id[qid])
            answer_str = variant["answer"].split("####")[-1].strip()
            result.append({"question": variant["question"], "answer": answer_str})
    return result


class SymbolicQuestionDataset(torch.utils.data.Dataset):
    """Minimal question-only prompt dataset for per-epoch symbolic samples.

    Indices are offset by `idx_offset` so they slot directly after the base
    training answers in the per-epoch combined answer list.
    """

    def __init__(self, samples, tokenizer, idx_offset):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.idx_offset = idx_offset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        tokens = self.tokenizer.encode(
            self.samples[i]["question"] + "\n", add_special_tokens=True
        )
        return {
            "input_ids":      tokens,
            "attention_mask": [1] * len(tokens),
            "position_ids":   list(range(len(tokens))),
            "idx":            self.idx_offset + i,
        }


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract the numerical answer following '###' from generated text."""
    return text.split("#")[-1].replace(",", "").strip()


def compute_combined_reward(generated_ids, tokenizer, ground_truth,
                            latent_id, question_len, coconut_mode=True):
    """Combined reward balancing correctness and latent-chain efficiency.

    Args:
        generated_ids: (1, L) tensor — full sequence including question tokens
        tokenizer:     tokenizer for decoding
        ground_truth:  expected answer string
        latent_id:     token id used for latent positions
        question_len:  number of question prompt tokens at the start of generated_ids
        coconut_mode:  whether to keep special tokens when decoding

    Returns:
        total_reward (float), n_latent (int)
    """
    text   = tokenizer.decode(generated_ids[0], skip_special_tokens=not coconut_mode)
    answer = extract_answer(text)
    correctness = 1.0 if answer == ground_truth else 0.0

    # Count latent tokens in the generated portion only
    gen_part = generated_ids[0, question_len:]
    n_latent = (gen_part == latent_id).sum().item()

    # Each latent token after the first costs 0.01
    latent_efficiency = max(0.0, 1.0 - 0.01 * max(0, n_latent - 1))

    total_reward = 0.5 * correctness + 0.5 * latent_efficiency
    return total_reward, n_latent


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # Init ####################################################################################################################
    parser = argparse.ArgumentParser(description="coconut-grpo")
    parser.add_argument("config_file")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs  = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier(device_ids=[local_rank])
    cur_ckpts    = os.listdir(save_dir)
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Check if resume ##########################################################################################################
    if (len([f for f in cur_ckpts if not f.endswith("txt") and not f.endswith("pt")]) > 0
            and not configs.only_eval and configs.resume == 0):
        if rank == 0:
            print("Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!")

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        latest_checkpoint    = checkpoints[-1] if checkpoints else None
        configs.resume       = int(latest_checkpoint.split("_")[1])
        configs.load_model_path = os.path.join(configs.save_path, configs.name, latest_checkpoint)
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        if configs.load_model_path == "None":
            print(f"Warning: skipping first {configs.resume} epochs without a checkpoint!")
        print(f"Loading from {configs.load_model_path} and skip first {configs.resume} epochs")

    # Define model #############################################################################################################
    if configs.sdpa_attention:
        if configs.bf16:
            model = AutoModelForCausalLM.from_pretrained(
                configs.model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                configs.model_id, attn_implementation="sdpa")
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
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded           = False
    previous_savepoint = None

    if configs.load_model_path != "None":
        saved_checkpoint = torch.load(configs.load_model_path, map_location='cpu')
        saved_weights    = saved_checkpoint["model_state_dict"]
        previous_savepoint = configs.load_model_path

        if configs.coconut and not any(k.startswith("base_causallm") for k in saved_weights):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))
        elif not configs.coconut and any(k.startswith("base_causallm") for k in saved_weights):
            raise ValueError("Cannot load coconut model weights into a causallm model")
        elif configs.coconut and any(k.startswith("base_causallm") for k in saved_weights):
            pass  # preempted run — handled below
        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id  = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding              = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id,
                        tokenizer.eos_token_id, configs.termination_gamma)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    # Reference model (frozen, for optional KL penalty) ########################################################################
    # Deep-copy BEFORE FSDP so the reference lives as a full replica on each rank.
    if configs.kl_coeff > 0.0:
        ref_model = copy.deepcopy(model)
        if configs.bf16:
            ref_model.to(torch.bfloat16)
        ref_model = ref_model.to(rank)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        if rank == 0:
            print("Reference model loaded for KL penalty.")
    else:
        ref_model = None

    # FSDP wrap policy model ###################################################################################################
    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    if configs.only_eval:
        parallel_model = DDP(model, device_ids=[rank])
    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy,
            device_id=rank, use_orig_params=True,
        )

    del model
    if rank == 0:
        print(parallel_model)

    # Data preparation ########################################################################################################
    question_val  = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val   = [d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))]
    cot_val       = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]
    answers_train = [d["answer"].replace(",", "").strip() for d in json.load(open(configs.train_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )
    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=3200 if configs.debug else 100000000
        )

    # Load GSM-symbolic augmentation datasets ###############################################################################
    # Each file is grouped by question id so we can sample 1 variant per question
    # each epoch without replacement across question ids.
    gsm_sym_by_id = {}
    gsm_p1_by_id  = {}
    gsm_p2_by_id  = {}

    gsm_symbolic_enabled = (
        not configs.only_eval
        and hasattr(configs, "gsm_symbolic_path")
        and hasattr(configs, "gsm_p1_path")
        and hasattr(configs, "gsm_p2_path")
    )
    if gsm_symbolic_enabled:
        gsm_sym_by_id = load_gsm_symbolic_by_id(configs.gsm_symbolic_path)
        gsm_p1_by_id  = load_gsm_symbolic_by_id(configs.gsm_p1_path)
        gsm_p2_by_id  = load_gsm_symbolic_by_id(configs.gsm_p2_path)
        if rank == 0:
            print(
                f"GSM-Symbolic loaded: {len(gsm_sym_by_id)} sym IDs, "
                f"{len(gsm_p1_by_id)} p1 IDs, {len(gsm_p2_by_id)} p2 IDs"
            )

    max_new_tokens = 128 if "gsm" in configs.val_path else 128

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
    else:
        wandb_run = None

    total_train_steps = 0
    best_acc          = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    optimizer = bnb.optim.Adam8bit(
        parallel_model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay,
    )

    # Main training loop ######################################################################################################
    for epoch in range(configs.resume, configs.num_epochs):

        # Remove old datasets
        for name in ('train_dataloader', 'dataset_train', 'valid_gen_dataloader', 'dataset_gen_val'):
            if name in locals():
                del locals()[name]
        gc.collect()

        # Validation dataset — question only; model generates latent chain + answer freely
        dataset_gen_val = get_question_only_dataset(base_dataset_valid)

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            # Training dataset — question only; model generates latent chain + answer on-the-fly
            base_grpo_dataset = get_question_only_dataset(base_dataset_train)

            # GSM-Symbolic augmentation: sample a fresh mixture every epoch
            if gsm_symbolic_enabled:
                n_sym, n_p1, n_p2 = get_gsm_symbolic_mixture_counts(
                    epoch, configs.num_epochs,
                    total=getattr(configs, "n_symbolic_per_epoch", 50),
                )
                epoch_rng    = random.Random(configs.seed + epoch)
                sym_sample   = sample_gsm_symbolic_mixture(
                    gsm_sym_by_id, gsm_p1_by_id, gsm_p2_by_id,
                    n_sym, n_p1, n_p2, epoch_rng,
                )
                # epoch_answers: base answers indexed 0..N-1, symbolic indexed N..N+49
                epoch_answers = answers_train + [s["answer"] for s in sym_sample]
                sym_dataset   = SymbolicQuestionDataset(
                    sym_sample, tokenizer, idx_offset=len(answers_train)
                )
                dataset_train = torch.utils.data.ConcatDataset(
                    [base_grpo_dataset, sym_dataset]
                )
                if rank == 0:
                    print(
                        f"Epoch {epoch + 1}: symbolic mixture — "
                        f"sym={n_sym}, p1={n_p1}, p2={n_p2} "
                        f"(total dataset size: {len(dataset_train)})"
                    )
            else:
                epoch_answers = answers_train
                dataset_train = base_grpo_dataset

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            if configs.reset_optimizer:
                del optimizer
                optimizer = bnb.optim.Adam8bit(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            # GRPO Training ###################################################################################################
            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"GRPO Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            # for step, batch in enumerate(train_dataloader):
            for step, batch in enumerate(train_dataloader):

                total_train_steps += 1

                # batch["input_ids"]:      (cur_bs, question_len) — raw question tokens
                # batch["attention_mask"]: (cur_bs, question_len)
                # batch["idx"]:            (cur_bs,)
                batch_idx_list       = batch["idx"]
                batch_input_ids      = batch["input_ids"].to(rank)
                batch_attention_mask = batch["attention_mask"].to(rank)
                cur_bs               = batch_input_ids.shape[0]

                # ---------------------------------------------------------------------------------
                # Phase 1: Rollout Generation
                #
                # Prompt = question only.  Coconut.generate() uses the termination head at each
                # step to decide latent vs. text, producing a variable-length latent chain
                # followed by the text answer — all driven by Dropout stochasticity.
                # ---------------------------------------------------------------------------------
                all_rollouts      = []   # list[list[Tensor(1, L_i)]]   (cur_bs × num_rollouts)
                all_rewards       = []   # list[list[float]]
                all_n_latents     = []   # list[list[int]]   latent count per rollout
                all_old_lp        = []   # list[list[Tensor(scalar)]]
                all_question_lens = []   # list[int] — actual (unpadded) question length per sample

                parallel_model.module.train()   # Dropout active

                with torch.no_grad():
                    for q_idx in range(cur_bs):
                        # Compute the actual (unpadded) question length for this sample.
                        # The collator right-pads questions in a batch; Coconut.generate()
                        # ignores the attention_mask (uses torch.ones_like internally), so
                        # we must strip padding before generating to avoid corrupted output.
                        q_len            = int(batch_attention_mask[q_idx].sum().item())
                        q_input_ids      = batch_input_ids[q_idx : q_idx + 1, :q_len]
                        q_attention_mask = batch_attention_mask[q_idx : q_idx + 1, :q_len]
                        ground_truth     = epoch_answers[batch_idx_list[q_idx].item()]

                        rollouts_q, rewards_q, n_lat_q, old_lp_q = [], [], [], []

                        for _ in range(configs.num_rollouts):
                            # Step A: generate one rollout (Dropout active, greedy decoding)
                            with FSDP.summon_full_params(parallel_model):
                                generated = parallel_model.module.generate(
                                    q_input_ids,
                                    q_attention_mask,
                                    max_new_tokens=max_new_tokens,
                                    synced_gpus=True,
                                )
                            # generated: (1, q_len + gen_len)

                            reward, n_latent = compute_combined_reward(
                                generated, tokenizer, ground_truth,
                                latent_id, q_len,
                                coconut_mode=configs.coconut,
                            )
                            rollouts_q.append(generated)
                            rewards_q.append(reward)
                            n_lat_q.append(n_latent)

                            # Step B: compute old log-probs (no_grad, same Dropout state is gone
                            #         but a fresh forward pass gives π_old for the ratio)
                            full_len = generated.shape[1]
                            if full_len > q_len:
                                full_attn = torch.ones(1, full_len, device=rank, dtype=torch.long)
                                full_pos  = torch.arange(0, full_len, device=rank).unsqueeze(0)
                                dummy_lbl = torch.full((1, full_len), -100, device=rank, dtype=torch.long)

                                fwd_out = parallel_model(
                                    input_ids=generated,
                                    attention_mask=full_attn,
                                    labels=dummy_lbl,
                                    position_ids=full_pos,
                                )
                                old_lp = compute_full_log_probs(
                                    fwd_out, generated, q_len, latent_id
                                ).detach()
                            else:
                                old_lp = torch.tensor(0.0, device=rank)

                            old_lp_q.append(old_lp)

                        all_rollouts.append(rollouts_q)
                        all_rewards.append(rewards_q)
                        all_n_latents.append(n_lat_q)
                        all_old_lp.append(old_lp_q)
                        all_question_lens.append(q_len)

                # ---------------------------------------------------------------------------------
                # Phase 2: Group-Normalised Advantages
                #
                # GRPO normalises within each question's G rollouts:
                #   A_i = (r_i − mean(r)) / (std(r) + ε)
                # ---------------------------------------------------------------------------------
                all_advantages = []
                for q_idx in range(cur_bs):
                    rewards    = torch.tensor(all_rewards[q_idx], dtype=torch.float32, device=rank)
                    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                    all_advantages.append(advantages)

                # ---------------------------------------------------------------------------------
                # Phase 3: GRPO Policy Loss (with gradients)
                #
                #   ratio        = exp(new_log_prob − old_log_prob)
                #   clipped      = clamp(ratio, 1−ε, 1+ε)
                #   rollout_loss = −min(ratio·A, clipped·A)
                #
                # log_prob covers BOTH termination decisions and lm-head token choices via
                # compute_full_log_probs(), so gradients flow through the entire policy.
                # ---------------------------------------------------------------------------------
                parallel_model.module.train()
                rollout_losses = []

                for q_idx in range(cur_bs):
                    q_len = all_question_lens[q_idx]
                    for r_idx in range(configs.num_rollouts):
                        generated = all_rollouts[q_idx][r_idx]
                        full_len  = generated.shape[1]

                        if full_len <= q_len:
                            continue    # no generated tokens; skip

                        advantage = all_advantages[q_idx][r_idx]
                        old_lp    = all_old_lp[q_idx][r_idx]

                        full_attn = torch.ones(1, full_len, device=rank, dtype=torch.long)
                        full_pos  = torch.arange(0, full_len, device=rank).unsqueeze(0)
                        dummy_lbl = torch.full((1, full_len), -100, device=rank, dtype=torch.long)

                        # New log-prob under current policy (grad enabled)
                        fwd_out = parallel_model(
                            input_ids=generated,
                            attention_mask=full_attn,
                            labels=dummy_lbl,
                            position_ids=full_pos,
                        )
                        new_lp = compute_full_log_probs(fwd_out, generated, q_len, latent_id)

                        # PPO-style clipped surrogate loss
                        ratio         = torch.exp(new_lp - old_lp.detach())
                        clipped_ratio = torch.clamp(ratio, 1.0 - configs.clip_eps, 1.0 + configs.clip_eps)
                        rollout_loss  = -torch.min(ratio * advantage, clipped_ratio * advantage)

                        # Optional KL penalty towards frozen reference model
                        if ref_model is not None:
                            with torch.no_grad():
                                ref_out = ref_model(
                                    input_ids=generated,
                                    attention_mask=full_attn,
                                    labels=dummy_lbl,
                                    position_ids=full_pos,
                                )
                                ref_lp = compute_full_log_probs(ref_out, generated, q_len, latent_id)

                            rollout_loss = rollout_loss + configs.kl_coeff * (new_lp - ref_lp.detach())

                        rollout_losses.append(rollout_loss)

                # Normalise and backprop
                if rollout_losses:
                    n_valid    = len(rollout_losses)
                    total_loss = sum(rollout_losses) / (n_valid * configs.gradient_accumulation_steps)
                    total_loss.backward()
                    loss_scalar = total_loss.item() * configs.gradient_accumulation_steps
                else:
                    loss_scalar = 0.0

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                # Logging
                if wandb_run and rank == 0:
                    flat_rewards  = [rv for grp in all_rewards   for rv in grp]
                    flat_n_latent = [nl for grp in all_n_latents  for nl in grp]
                    log_dict = {
                        "train/epoch":         epoch + 1,
                        "train/step":          total_train_steps,
                        "train/grpo_loss":     loss_scalar,
                        "train/mean_reward":   sum(flat_rewards) / len(flat_rewards),
                        "train/correct_rate":  sum(1 for rv in flat_rewards if rv >= 0.5) / len(flat_rewards),
                        "train/mean_n_latent": sum(flat_n_latent) / len(flat_n_latent),
                    }
                    if gsm_symbolic_enabled:
                        log_dict["train/sym_mix_sym"] = n_sym
                        log_dict["train/sym_mix_p1"]  = n_p1
                        log_dict["train/sym_mix_p2"]  = n_p2
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"GRPO Epoch: {epoch+1}/{configs.num_epochs}, "
                    f"step {step}/{len(train_dataloader)} "
                    f"(loss: {round(loss_scalar, 4)})"
                )

                del all_rollouts, all_rewards, all_n_latents, all_old_lp, all_advantages, all_question_lens
                gc.collect()

            pbar.close()
            dist.barrier()

            # Save checkpoint
            if not configs.save_only_improve and not configs.debug and not configs.only_eval:
                checkpoint = {
                    "epoch":            epoch,
                    "model_state_dict": parallel_model.state_dict(),
                }
                if rank == 0:
                    ckpt_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}_{current_time}.pt")
                    torch.save(checkpoint, ckpt_path)
                    print("saving model.")
                    previous_savepoint = ckpt_path

                dist.barrier()
                del checkpoint
                gc.collect()
                torch.cuda.empty_cache()

        # Evaluation ##############################################################################################################
        pbar = tqdm(
            colour="blue", desc="Test Accuracy",
            total=len(valid_gen_dataloader), dynamic_ncols=True,
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
            torch.tensor(0, device=rank),
        )

        with torch.no_grad():
            parallel_model.module.eval()
            eval_savepath = os.path.join(
                save_dir, f"checkpoint_{epoch + 1}_evaluation_{current_time}.txt"
            )
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                # question-only batch: no latent_tokens field, just input_ids + attention_mask
                eval_batch = {
                    k: v.to(rank)
                    for k, v in batch.items()
                    if v is not None and k not in ["idx", "position_ids"]
                }

                assert len(eval_batch["input_ids"]) == 1
                answer     = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question   = question_val[test_idx.cpu().item()]
                total     += 1

                with FSDP.summon_full_params(parallel_model):
                    outputs = parallel_model.module.generate(
                        **eval_batch,
                        max_new_tokens=max_new_tokens,
                        synced_gpus=not configs.only_eval,
                    )

                text_output   = tokenizer.decode(outputs[0], skip_special_tokens=not configs.coconut)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output    = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

                if idx < 5 and rank == 0:
                    print(f"Question {test_idx}: Answer = '{answer}'")
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                cor     += answer_output == answer
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
        dist.all_reduce(cor,     op=dist.ReduceOp.SUM)
        dist.all_reduce(total,   op=dist.ReduceOp.SUM)

        cor_cot, cor, total = cor_cot.item(), cor.item(), total.item()
        if rank == 0:
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        if configs.only_eval:
            break

        dist.barrier()

        if (cor / total > best_acc
                and configs.save_only_improve
                and not configs.debug
                and not configs.only_eval):
            checkpoint = {
                "epoch":            epoch,
                "model_state_dict": parallel_model.state_dict(),
            }
            if rank == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}_{current_time}.pt")
                torch.save(checkpoint, ckpt_path)
                print("saving model.")
                previous_savepoint = ckpt_path
            best_acc = cor / total

            dist.barrier()
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
