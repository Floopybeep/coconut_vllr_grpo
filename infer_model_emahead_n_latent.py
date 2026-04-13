"""
Inference script for the EMAHead Coconut model with forced variable-length latent steps.

For each question in the validation set:
  1. Generates answers with forced n_latent = 3..16 (14 runs), recording correctness at each.
  2. Runs one "normal" generation (forced_min=3, greedy termination) to obtain the latent
     length the termination head would choose.
  3. Runs n=16 with output_embedding=True to collect the 16 latent hidden states.
  4. Produces a 3-D PCA scatter plot showing the 16-step chain, color-coded by per-length
     correctness, with the "normal" termination point outlined in black dotted lines.
  5. Saves plots under {save_path}/{project}/{YMD_hms}/plots/{correct|incorrect|wrong_format}/

Usage:
    python infer_model_emahead_n_latent.py args/gsm_vllr_grpo_emahead.yaml [--batch_size 64][--max_samples N] [--n_min 3] [--n_max 16]
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import yaml
import datetime
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D projection)
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import bitsandbytes as bnb

from coconut_grpo_emahead import Coconut
from dataset import get_dataset, get_grpo_dataset
from utils import Config, set_seed


# ---------------------------------------------------------------------------
# Answer extraction & reward helpers
# ---------------------------------------------------------------------------

def extract_answer(text: str, eot_token: str = "<|endoftext|>") -> str:
    """Return the numerical answer following the last '#' in *text*."""
    return text.split("#")[-1].replace(",", "").replace(eot_token, "").strip()


def check_single_output(generated_ids: torch.Tensor, tokenizer, ground_truth: str,
                         latent_id: int, coconut_mode: bool = True):
    """
    Check a single generated sequence for format violation and correctness.

    Returns:
        is_correct      (bool)
        format_violation (bool)
        extracted_answer (str)
        num_latent       (int)
    """
    text = tokenizer.decode(generated_ids, skip_special_tokens=not coconut_mode)
    answer = extract_answer(text)

    num_latent = (generated_ids == latent_id).sum().item()

    format_violation = False

    # 1. Latent markers leaked into text answer
    if "<|start-latent|>" in answer or "<|end-latent|>" in answer:
        format_violation = True
        answer = answer.replace("<|start-latent|>", "").replace("<|end-latent|>", "")

    # 2. Empty / missing answer
    cleaned = answer.strip()
    if not cleaned or cleaned == "#":
        format_violation = True

    # 3. No latent tokens (skipped reasoning)
    if num_latent == 0:
        format_violation = True

    # 4. CoT steps leaked
    if "<<" in answer or ">>" in answer:
        format_violation = True

    is_correct = (answer == ground_truth)
    return is_correct, format_violation, answer, num_latent


# ---------------------------------------------------------------------------
# Custom PCA visualisation
# ---------------------------------------------------------------------------

def visualize_n_latent_pca(
    latent_embeds_16: np.ndarray,
    correctness_per_n: dict,          # {n: True/False/None} for n in 3..16; None = format violation
    normal_n_latent: int,
    output_path: str,
    question: str,
    answer_gt: str,
    normal_answer: str,
    normal_correct: bool,
    normal_format_violation: bool,
    n_min: int = 3,
    n_max: int = 16,
    verbose: bool = False,
):
    """
    3-D PCA scatter plot for the n-latent experiment.

    The 16-step chain (from the n=16 run) is shown.  Points are coloured:
      - index 0           → green  (starting point)
      - index 1           → grey   (n=2, untested)
      - indices 2..15     → blue (correct) / red (incorrect) / orange (wrong format)
    The "normal" termination point is additionally outlined with a black dotted circle.

    Args:
        latent_embeds_16 : (16, hidden_dim) numpy array
        correctness_per_n: {n (int): True/False/None} for tested latent lengths
        normal_n_latent  : latent count under normal (forced-min-3, greedy) operation
        output_path      : full file path to save the figure
        n_min / n_max    : tested range (3 / 16)
    """
    if latent_embeds_16.shape[0] < 3:
        print(f"  [skip PCA] only {latent_embeds_16.shape[0]} latent steps — need ≥ 3")
        return

    # PCA → 3-D
    pca = PCA(n_components=3)
    pts = pca.fit_transform(latent_embeds_16)       # (16, 3)

    if verbose:
        total_var = sum(pca.explained_variance_ratio_)
        print(f"  PCA explained variance: {total_var:.3f}")

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw connecting lines
    for i in range(len(pts) - 1):
        ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], pts[i:i+2, 2],
                color='grey', alpha=0.35, linewidth=2)

    # Determine point colour for each latent step (1-indexed n)
    colors = []
    n_correct = 0
    for n in range(1, n_max + 1):
        if n == 1:
            colors.append('green')   # starting point
        elif n == 2:
            colors.append('grey')    # untested (between start and min)
        else:
            result = correctness_per_n.get(n)
            if result is None:
                colors.append('orange')   # format violation
            elif result:
                colors.append('blue')     # correct
                n_correct += 1
            else:
                colors.append('red')      # incorrect

    output_filename, ext = os.path.splitext(output_path)
    output_root, output_file = os.path.split(output_filename)
    output_path = os.path.join(output_root, f"{n_correct / n_max:.2f}_{output_file}{ext}")

    # Draw points (one per latent step)
    for i, (pt, c) in enumerate(zip(pts, colors)):
        n = i + 1
        size = 120 if n == 1 else 70
        ax.scatter(pt[0], pt[1], pt[2], c=c, s=size, alpha=0.85,
                   zorder=5, edgecolors='none')

    # Overlay black dotted circle for "normal" termination
    norm_idx = normal_n_latent - 1   # 0-indexed
    if 0 <= norm_idx < n_max:
        npt = pts[norm_idx]
        ax.scatter(npt[0], npt[1], npt[2],
                   s=300, facecolors='none', edgecolors='black',
                   linewidths=2, linestyles='dashed', zorder=6,
                   label=f'Normal termination (n={normal_n_latent})')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=9, label='Start (n=1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',  markersize=9, label='n=2 (untested)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',  markersize=9, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   markersize=9, label='Incorrect'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',markersize=9, label='Wrong format'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='black', markeredgewidth=2, markersize=12,
               linestyle='dashed', label=f'Normal term. (n={normal_n_latent})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=8)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=8)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=8)

    # Build title
    normal_status = ('CORRECT' if normal_correct
                     else ('WRONG FORMAT' if normal_format_violation else 'INCORRECT'))
    ax.set_title(f'Latent Chain PCA  |  normal termination: n={normal_n_latent} → {normal_status}',
                 fontsize=10)

    # Text box below plot
    q_short = (question[:120] + '…') if len(question) > 120 else question
    text_str = (f"Q: {q_short}\n"
                f"GT: {answer_gt}   |   Normal pred: {normal_answer}   |   Normal n_latent: {normal_n_latent}")
    text_str = text_str.replace('$', r'\$')
    fig.text(0.05, -0.04, text_str, fontsize=8, verticalalignment='top',
             family='monospace', clip_on=False, wrap=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------

def collate_batch(questions_batch, tokenizer, start_id, device):
    """
    Tokenise and left-pad a list of question dicts so they can be fed as a
    single (B, padded_len) tensor to the model.

    Returns:
        input_ids      : (B, padded_len) long tensor, left-padded
        attention_mask : (B, padded_len) long tensor
        padded_len     : int — shared column index where generated tokens begin
    """
    pad_id = tokenizer.eos_token_id
    all_tokens = [
        tokenizer.encode(q["question"] + "\n", add_special_tokens=True) + [start_id]
        for q in questions_batch
    ]
    padded_len = max(len(t) for t in all_tokens)

    input_rows, mask_rows = [], []
    for tokens in all_tokens:
        pad = padded_len - len(tokens)
        input_rows.append([pad_id] * pad + tokens)
        mask_rows.append([0] * pad + [1] * len(tokens))

    input_ids      = torch.tensor(input_rows, dtype=torch.long,  device=device)
    attention_mask = torch.tensor(mask_rows,  dtype=torch.long,  device=device)
    return input_ids, attention_mask, padded_len


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(configs, device):
    """Load the Coconut + EMAHead model from a checkpoint."""
    model_config = AutoConfig.from_pretrained(configs.model_id)
    model_config.attention_dropout = getattr(configs, 'dropout', 0.0)

    if getattr(configs, 'bf16', True):
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id, config=model_config,
            attn_implementation="sdpa" if getattr(configs, 'sdpa_attention', False) else None,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(configs.model_id, config=model_config)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Resize & initialise new token embeddings
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id  = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
        model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id].clone()

    # Wrap in Coconut
    model = Coconut(
        model, latent_id, start_id, end_id,
        tokenizer.eos_token_id,
        getattr(configs, 'termination_gamma', 1.0),
        ema_decay=getattr(configs, 'ema_decay', 0.9),
        bottleneck_ratio=getattr(configs, 'bottleneck_ratio', 4),
    )

    # Load checkpoint
    if configs.load_model_path != "None":
        saved = torch.load(configs.load_model_path, map_location='cpu')
        result = model.load_state_dict(saved["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {configs.load_model_path}")
        print(f"  missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
        del saved
        gc.collect()

    if getattr(configs, 'bf16', True):
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    return model, tokenizer, latent_id, start_id, end_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMAHead Coconut inference with variable latent lengths")
    parser.add_argument("config_file", help="Path to YAML config file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of questions to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of questions to process simultaneously (default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max_new_tokens from config")
    parser.add_argument("--n_min", type=int, default=3,
                        help="Minimum forced latent length to test (default: 3)")
    parser.add_argument("--n_max", type=int, default=16,
                        help="Maximum forced latent length to test (default: 16)")
    args = parser.parse_args()

    # ---- Config ----
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)
    set_seed(configs.seed)

    n_min = args.n_min
    n_max = args.n_max
    assert 1 <= n_min <= n_max, "n_min must be ≥ 1 and ≤ n_max"

    batch_size     = args.batch_size
    max_new_tokens = args.max_new_tokens or getattr(configs, 'max_new_tokens', 128)
    # Add extra room for the latent tokens themselves (n_max tokens before text starts)
    gen_budget = max_new_tokens + n_max

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  batch_size: {batch_size}")

    # ---- Output paths ----
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    save_path = getattr(configs, 'save_path', 'checkpoints')
    project   = getattr(configs, 'project', 'coconut-grpo-emahead')
    base_dir  = os.path.join(save_path, project, current_time, "plots")
    dirs = {
        'correct':             os.path.join(base_dir, 'correct'),
        'correct_mixed':       os.path.join(base_dir, 'correct_mixed'),
        'incorrect':           os.path.join(base_dir, 'incorrect'),
        'incorrect_mixed':     os.path.join(base_dir, 'incorrect_mixed'),
        'wrong_format':        os.path.join(base_dir, 'wrong_format'),
        'wrong_format_mixed':  os.path.join(base_dir, 'wrong_format_mixed'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Saving plots to: {base_dir}/{{correct|correct_mixed|incorrect|incorrect_mixed|wrong_format|wrong_format_mixed}}/")

    # ---- Load model ----
    model, tokenizer, latent_id, start_id, end_id = load_model_and_tokenizer(configs, device)

    # ---- Load dataset ----
    val_data = json.load(open(configs.val_path))
    if args.max_samples is not None:
        val_data = val_data[:args.max_samples]

    questions = [
        {
            "question": d["question"],
            "answer": d["answer"].replace(",", "").strip(),
            "idx": i,
        }
        for i, d in enumerate(val_data)
    ]
    print(f"Evaluating {len(questions)} questions in batches of {batch_size}, "
          f"n_latent = {n_min}..{n_max}")

    # ---- Summary accumulators ----
    accuracy_by_n = {n: {"correct": 0, "total": 0} for n in range(n_min, n_max + 1)}
    normal_stats  = {
        "correct": 0, "correct_mixed": 0,
        "incorrect": 0, "incorrect_mixed": 0,
        "wrong_format": 0, "wrong_format_mixed": 0,
    }

    n_batches = (len(questions) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(questions), batch_size),
                                total=n_batches, desc="Batches"):
            batch = questions[batch_start : batch_start + batch_size]
            actual_bs = len(batch)
            answers_batch = [q["answer"] for q in batch]

            # Left-pad the batch; generated tokens for ALL samples start at padded_len.
            input_ids, attention_mask, padded_len = collate_batch(
                batch, tokenizer, start_id, device
            )

            # ----------------------------------------------------------------
            # 1) "Normal" operation: forced_min=n_min, greedy termination
            # ----------------------------------------------------------------
            forced_min_vec = torch.full((actual_bs,), n_min, dtype=torch.long, device=device)
            normal_tokens = model.generate_batched(
                input_ids, attention_mask,
                max_new_tokens=gen_budget,
                output_embedding=False,
                synced_gpus=False,
                term_temperature=0.0,
                forced_min_latents=forced_min_vec,
            )

            # Check each sample in the batch
            normal_results = []  # list of (is_correct, fmt_viol, answer, n_latent)
            for i in range(actual_bs):
                res = check_single_output(normal_tokens[i], tokenizer, answers_batch[i], latent_id)
                normal_results.append(res)

            del normal_tokens

            # ----------------------------------------------------------------
            # 2) Correctness sweep: one call per n, whole batch at once
            # ----------------------------------------------------------------
            # correctness_per_n_batch[i][n] = True/False/None
            correctness_per_n_batch = [{} for _ in range(actual_bs)]

            for n in range(n_min, n_max + 1):
                tokens_n = model.generate_batched_n(
                    input_ids, attention_mask,
                    max_new_tokens=gen_budget,
                    num_latent=n,
                    output_embedding=False,
                    synced_gpus=False,
                )
                for i in range(actual_bs):
                    is_correct, fmt_viol, _, _ = check_single_output(
                        tokens_n[i], tokenizer, answers_batch[i], latent_id
                    )
                    correctness_per_n_batch[i][n] = None if fmt_viol else is_correct
                    accuracy_by_n[n]["total"] += 1
                    if is_correct and not fmt_viol:
                        accuracy_by_n[n]["correct"] += 1

                if hasattr(model, 'kv_cache'):
                    model.kv_cache = None
                del tokens_n
                gc.collect()

            # ----------------------------------------------------------------
            # 3) Full-chain embedding run with n = n_max (whole batch at once)
            # ----------------------------------------------------------------
            tokens_max, all_embeds, _ = model.generate_batched_n(
                input_ids, attention_mask,
                max_new_tokens=gen_budget,
                num_latent=n_max,
                output_embedding=True,
                synced_gpus=False,
            )

            # After left-padding, generated tokens for every sample start at the
            # same column index (padded_len).  Slice latent positions in one go.
            latent_embeds_batch = (
                all_embeds[:, padded_len : padded_len + n_max, :]
                .detach().cpu().float().numpy()
            )   # (actual_bs, n_max, hidden)

            if hasattr(model, 'kv_cache'):
                model.kv_cache = None
            del tokens_max, all_embeds
            gc.collect()

            # ----------------------------------------------------------------
            # 4) PCA visualisation — one plot per sample (CPU-bound, cheap)
            # ----------------------------------------------------------------
            for i in range(actual_bs):
                q_idx = batch[i]["idx"]
                question = batch[i]["question"]
                answer_gt = answers_batch[i]

                normal_correct, normal_fmt_viol, normal_answer, normal_n_latent = normal_results[i]
                normal_n_latent_clamped = max(n_min, min(normal_n_latent, n_max))

                chain = correctness_per_n_batch[i]   # {n: True/False/None}
                has_correct  = any(v is True  for v in chain.values())
                has_incorrect = any(v is False for v in chain.values())
                has_format   = any(v is None   for v in chain.values())

                if normal_fmt_viol:
                    is_mixed = has_correct or has_incorrect
                    subdir_key = "wrong_format_mixed" if is_mixed else "wrong_format"
                elif normal_correct:
                    is_mixed = has_incorrect or has_format
                    subdir_key = "correct_mixed" if is_mixed else "correct"
                else:
                    is_mixed = has_correct or has_format
                    subdir_key = "incorrect_mixed" if is_mixed else "incorrect"

                normal_stats[subdir_key] += 1
                save_subdir = dirs[subdir_key]

                latent_embeds = latent_embeds_batch[i]   # (n_max, hidden)

                if latent_embeds.shape[0] < 3:
                    print(f"  Q{q_idx}: fewer than 3 latent embeds — skipping PCA")
                    continue

                visualize_n_latent_pca(
                    latent_embeds_16=latent_embeds,
                    correctness_per_n=correctness_per_n_batch[i],
                    normal_n_latent=normal_n_latent_clamped,
                    output_path=os.path.join(save_subdir, f"Q{q_idx:04d}_pca.png"),
                    question=question,
                    answer_gt=answer_gt,
                    normal_answer=normal_answer,
                    normal_correct=normal_correct,
                    normal_format_violation=normal_fmt_viol,
                    n_min=n_min,
                    n_max=n_max,
                )

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'n_latent':>10}  {'correct':>8}  {'total':>8}  {'accuracy':>10}")
    print("-" * 40)
    for n in range(n_min, n_max + 1):
        s   = accuracy_by_n[n]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"{n:>10}  {s['correct']:>8}  {s['total']:>8}  {acc:>10.4f}")

    total = sum(normal_stats.values())
    print("\nNormal operation (forced_min=3, greedy):")
    for k, v in normal_stats.items():
        pct = 100.0 * v / total if total else 0
        print(f"  {k:<14}: {v:4d}  ({pct:.1f}%)")

    print(f"\nPlots saved to: {base_dir}")


if __name__ == "__main__":
    main()
