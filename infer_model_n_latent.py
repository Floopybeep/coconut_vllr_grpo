# Inference script for Coconut model with forced variable-length latent steps.
# For each query, generates answers with n_latent = 1, 2, ..., n_latent_max latent tokens,
# logging all results to logs/n_latent/coconut-baseline_{datetime}.log

import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import gc
import yaml
import datetime
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np

from coconut_baseline import Coconut
from utils import Config, set_seed, visualize_latent_pca


def extract_answer(text):
    """Extract the final answer after the last '#' marker."""
    return text.split("#")[-1].replace(",", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Coconut inference with variable latent lengths")
    parser.add_argument("config_file", help="Path to YAML config file")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query string to evaluate (if not set, uses eval dataset)")
    parser.add_argument("--n_latent_max", type=int, default=None,
                        help="Max number of latent tokens to test (default: max_latent_stage * c_thought from config)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max new tokens to generate per sample")
    args = parser.parse_args()

    # Load config
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)
    set_seed(configs.seed)

    # Determine n_latent_max
    n_latent_max = args.n_latent_max
    if n_latent_max is None:
        n_latent_max = configs.max_latent_stage * configs.c_thought
    print(f"Will test n_latent = 1 to {n_latent_max}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    if configs.bf16:
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id,
            attn_implementation="sdpa" if configs.sdpa_attention else None,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(configs.model_id)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Resize embeddings for new tokens
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, configs.termination_gamma)

    # Load checkpoint
    if configs.load_model_path != "None":
        saved_checkpoint = torch.load(configs.load_model_path, map_location="cpu")
        saved_weights = saved_checkpoint["model_state_dict"]
        print(model.load_state_dict(saved_weights, strict=False))
        del saved_weights, saved_checkpoint
        gc.collect()

    model = model.to(device)
    if configs.bf16:
        model.to(torch.bfloat16)
    model.eval()

    # Prepare log directory and file
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", "n_latent", current_time)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"coconut-baseline_{current_time}.log")
    print(f"Logging to: {log_path}")
    print(f"PCA plots to: {run_dir}/<query_num>/")

    # Build queries
    if args.query is not None:
        # Single query mode
        queries = [{"question": args.query, "answer": "N/A", "idx": 0}]
    else:
        # Dataset mode
        val_data = json.load(open(configs.val_path))
        queries = [
            {
                "question": d["question"],
                "answer": d["answer"].replace(",", "").strip(),
                "idx": i,
            }
            for i, d in enumerate(val_data)
        ]

    # Run inference
    with open(log_path, "w") as log_fp:
        header = (
            f"Coconut Inference - Variable Latent Length\n"
            f"Config: {args.config_file}\n"
            f"Checkpoint: {configs.load_model_path}\n"
            f"n_latent_max: {n_latent_max}\n"
            f"max_new_tokens: {args.max_new_tokens}\n"
            f"Num queries: {len(queries)}\n"
            f"Date: {current_time}\n"
            f"{'=' * 80}\n\n"
        )
        log_fp.write(header)
        print(header, end="")

        # Track accuracy per n_latent
        accuracy_by_n = {n: {"correct": 0, "total": 0} for n in range(1, n_latent_max + 1)}

        with torch.no_grad():
            for q_idx, query in enumerate(tqdm(queries, desc="Queries")):
                question = query["question"]
                answer_gt = query["answer"]

                # Tokenize the question
                question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)
                input_ids = torch.tensor([question_tokens], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, device=device)

                log_fp.write(f"Question {query['idx']+1}: {question}\n")
                log_fp.write(f"Ground truth: {answer_gt}\n")
                log_fp.write(f"{'-' * 60}\n")

                input_len = input_ids.shape[1]
                pca_dir = os.path.join(run_dir, str(query["idx"] + 1))

                for n_latent in range(1, n_latent_max + 1):
                    token_output, all_embeds = model.generate_n(
                        input_ids,
                        attention_mask,
                        num_latents=n_latent,
                        max_new_tokens=args.max_new_tokens,
                        output_embedding=True,
                        synced_gpus=False,
                    )

                    full_text = tokenizer.decode(token_output[0], skip_special_tokens=False)
                    text_no_special = tokenizer.decode(token_output[0], skip_special_tokens=True)
                    answer_extracted = extract_answer(text_no_special)
                    is_correct = answer_extracted == answer_gt

                    accuracy_by_n[n_latent]["total"] += 1
                    if is_correct:
                        accuracy_by_n[n_latent]["correct"] += 1

                    log_fp.write(f"  n_latent={n_latent:3d} | correct={is_correct} | extracted='{answer_extracted}' | full='{full_text}'\n")

                    # Extract latent embeddings and generate PCA plot
                    # Latent embeddings sit at positions input_len .. input_len + n_latent - 1
                    latent_embeds = all_embeds[0, input_len:input_len + n_latent, :].detach().cpu().float().numpy()
                    if latent_embeds.shape[0] >= 3:
                        visualize_latent_pca(
                            latent_embeds,
                            title=f"Q{query['idx']+1} n_latent={n_latent}",
                            output_dir=pca_dir,
                            question=question,
                            answer=answer_gt,
                            predicted=answer_extracted,
                            idx=n_latent,
                        )

                    # Clear KV cache
                    if hasattr(model, 'kv_cache'):
                        model.kv_cache = None
                    del token_output, all_embeds

                log_fp.write(f"{'=' * 80}\n\n")
                log_fp.flush()

                # Print periodic progress
                if (q_idx + 1) % 50 == 0 or q_idx == 0:
                    print(f"\n[{q_idx+1}/{len(queries)}] Running accuracies:")
                    for n in range(1, n_latent_max + 1):
                        stats = accuracy_by_n[n]
                        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                        print(f"  n_latent={n}: {stats['correct']}/{stats['total']} = {acc:.4f}")

        # Write summary
        summary = f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}\n"
        summary += f"{'n_latent':>10} {'correct':>10} {'total':>10} {'accuracy':>12}\n"
        summary += f"{'-' * 44}\n"
        for n in range(1, n_latent_max + 1):
            stats = accuracy_by_n[n]
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            summary += f"{n:>10} {stats['correct']:>10} {stats['total']:>10} {acc:>12.4f}\n"
        summary += f"{'=' * 80}\n"

        log_fp.write(summary)
        print(summary)

    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    main()
