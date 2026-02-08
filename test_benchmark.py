"""
Benchmark script for comparing old vs optimized Coconut forward pass.

Measures wall-clock time and peak GPU memory for both implementations.

Usage: python coconut_grpo/test_benchmark.py
Requires: 1 GPU with enough VRAM for GPT-2 (~1-2GB depending on batch size)
"""

import time
import torch
import torch.nn as nn
from collections import namedtuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel

from coconut import Coconut, Outputs


class CoconutOriginal(nn.Module):
    """Original (pre-optimization) implementation for benchmarking."""

    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id,
                 eos_token_id, termination_gamma):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.termination_gamma = termination_gamma
        self.latent_termination_head = nn.Linear(base_causallm.config.hidden_size, 2, dtype=torch.bfloat16)

        if isinstance(base_causallm, GPT2LMHeadModel):
            self.embedding = base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids):
        from torch.nn import CrossEntropyLoss

        hidden_states_list = []
        logits = []
        termination_labels = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1]),
            dtype=torch.long, device=input_ids.device
        )

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            hidden_states_list.append(torch.Tensor(outputs.hidden_states[-1]))

            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1,
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # ORIGINAL: O(B*L) tensor decompose/recompose
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                termination_labels[batch_idx][token_idx - 1] = 1
            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=(
                [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                if kv_cache else None
            ),
            output_hidden_states=True,
        )

        hidden_states_list.append(torch.Tensor(outputs.hidden_states[-1]))
        hidden_states_total = torch.cat(hidden_states_list, dim=1)
        logits.append(outputs.logits)

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        termination_logits = self.latent_termination_head(hidden_states_total)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ) + loss_fct(termination_logits.view(-1, 2), termination_labels.view(-1)) * self.termination_gamma

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits,
                      termination_logits=termination_logits, termination_labels=termination_labels)


def create_test_input(tokenizer, latent_id, start_id, end_id, batch_size, n_latents, seq_len, device):
    """Create a synthetic input batch with configurable size."""
    q_tokens = tokenizer.encode("John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 hours but spends the first 2 hours in standstill traffic.  He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph.  How far is he from home at the end of those 4 hours?\n", add_special_tokens=True)
    a_tokens = tokenizer.encode("### 45", add_special_tokens=False) + [tokenizer.eos_token_id]

    # Pad with filler tokens to reach desired seq_len
    filler = tokenizer.encode("step " * 20, add_special_tokens=False)
    n_filler_needed = seq_len - len(q_tokens) - n_latents - 2 - len(a_tokens)
    filler_tokens = (filler * (n_filler_needed // len(filler) + 1))[:max(0, n_filler_needed)]

    tokens = q_tokens + [start_id] + [latent_id] * n_latents + [end_id] + filler_tokens + a_tokens

    input_ids = torch.tensor([tokens] * batch_size, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(len(tokens), device=device).unsqueeze(0).expand(batch_size, -1)
    n_masked = len(q_tokens) + n_latents + 2
    labels_list = [-100] * n_masked + tokens[n_masked:]
    labels = torch.tensor([labels_list] * batch_size, device=device)

    return input_ids, attention_mask, labels, position_ids


def benchmark_forward(model, input_ids, attention_mask, labels, position_ids, n_warmup=3, n_runs=10):
    """Benchmark forward pass timing and memory."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids, attention_mask, labels, position_ids)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, labels, position_ids)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        "min_ms": min(times) * 1000,
        "peak_mem_mb": peak_mem,
    }


def benchmark_backward(model, input_ids, attention_mask, labels, position_ids, n_warmup=2, n_runs=5):
    """Benchmark forward+backward pass timing and memory."""
    # Warmup
    for _ in range(n_warmup):
        outputs = model(input_ids, attention_mask, labels, position_ids)
        outputs.loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(input_ids, attention_mask, labels, position_ids)
        outputs.loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        model.zero_grad()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        "min_ms": min(times) * 1000,
        "peak_mem_mb": peak_mem,
    }


def main():
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required for benchmarking"

    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Test configurations
    configs = [
        {"batch_size": 1, "n_latents": 2, "seq_len": 128},
        {"batch_size": 4, "n_latents": 2, "seq_len": 128},
        {"batch_size": 8, "n_latents": 4, "seq_len": 256},
        {"batch_size": 4, "n_latents": 6, "seq_len": 512},
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"Config: batch_size={cfg['batch_size']}, n_latents={cfg['n_latents']}, seq_len={cfg['seq_len']}")
        print(f"{'='*70}")

        # Create fresh models for each config to avoid memory fragmentation
        base_model_orig = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        base_model_orig.resize_token_embeddings(len(tokenizer))
        original = CoconutOriginal(base_model_orig, latent_id, start_id, end_id, tokenizer.eos_token_id, 0.1).to(device).to(torch.bfloat16)

        base_model_opt = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        base_model_opt.resize_token_embeddings(len(tokenizer))
        optimized = Coconut(base_model_opt, latent_id, start_id, end_id, tokenizer.eos_token_id, 0.1).to(device).to(torch.bfloat16)

        input_ids, attention_mask, labels, position_ids = create_test_input(
            tokenizer, latent_id, start_id, end_id, **cfg, device=device
        )

        # Forward-only benchmark
        print("\n  Forward pass (inference):")
        torch.cuda.empty_cache()
        orig_fwd = benchmark_forward(original, input_ids, attention_mask, labels, position_ids)
        torch.cuda.empty_cache()
        opt_fwd = benchmark_forward(optimized, input_ids, attention_mask, labels, position_ids)

        speedup_fwd = orig_fwd["mean_ms"] / opt_fwd["mean_ms"]
        mem_save_fwd = orig_fwd["peak_mem_mb"] - opt_fwd["peak_mem_mb"]

        print(f"    Original:  {orig_fwd['mean_ms']:.1f} +/- {orig_fwd['std_ms']:.1f} ms | Peak mem: {orig_fwd['peak_mem_mb']:.0f} MB")
        print(f"    Optimized: {opt_fwd['mean_ms']:.1f} +/- {opt_fwd['std_ms']:.1f} ms | Peak mem: {opt_fwd['peak_mem_mb']:.0f} MB")
        print(f"    Speedup: {speedup_fwd:.2f}x | Memory saved: {mem_save_fwd:.0f} MB")

        # Forward+backward benchmark
        print("\n  Forward+backward pass (training):")
        torch.cuda.empty_cache()
        orig_bwd = benchmark_backward(original, input_ids, attention_mask, labels, position_ids)
        torch.cuda.empty_cache()
        opt_bwd = benchmark_backward(optimized, input_ids, attention_mask, labels, position_ids)

        speedup_bwd = orig_bwd["mean_ms"] / opt_bwd["mean_ms"]
        mem_save_bwd = orig_bwd["peak_mem_mb"] - opt_bwd["peak_mem_mb"]

        print(f"    Original:  {orig_bwd['mean_ms']:.1f} +/- {orig_bwd['std_ms']:.1f} ms | Peak mem: {orig_bwd['peak_mem_mb']:.0f} MB")
        print(f"    Optimized: {opt_bwd['mean_ms']:.1f} +/- {opt_bwd['std_ms']:.1f} ms | Peak mem: {opt_bwd['peak_mem_mb']:.0f} MB")
        print(f"    Speedup: {speedup_bwd:.2f}x | Memory saved: {mem_save_bwd:.0f} MB")

        del original, optimized, base_model_orig, base_model_opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
