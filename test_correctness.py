"""
Correctness test for optimized Coconut forward pass.

Compares the optimized forward() against a reference implementation
to verify that Changes 1, 2, 4 produce numerically equivalent results.

Usage: python coconut_grpo/test_correctness.py
Requires: 1 GPU with enough VRAM for GPT-2 (~500MB)
"""

import torch
import torch.nn as nn
from collections import namedtuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel

# Import the optimized version
from coconut import Coconut, Outputs


class CoconutReference(nn.Module):
    """Reference implementation matching the ORIGINAL (pre-optimization) code."""

    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id,
                 eos_token_id, termination_gamma):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.termination_gamma = termination_gamma
        self.latent_termination_head = nn.Linear(base_causallm.config.hidden_size, 2, dtype=torch.bfloat16)

        if isinstance(base_causallm, GPT2LMHeadModel):
            self.embedding = base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids):
        """Original forward logic (pre-optimization) for comparison."""
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

        # final pass
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


def create_test_input(tokenizer, latent_id, start_id, end_id, batch_size=2, n_latents=2, device="cuda"):
    """Create a synthetic input batch with latent tokens."""
    question = "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 hours but spends the first 2 hours in standstill traffic.  He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph.  How far is he from home at the end of those 4 hours?\n"
    steps = [
            "<<3*60=180>>",
            "<<4-2=2>>",
            "<<30*.5=15>>",
            "<<2-.5=1.5>>",
            "<<80*1.5=120>>",
            "<<120+15=135>>",
            "<<180-135=45>>"
        ]
    answer = "### 45"

    q_tokens = tokenizer.encode(question, add_special_tokens=True)
    s_tokens = tokenizer.encode("\n".join(steps), add_special_tokens=False)
    a_tokens = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]

    tokens = q_tokens + [start_id] + [latent_id] * n_latents + [end_id] + s_tokens + a_tokens

    input_ids = torch.tensor([tokens] * batch_size, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(len(tokens), device=device).unsqueeze(0).expand(batch_size, -1)

    # Labels: mask question + latent tokens, predict rest
    n_masked = len(q_tokens) + n_latents + 2  # +2 for start/end markers
    labels = [-100] * n_masked + tokens[n_masked:]
    labels = torch.tensor([labels] * batch_size, device=device)

    return input_ids, attention_mask, labels, position_ids


def test_forward_equivalence():
    """Test that optimized forward produces same outputs as reference."""
    print("=" * 60)
    print("Test: Forward pass equivalence (optimized vs reference)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B"

    print(f"Loading {model_id} on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    base_model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Create both models with shared weights
    optimized = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id, 0.1).to(device)

    # Create reference with same base model and copy termination head weights
    ref = CoconutReference(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id, 0.1).to(device)
    ref.latent_termination_head.load_state_dict(optimized.latent_termination_head.state_dict())

    optimized.to(torch.bfloat16)

    for batch_size in [4, 8]:
        for n_latents in [0, 1, 2, 3]:
            print(f"\n  batch_size={batch_size}, n_latents={n_latents}:")
            input_ids, attention_mask, labels, position_ids = create_test_input(
                tokenizer, latent_id, start_id, end_id,
                batch_size=batch_size, n_latents=n_latents, device=device
            )

            with torch.no_grad():
                out_opt = optimized(input_ids, attention_mask, labels, position_ids)
                out_ref = ref(input_ids, attention_mask, labels, position_ids)

            # Compare logits
            logits_close = torch.allclose(out_opt.logits, out_ref.logits, atol=1e-4, rtol=1e-3)
            logits_max_diff = (out_opt.logits - out_ref.logits).abs().max().item()
            print(f"    Logits match: {logits_close} (max diff: {logits_max_diff:.6f})")

            # Compare termination logits
            term_close = torch.allclose(out_opt.termination_logits, out_ref.termination_logits, atol=1e-4, rtol=1e-3)
            term_max_diff = (out_opt.termination_logits - out_ref.termination_logits).abs().max().item()
            print(f"    Termination logits match: {term_close} (max diff: {term_max_diff:.6f})")

            # Compare termination labels
            labels_match = torch.equal(out_opt.termination_labels, out_ref.termination_labels)
            print(f"    Termination labels match: {labels_match}")

            # Compare loss
            loss_close = torch.allclose(out_opt.loss, out_ref.loss, atol=1e-4, rtol=1e-3)
            loss_diff = (out_opt.loss - out_ref.loss).abs().item()
            print(f"    Loss match: {loss_close} (diff: {loss_diff:.6f})")

            if not (logits_close and term_close and labels_match and loss_close):
                print("    *** MISMATCH DETECTED ***")
            else:
                print("    PASSED")


def test_gradient_flow():
    """Test that gradients flow correctly through the optimized forward pass."""
    print("\n" + "=" * 60)
    print("Test: Gradient flow through optimized forward")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B"

    base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    base_model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id, 0.1).to(device).to(torch.bfloat16)

    input_ids, attention_mask, labels, position_ids = create_test_input(
        tokenizer, latent_id, start_id, end_id,
        batch_size=2, n_latents=2, device=device
    )

    outputs = model(input_ids, attention_mask, labels, position_ids)
    outputs.loss.backward()

    # Check that gradients exist on key parameters
    has_grads = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  WARNING: No gradient for {name}")
            has_grads = False

    # Check specific components
    term_head_grad = model.latent_termination_head.weight.grad is not None
    lm_head_grad = model.lm_head.weight.grad is not None
    embedding_grad = any(p.grad is not None for p in model.embedding.parameters())

    print(f"  Termination head has gradients: {term_head_grad}")
    print(f"  LM head has gradients: {lm_head_grad}")
    print(f"  Embedding has gradients: {embedding_grad}")
    print(f"  All parameters have gradients: {has_grads}")

    if term_head_grad and lm_head_grad:
        print("  PASSED")
    else:
        print("  *** GRADIENT FLOW ISSUE ***")


def test_dataset_lazy():
    """Test that the lazy CotLatentDataset produces valid outputs."""
    print("\n" + "=" * 60)
    print("Test: Lazy CotLatentDataset correctness")
    print("=" * 60)

    from dataset import CotLatentDataset
    from datasets import Dataset

    # Create a mock base dataset
    base_data = {
        "question_tokenized": [[1, 2, 3, 4, 5], [10, 11, 12]],
        "steps_tokenized": [[[6, 7], [8, 9]], [[13, 14], [15, 16], [17, 18]]],
        "answer_tokenized": [[100, 101, 0], [200, 201, 0]],
        "idx": [0, 1],
    }
    base_dataset = Dataset.from_dict(base_data)

    class MockConfig:
        uniform_prob = 0.0  # deterministic for testing
        max_latent_stage = 3
        pad_latent_to_max = False
        c_thought = 1
        no_cot = False

    configs = MockConfig()

    for stage in [0, 1, 2]:
        dataset = CotLatentDataset(
            base_dataset, scheduled_stage=stage, configs=configs,
            start_id=50, latent_id=51, end_id=52,
        )

        print(f"\n  Stage {stage}, dataset length: {len(dataset)}")
        for i in range(len(dataset)):
            sample = dataset[i]
            assert "input_ids" in sample
            assert "labels" in sample
            assert "attention_mask" in sample
            assert "idx" in sample
            assert "position_ids" in sample
            assert len(sample["input_ids"]) == len(sample["labels"])
            assert len(sample["input_ids"]) == len(sample["attention_mask"])
            assert len(sample["input_ids"]) == len(sample["position_ids"])

            # Count latent tokens
            n_latent = sample["input_ids"].count(51)
            expected_latent = min(stage, len(base_data["steps_tokenized"][i])) * configs.c_thought
            assert n_latent == expected_latent, f"Expected {expected_latent} latents, got {n_latent}"

            print(f"    Sample {i}: len={len(sample['input_ids'])}, latents={n_latent}, "
                  f"masked_labels={sample['labels'].count(-100)}")

    print("  PASSED")


if __name__ == "__main__":
    test_dataset_lazy()
    test_forward_equivalence()
    test_gradient_flow()
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
