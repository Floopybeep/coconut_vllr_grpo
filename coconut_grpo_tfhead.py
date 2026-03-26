# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modification: replaces the binary linear termination head with a 2-layer
# causal transformer head.  Hidden size and number of attention heads match
# the base LLM for embedding compatibility, but the FFN uses a standard
# 2-layer MLP (Linear → GELU → Linear, 4× expansion) rather than the base
# model's large SwiGLU, keeping parameter count modest.

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "output_embeds", "inputs_embeds", "logits", "termination_logits", "termination_labels", "past_key_values"])
MAX_N_LATENT = 8


# ---------------------------------------------------------------------------
# 2-layer Transformer Termination Head
# ---------------------------------------------------------------------------

class _TerminationLayer(nn.Module):
    """Single causal transformer layer for the termination head.

    Uses the base model's hidden_size and num_attention_heads for embedding
    compatibility, but a simple 2-layer MLP FFN (Linear → GELU → Linear,
    4× expansion) instead of the base model's large SwiGLU.  Supports
    incremental KV-cache decoding for efficient autoregressive generation.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        ffn_dim = 4 * hidden_size

        self.norm1 = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)

        self.norm2 = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
        self.fc1 = nn.Linear(hidden_size, ffn_dim, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, dtype=torch.bfloat16)

    def forward(self, x, past_key_value=None, use_cache: bool = False):
        """
        Args:
            x:               (B, L, H) input hidden states
            past_key_value:  tuple (K, V) each (B, num_heads, S, head_dim) or None
            use_cache:       whether to return the updated KV cache

        Returns:
            output:   (B, L, H)
            new_kv:   (K, V) if use_cache else None
        """
        B, L, H = x.shape

        # --- Pre-norm self-attention ---
        residual = x
        x_norm = self.norm1(x)

        Q = self.q_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Append past KV (incremental decoding)
        if past_key_value is not None:
            K = torch.cat([past_key_value[0], K], dim=2)  # (B, nh, S+L, hd)
            V = torch.cat([past_key_value[1], V], dim=2)

        new_kv = (K, V) if use_cache else None

        # Causal SDPA: apply causal mask only for full-sequence pass (no past KV, L > 1)
        is_causal = (past_key_value is None and L > 1)
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H)
        x = residual + self.o_proj(attn_out)

        # --- Pre-norm MLP FFN (Linear → GELU → Linear) ---
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))

        return x, new_kv


class TransformerTerminationHead(nn.Module):
    """2-layer causal transformer that predicts latent-continue vs. terminate.

    hidden_size and num_attention_heads match the base LLM so the head can
    directly consume the base model's last-layer hidden states.  The FFN uses
    a simple Linear → GELU → Linear with 4× expansion (not the base model's
    large SwiGLU), keeping the parameter count small relative to the backbone.

    Supports KV-cache mode (use_cache=True) for O(T) per-step cost during
    autoregressive generation instead of O(T²).
    """

    def __init__(self, config, num_layers: int = 2):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads   = config.num_attention_heads

        self.layers = nn.ModuleList([
            _TerminationLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
        self.proj = nn.Linear(hidden_size, 2, dtype=torch.bfloat16)

    def forward(self, hidden_states, past_key_values=None, use_cache: bool = False):
        """
        Args:
            hidden_states:   (B, L, hidden_size)
            past_key_values: list of (K, V) per layer, or None
            use_cache:       if True, return accumulated KV cache

        Returns:
            logits:          (B, L, 2)
            pre_proj:        (B, L, hidden_size) — post-LayerNorm, pre-classification
                             hidden state that retains full dimensionality with
                             trajectory-level context from causal attention.
            new_kvs:         list[(K,V)] per layer  (only when use_cache=True)
        """
        new_kvs = [] if use_cache else None
        x = hidden_states

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(x, past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        pre_proj = self.norm(x)            # (B, L, hidden_size)
        logits = self.proj(pre_proj)       # (B, L, 2)

        if use_cache:
            return logits, pre_proj, new_kvs
        return logits, pre_proj


# ---------------------------------------------------------------------------
# Coconut model (transformer termination head variant)
# ---------------------------------------------------------------------------

class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        termination_gamma,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.pad_id = eos_token_id

        # Determine how to access the base transformer and lm_head for direct calls (Change 4).
        if isinstance(base_causallm, GPT2LMHeadModel):
            self._base_transformer_attr = "transformer"
        else:
            self._base_transformer_attr = "model"

        self.kv_cache = None
        self.term_kv_cache = None  # KV cache for the transformer termination head

        # Latent Termination Head — 2-layer transformer matching base model config
        self.termination_gamma = termination_gamma
        self.latent_termination_head = TransformerTerminationHead(base_causallm.config, num_layers=2)

        # Approach 2: projects the term head's pre-projection hidden state (which
        # retains full hidden_size dimensionality with trajectory-level context from
        # causal attention) into a modulation signal for each latent step's feedback.
        # Zero-init so the model starts identical to baseline.
        hidden_size = base_causallm.config.hidden_size
        self.term_signal_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        nn.init.zeros_(self.term_signal_proj.weight)

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            model_embedding = self.base_causallm.get_input_embeddings()
            self.embedding = bnb.nn.StableEmbedding(
                model_embedding.num_embeddings,
                model_embedding.embedding_dim,
                padding_idx=model_embedding.padding_idx
            )
            # Copy pretrained weights into StableEmbedding BEFORE tying lm_head.
            # set_input_embeddings() does not auto-retie lm_head for Qwen2/Llama,
            # so we must tie explicitly — but only after the embedding holds the
            # correct pretrained weights, otherwise lm_head would be overwritten
            # with random values.
            self.embedding.weight.data.copy_(model_embedding.weight.data)
            self.embedding.norm = nn.Identity()
            self.base_causallm.set_input_embeddings(self.embedding)
            self.base_causallm.lm_head.weight = self.embedding.weight

    def _forward_base(self, inputs_embeds, attention_mask=None, position_ids=None, past_key_values=None):
        """Call base transformer directly and apply lm_head manually."""
        base_transformer = getattr(self.base_causallm, self._base_transformer_attr)
        transformer_outputs = base_transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = transformer_outputs[0]
        kv_cache = transformer_outputs.past_key_values
        logits = self.base_causallm.lm_head(hidden_states)
        return logits, hidden_states, kv_cache

    @staticmethod
    def _trim_kv_cache(kv_cache, max_pos):
        """Trim KV cache to max_pos positions. Handles both DynamicCache and tuple formats."""
        if hasattr(kv_cache, 'key_cache'):
            return [
                (kv_cache.key_cache[i][:, :, :max_pos, :],
                 kv_cache.value_cache[i][:, :, :max_pos, :])
                for i in range(len(kv_cache.key_cache))
            ]
        else:
            return [
                (k[:, :, :max_pos, :], v[:, :, :max_pos, :])
                for k, v in kv_cache
            ]

    def _compute_term_logits_latent_only(self, hidden_states_total, latent_lists):
        """Run the termination head on the latent-only hidden-state subsequence.

        The term head is a "judge" of the latent reasoning trajectory.  It should
        attend only to:
          • h_{q_last}: the last question hidden state — a compressed representation
            of the full question, already encoding all prompt context through the
            base model's causal attention.
          • h_{lat_k}: the hidden state at each latent token position — the result
            of each iterative latent computation step.

        Prompt tokens and text-generation tokens are deliberately excluded:
          • Prompt tokens are already summarised in h_{q_last}; re-attending to them
            adds noise and lengthens the attention window unnecessarily.
          • Text tokens appear after the latent chain ends and are irrelevant to the
            termination decision (which is made only during latent mode).

        This design also decouples gradient flow: the term head's gradient only
        propagates through latent-step hidden states, not through the LM head's
        text predictions.

        Args:
            hidden_states_total: (B, L, H) full hidden states from the base model
            latent_lists:        list[list[int]] — latent token positions per batch item

        Returns:
            termination_logits: (B, L, 2) with valid logits ONLY at decision positions:
                • position q_last  (last question position)
                • position lat_k   for each latent position k
              All other positions are zero (not used in loss or log-prob computation).
        """
        B, L, H = hidden_states_total.shape
        max_n_latents = max(len(ll) for ll in latent_lists)

        # q_last: the last question position, immediately before the first latent token.
        # With left-padding, all batch items share the same first latent position.
        has_latents = [ll for ll in latent_lists if len(ll) > 0]
        q_last = has_latents[0][0] - 1 if has_latents else L - 1

        # Build compact latent-only sequence: (B, 1 + max_n_latents, H)
        #   slot 0         → h_{q_last}   (question summary)
        #   slot k+1       → h_{lat_k}    (k-th latent hidden state)
        term_seq = torch.zeros(B, 1 + max_n_latents, H,
                               device=hidden_states_total.device,
                               dtype=hidden_states_total.dtype)
        term_seq[:, 0, :] = hidden_states_total[:, q_last, :]
        for b, lat_list in enumerate(latent_lists):
            for k, lat_pos in enumerate(lat_list):
                term_seq[b, k + 1, :] = hidden_states_total[b, lat_pos, :]

        # Run term head with causal masking over this short sequence.
        term_logits_compact, _ = self.latent_termination_head(term_seq)  # (B, 1+max_n_latents, 2)

        # Scatter compact outputs back to full (B, L, 2) tensor.
        # Positions without a term-head output stay zero (masked out in loss computation).
        termination_logits = torch.zeros(B, L, 2,
                                         device=hidden_states_total.device,
                                         dtype=term_logits_compact.dtype)
        for b, lat_list in enumerate(latent_lists):
            termination_logits[b, q_last, :] = term_logits_compact[b, 0, :]
            for k, lat_pos in enumerate(lat_list):
                termination_logits[b, lat_pos, :] = term_logits_compact[b, k + 1, :]

        return termination_logits

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, input_embeds=None, labels=None, token_ids=None, reset_kv_cache=False, **kwargs):
        assert input_ids is not None or input_embeds is not None, "Input IDs or Input Embeds must be given"

        if reset_kv_cache:
            self.kv_cache = None

        if input_ids is not None:
            hidden_states_list = []
            logits_list = []
            termination_labels = torch.zeros((input_ids.shape[0], input_ids.shape[1]), dtype=torch.long, device=input_ids.device)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=input_ids.device)

            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
                if input_ids.dtype == torch.bfloat16:
                    position_ids = position_ids.to(torch.bfloat16)
                else:
                    position_ids = position_ids.to(torch.long)

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
            term_kv_train = None  # incremental KV cache for term head signal injection

            for pass_idx in range(max_n_latents):
                if kv_cache is None:
                    logits_chunk, hidden_states, kv_cache = self._forward_base(
                        inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                        attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                        position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    )
                    hidden_states_offset = 0
                else:
                    past_kv = self._trim_kv_cache(kv_cache, next_compute_range[0])
                    logits_chunk, hidden_states, kv_cache = self._forward_base(
                        inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                        attention_mask[:, : next_compute_range[1]],
                        position_ids[:, next_compute_range[0] : next_compute_range[1]],
                        past_key_values=past_kv,
                    )
                    hidden_states_offset = next_compute_range[0]

                logits_list.append(logits_chunk)
                hidden_states_list.append(hidden_states)

                # --- Approach 2: incremental term head → signal injection ---
                # Last hidden state in each chunk is h_q_last (pass 0) or
                # h_lat_{k-1} (pass k), matching the compact term-head sequence.
                with torch.no_grad():
                    step_term_logits, step_pre_proj, term_kv_train = self.latent_termination_head(
                        hidden_states[:, -1:, :],
                        past_key_values=term_kv_train,
                        use_cache=True,
                    )
                term_signal = self.term_signal_proj(step_pre_proj[:, -1, :])  # (B, H)

                next_compute_range = (
                    next_compute_range[1],
                    (
                        input_ids.shape[1]
                        if pass_idx + 1 >= max_n_latents
                        else next_compute_range[1] + 1
                    ),
                )

                filling_indices = [
                    (instance_idx, mask_list[pass_idx])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > pass_idx
                ]

                inputs_embeds = inputs_embeds.clone()
                for batch_idx, token_idx in filling_indices:
                    inputs_embeds[batch_idx, token_idx] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :
                    ] + term_signal[batch_idx]
                    termination_labels[batch_idx][token_idx-1] = 1

            # final pass
            logits_chunk, hidden_states, final_kv_cache = self._forward_base(
                inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                attention_mask[:, : next_compute_range[1]],
                position_ids[:, next_compute_range[0] : next_compute_range[1]],
                past_key_values=(
                    self._trim_kv_cache(kv_cache, next_compute_range[0])
                    if kv_cache
                    else None
                ),
            )

            hidden_states_list.append(hidden_states)
            hidden_states_total = torch.cat(hidden_states_list, dim=1)

            logits_list.append(logits_chunk)

            self.gen_forward_cnt += max_n_latents + 1

            logits = torch.cat(logits_list, dim=-2)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = (labels if labels is not None else input_ids)[..., 1:].contiguous()

            # Transformer termination head — latent-only subsequence for termination loss.
            if max_n_latents > 0:
                termination_logits = self._compute_term_logits_latent_only(hidden_states_total, latent_lists)

                # Build decision mask: q_last + all latent positions per batch item.
                has_latents_any = [ll for ll in latent_lists if len(ll) > 0]
                q_last_pos = has_latents_any[0][0] - 1
                term_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
                for b, lat_list in enumerate(latent_lists):
                    if len(lat_list) > 0:
                        term_mask[b, q_last_pos] = True
                        for lat_pos in lat_list:
                            term_mask[b, lat_pos] = True

                term_loss = F.cross_entropy(termination_logits[term_mask], termination_labels[term_mask])

                # --- Approach 1: termination-weighted LM loss ---
                # One more incremental term head step on h_lat_{last} for readiness.
                with torch.no_grad():
                    final_term_logits, _, _ = self.latent_termination_head(
                        hidden_states[:, 0:1, :],  # h at last latent pos from final pass
                        past_key_values=term_kv_train,
                        use_cache=True,
                    )
                    # P(terminate): higher = term head thinks reasoning is done
                    term_readiness = final_term_logits[:, -1, :].softmax(dim=-1)[:, 0]  # (B,)

                per_token_lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none',
                    ignore_index=-100,
                ).view(shift_labels.shape)

                # Weight: [0.5, 1.5] — stronger LM signal when term head is confident
                lm_weights = (0.5 + term_readiness).unsqueeze(-1)  # (B, 1)
                valid_count = (shift_labels != -100).sum().clamp(min=1)
                lm_loss = (per_token_lm_loss * lm_weights).sum() / valid_count
            else:
                termination_logits = torch.zeros(*hidden_states_total.shape[:2], 2,
                                                 device=hidden_states_total.device,
                                                 dtype=hidden_states_total.dtype)
                term_loss = termination_logits.new_tensor(0.0)
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            loss = lm_loss + term_loss * self.termination_gamma

            return Outputs(loss=loss, inputs_embeds=inputs_embeds, output_embeds=hidden_states_total, logits=logits, termination_logits=termination_logits, termination_labels=termination_labels, past_key_values=final_kv_cache)

        else:           # input_embeds path — GRPO log-prob computation and legacy generate()
            logits, hidden_states, kv_cache = self._forward_base(inputs_embeds=input_embeds)
            self.kv_cache = kv_cache

            if token_ids is not None:
                # GRPO log-prob forward: use latent-only term head for consistency with training.
                # token_ids identifies latent positions so the compact subsequence can be built.
                # Vectorised: single comparison + per-row nonzero, avoids O(B*L) .item() calls.
                latent_mask = (token_ids == self.latent_token_id)
                latent_lists_emb = [
                    latent_mask[b].nonzero(as_tuple=True)[0].tolist()
                    for b in range(token_ids.shape[0])
                ]
                max_n_lat_emb = max(len(ll) for ll in latent_lists_emb)
                if max_n_lat_emb > 0:
                    termination_logits = self._compute_term_logits_latent_only(hidden_states, latent_lists_emb)
                else:
                    termination_logits = torch.zeros(*hidden_states.shape[:2], 2,
                                                     device=hidden_states.device,
                                                     dtype=hidden_states.dtype)
            else:
                # Legacy generate() path: full-sequence causal forward (no KV cache)
                termination_logits, _ = self.latent_termination_head(hidden_states)

            return Outputs(loss=None, output_embeds=hidden_states, inputs_embeds=input_embeds, logits=logits,
                        termination_logits=termination_logits, termination_labels=None, past_key_values=kv_cache)


    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        latent_decision = torch.argmax(outputs.termination_logits[0, -1]).item()

        if latent_decision == 1:
            next_token = self.latent_token_id
            new_token_embed = outputs.output_embeds[0, -1].view(1, 1, -1)
        else:
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)

        tokens.append(next_token)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.forward(input_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1

            latent_decision = torch.argmax(outputs.termination_logits[0, -1]).item()

            if latent_decision == 1:
                next_token = self.latent_token_id
                new_token_embed = outputs.output_embeds[0, -1].view(1, 1, -1)
            else:
                next_token = torch.argmax(outputs.logits[0, -1]).item()
                if next_token == self.eos_token_id:
                    break
                new_token_embed = self.embedding(
                    torch.tensor(next_token, device=input_ids.device)
                ).view(1, 1, -1)

            tokens.append(next_token)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)


    @staticmethod
    def _sample_decision(term_logits, temperature: float):
        """Sample the binary termination decision (0=text, 1=latent).

        temperature <= ~0  →  greedy argmax (deterministic)
        temperature  > 0   →  multinomial sample from softmax(logits / T)

        Stochastic termination produces varied chain lengths across rollouts,
        which is the primary driver of rollout diversity for GRPO advantage
        estimation.  The sampled decisions are discrete and recorded in token_ids,
        so the log-probability computation is exact — unlike dropout, this
        diversity is directly attributable by the policy gradient.
        """
        if temperature <= 1e-4:
            return torch.argmax(term_logits, dim=-1)
        probs = torch.softmax(term_logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate_batched(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=100,
        output_embedding=False,
        synced_gpus=False,
        term_temperature=0.0,
        **kwargs
    ):
        """Generate a batch of sequences autoregressively.

        Diversity strategy:
        - Termination decisions: sampled with temperature `term_temperature`.
          This is the primary diversity lever — different chain lengths produce
          different latent trajectories and therefore different text answers.
          The sampled decisions are discrete, so GRPO can directly attribute
          reward to the chain-length choice via exact log-probabilities.
        - Latent hidden-state content: additionally diversified by dropout noise
          in the base model (must remain in train() mode during generation).
        - Text tokens: greedy argmax.  Chain-length diversity from the term head
          naturally produces varied text outputs without needing text sampling.
        """
        batch_size = input_ids.shape[0]
        self.gen_forward_cnt = 0
        self.kv_cache = None
        self.term_kv_cache = None  # reset transformer termination head KV cache

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        tokens = input_ids.clone()
        is_terminated = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        in_latent_mode = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

        try:
          return self._generate_batched_inner(
              input_ids, attention_mask, tokens, is_terminated, in_latent_mode, batch_size,
              max_new_tokens, output_embedding, synced_gpus, term_temperature,
          )
        finally:
            self.kv_cache = None
            self.term_kv_cache = None

    def _generate_batched_inner(
        self, input_ids, attention_mask, tokens, is_terminated, in_latent_mode, batch_size,
        max_new_tokens, output_embedding, synced_gpus, term_temperature,
    ):
        # First pass — full question through base model + coconut latent filling
        # Compute position_ids from attention_mask so pad tokens get position 0
        # and real tokens get consecutive positions.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)

        outputs = self.forward(
            input_ids,
            attention_mask,
            position_ids,
        )

        inputs_embeds = outputs.inputs_embeds
        kv_cache = outputs.past_key_values

        # Seed term KV cache with only the last question hidden state (h_{q_last}).
        # The term head is a latent-only judge: it should not attend to the full prompt.
        # Subsequent latent hidden states are appended incrementally in the loop below.
        init_term_logits, init_pre_proj, self.term_kv_cache = self.latent_termination_head(
            outputs.output_embeds[:, -1:, :], use_cache=True
        )

        latent_decision = self._sample_decision(init_term_logits[:, -1, :], term_temperature)
        latent_decision = torch.where(in_latent_mode, latent_decision, torch.zeros_like(latent_decision))
        in_latent_mode = in_latent_mode & (latent_decision == 1)

        raw_next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        # Approach 2: add term signal to latent hidden state feedback
        term_signal = self.term_signal_proj(init_pre_proj[:, -1, :].detach())
        latent_embeds = outputs.output_embeds[:, -1, :] + term_signal
        next_embeds = torch.where((latent_decision == 1).unsqueeze(-1), latent_embeds, self.embedding(raw_next_tokens))
        next_tokens = torch.where(latent_decision == 1, self.latent_token_id, raw_next_tokens)

        last_embed = next_embeds.unsqueeze(1).detach()
        embed_list = [inputs_embeds.detach(), last_embed]
        current_len = inputs_embeds.shape[1] + 1
        tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        # Track growing attention mask: original mask + 1 for each generated token so far
        # next_pos tracks the per-sequence position for the next generated token
        gen_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=input_ids.device)], dim=1)
        next_pos = attention_mask.sum(dim=1)  # (B,) — number of real tokens per sequence

        # Incremental generation loop — base model uses its KV cache; term head uses term_kv_cache
        for _ in range(max_new_tokens - 1):
            if is_terminated.all():
                break

            self.gen_forward_cnt += 1

            attn_mask = gen_mask
            pos_ids = next_pos.unsqueeze(1)  # (B, 1)

            logits, hidden_states, kv_cache = self._forward_base(
                last_embed,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=kv_cache,
            )

            # Transformer termination head — query only while any sequence is in latent mode.
            # Consistent with the latent-only training objective: the term head should not
            # attend to text-generation hidden states.  Once all sequences exit latent mode,
            # the KV cache is frozen and termination decisions are forced to "text" (0).
            if in_latent_mode.any():
                term_logits, term_pre_proj, self.term_kv_cache = self.latent_termination_head(
                    hidden_states,
                    past_key_values=self.term_kv_cache,
                    use_cache=True,
                )
                latent_decision = self._sample_decision(term_logits[:, -1, :], term_temperature)
                # Approach 2: term signal for latent feedback
                term_signal = self.term_signal_proj(term_pre_proj[:, -1, :].detach())
                latent_embeds = hidden_states[:, -1, :] + term_signal
            else:
                latent_decision = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
                latent_embeds = hidden_states[:, -1, :]  # no signal needed, won't be selected

            latent_decision = torch.where(in_latent_mode, latent_decision, torch.zeros_like(latent_decision))
            in_latent_mode = in_latent_mode & (latent_decision == 1)
            raw_next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

            next_embeds = torch.where((latent_decision == 1).unsqueeze(-1), latent_embeds, self.embedding(raw_next_tokens))
            next_tokens = torch.where(latent_decision == 1, self.latent_token_id, raw_next_tokens)

            next_tokens = torch.where(is_terminated, self.pad_id, next_tokens)
            is_terminated = is_terminated | (next_tokens == self.eos_token_id)

            last_embed = next_embeds.unsqueeze(1).detach()
            embed_list.append(last_embed)
            current_len += 1
            gen_mask = torch.cat([gen_mask, torch.ones(batch_size, 1, dtype=gen_mask.dtype, device=input_ids.device)], dim=1)
            next_pos = next_pos + 1
            tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        new_inputs_embeds = torch.cat(embed_list, dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1

        if output_embedding:
            return tokens, new_inputs_embeds, None

        else:
            return tokens
