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
            new_kvs:         list[(K,V)] per layer  (only when use_cache=True)
        """
        new_kvs = [] if use_cache else None
        x = hidden_states

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(x, past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        logits = self.proj(self.norm(x))   # (B, L, 2)

        if use_cache:
            return logits, new_kvs
        return logits


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

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, input_embeds=None, labels=None, reset_kv_cache=False, **kwargs):
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
                    ]
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

            # Transformer termination head — full-sequence causal forward (no KV cache during training)
            termination_logits = self.latent_termination_head(hidden_states_total)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ) + loss_fct(termination_logits.view(-1, 2), termination_labels.view(-1)) * self.termination_gamma

            return Outputs(loss=loss, inputs_embeds=inputs_embeds, output_embeds=hidden_states_total, logits=logits, termination_logits=termination_logits, termination_labels=termination_labels, past_key_values=final_kv_cache)

        else:           # Only for generation (legacy single-sequence generate())
            logits, hidden_states, kv_cache = self._forward_base(inputs_embeds=input_embeds)
            self.kv_cache = kv_cache

            # Full-sequence causal forward (no incremental KV cache; used only in legacy generate())
            termination_logits = self.latent_termination_head(hidden_states)
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


    def generate_batched(
        self,
        input_ids,
        max_new_tokens=100,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        batch_size = input_ids.shape[0]
        self.gen_forward_cnt = 0
        self.kv_cache = None
        self.term_kv_cache = None  # reset transformer termination head KV cache

        tokens = input_ids.clone()
        is_terminated = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        in_latent_mode = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

        # First pass — full question through base model + coconut latent filling
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )

        inputs_embeds = outputs.inputs_embeds
        kv_cache = outputs.past_key_values

        # Run transformer termination head on the full initial sequence to seed KV cache.
        # This re-runs the head (already called inside forward()), but we need the KV cache
        # for efficient O(1) per-step generation in the loop below.
        init_term_logits, self.term_kv_cache = self.latent_termination_head(
            outputs.output_embeds, use_cache=True
        )

        latent_decision = torch.argmax(init_term_logits[:, -1, :], dim=-1)
        latent_decision = torch.where(in_latent_mode, latent_decision, torch.zeros_like(latent_decision))
        in_latent_mode = in_latent_mode & (latent_decision == 1)
        raw_next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        next_embeds = torch.where((latent_decision == 1).unsqueeze(-1), outputs.output_embeds[:, -1, :], self.embedding(raw_next_tokens))
        next_tokens = torch.where(latent_decision == 1, self.latent_token_id, raw_next_tokens)

        last_embed = next_embeds.unsqueeze(1)
        embed_list = [inputs_embeds, last_embed]
        current_len = inputs_embeds.shape[1] + 1
        tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        # Incremental generation loop — base model uses its KV cache; term head uses term_kv_cache
        for _ in range(max_new_tokens - 1):
            if is_terminated.all():
                break

            self.gen_forward_cnt += 1

            attn_mask = torch.ones(batch_size, current_len, dtype=torch.long, device=input_ids.device)
            pos_ids = torch.full((batch_size, 1), current_len - 1, dtype=torch.long, device=input_ids.device)

            logits, hidden_states, kv_cache = self._forward_base(
                last_embed,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=kv_cache,
            )

            # Transformer termination head: single token query + accumulated KV cache → O(S) cost
            term_logits, self.term_kv_cache = self.latent_termination_head(
                hidden_states,
                past_key_values=self.term_kv_cache,
                use_cache=True,
            )

            latent_decision = torch.argmax(term_logits[:, -1, :], dim=-1)
            latent_decision = torch.where(in_latent_mode, latent_decision, torch.zeros_like(latent_decision))
            in_latent_mode = in_latent_mode & (latent_decision == 1)
            raw_next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

            next_embeds = torch.where((latent_decision == 1).unsqueeze(-1), hidden_states[:, -1, :], self.embedding(raw_next_tokens))
            next_tokens = torch.where(latent_decision == 1, self.latent_token_id, raw_next_tokens)

            next_tokens = torch.where(is_terminated, self.pad_id, next_tokens)
            is_terminated = is_terminated | (next_tokens == self.eos_token_id)

            last_embed = next_embeds.unsqueeze(1)
            embed_list.append(last_embed)
            current_len += 1
            tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        new_inputs_embeds = torch.cat(embed_list, dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1

        self.kv_cache = None
        self.term_kv_cache = None  # release transformer termination head KV cache

        if output_embedding:
            return tokens, new_inputs_embeds, None

        else:
            return tokens
