# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "output_embeds", "inputs_embeds", "logits", "termination_logits", "termination_labels", "past_key_values"])
MAX_N_LATENT = 8


class DifferentialTerminationHead(nn.Module):
    """Termination head that uses current hidden state, EMA of previous states,
    and the step-wise change (last_hidden - current_hidden) to decide whether
    to continue latent reasoning.

    The step-wise diff captures how much the hidden state changed from the
    previous step.  When diff → 0, the model has converged and further latent
    steps add no new information.
    """
    def __init__(self, hidden_size, bottleneck_ratio=4, ema_decay=0.9):
        super().__init__()
        self.ema_decay = ema_decay
        bottleneck = hidden_size // bottleneck_ratio
        # Input: [current_hidden; running_mean; last_hidden - current_hidden]
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 2),
        )
        self.last_hidden = None  # cached for step-by-step generation

    def forward(self, current_hidden, running_mean, last_hidden=None):
        """
        current_hidden: (..., hidden_size)
        running_mean:   (..., hidden_size) — same shape as current_hidden
        last_hidden:    (..., hidden_size) — previous step's hidden state.
                        Falls back to self.last_hidden, then to current_hidden (diff=0).
        Returns: (..., 2) logits for [text, latent]
        """
        if last_hidden is None:
            last_hidden = self.last_hidden if self.last_hidden is not None else current_hidden
        diff = last_hidden - current_hidden
        features = torch.cat([current_hidden, running_mean, diff], dim=-1)
        return self.head(features)

    def update_ema(self, running_mean, current_hidden, detach_current=True):
        """Update EMA in-place."""
        source = current_hidden.detach() if detach_current else current_hidden
        return self.ema_decay * running_mean + (1.0 - self.ema_decay) * source


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        termination_gamma,
        ema_decay=0.9,
        bottleneck_ratio=4,
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
        # Calling base_transformer + lm_head directly avoids output_hidden_states=True,
        # which would store hidden states from ALL intermediate layers.
        # NOTE: We store the accessor name, NOT the module reference, to avoid
        # nn.Module registering duplicate submodules (which breaks checkpoint loading).
        if isinstance(base_causallm, GPT2LMHeadModel):
            self._base_transformer_attr = "transformer"
        else:
            self._base_transformer_attr = "model"

        self.kv_cache = None

        # Differential EMA Termination Head
        self.termination_gamma = termination_gamma  # hyperparameter for loss
        self.latent_termination_head = DifferentialTerminationHead(
            self.base_causallm.config.hidden_size,
            bottleneck_ratio=bottleneck_ratio,
            ema_decay=ema_decay,
        )

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            # self.embedding = self.base_causallm.get_input_embeddings()
            model_embedding = self.base_causallm.get_input_embeddings()     # change for 8bitoptimizer
            self.embedding = bnb.nn.StableEmbedding(
                model_embedding.num_embeddings,
                model_embedding.embedding_dim,
                padding_idx=model_embedding.padding_idx
            )
            self.embedding.weight.data.copy_(model_embedding.weight.data)
            self.embedding.norm = nn.Identity()
            self.base_causallm.set_input_embeddings(self.embedding)
            # self.base_causallm.lm_head.weight = self.embedding.weight                   # When training, NEED to turn this on!

    def _forward_base(self, inputs_embeds, attention_mask=None, position_ids=None, past_key_values=None):
        """Call base transformer directly and apply lm_head manually.

        This avoids output_hidden_states=True which stores hidden states from ALL
        intermediate transformer layers. We only need the last layer's hidden state.

        Returns: (logits, last_hidden_states, past_key_values)
        """
        base_transformer = getattr(self.base_causallm, self._base_transformer_attr)
        transformer_outputs = base_transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = transformer_outputs[0]  # last hidden state (works for GPT2Model, LlamaModel, Qwen2Model)
        kv_cache = transformer_outputs.past_key_values
        logits = self.base_causallm.lm_head(hidden_states)
        return logits, hidden_states, kv_cache

    @staticmethod
    def _trim_kv_cache(kv_cache, max_pos):
        """Trim KV cache to max_pos positions. Handles both DynamicCache and tuple formats."""
        if hasattr(kv_cache, 'key_cache'):
            # DynamicCache format (newer transformers versions)
            return [
                (kv_cache.key_cache[i][:, :, :max_pos, :],
                 kv_cache.value_cache[i][:, :, :max_pos, :])
                for i in range(len(kv_cache.key_cache))
            ]
        else:
            # Tuple format (older transformers versions)
            return [
                (k[:, :, :max_pos, :], v[:, :, :max_pos, :])
                for k, v in kv_cache
            ]

    def _replay_generate_batched(self, input_ids, attention_mask, replay_generated_ids, term_temperature=1.0):
        """Teacher-forced replay that mirrors generate_batched step-by-step.

        term_temperature must match the value used during rollout so that the
        torch.multinomial calls consume the exact same CUDA RNG draws as
        generate_batched did, keeping dropout masks identical.
        """
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        full_len = replay_generated_ids.shape[1]
        gen_len = full_len - prompt_len

        # Mirror the RNG consumption of _sample_term_batched in generate_batched.
        # generate_batched calls torch.multinomial once before the loop and once
        # per loop step (when term_temperature > 0.1).  We must consume the same
        # RNG draws here even though we ignore the result (we use stored token IDs).
        def _consume_term_rng(term_logits_last):
            if term_temperature <= 0.1:
                return  # generate_batched uses argmax; no RNG consumed
            probs = torch.softmax(term_logits_last.float() / term_temperature, dim=-1)
            torch.multinomial(probs, num_samples=1)  # consume RNG, discard result

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)

        prompt_outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if gen_len <= 0:
            return prompt_outputs

        # Mirror the pre-loop _sample_term_batched call in generate_batched (line 672).
        _consume_term_rng(prompt_outputs.termination_logits[:, -1, :])

        logits_list = [prompt_outputs.logits]
        termination_logits_list = [prompt_outputs.termination_logits]

        kv_cache = prompt_outputs.past_key_values
        running_mean = prompt_outputs.output_embeds[:, -1, :].clone()
        last_hidden = running_mean.clone()
        is_terminated = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        gen_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=input_ids.device),
            ],
            dim=1,
        )
        next_pos = attention_mask.sum(dim=1)

        first_expected = replay_generated_ids[:, prompt_len]
        first_active = ~is_terminated
        first_latent = first_active & (first_expected == self.latent_token_id)
        first_raw_tokens = torch.argmax(prompt_outputs.logits[:, -1, :], dim=-1)
        first_text_tokens = torch.where(first_active, first_expected, first_raw_tokens)
        next_embeds = torch.where(
            first_latent.unsqueeze(-1),
            prompt_outputs.output_embeds[:, -1, :],
            self.embedding(first_text_tokens),
        )
        is_terminated = is_terminated | (first_active & (first_expected == self.eos_token_id))
        last_embed = next_embeds.unsqueeze(1)

        for step in range(1, gen_len):
            logits, hidden_states, kv_cache = self._forward_base(
                last_embed,
                attention_mask=gen_mask,
                position_ids=next_pos.unsqueeze(1),
                past_key_values=kv_cache,
            )
            current_h = hidden_states[:, -1, :]
            termination_logits = self.latent_termination_head(
                current_h, running_mean, last_hidden
            ).unsqueeze(1)

            # Mirror the per-step _sample_term_batched call in generate_batched (line 717).
            _consume_term_rng(termination_logits[:, -1, :])

            logits_list.append(logits)
            termination_logits_list.append(termination_logits)

            expected_tokens = replay_generated_ids[:, prompt_len + step]
            active = ~is_terminated
            raw_next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            # generate_batched uses embedding(raw_next_tokens) for ALL sequences (active and
            # terminated alike) — only the stored token ID is forced to pad_id, not the embedding.
            force_latent = active & (expected_tokens == self.latent_token_id)
            text_embed_tokens = torch.where(active, expected_tokens, raw_next_tokens)
            next_embeds = torch.where(
                force_latent.unsqueeze(-1),
                current_h,
                self.embedding(text_embed_tokens),
            )

            is_terminated = is_terminated | (active & (expected_tokens == self.eos_token_id))
            last_hidden = current_h
            running_mean = self.latent_termination_head.update_ema(
                running_mean, current_h, detach_current=False
            )
            last_embed = next_embeds.unsqueeze(1)
            gen_mask = torch.cat(
                [
                    gen_mask,
                    torch.ones(batch_size, 1, dtype=gen_mask.dtype, device=input_ids.device),
                ],
                dim=1,
            )
            next_pos = next_pos + 1

        return Outputs(
            loss=None,
            output_embeds=None,
            inputs_embeds=None,
            logits=torch.cat(logits_list, dim=1),
            termination_logits=torch.cat(termination_logits_list, dim=1),
            termination_labels=None,
            past_key_values=kv_cache,
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, input_embeds=None, labels=None, reset_kv_cache=False, gen_token_ids=None, replay_generated_ids=None, term_temperature=1.0, **kwargs):
        # In training settings, input_ids would be tensor of shape (batch_size, padded_len)
        # gen_token_ids: optional (B, L) token IDs for the input_embeds path, used to compute
        #                proper EMA running_mean for latent positions during GRPO policy forward.
        assert input_ids is not None or input_embeds is not None, "Input IDs or Input Embeds must be given"

        if replay_generated_ids is not None:
            assert input_ids is not None, "Prompt input_ids are required for teacher-forced replay"
            return self._replay_generate_batched(input_ids, attention_mask, replay_generated_ids, term_temperature=term_temperature)

        if reset_kv_cache:
            self.kv_cache = None

        if input_ids is not None:
            hidden_states_list = []
            logits_list = []
            termination_labels = torch.zeros((input_ids.shape[0], input_ids.shape[1]), dtype=torch.long, device=input_ids.device)        # (B, L) mask for latent termination head calculation

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=input_ids.device)

            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
                if input_ids.dtype == torch.bfloat16:
                    position_ids = position_ids.to(torch.bfloat16)
                else:
                    position_ids = position_ids.to(torch.long)

            latent_indices = (
                input_ids == self.latent_token_id
            ).nonzero()  # (num_latent_tokens_in_the_batch, 2)          # not just single sample, but the WHOLE batch (ex output - [[0, 128], [0, 129], [1, 128], [1, 129]...] - [batch_num, latent_idx])

            latent_lists = [
                [idx[1].item() for idx in latent_indices if idx[0] == i]        # latent_lists = [[128, 129], [128, 129], ...]
                for i in range(input_ids.shape[0])
            ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

            max_n_latents = max([len(l) for l in latent_lists])                 # maximum number of latents in a sample in the whole batch

            next_compute_range = (0, input_ids.shape[1])                        # (0, padded_len) for when latent thinking is not used
            inputs_embeds = self.embedding(input_ids)                           # ONLY vocab embeddings (input_embs is of shape (batch_size, padded_len, vocab_size))

            if max_n_latents > 0:
                next_compute_range = (0, latent_indices[:, 1].min().item())     # (0, 128) (second number is the smallest number in latent_indices)
                # before the earliest latent token position

            kv_cache = None

            for pass_idx in range(max_n_latents):           # compute latents
                if kv_cache is None:
                    # first forward pass - use _forward_base (Change 4: avoids storing all intermediate hidden states)
                    logits_chunk, hidden_states, kv_cache = self._forward_base(
                        inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                        attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                        position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    )
                    hidden_states_offset = 0

                else:
                    # extract and trim kv cache to reuse (handles both DynamicCache and tuple formats)
                    past_kv = self._trim_kv_cache(kv_cache, next_compute_range[0])

                    logits_chunk, hidden_states, kv_cache = self._forward_base(
                        inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                        attention_mask[:, : next_compute_range[1]],
                        position_ids[:, next_compute_range[0] : next_compute_range[1]],
                        past_key_values=past_kv,
                    )

                    hidden_states_offset = next_compute_range[0]

                logits_list.append(logits_chunk)
                hidden_states_list.append(hidden_states)                        # Change 2: direct reference, no torch.Tensor() copy

                next_compute_range = (
                    next_compute_range[1],                      # next_compute_range = (0, 128) -> (128, 129) -> (129, padded_len)
                    (
                        input_ids.shape[1]
                        if pass_idx + 1 >= max_n_latents
                        else next_compute_range[1] + 1
                    ),
                )

                # feedback the continuous thoughts to the input_embeds

                # first decide the positions to feedback
                filling_indices = [                                                         # [[batch_num, 128], ...] -> [[batch_num, 129], ...] (if there are different number of latents per batch, then only process the ones with N latents where N == pass_idx)
                    (instance_idx, mask_list[pass_idx])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > pass_idx
                ]

                # Change 1: Replace O(B*L) tensor decompose/recompose with clone + indexed assignment
                # clone() creates a new tensor (avoids in-place modification on the compute graph),
                # then only the latent positions are overwritten - O(num_latents) instead of O(B*L)
                inputs_embeds = inputs_embeds.clone()
                for batch_idx, token_idx in filling_indices:
                    # replace latent token embedding with preceding position's hidden state
                    inputs_embeds[batch_idx, token_idx] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :      # -1 here is correct because it needs to take the PREVIOUS output (128th emb is replaced with 127th)
                    ]
                    termination_labels[batch_idx][token_idx-1] = 1        # set termination labels (1 == latent operation for NEXT step, 0 == non-latent operation for NEXT step)

            # final pass
            logits_chunk, hidden_states, final_kv_cache = self._forward_base(
                inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],                  # Only pass steps after latent calculation, because kv cache will exist if it was calculated
                attention_mask[:, : next_compute_range[1]],                                          # Automatically passes the whole input tokens if latent is not used (= kv cache doesn't exist)
                position_ids[:, next_compute_range[0] : next_compute_range[1]],
                past_key_values=(
                    self._trim_kv_cache(kv_cache, next_compute_range[0])
                    if kv_cache
                    else None
                ),
            )

            hidden_states_list.append(hidden_states)                            # Change 2: direct reference, no torch.Tensor() copy
            hidden_states_total = torch.cat(hidden_states_list, dim=1)

            logits_list.append(logits_chunk)

            self.gen_forward_cnt += max_n_latents + 1

            logits = torch.cat(logits_list, dim=-2)                  # B, L, vocab_size

            # Build last_hidden and running_mean for the differential termination head.
            # Non-latent positions: last_hidden = current (diff=0), running_mean = current.
            # Latent positions: last_hidden = previous position's hidden state,
            #                   running_mean = EMA across the latent chain.
            last_hidden_total = hidden_states_total.clone()       # (B, L, hidden)
            running_mean_total = hidden_states_total.clone()      # (B, L, hidden)

            if max_n_latents > 0:
                ema_decay = self.latent_termination_head.ema_decay
                B = input_ids.shape[0]
                L = input_ids.shape[1]
                batch_range = torch.arange(B, device=input_ids.device)

                # 1. Mask of latent positions
                latent_mask = (input_ids == self.latent_token_id)          # (B, L)
                has_latent = latent_mask.any(dim=1)                        # (B,)
                lat_counts = latent_mask.sum(dim=1)                        # (B,)
                first_lat = latent_mask.long().argmax(dim=1)               # (B,) — first latent col per sample

                # 2-4. last_hidden at latent positions = hidden state at (position - 1)
                lat_pos = latent_mask.nonzero()                            # (N, 2) [batch, seq]
                if lat_pos.numel() > 0:
                    b_idx, s_idx = lat_pos[:, 0], lat_pos[:, 1]
                    prev_s_idx = (s_idx - 1).clamp(min=0)
                    last_hidden_total[b_idx, s_idx] = hidden_states_total[b_idx, prev_s_idx]

                # 5. Running mean via EMA — loop over latent steps (small: 6-12), not batch
                #    Latent positions are contiguous, so first_lat + step gives each position.
                seed_pos = (first_lat - 1).clamp(min=0)                    # (B,)
                ema = hidden_states_total[batch_range, seed_pos]   # (B, hidden)
                for step in range(max_n_latents):
                    pos = (first_lat + step).clamp(max=L - 1)              # (B,)
                    valid = has_latent & (lat_counts > step)               # (B,)
                    h = hidden_states_total[batch_range, pos]      # (B, hidden)
                    ema = torch.where(valid.unsqueeze(-1), ema_decay * ema + (1.0 - ema_decay) * h, ema)
                    running_mean_total[batch_range[valid], pos[valid]] = ema[valid]

            # 6. Termination head inference
            termination_logits = self.latent_termination_head(hidden_states_total, running_mean_total, last_hidden_total)  # B, L, 2

            # Only compute loss when labels are provided (i.e., during training).
            # Skips the expensive cross-entropy during generation, avoiding a
            # ~1.8 GB allocation for shift_logits that would be immediately discarded.
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                lm_loss = CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                # Termination loss: only over latent-relevant positions (q_last + latent positions)
                # to avoid severe class imbalance from the overwhelming majority of non-latent positions.
                if max_n_latents > 0:
                    has_latents_any = [ll for ll in latent_lists if len(ll) > 0]
                    q_last_pos = has_latents_any[0][0] - 1
                    term_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
                    for b, lat_list in enumerate(latent_lists):
                        if len(lat_list) > 0:
                            term_mask[b, q_last_pos] = True
                            for lat_pos in lat_list:
                                term_mask[b, lat_pos] = True
                    term_loss = CrossEntropyLoss()(termination_logits[term_mask], termination_labels[term_mask])
                else:
                    term_loss = torch.tensor(0.0, device=input_ids.device)

                loss = lm_loss + term_loss * self.termination_gamma
            else:
                loss = None

            return Outputs(loss=loss, inputs_embeds=inputs_embeds, output_embeds=hidden_states_total, logits=logits, termination_logits=termination_logits, termination_labels=termination_labels, past_key_values=final_kv_cache)

        else:           # Only for generation / GRPO policy forward
            logits, hidden_states, kv_cache = self._forward_base(inputs_embeds=input_embeds)

            self.kv_cache = kv_cache

            # Build last_hidden and running_mean for differential termination head
            last_hidden = hidden_states.clone()      # default: current (diff=0)
            running_mean = hidden_states.clone()     # default: current (non-latent)

            if gen_token_ids is not None:
                ema_decay = self.latent_termination_head.ema_decay
                B = gen_token_ids.shape[0]
                L = gen_token_ids.shape[1]
                is_latent = (gen_token_ids == self.latent_token_id)  # (B, L)
                has_latent = is_latent.any(dim=1)                    # (B,)

                if has_latent.any():
                    batch_range = torch.arange(B, device=gen_token_ids.device)
                    lat_counts = is_latent.sum(dim=1)                          # (B,)
                    first_lat = is_latent.long().argmax(dim=1)                 # (B,)
                    max_lat_count = lat_counts.max().item()

                    # last_hidden at latent positions = previous position's hidden state
                    lat_pos = is_latent.nonzero()                              # (N, 2)
                    if lat_pos.numel() > 0:
                        b_idx, s_idx = lat_pos[:, 0], lat_pos[:, 1]
                        last_hidden[b_idx, s_idx] = hidden_states[b_idx, (s_idx - 1).clamp(min=0)]

                    # Running mean via EMA (contiguous latent positions: first_lat + step)
                    seed_pos = (first_lat - 1).clamp(min=0)
                    ema = hidden_states[batch_range, seed_pos]         # (B, hidden)
                    for step in range(max_lat_count):
                        pos = (first_lat + step).clamp(max=L - 1)              # (B,)
                        valid = has_latent & (lat_counts > step)               # (B,)
                        h = hidden_states[batch_range, pos]            # (B, hidden)
                        ema = torch.where(valid.unsqueeze(-1), ema_decay * ema + (1.0 - ema_decay) * h, ema)
                        running_mean[batch_range[valid], pos[valid]] = ema[valid]

            termination_logits = self.latent_termination_head(hidden_states, running_mean, last_hidden)
            return Outputs(loss=None, output_embeds=hidden_states, inputs_embeds=input_embeds, logits=logits,
                        termination_logits=termination_logits, termination_labels=None, past_key_values=kv_cache)


    # def train(self):
    #     self.base_causallm.train()
    #     self.latent_termination_head.train()

    # def eval(self):
    #     self.base_causallm.eval()
    #     self.latent_termination_head.eval()

    def generate(
        self,
        input_ids,
        max_new_tokens=100,     # expanded due to variable length latent generation (16 -> 100)
        output_embedding=False,
        synced_gpus=False,
        term_temperature=1.0,   # > 1.0 → stochastic termination for exploration
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        def _sample_termination(logits_1d):
            """Greedy at temp=1.0 (argmax); sampled otherwise."""
            if term_temperature <= 0.1:
                return torch.argmax(logits_1d).item()
            probs = torch.softmax(logits_1d.float() / term_temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()

        tokens = input_ids[0].detach().tolist()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # Decide whether first token is latent or not
        latent_decision = _sample_termination(outputs.termination_logits[0, -1])

        if latent_decision == 1:        # Latent mode
            next_token = self.latent_token_id
            new_token_embed = outputs.output_embeds[0, -1].view(1, 1, -1)

        else:                           # Non-latent mode
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            # if next_token == self.eos_token_id:                           # does this need an edge case? Probably not...
            #     break
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)

        tokens.append(next_token)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            gen_tids = torch.tensor(tokens, device=input_ids.device).view(1, -1)
            outputs = self.forward(input_embeds=new_inputs_embeds, gen_token_ids=gen_tids)
            self.gen_forward_cnt += 1

            # Decide whether NEXT token is latent or not
            latent_decision = _sample_termination(outputs.termination_logits[0, -1])

            if latent_decision == 1:        # Latent mode
                next_token = self.latent_token_id
                new_token_embed = outputs.output_embeds[0, -1].view(1, 1, -1)

            else:                           # Non-latent mode
                next_token = torch.argmax(outputs.logits[0, -1]).item()
                if next_token == self.eos_token_id:
                    break
                new_token_embed = self.embedding(
                    torch.tensor(next_token, device=input_ids.device)
                ).view(1, 1, -1)
            # print(new_inputs_embeds.shape, new_token_embed.shape)
            tokens.append(next_token)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)


    def generate_batched(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=100,
        output_embedding=False,
        synced_gpus=False,
        term_temperature=1.0,           # > 1.0 → stochastic termination for exploration
        forced_min_latents=None,        # (B,) long tensor: minimum latent steps per sequence; 0 = no forcing
        **kwargs
    ):
        batch_size = input_ids.shape[0]
        self.gen_forward_cnt = 0
        self.kv_cache = None  # Fix #3: clear stale KV cache from any previous call

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        def _sample_term_batched(term_logits_last):
            """term_logits_last: (B, 2). Returns (B,) int tensor of 0/1 decisions."""
            if term_temperature <= 0.1:
                return torch.argmax(term_logits_last, dim=-1)
            probs = torch.softmax(term_logits_last.float() / term_temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        tokens = input_ids.clone()      # for outputting token_ids
        is_terminated = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)        # (batch_size,) keep track of which sequences in the batch have generated EOS token
        in_latent_mode = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)               # once a text token is generated, lock out latent mode for that sequence
        latent_step_count = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)            # (batch_size,) number of latent tokens generated so far per sequence

        # First pass — compute position_ids from attention_mask so pad tokens
        # get position 0 and real tokens get consecutive positions.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)

        outputs = self.forward(
            input_ids,
            attention_mask,
            position_ids,
        )
        # Termination_logits shape: (B, L, 2)
        # Outputs_embeds shape: (B, L, d_emb)
        # Logits shape: (B, L, vocab_size)

        inputs_embeds = outputs.inputs_embeds
        kv_cache = outputs.past_key_values  # Fix #1: save KV cache covering the full question
        # Initialize EMA running_mean and last_hidden from the question's last hidden state
        running_mean = outputs.output_embeds[:, -1, :].detach().clone()  # (B, hidden)
        self.latent_termination_head.last_hidden = running_mean.clone()  # seed last_hidden for first gen step
        latent_decision = _sample_term_batched(outputs.termination_logits[:, -1, :])        # Decide whether next token is latent or not
        latent_decision = torch.where(in_latent_mode, latent_decision, torch.zeros_like(latent_decision))  # enforce no return to latent after text
        if forced_min_latents is not None:                                                   # force continuation for sequences below their minimum
            force_continue = in_latent_mode & (latent_step_count < forced_min_latents)
            latent_decision = torch.where(force_continue, torch.ones_like(latent_decision), latent_decision)
        in_latent_mode = in_latent_mode & (latent_decision == 1)                             # transition to text mode if this step generated text
        raw_next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        next_embeds = torch.where((latent_decision == 1).unsqueeze(-1), outputs.output_embeds[:, -1, :], self.embedding(raw_next_tokens))      # Select next token embedding based on latent decision
        next_tokens = torch.where(latent_decision == 1, self.latent_token_id, raw_next_tokens)                              # Select next token IDs based on latent decision
        latent_step_count = latent_step_count + (next_tokens == self.latent_token_id).long()

        # Fix #2: collect embeddings in a list; single torch.cat at the end avoids O(T²) reallocations
        last_embed = next_embeds.unsqueeze(1)       # (B, 1, hidden) — token to process in next iteration
        embed_list = [inputs_embeds, last_embed]    # assembled into new_inputs_embeds after the loop
        current_len = inputs_embeds.shape[1] + 1    # tracks full sequence length without a live tensor
        tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        # Track growing attention mask: original mask + 1 for each generated token so far
        # next_pos tracks the per-sequence position for the next generated token
        gen_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=input_ids.device)], dim=1)
        next_pos = attention_mask.sum(dim=1)  # (B,) — number of real tokens per sequence

        # next passes — Fix #1: process only the last token per step using the KV cache
        for _ in range(max_new_tokens - 1):
            if is_terminated.all():        # If all sequences in the batch have generated EOS token, stop generation
                break

            self.gen_forward_cnt += 1

            # KV cache covers positions 0..current_len-2; process only the current last token
            attn_mask = gen_mask
            pos_ids = next_pos.unsqueeze(1)  # (B, 1)

            logits, hidden_states, kv_cache = self._forward_base(
                last_embed,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=kv_cache,
            )
            # Termination head with EMA running_mean and stored last_hidden
            current_h = hidden_states[:, -1, :]                                  # (B, hidden)
            termination_logits = self.latent_termination_head(current_h, running_mean).unsqueeze(1)  # (B, 1, 2) — uses self.last_hidden

            # Decide whether NEXT token is latent or not
            latent_decision = _sample_term_batched(termination_logits[:, -1, :])
            latent_decision = torch.where(in_latent_mode, latent_decision, torch.zeros_like(latent_decision))  # enforce no return to latent after text
            if forced_min_latents is not None:                                                                  # force continuation for sequences below their minimum
                force_continue = in_latent_mode & (latent_step_count < forced_min_latents)
                latent_decision = torch.where(force_continue, torch.ones_like(latent_decision), latent_decision)
            in_latent_mode = in_latent_mode & (latent_decision == 1)
            raw_next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

            next_embeds = torch.where((latent_decision == 1).unsqueeze(-1), hidden_states[:, -1, :], self.embedding(raw_next_tokens))
            next_tokens = torch.where(latent_decision == 1, self.latent_token_id, raw_next_tokens)

            # Update Termination Status
            next_tokens = torch.where(is_terminated, self.pad_id, next_tokens)
            is_terminated = is_terminated | (next_tokens == self.eos_token_id)        # Update termination status for each sequence in the batch
            latent_step_count = latent_step_count + (next_tokens == self.latent_token_id).long()

            # Update last_hidden and EMA running_mean
            self.latent_termination_head.last_hidden = current_h.detach().clone()
            running_mean = self.latent_termination_head.update_ema(running_mean, current_h)

            last_embed = next_embeds.unsqueeze(1)
            embed_list.append(last_embed)
            current_len += 1
            gen_mask = torch.cat([gen_mask, torch.ones(batch_size, 1, dtype=gen_mask.dtype, device=input_ids.device)], dim=1)
            next_pos = next_pos + 1
            tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        # Fix #2: single cat to assemble the full embedding sequence
        new_inputs_embeds = torch.cat(embed_list, dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            # Fix #5: pass single token + accumulated KV cache instead of the full growing sequence
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                # _ = self.base_causallm(inputs_embeds=last_embed, past_key_values=kv_cache)

        self.kv_cache = None  # Fix #3: release KV cache after generation
        self.latent_termination_head.last_hidden = None  # release stored state

        if output_embedding:
            # for analysis purpose
            return tokens, new_inputs_embeds, None  # third value unused; caller deletes it immediately

        else:
            return tokens

    def generate_batched_n(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=100,
        num_latent=6,           # fixed number of latent steps; termination head is bypassed entirely
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        """Batched generation with a fixed number of latent steps.

        Bypasses the termination head: the first num_latent generated tokens are
        always latent (hidden-state feedback), then the model switches to normal
        text generation until EOS or max_new_tokens is reached.
        Intended for evaluation/analysis to probe model behaviour at a specific
        latent depth.
        """
        batch_size = input_ids.shape[0]
        self.gen_forward_cnt = 0
        self.kv_cache = None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        tokens = input_ids.clone()
        is_terminated = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # First pass over the full question (same as generate_batched)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)

        outputs = self.forward(input_ids, attention_mask, position_ids)
        inputs_embeds = outputs.inputs_embeds
        kv_cache = outputs.past_key_values

        # Decide first generated token: latent if num_latent > 0, otherwise text
        latent_count = 0
        if num_latent > 0:
            next_token_embed = outputs.output_embeds[:, -1, :]      # hidden state → latent embedding
            next_tokens = torch.full(
                (batch_size,), self.latent_token_id, dtype=torch.long, device=input_ids.device
            )
            latent_count = 1
        else:
            raw_next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            next_token_embed = self.embedding(raw_next_tokens)
            next_tokens = raw_next_tokens

        last_embed = next_token_embed.unsqueeze(1)
        embed_list = [inputs_embeds, last_embed]
        current_len = inputs_embeds.shape[1] + 1
        tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        gen_mask = torch.cat(
            [attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=input_ids.device)], dim=1
        )
        next_pos = attention_mask.sum(dim=1)  # (B,) — position index of last real token

        for _ in range(max_new_tokens - 1):
            if is_terminated.all():
                break

            self.gen_forward_cnt += 1

            logits, hidden_states, kv_cache = self._forward_base(
                last_embed,
                attention_mask=gen_mask,
                position_ids=next_pos.unsqueeze(1),
                past_key_values=kv_cache,
            )

            if latent_count < num_latent:
                # Latent phase: feed hidden state back as next embedding
                next_token_embed = hidden_states[:, -1, :]
                next_tokens = torch.full(
                    (batch_size,), self.latent_token_id, dtype=torch.long, device=input_ids.device
                )
                latent_count += 1
            else:
                # Text phase: pick token from lm_head
                raw_next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
                next_tokens = torch.where(is_terminated, self.pad_id, raw_next_tokens)
                is_terminated = is_terminated | (next_tokens == self.eos_token_id)
                next_token_embed = self.embedding(next_tokens)

            last_embed = next_token_embed.unsqueeze(1)
            embed_list.append(last_embed)
            current_len += 1
            gen_mask = torch.cat(
                [gen_mask, torch.ones(batch_size, 1, dtype=gen_mask.dtype, device=input_ids.device)], dim=1
            )
            next_pos = next_pos + 1
            tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        new_inputs_embeds = torch.cat(embed_list, dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + num_latent:
                self.gen_forward_cnt += 1

        self.kv_cache = None

        if output_embedding:
            return tokens, new_inputs_embeds, None
        else:
            return tokens
