# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

import bitsandbytes as bnb

Outputs = namedtuple(
    "Outputs",
    ["loss", "inputs_embeds", "output_embeds", "logits", "past_key_values"],
)
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):
        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.pad_id = eos_token_id
        self.kv_cache = None

        if isinstance(base_causallm, GPT2LMHeadModel):
            self._base_transformer_attr = "transformer"
        else:
            self._base_transformer_attr = "model"

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            model_embedding = self.base_causallm.get_input_embeddings()
            self.embedding = bnb.nn.StableEmbedding(
                model_embedding.num_embeddings,
                model_embedding.embedding_dim,
                padding_idx=model_embedding.padding_idx,
            )
            self.embedding.weight.data.copy_(model_embedding.weight.data)
            self.embedding.norm = nn.Identity()
            self.base_causallm.set_input_embeddings(self.embedding)
            self.base_causallm.lm_head.weight = self.embedding.weight

    def _forward_base(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
    ):
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
        if hasattr(kv_cache, "key_cache"):
            return [
                (
                    kv_cache.key_cache[i][:, :, :max_pos, :],
                    kv_cache.value_cache[i][:, :, :max_pos, :],
                )
                for i in range(len(kv_cache.key_cache))
            ]

        return [
            (k[:, :, :max_pos, :], v[:, :, :max_pos, :])
            for k, v in kv_cache
        ]

    def _replay_generate_batched(self, input_ids, attention_mask, replay_generated_ids):
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        full_len = replay_generated_ids.shape[1]
        gen_len = full_len - prompt_len

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

        logits_list = [prompt_outputs.logits]
        kv_cache = prompt_outputs.past_key_values

        is_terminated = torch.zeros(
            batch_size, dtype=torch.bool, device=input_ids.device
        )
        first_expected = replay_generated_ids[:, prompt_len]
        first_next_embeds = torch.where(
            (first_expected == self.latent_token_id).unsqueeze(-1),
            prompt_outputs.output_embeds[:, -1, :],
            self.embedding(first_expected),
        )
        is_terminated = is_terminated | (first_expected == self.eos_token_id)

        last_embed = first_next_embeds.unsqueeze(1)
        embed_list = [prompt_outputs.inputs_embeds, last_embed]
        gen_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    batch_size,
                    1,
                    dtype=attention_mask.dtype,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )
        next_pos = attention_mask.sum(dim=1)

        for step in range(1, gen_len):
            logits, hidden_states, kv_cache = self._forward_base(
                last_embed,
                attention_mask=gen_mask,
                position_ids=next_pos.unsqueeze(1),
                past_key_values=kv_cache,
            )
            logits_list.append(logits)

            expected_tokens = replay_generated_ids[:, prompt_len + step]
            active = ~is_terminated
            next_embeds = torch.where(
                (active & (expected_tokens == self.latent_token_id)).unsqueeze(-1),
                hidden_states[:, -1, :],
                self.embedding(expected_tokens),
            )

            is_terminated = is_terminated | (active & (expected_tokens == self.eos_token_id))
            last_embed = next_embeds.unsqueeze(1)
            embed_list.append(last_embed)
            gen_mask = torch.cat(
                [
                    gen_mask,
                    torch.ones(
                        batch_size,
                        1,
                        dtype=gen_mask.dtype,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            next_pos = next_pos + 1

        return Outputs(
            loss=None,
            inputs_embeds=torch.cat(embed_list, dim=1),
            output_embeds=None,
            logits=torch.cat(logits_list, dim=1),
            past_key_values=kv_cache,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        input_embeds=None,
        reset_kv_cache=False,
        replay_generated_ids=None,
        **kwargs,
    ):
        assert input_ids is not None or input_embeds is not None, (
            "Input IDs or Input Embeds must be given"
        )

        if replay_generated_ids is not None:
            assert input_ids is not None, "Prompt input_ids are required for replay"
            return self._replay_generate_batched(
                input_ids, attention_mask, replay_generated_ids
            )

        if reset_kv_cache:
            self.kv_cache = None

        if input_ids is not None:
            hidden_states_list = []
            logits_list = []

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=input_ids.device)

            if position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.clamp_(min=0)

            latent_indices = (input_ids == self.latent_token_id).nonzero()
            latent_lists = [
                [idx[1].item() for idx in latent_indices if idx[0] == i]
                for i in range(input_ids.shape[0])
            ]
            max_n_latents = max(len(latent_list) for latent_list in latent_lists)

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
                        else min(
                            mask_list[pass_idx + 1]
                            for mask_list in latent_lists
                            if len(mask_list) > pass_idx + 1
                        )
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
                        batch_idx,
                        token_idx - 1 - hidden_states_offset,
                        :,
                    ]

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

            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            else:
                loss = None

            return Outputs(
                loss=loss,
                inputs_embeds=inputs_embeds,
                output_embeds=hidden_states_total,
                logits=logits,
                past_key_values=final_kv_cache,
            )

        logits, hidden_states, kv_cache = self._forward_base(inputs_embeds=input_embeds)
        self.kv_cache = kv_cache
        return Outputs(
            loss=None,
            inputs_embeds=input_embeds,
            output_embeds=hidden_states,
            logits=logits,
            past_key_values=kv_cache,
        )

    def generate(
        self,
        input_ids,
        attention_mask=None,
        num_latents=0,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs,
    ):
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        outputs = self.generate_batched(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_latents=num_latents,
            max_new_tokens=max_new_tokens,
            output_embedding=output_embedding,
            synced_gpus=synced_gpus,
            **kwargs,
        )
        return outputs

    def generate_batched(
        self,
        input_ids,
        attention_mask=None,
        num_latents=0,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        self.gen_forward_cnt = 0
        self.kv_cache = None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        inputs_embeds = outputs.inputs_embeds
        kv_cache = outputs.past_key_values

        latent_count = 0
        end_latent_emitted = num_latents == 0
        if num_latents > 0:
            next_embeds = outputs.output_embeds[:, -1, :]
            next_tokens = torch.full(
                (batch_size,),
                self.latent_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            latent_count = 1
        else:
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            next_embeds = self.embedding(next_tokens)

        is_terminated = next_tokens == self.eos_token_id
        last_embed = next_embeds.unsqueeze(1)
        embed_list = [inputs_embeds, last_embed]
        tokens = torch.cat((input_ids, next_tokens.unsqueeze(1)), dim=1)
        gen_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    batch_size,
                    1,
                    dtype=attention_mask.dtype,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )
        next_pos = attention_mask.sum(dim=1)

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

            if latent_count < num_latents:
                next_tokens = torch.full(
                    (batch_size,),
                    self.latent_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                next_embeds = hidden_states[:, -1, :]
                latent_count += 1
            elif not end_latent_emitted:
                next_tokens = torch.full(
                    (batch_size,),
                    self.end_latent_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                next_embeds = self.embedding(next_tokens)
                end_latent_emitted = True
            else:
                raw_next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
                next_tokens = torch.where(
                    is_terminated,
                    torch.full_like(raw_next_tokens, self.pad_id),
                    raw_next_tokens,
                )
                next_embeds = self.embedding(next_tokens)
                is_terminated = is_terminated | (next_tokens == self.eos_token_id)

            last_embed = next_embeds.unsqueeze(1)
            embed_list.append(last_embed)
            gen_mask = torch.cat(
                [
                    gen_mask,
                    torch.ones(
                        batch_size,
                        1,
                        dtype=gen_mask.dtype,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            next_pos = next_pos + 1
            tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)

        new_inputs_embeds = torch.cat(embed_list, dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1

        self.kv_cache = None

        if output_embedding:
            return tokens, new_inputs_embeds
        return tokens
