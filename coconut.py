# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "output_embeds", "inputs_embeds", "logits", "termination_logits","termination_labels"])
MAX_N_LATENT = 8


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

        # Latent Termination Head
        self.termination_gamma = termination_gamma  # hyperparameter for loss
        self.latent_termination_head = nn.Linear(self.base_causallm.config.hidden_size, 2, dtype=torch.bfloat16)      # May not work for some models, need checking

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
            self.embedding.norm = nn.Identity()
            self.base_causallm.set_input_embeddings(self.embedding)

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

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, input_embeds=None, reset_kv_cache=False, **kwargs):
        # In training settings, input_ids would be tensor of shape (batch_size, padded_len)
        assert input_ids is not None or input_embeds is not None, "Input IDs or Input Embeds must be given"

        if reset_kv_cache:
            self.kv_cache = None

        if input_ids is not None:
            hidden_states_list = []
            logits_list = []
            termination_labels = torch.zeros((input_ids.shape[0], input_ids.shape[1]), dtype=torch.long, device=input_ids.device)        # (B, L) mask for latent termination head calculation

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
                    # when we use kv_cache for the first k tokens
                    # in hidden_states, [0, k) will be skipped
                    # so we need to keep this offset to correctly use the last hidden states

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
            logits_chunk, hidden_states, _ = self._forward_base(
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

            logits = torch.cat(logits_list, dim=-2)                  # B, L+1, d_emb
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # termination_logits = self.latent_termination_head(logits[:, :-1, :])       # B, L, d_emb
            termination_logits = self.latent_termination_head(hidden_states_total)       # B, L, d_emb

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ) + loss_fct(termination_logits.view(-1, 2), termination_labels.view(-1)) * self.termination_gamma

            return Outputs(loss=loss, inputs_embeds=inputs_embeds, output_embeds=hidden_states_total, logits=logits, termination_logits=termination_logits, termination_labels=termination_labels)

        else:           # Only for generation
            # Use _forward_base to avoid output_hidden_states=True (Change 4)
            logits, hidden_states, kv_cache = self._forward_base(inputs_embeds=input_embeds)

            self.kv_cache = kv_cache

            termination_logits = self.latent_termination_head(hidden_states)
            return Outputs(loss=None, output_embeds=hidden_states, inputs_embeds=input_embeds, logits=logits,
                        termination_logits=termination_logits, termination_labels=None)


    def train(self):
        self.base_causallm.train()
        self.latent_termination_head.train()

    def eval(self):
        self.base_causallm.eval()
        self.latent_termination_head.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=100,     # expanded due to variable length latent generation (16 -> 100)
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # Decide whether first token is latent or not
        latent_decision = torch.argmax(outputs.termination_logits[0, -1]).item()

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
            outputs = self.forward(input_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1

            # Decide whether NEXT token is latent or not
            latent_decision = torch.argmax(outputs.termination_logits[0, -1]).item()

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
        attention_mask,  # attention_mask is not used
        num_outputs_per_batch=4,
        max_new_tokens=100,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        # NEEDS WORK, NOT YET IMPLEMENTED!!

        self.gen_forward_cnt = 0

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1

            # Decide whether current token is latent or not
            latent_decision = torch.argmax(outputs.termination_logits[0, -1]).item()

            if latent_decision == 1:        # Latent mode
                next_token = self.latent_token_id
                new_token_embed = outputs.logits[0, -1]

            else:                           # Non-latent mode
                next_token = torch.argmax(outputs.logits[0, -1]).item()
                if next_token == self.eos_token_id:
                    break
                new_token_embed = self.embedding(
                    torch.tensor(next_token, device=input_ids.device)
                ).view(1, 1, -1)
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
