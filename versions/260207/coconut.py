# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "termination_logits","termination_labels"])
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, input_embeds=None, reset_kv_cache=False, **kwargs):
        # In training settings, input_ids would be tensor of shape (batch_size, padded_len)
        assert input_ids is not None or input_embeds is not None, "Input IDs or Input Embeds must be given"

        if reset_kv_cache:
            self.kv_cache = None

        logits = []

        if input_ids is not None:
            hidden_states_list = []
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

            for pass_idx in range(max_n_latents):           # compute latents?
                if kv_cache == None:
                    # first forward pass
                    outputs = self.base_causallm(           # outputs: loss, logits, past_key_values, hidden_states(because output_hidden_states), attentions(because output_hidden_states), cross_attentions
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        output_hidden_states=True,
                    )
                    hidden_states_offset = 0

                else:
                    # extract kv cache to reuse
                    past_key_values = [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in kv_cache
                    ]

                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[:, : next_compute_range[1]],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                    )

                    hidden_states_offset = next_compute_range[0]
                    # when we use kv_cache for the first k tokens
                    # in `outputs.hidden_states`, [0, k) will be skipped
                    # so we need to keep this offset to correctly use the last hidden states

                logits.append(outputs.logits)
                hidden_states_list.append(torch.Tensor(outputs.hidden_states[-1]))

                next_compute_range = (
                    next_compute_range[1],                      # next_compute_range = (0, 128) -> (128, 129) -> (129, padded_len) 
                    (
                        input_ids.shape[1]
                        if pass_idx + 1 >= max_n_latents
                        else next_compute_range[1] + 1
                    ),
                )

                hidden_states = outputs.hidden_states[                  # ONLY the last hidden output (which is the hidden embeddings for NEXT token prediction)
                    -1
                ]  # Get the last layer hidden states
                kv_cache = outputs.past_key_values

                # print(type(outputs.past_key_values))
                # print(outputs.past_key_values.shape)

                # feedback the continuous thoughts to the input_embeds

                # first decide the positions to feedback
                filling_indices = [                                                         # [[batch_num, 128], ...] -> [[batch_num, 129], ...] (if there are different number of latents per batch, then only process the ones with N latents where N == pass_idx)
                    (instance_idx, mask_list[pass_idx])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > pass_idx
                ]

                # to avoid in-place operations
                # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
                tensor_list = [                                         # [[[hidden_embs], [hidden_embs], ... (padded_len)], ... (batch_size)]
                    [                                                   # here, hidden_embs = model.vocab_emb(input_token_ids)
                        inputs_embeds[batch_idx, pos, :]
                        for pos in range(inputs_embeds.shape[1])
                    ]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]

                # replace some of them with continuous thoughts
                for idx_pair in filling_indices:                        # Replace <latent> token vocab embeddings with previous state outputs
                    batch_idx, token_idx = idx_pair

                    # replace it with the preceding last hidden states
                    tensor_list[batch_idx][token_idx] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :      # -1 here is correct because it needs to take the PREVIOUS output (128th emb is replaced with 127th)
                    ]
                    termination_labels[batch_idx][token_idx-1] = 1        # set termination labels (1 == latent operation for NEXT step, 0 == non-latent operation for NEXT step)

                # assemble the new inputs_embeds
                inputs_embeds = torch.stack(                            # Replace inputs_embs <latent> token vocab embs with output hidden embs from last layer instead
                    [
                        torch.stack(tensor_list[batch_idx])
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]
                )

            # final pass
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[                                                    # Only pass steps after latent calculation, because kv cache will exist if it was calculated
                    :, next_compute_range[0] : next_compute_range[1], :                         # Automatically passes the whole input tokens if latent is not used (= kv cache doesn't exist)
                ],
                attention_mask=attention_mask[:, : next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                past_key_values=(
                    [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in kv_cache
                    ]
                    if kv_cache
                    else None
                ),
                output_hidden_states=True,
            )

            hidden_states_list.append(torch.Tensor(outputs.hidden_states[-1]))
            hidden_states_total = torch.cat(hidden_states_list, dim=1)

            logits.append(outputs.logits)

            self.gen_forward_cnt += max_n_latents + 1

            logits = torch.cat(logits, dim=-2)                  # B, L+1, d_emb
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # termination_logits = self.latent_termination_head(logits[:, :-1, :])       # B, L, d_emb
            termination_logits = self.latent_termination_head(hidden_states_total)       # B, L, d_emb

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ) + loss_fct(termination_logits.view(-1, 2), termination_labels.view(-1)) * self.termination_gamma

            return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits, termination_logits=termination_logits, termination_labels=termination_labels)
        
        else:           # Only for generation 
            # first forward pass
            outputs = self.base_causallm(           # outputs: loss, logits, past_key_values, hidden_states(because output_hidden_states), attentions(because output_hidden_states), cross_attentions
                inputs_embeds=input_embeds,
                output_hidden_states=True,
            )

            self.kv_cache = outputs.past_key_values

            logits.append(outputs.logits)

            termination_logits = self.latent_termination_head(outputs.hidden_states[-1])
            return Outputs(loss=None, inputs_embeds=input_embeds, logits=outputs.logits, 
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

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.forward(input_embeds=new_inputs_embeds)
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
            # print(new_inputs_embeds.shape, new_token_embed.shape)
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
