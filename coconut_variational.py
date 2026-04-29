# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Variational-dropout variant of Coconut.
#
# Implementation strategy: each call to generate_batched / _replay_generate_batched
# captures the CUDA + CPU RNG state on entry and stores it as the "chain seed".
# The overridden _forward_base then restores that seed BEFORE every latent
# recurrence step (identified by inputs_embeds.shape[1] == 1). Dropout layers
# acting on tensors of shape (B, 1, H) therefore draw the same Bernoulli mask
# at every step, realizing variational dropout in the sense of
# Gal & Ghahramani (2016): a single sample theta_tilde(xi) of the perturbed
# parameters is reused across the whole latent recurrence within one chain.
#
# Caveat 1 (attention dropout): attention dropout acts on a tensor whose
# kv-length grows with the step index, so its mask shape is not constant
# across steps. Even with the RNG restored, attention dropout will draw a
# different number of Bernoullis at each step, breaking the variational
# interpretation for that path. For a clean Bayesian story, set
# attention_dropout = 0 in the model config (this is the default for
# LLaMA / Mistral / Qwen-style decoders).
#
# Caveat 2 (first latent state): the prompt forward (seq_len > 1) is not
# affected by the override; it sees regular per-position dropout. In Coconut,
# the first latent state h_1 is the last hidden of the prompt forward, so
# h_1 is computed under the prompt's dropout pattern, not the chain seed.
# Latent states h_2 ... h_T are produced inside the recurrent loop and DO
# go through the overridden _forward_base, so they share one variational
# mask. The Bayesian "single posterior sample theta_tilde(xi) reused across
# the recurrence" interpretation therefore holds for the (T-1) interior
# steps of the chain, with h_1 acting as the encoder output. For typical
# T = 4..8 this is acceptable; the boundary effect is one step out of T.

import torch

from coconut import Coconut


def _capture_chain_rng(device):
    state = {"cpu": torch.get_rng_state().clone()}
    if device is not None and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(device=device).clone().cpu()
    return state


def _restore_chain_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and device is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"], device=device)


class CoconutVariational(Coconut):
    """Coconut with per-chain locked dropout masks (variational dropout)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chain_rng = None
        self._chain_rng_device = None

    def _forward_base(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
    ):
        # Latent recurrence step: input is a single new token's hidden state,
        # shape (B, 1, H). Restore the chain's RNG seed so that this step's
        # dropout draws produce the same mask as every other step in the
        # chain. Prompt forwards (seq_len > 1) bypass this branch.
        if (
            self._chain_rng is not None
            and inputs_embeds.dim() >= 2
            and inputs_embeds.shape[1] == 1
        ):
            _restore_chain_rng(self._chain_rng, self._chain_rng_device)
        return super()._forward_base(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def _begin_chain(self, device):
        self._chain_rng = _capture_chain_rng(device)
        self._chain_rng_device = device

    def _end_chain(self):
        self._chain_rng = None
        self._chain_rng_device = None

    def _replay_generate_batched(
        self,
        input_ids,
        attention_mask,
        replay_generated_ids,
        return_inputs_embeds=False,
        return_past_key_values=False,
    ):
        self._begin_chain(input_ids.device)
        try:
            return super()._replay_generate_batched(
                input_ids,
                attention_mask,
                replay_generated_ids,
                return_inputs_embeds=return_inputs_embeds,
                return_past_key_values=return_past_key_values,
            )
        finally:
            self._end_chain()

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
        self._begin_chain(input_ids.device)
        try:
            return super().generate_batched(
                input_ids,
                attention_mask=attention_mask,
                num_latents=num_latents,
                max_new_tokens=max_new_tokens,
                output_embedding=output_embedding,
                synced_gpus=synced_gpus,
                **kwargs,
            )
        finally:
            self._end_chain()
