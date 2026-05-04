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
#
# Caveat 3 (modern decoder architectures): Qwen2 / LLaMA / Mistral / Gemma
# decoders have NO residual, MLP, or embedding dropout layers. Their only
# dropout point is attention dropout (caveat 1), which has variable mask
# shape across t and so cannot be locked by RNG restore. On these models,
# the variational override effectively becomes a no-op for the constant-
# shape dropout path (there is none), and attention dropout reverts to
# regular per-step behavior. The Bayesian "single posterior sample"
# interpretation does not hold cleanly on these architectures; it holds
# only for GPT-2-family models that ship with embd_pdrop / resid_pdrop.
# The GRPO proof framework (Theorem 4.4 rho_old==1, Prop 4.5 CRN coupling)
# remains valid in either case because those theorems only require xi to
# be theta-independent and replayable -- which holds for any dropout
# scheme combined with this RNG-restore implementation.

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


class _DropoutMaskAuditor:
    """Auditor that verifies dropout masks are consistent across latent
    recurrence steps within a chain. Covers BOTH:

      (a) nn.Dropout modules (constant-shape masks) -- via forward hooks;
      (b) torch.nn.functional.dropout calls (variable-shape masks, e.g.
          attention dropout in Qwen2/LLaMA/Mistral whose mask shape is
          (B, H, 1, K_t) with K_t growing across steps) -- via a temporary
          monkey-patch of torch.nn.functional.dropout that wraps the call
          and records the mask while the chain is active.

    Mask recovery: `(output != 0)`. Exact at positions where input is
    non-zero; positions with input == 0 give no info about the Bernoulli
    draw. Comparison restricts to positions that were non-zero in BOTH
    the current and previously recorded step.

    Variable-shape masks (case b): compared on the shape-overlap slice
    (per-dim min). With PyTorch's Philox-based CUDA dropout kernel,
    element i of the mask is determined by (seed, offset, i) independent
    of total tensor length -- so when RNG state is restored at the start
    of each step, the prefix masks at step t and step t-1 must be
    bit-identical on the overlap. Any divergence on the overlap is a
    real mask drift.

    Step boundaries: comparison is enabled only during latent recurrence
    steps (shape[1] == 1 forwards). The prompt forward is ignored. The
    functional call counter resets at each step boundary; call index k
    at step t is compared against call index k at step t-1.
    """

    def __init__(self, strict=False):
        self.strict = strict
        self.module_records = {}      # name -> (mask, valid)
        self.functional_records = {}  # call_idx -> (mask, valid)
        self.mismatches = []
        self.checks = 0
        self.call_counter = 0
        self.step_counter = 0
        self.in_step = False
        self.handles = []
        self.enabled = False
        self._orig_f_dropout = None

    def _compare(self, key, input_tensor, output_tensor, store):
        valid = input_tensor != 0
        mask = output_tensor != 0
        prev = store.get(key)
        if prev is not None:
            pmask, pvalid = prev
            slices = tuple(
                slice(0, min(a, b))
                for a, b in zip(pmask.shape, mask.shape)
            )
            cur_m = mask[slices]
            cur_v = valid[slices]
            prv_m = pmask[slices]
            prv_v = pvalid[slices]
            both = cur_v & prv_v
            total = int(both.sum().item())
            if total > 0:
                self.checks += 1
                diff = int(((cur_m ^ prv_m) & both).sum().item())
                if diff > 0:
                    info = {
                        "key": key,
                        "step": self.step_counter,
                        "diff": diff,
                        "checked": total,
                        "frac": diff / total,
                        "prev_shape": tuple(pmask.shape),
                        "cur_shape": tuple(mask.shape),
                    }
                    self.mismatches.append(info)
                    if self.strict:
                        raise AssertionError(
                            f"dropout mask mismatch at {key} (step "
                            f"{self.step_counter}): {diff}/{total} positions "
                            f"differ; prev_shape={tuple(pmask.shape)} "
                            f"cur_shape={tuple(mask.shape)}"
                        )
        store[key] = (mask.detach(), valid.detach())

    def _module_hook(self, name):
        def fn(module, inputs, output):
            if not self.enabled or not self.in_step:
                return
            if not module.training or float(getattr(module, "p", 0.0)) == 0.0:
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.dim() < 2 or x.shape[1] != 1:
                return
            self._compare(f"module:{name}", x, output, self.module_records)
        return fn

    def _wrap_functional(self, orig):
        def wrapped(input, p=0.5, training=True, inplace=False):
            out = orig(input, p=p, training=training, inplace=inplace)
            if (
                self.enabled
                and self.in_step
                and training
                and p > 0.0
                and torch.is_tensor(input)
            ):
                self.call_counter += 1
                key = f"functional:{self.call_counter}"
                self._compare(key, input, out, self.functional_records)
            return out
        return wrapped

    def attach(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Dropout):
                self.handles.append(
                    mod.register_forward_hook(self._module_hook(name))
                )

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._unpatch_functional()

    def _patch_functional(self):
        if self._orig_f_dropout is None:
            self._orig_f_dropout = torch.nn.functional.dropout
            torch.nn.functional.dropout = self._wrap_functional(
                self._orig_f_dropout
            )

    def _unpatch_functional(self):
        if self._orig_f_dropout is not None:
            torch.nn.functional.dropout = self._orig_f_dropout
            self._orig_f_dropout = None

    def begin_chain(self):
        self.module_records.clear()
        self.functional_records.clear()
        self.mismatches.clear()
        self.checks = 0
        self.call_counter = 0
        self.step_counter = 0
        self.in_step = False
        self.enabled = True
        self._patch_functional()

    def end_chain(self):
        self.enabled = False
        self.in_step = False
        self._unpatch_functional()

    def begin_step(self):
        """Mark the start of a latent recurrence step. Resets the functional
        call counter so that call index k at this step is compared against
        call index k at the previous step. Prompt forwards must NOT call
        this -- only seq_len == 1 forwards do."""
        self.call_counter = 0
        self.step_counter += 1
        self.in_step = True


class CoconutVariational(Coconut):
    """Coconut with per-chain locked dropout masks (variational dropout)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chain_rng = None
        self._chain_rng_device = None
        self._mask_auditor = None

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
            if self._mask_auditor is not None:
                self._mask_auditor.begin_step()
        return super()._forward_base(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def _begin_chain(self, device):
        self._chain_rng = _capture_chain_rng(device)
        self._chain_rng_device = device
        if self._mask_auditor is not None:
            self._mask_auditor.begin_chain()

    def _end_chain(self):
        self._chain_rng = None
        self._chain_rng_device = None
        if self._mask_auditor is not None:
            self._mask_auditor.end_chain()

    def enable_mask_audit(self, strict=False):
        """Attach forward hooks that verify dropout masks are bit-identical
        across latent recurrence steps within a chain. Call once after model
        construction. Use strict=True to raise on first mismatch."""
        if self._mask_auditor is None:
            self._mask_auditor = _DropoutMaskAuditor(strict=strict)
            self._mask_auditor.attach(self)

    def disable_mask_audit(self):
        if self._mask_auditor is not None:
            self._mask_auditor.detach()
            self._mask_auditor = None

    def get_mask_audit_report(self):
        """Returns (num_checks, list_of_mismatches) for the most recent chain.
        Covers nn.Dropout modules AND torch.nn.functional.dropout calls
        (e.g. Qwen2/LLaMA attention dropout). Empty mismatch list with
        checks > 0 means all dropouts agreed across recurrence steps on
        the shape-overlap. checks == 0 means dropout fired fewer than
        twice during the chain (e.g. only one latent step, or all p=0)."""
        if self._mask_auditor is None:
            return 0, []
        return self._mask_auditor.checks, list(self._mask_auditor.mismatches)

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
