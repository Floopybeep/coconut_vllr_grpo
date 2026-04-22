# Dropout GRPO Ablations

These configs target `run_grpo_dropout.py` and map directly to the ablations listed in `research_plan.md`.

Recommended run pattern:

```bash
torchrun --nnodes 1 --nproc_per_node 1 run_grpo_dropout.py ablations/replay_consistent.yaml
```

Included configs:

- `replay_consistent.yaml`: main method, deterministic reference model.
- `naive_dropout.yaml`: dropout during rollout/update without RNG-restored policy replay.
- `no_dropout.yaml`: negative control with dropout disabled. Under greedy decoding this is expected to collapse rollout diversity.
- `dropout_low.yaml`: replay-consistent dropout with `dropout=0.05`.
- `dropout_high.yaml`: replay-consistent dropout with `dropout=0.20`.
- `group_rollouts_8.yaml`: replay-consistent dropout with `num_rollouts=8`.
- `group_rollouts_32.yaml`: replay-consistent dropout with `num_rollouts=32`.
- `chunk_mismatch.yaml`: intentionally mismatched rollout/policy chunk sizes. This tests the replay-structure sensitivity claim, so replay verification is disabled on purpose.
- `reference_dropout_matched.yaml`: replay-consistent dropout with a dropout-matched reference replay instead of a deterministic reference.
