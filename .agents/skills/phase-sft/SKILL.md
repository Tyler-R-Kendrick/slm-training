---
name: phase-sft
description: Phase A supervised model-build training of TwoTower/grammar models, locally or via remote pod and managed Hugging Face Jobs GPU, with checkpoint-bucket sync. Use when running SFT / train_model / hf_jobs_train / remote_train.
---

# SFT (model build) phase

Supervised training from a versioned train snapshot. Owner:
`src/slm_training/harnesses/model_build/`.

## Prerequisites

- `outputs/data/train/<version>/` (train-data phase).
- `HF_TOKEN` (or `hf auth login`) for HF context backend and bucket sync; never
  commit tokens.

## Commands

```bash
# Local/full train (full HF-context runs sync checkpoints to the OpenUI bucket)
python -m scripts.train_model --train-dir outputs/data/train/v1 \
  --model twotower --context-backend hf --steps 200 --run-id twotower_v1

# Managed GPU job (A10G+) — always inspect the plan first
python -m scripts.hf_jobs_train --run-id twotower_v1 --steps 200 --branch main --dry-run
python -m scripts.hf_jobs_train --run-id twotower_v1 --steps 200 --branch main

# Remote pod
python -m scripts.remote_train --host <pod> --run-id twotower_v1 --steps 200
```

Scratch/CI runs add `--no-sync-checkpoints`. Do not use Spaces ZeroGPU for full
trains (short quotas, no `torch.compile`).

## Key flags

`--fast-train`, `--context-backend hf|scratch`, `--device`, `--batch-size`,
`--seed`, `--checkpoint-bucket` / `--checkpoint-bucket-dry-run`.

## Outputs

`outputs/runs/<run-id>/` with `train_summary.json` and `trace.json`. A full
HF-context train is **not done** until `train_summary.json` contains a
successful `checkpoint_bucket` URI
(`hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/`) or a documented
scratch reason.

## Gates & invariants

- Every created/synced checkpoint updates `docs/MODEL_CARD.md` **and** the
  README "Model card (summary)".
- Fixture/scratch trains are wiring evidence only — never ship claims.

## Close out

- Iron law: `docs/design/` JSON + markdown for the run
  (`documenting-experiment-results`).
- Docs: `docs/design/hf-jobs-train.md`, `docs/design/checkpoint-bucket.md`.
  Checks: `pytest -q tests/test_harnesses/model_build tests/test_models`.
- Next: evaluate with the eval phase; changing the harness →
  `improve-openui-harnesses`.
