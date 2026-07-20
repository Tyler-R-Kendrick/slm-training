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
slm sft train --train-dir outputs/data/train/v1 \
  --model twotower --context-backend hf --steps 200 --run-id twotower_v1

# Managed GPU job (A10G+) — inspect the dry-run plan first, then get explicit
# user approval before the paid execute (approvals contract, below)
slm sft hf-jobs --run-id twotower_v1 --steps 200 --branch main --dry-run
slm sft hf-jobs --run-id twotower_v1 --steps 200 --branch main   # paid — approval required

# Remote pod (remote job — approval required)
slm sft remote --host <pod> --run-id twotower_v1 --steps 200
```

(`slm sft train|hf-jobs|remote` ≡ `python -m scripts.train_model` /
`scripts.hf_jobs_train` / `scripts.remote_train`.)

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

## Close out

- Shared duties (iron law, model-card, fixture ≠ ship): [contracts.md](contracts.md).
- Docs: `docs/design/hf-jobs-train.md`, `docs/design/checkpoint-bucket.md`.
  Checks: `pytest -q tests/test_harnesses/model_build tests/test_models`.
- Next: evaluate with [eval.md](eval.md); changing the harness →
  `improve-openui-harnesses`.
