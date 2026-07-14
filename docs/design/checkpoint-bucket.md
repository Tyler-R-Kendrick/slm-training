# Checkpoint bucket (Hugging Face)

Durable storage for **real full training runs** (HF-context ship track).

| Field | Value |
| --- | --- |
| Bucket | [TKendrick/OpenUI](https://huggingface.co/buckets/TKendrick/OpenUI) |
| URI | `hf://buckets/TKendrick/OpenUI` |
| Layout | `checkpoints/<run_id>/…` |

## What gets synced

From `outputs/runs/<run_id>/checkpoints/`:

- `last.pt` + `.tokenizer.json` + `.meta.json` (+ optional `.context.tokenizer.json`)
- `best_ship_score.pt` / `best_weighted_nll.pt` (+ sidecars) when present
- `last_full_state.pt` (resume payload) when present
- `promoted.pt` / `promoted.json` when present
- `train_summary.json` (copied from the run dir)

## When sync runs

Auto-enabled when `ModelBuildConfig.context_backend == "hf"` and sync is not
disabled. Scratch / quality-matrix CPU demos stay local-only.

| Entry | Default |
| --- | --- |
| `scripts.train_model` (default `--context-backend hf`) | sync on |
| `scripts.remote_train` | sync on (`--sync-checkpoints`) |
| `scripts.run_quality_matrix` (default scratch) | sync off |
| CI / fixture demos | use `--no-sync-checkpoints` or scratch |

Disable: `--no-sync-checkpoints`, `SLM_DISABLE_CHECKPOINT_BUCKET=1`, or
`checkpoint_bucket=""`.

## Auth

Write access requires a Hub token with bucket permissions:

```bash
export HF_TOKEN=hf_...   # or: hf auth login
```

Also accepted: `HUGGING_FACE_HUB_TOKEN`, `SLM_CHECKPOINT_BUCKET` (override URI).

## Commands

```bash
# Full HF train (auto-sync at end)
python -m scripts.train_model \
  --train-dir outputs/train_data/v1 \
  --run-id twotower_v1 \
  --context-backend hf \
  --steps 200

# Manual / recover a local run
python -m scripts.sync_checkpoints \
  --run-dir outputs/runs/twotower_v1 \
  --ensure-bucket

# Plan only
python -m scripts.sync_checkpoints --run-dir outputs/runs/twotower_v1 --dry-run
```

Artifacts: `outputs/runs/<id>/checkpoint_bucket.json` plus
`train_summary.json` → `checkpoint_bucket` field.
