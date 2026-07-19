# Checkpoint bucket (Hugging Face)

Durable storage for **real full training runs** (HF-context ship track).

| Field | Value |
| --- | --- |
| Bucket | [TKendrick/OpenUI](https://huggingface.co/buckets/TKendrick/OpenUI) |
| URI | `hf://buckets/TKendrick/OpenUI` |
| Layout | `checkpoints/<run_id>/…` |

Autoresearch evidence bundles use a separate prefix,
`autoresearch/<campaign_id>/`. They contain campaign specs, source/evidence
snapshots, decisions, outcomes, telemetry, and checksums—not serving checkpoints.
The checkpoint and model-card rules below still apply to any full training run
launched by a campaign.

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
| `scripts.train_model` (default `--context-backend hf`) | sync on (CLI sets `sync_checkpoints=True`) |
| `scripts.hf_jobs_train` | sync on (Jobs entrypoint `--sync-checkpoints`) |
| `scripts.remote_train` | sync on (`--sync-checkpoints`) |
| Programmatic `ModelBuildConfig` / pytest | sync off |
| `scripts.run_quality_matrix` (default scratch) | sync off |
| CI / fixture demos | local-only (no CLI sync flags) |

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
  --train-dir outputs/data/train/v1 \
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

## Checkpoint references (fail-closed provenance)

Every sync hashes each artifact before upload, writes a canonical
`CheckpointReferenceV1` sidecar (`<checkpoint>.ref.json`) plus an aggregate
`checkpoint_references.json` manifest into the uploaded set, and — for a real
sync — re-verifies that the files landed remotely before stamping the reference
`verified`. A verification mismatch raises (the train fails closed); a dry run is
never persistence evidence. Frontier / ship-candidate references must be fully
provenanced and verified to be publishable. Declare the class with
`--claim-class` (default `diagnostic`). Full contract:
[checkpoint-provenance.md](checkpoint-provenance.md); CI audit
`python -m scripts.verify_checkpoint_references --check`.

## Model card (required)

After every successful sync (or fixture bootstrap that writes a checkpoint),
update:

1. [`docs/MODEL_CARD.md`](../MODEL_CARD.md) — roster, eval, history, URI
2. [`README.md`](../../README.md) — “Model card (summary)” table only

Agent process: [`AGENTS.md`](../../AGENTS.md) + skill
`documenting-experiment-results`.

## Measured results

| Date (UTC) | Run | Sync? | Notes |
| --- | --- | --- | --- |
| 2026-07-19 | `e513-e396-e500-replay050-slotrole4-focal2-r3-5k` | Yes | Automatic sync and resync verification passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e513-e396-e500-replay050-slotrole4-focal2-r3-5k`; serving checkpoint SHA `59253c679477060694370c5e2d8cd9fce5d7accc7d71df3b6d56edf0a88a9548`, full-state SHA `98b6d71321add3962faa2d717a5963f17d53719f166af17ee0c6120ed7fe5133`. The run completed 101 CPU HF-context steps / 5,000 target tokens in 79.6s under `max_wall_minutes=3`. E514 OOD gates and AgentV fail, so this is durable diagnostic evidence, not a promotion or ship checkpoint. |
| 2026-07-19 | E357 training-data snapshot for E504 replay | Yes (data only) | Exact eight-file, 998-row corpus persisted at `hf://buckets/TKendrick/OpenUI/data/train/e357_card_hierarchy_v1/`. Post-upload sync found all eight files identical; independent download verified semantic manifest SHA `a4f212a3444d0f219fe1b3604f70929fe1a1b91d4fdc11a73167cb74c55b6a51` and records SHA `b1b2c3d0c1965bd9829edfc6ae34b5dce916a68c33bb17497a6392c80d7ea6ef`. E504's five rejected checkpoints were explicitly not synced. |
| 2026-07-18 | `e396-balanced-type-head-continuation-r1` | Yes | Manual recovery sync verified at `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1`; checkpoint SHA `feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`. E498 restores current-main loading and learned-head application, but smoke semantic gates and AgentV remain red. Persistence and compatibility are verified; this is not a champion or serving promotion. |
| 2026-07-14 | `restructure_cpu_scratch_v0` | No (`--no-sync-checkpoints`) | Post-package-restructure CPU fixture/scratch train on a 4c/15GB host with **no** `HF_TOKEN`. Validates harness wiring after the dsl/harnesses/runtime move. Smoke parse **0.0** @ 80 steps — not a ship claim. JSON: [restructure-cpu-train-results.json](restructure-cpu-train-results.json). |
| 2026-07-14 | `restructure_cpu_scratch_v0_cont` | No | Resume from v0 full-state; +200 CPU scratch steps. Smoke parse still 0.0. HF Jobs blocked: no Cloud Agent HF_TOKEN. JSON: [restructure-cpu-train-cont-results.json](restructure-cpu-train-cont-results.json). |
| 2026-07-14 | `local_directml_adreno_20260714` | No (`--no-sync-checkpoints`) | Five-step local scratch train on Qualcomm Adreno X1-85 through Torch-DirectML. Checkpoint and CPU reload verified; no eval/ship claim. JSON: [local-directml-train-results.json](local-directml-train-results.json). |
