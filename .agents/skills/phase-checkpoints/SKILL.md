---
name: phase-checkpoints
description: Checkpoint lifecycle operations — sync runs to the OpenUI HF bucket, migrate checkpoint formats, and drive the immutable model_cycle lineage (snapshot, branch, train, evaluate, promote, merge, deploy). Use when syncing, promoting, or reconciling checkpoints.
---

# Checkpoints & lineage phase

Durable checkpoints + immutable lineage. Owners:
`src/slm_training/lineage/` (cycle), bucket flows in
`scripts/sync_checkpoints.py`.

## Prerequisites

- `HF_TOKEN` with write access for bucket sync
  (`hf://buckets/TKendrick/OpenUI`); never commit tokens.

## Commands

```bash
# Manual / rescue sync of a run's checkpoints to the bucket
python -m scripts.sync_checkpoints --run-dir outputs/runs/<id> --ensure-bucket

# Format migration
python -m scripts.migrate_checkpoint --checkpoint <old.pt> --output <new.pt>

# Immutable lifecycle (16 subcommands)
python -m scripts.model_cycle snapshot-data ...
python -m scripts.model_cycle init|branch|train|evaluate|promote|merge|deploy ...
python -m scripts.model_cycle submit-nemo|reconcile-nemo|submit-molt|reconcile-molt ...
```

Hygiene: `python -m scripts.verify_checkpoint_references --check` (fail-closed
provenance; runs in CI). `python -m scripts.bootstrap_playground` copies a
serving checkpoint — the model-card duty applies to it too.

## Key flags

`sync`: `--dry-run`, `--claim-class`, `--provenance-json`,
`--training-source-commit`; `cycle`: see `model_cycle --help` per subcommand.

## Outputs

Bucket URIs under `checkpoints/<run_id>/`; verified `CheckpointReferenceV1`
provenance records; lineage state under the cycle's roots.

## Gates & invariants

- **Every** created/synced/bootstrapped/promoted checkpoint updates
  `docs/MODEL_CARD.md` and the README "Model card (summary)" — a checkpoint
  without both is incomplete work.
- Promotion registers only fully evaluated checkpoints; promotion records are
  immutable.
- `frontier`/`ship_candidate` citations must resolve from a fresh clone.

## Close out

- Docs: `docs/design/checkpoint-bucket.md`,
  `docs/design/checkpoint-provenance.md`, `docs/MODEL_CARD.md`
  (`documenting-experiment-results`).
- Inspect with `hf-cli` bucket skills (`hf buckets list TKendrick/OpenUI -R`).
