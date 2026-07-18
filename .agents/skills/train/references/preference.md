# Preference phase

Composite-reward pair building + surrogate-DPO training. Owner:
`src/slm_training/harnesses/preference/`.

## Prerequisites

- A checkpoint (SFT phase) and pair sources: train records, exported
  annotations (annotations phase), or local decision events.

## Commands

```bash
# Build preference pairs from train records (soft-corrupt negatives)
slm preference build-pairs --limit 48 --out <pairs.jsonl>

# Bounded preference training from a checkpoint
slm preference train --from-checkpoint outputs/runs/<id>/last.pt \
  --pairs <pairs.jsonl> --steps 20

# Variants: annotation/decision-event driven
slm preference train-events --events <events.jsonl> ...
slm preference train-local ...
```

(`slm preference <subcommand>` passes through to
`python -m scripts.train_preference`.)

## Key flags

`--balanced`, `--corrupt`, `--epsilon`, `--lr`, `--limit`,
`--allow-invalid-rejects` (diagnostic only), `--evidence-out`.

## Outputs

Pairs and preference checkpoints under the owning run — never a second corpus
tree.

## Gates & invariants

- Pair validity + provenance recorded; keep the documented surrogate-DPO
  distinction honest (it is not full DPO).
- Never train on eval-feedback holdouts.

## Close out

- Shared duties: [contracts.md](contracts.md).
- Checks: `pytest -q tests/test_harnesses/quality/test_preference_corpora.py
  tests/test_harnesses/quality/test_quality_helpers.py`.
