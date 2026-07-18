---
name: phase-preference
description: Phase B preference learning (surrogate-DPO) — build preference pairs from train records, annotations, or decision events, then run bounded preference training from a checkpoint. Use when building pairs or training preference.
---

# Preference phase

Composite-reward pair building + surrogate-DPO training. Owner:
`src/slm_training/harnesses/preference/`.

## Prerequisites

- A checkpoint (SFT phase) and pair sources: train records, exported
  annotations (annotations phase), or local decision events.

## Commands

```bash
# Build preference pairs from train records (soft-corrupt negatives)
python -m scripts.train_preference build-pairs --limit 48 --out <pairs.jsonl>

# Bounded preference training from a checkpoint
python -m scripts.train_preference train --from-checkpoint outputs/runs/<id>/last.pt \
  --pairs <pairs.jsonl> --steps 20

# Variants: annotation/decision-event driven
python -m scripts.train_preference train-events --events <events.jsonl> ...
python -m scripts.train_preference train-local ...
```

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

- Iron law docs; model-card duty if a new serving checkpoint is written
  (`documenting-experiment-results`).
- Checks: `pytest -q tests/test_harnesses/quality/test_preference_corpora.py
  tests/test_harnesses/quality/test_quality_helpers.py`.
