---
name: phase-experiments
description: Run experiment matrices and structured campaigns — quality (E*), grammar (X*), perf (P/Q/R), the A→B→C phase pipeline, scaling ladder, mixture search, recipe evolution, and baseline reproduction. Use when running or extending experiment matrices.
---

# Experiments phase

Matrices are the intended way to land levers. Owner:
`src/slm_training/harnesses/experiments/`. Methodology (IDs, gate policy,
interpretation): **`running-experiment-matrices`** — follow it alongside this
phase.

## Prerequisites

- Versioned train/eval data; matrix IDs chosen from the matching design doc.

## Commands

```bash
# Quality matrix (honest ship path example)
python -m scripts.run_quality_matrix --matrix v6 --only E53 --steps 80 \
  --device cpu --context-backend scratch --no-design-md-context --scratch-control

python -m scripts.run_grammar_matrix --only X22 ...
python -m scripts.run_perf_matrix --list        # then --only <ids>

# Phases A→B→C to completion (RL leg gated)
python -m scripts.run_phase_pipeline --rl-readiness-report <approved.json>

# Scaling / mixture / recipe evolution / reproduction
python -m scripts.run_scaling_ladder --family <f> --arms <n>
python -m scripts.run_mixture_search --train-dir ... --test-dir ...
python -m scripts.run_recipe_evolution --campaign-id g2 --dry-run
python -m scripts.reproduce_baseline --train-dir ... --test-dir ... --seeds 3
```

## Key flags

`--only`, `--matrix`, `--describe`/`--list`, `--docs-out`, `--steps`,
`--device`, `--context-backend`, `--scratch-control`, `--seeds`.

## Outputs

Run roots under `outputs/runs/`, matrix scoreboards mirrored to
`docs/design/*results.json` + the matching markdown matrix.

## Gates & invariants

- Isolatable experiment IDs; shared gate policy; matched controls.
- Register/promote only fully evaluated checkpoints; quality gates outrank
  speed and cost.

## Close out

- Iron law docs are the point of a matrix run
  (`documenting-experiment-results`); doc homes:
  `quality-experiment-matrix.md`, `grammar-experiment-matrix.md`,
  `perf-experiment-matrix.md`.
- Checks: `pytest -q tests/test_harnesses/experiments`.
