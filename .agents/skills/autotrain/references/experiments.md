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
slm experiments quality-matrix --matrix v6 --only E53 --steps 80 \
  --device cpu --context-backend scratch --no-design-md-context --scratch-control

slm experiments grammar-matrix --only X22 ...
slm experiments perf-matrix --list        # then --only <ids>

# Phases A→B→C to completion (RL leg gated)
slm experiments phase-pipeline --rl-readiness-report <approved.json>

# Scaling / mixture / recipe evolution / reproduction
slm experiments scaling-ladder --family <f> --arms <n>
slm experiments mixture-search --train-dir ... --test-dir ...
slm experiments recipe-evolution --campaign-id g2 --dry-run
slm experiments reproduce-baseline --train-dir ... --test-dir ... --seeds 3
```

(`slm experiments <action>` ≡ `python -m scripts.run_*` — `slm list` shows the
mapping.)

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

- Shared duties: [contracts.md](contracts.md). Doc homes:
  `quality-experiment-matrix.md`, `grammar-experiment-matrix.md`,
  `perf-experiment-matrix.md`.
- Checks: `pytest -q tests/test_harnesses/experiments`.
