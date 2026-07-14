---
name: running-experiment-matrices
description: Use when running, extending, or interpreting quality (E*), grammar (X*), perf (P/Q/R), phase, scaling, or mixture experiment matrices
---

# Running experiment matrices

## Overview

Matrices are the intended way to land levers: isolatable IDs, shared gate
policy, JSON + markdown scoreboards. Implement a lever, run the matching
matrix subset, then document results.

**REQUIRED AFTERWARD:** `documenting-experiment-results`.
**REQUIRED FOR SHIP CLAIMS:** `honest-ship-eval`.

## Matrix index

| Matrix | Script | Spec | Results JSON |
| --- | --- | --- | --- |
| Quality E0–E75 (Vn sets) | `python -m scripts.run_quality_matrix` | `docs/design/quality-experiment-matrix.md` | `docs/design/quality-matrix-results.json` |
| Grammar X0–X8 | `python -m scripts.run_grammar_matrix` | same (X section) | `docs/design/grammar-matrix-results.json` |
| Perf P/Q/R/PG | `python -m scripts.run_perf_matrix` | `docs/design/perf-experiment-matrix.md` | `docs/design/perf-matrix-results.json` |
| Phase A/B/C | `python -m scripts.run_phase_pipeline` | quality matrix notes | `docs/design/phase-abc-results.json` |
| Baseline seeds | `python -m scripts.reproduce_baseline` | quality matrix | `docs/design/baseline-reproduction-results.json` |

Design contracts for levers live beside research docs (`research-lineage.md`,
`speculative-denoising.md`, `dsl-native-tokenizer.md`, …).

## How to run (typical)

```bash
# Quality subset (prefer modern matrix set for ship claims)
python -m scripts.run_quality_matrix --matrix v6 --only E53 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context

# Grammar X
python -m scripts.run_grammar_matrix --only X0,X2,X7 --steps 80

# Perf decode matrix (quality guardrails vs P0)
python -m scripts.run_perf_matrix --only P0,Q9,R9,PG --limit 4
```

Use `--only` for focused work; full matrices are expensive. Keep
`--rico-limit` explicit in docs when below full 1500.

## Extending a matrix

1. Add the lever in code + config/factory flags.
2. Register a stable experiment ID and run id in the matrix script **and** the
   markdown table in the design doc.
3. Run the new ID (plus a baseline control when comparing).
4. Update JSON + measured-results markdown.
5. Link research tags (Implemented / Adapted / …) in `research-lineage.md`
   when the lever maps to a paper.

Do not add matrix rows without a runnable script path.

## Interpretation rules

- Compare only against runs that share honesty mode and suite sizes.
- Perf optimizations fail if parse/fidelity drop >5 points abs vs P0.
- Vacuous guardrails (broken OpenUI bridge zeroing parse) are hard errors —
  fix the bridge, do not accept empty scoreboards.
- Historical curriculum / gold-leak runs stay labeled invalid for selection.

## Completion checklist

- [ ] Intended IDs ran (or blockers recorded)
- [ ] Results JSON under `docs/design/` current
- [ ] Measured-results markdown updated
- [ ] Ship/perf pass-fail stated with caveats
- [ ] No silent gold channels on honest rows
