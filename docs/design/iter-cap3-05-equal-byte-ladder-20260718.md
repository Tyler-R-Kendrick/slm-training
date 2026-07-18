# CAP3-05: Width × precision ladder at equal deployed bytes (SLM-94)

**Linear issue:** SLM-94
**Branch:** `agent/slm-94-cap3-05-equal-byte-ladder`
**Date:** 2026-07-18
**Status:** partial planner wiring / plan-only dry-run; SLM-94 acceptance incomplete

Evidence: [iter-cap3-05-equal-byte-ladder-20260718.json](iter-cap3-05-equal-byte-ladder-20260718.json).
Harness: [`src/slm_training/harnesses/experiments/ladder.py`](../../src/slm_training/harnesses/experiments/ladder.py),
runner: [`scripts/run_scaling_ladder.py`](../../scripts/run_scaling_ladder.py).
Tests: [`tests/test_harnesses/experiments/test_equal_byte_ladder.py`](../../tests/test_harnesses/experiments/test_equal_byte_ladder.py).

## What changed

Extended the canonical scaling ladder with CAP3-05 equal-byte / precision metadata:

- `src/slm_training/harnesses/experiments/ladder.py`
  - Added optional byte/precision, matching-regime, and training-status metadata to `LadderPoint` in a backward-compatible way.
  - `point_id` now appends `_b{byte_budget}` and `_p{precision_format}` only when those fields are set; legacy IDs are unchanged.
  - Added `_SyntheticTwoTower` module and `estimate_bytes()` so the planner can call the existing CAP3-01 `build_model_ledger` without requiring a real checkpoint or tokenizer.
  - Added `plan_equal_byte_ladder()` that searches a width/depth grid and picks the candidate closest to each target byte budget within tolerance; infeasible formats are still emitted with a reason.
  - Fixed selected arms to retain their own width-derived token budget instead of the final grid candidate's budget.
- `src/slm_training/models/quantization/cost.py`
  - Corrected double counting of excluded tensors and per-tensor metadata; `actual_bytes` now uses checkpoint/static bytes including alignment.
- `src/slm_training/harnesses/model_build/config.py`
  - Added `byte_budget`; reference-only low-bit points now fail closed instead of being mislabeled as native/QAT training.
- `scripts/run_scaling_ladder.py`
  - Added `--family equal-byte-precision`.
  - Added `--byte-budgets`, `--formats`, `--plan-only`, `--tolerance`, and `--group-size` arguments.
  - The family requires `--plan-only` until native/QAT training exists and writes a version-stamped manifest plus an optional canonical `--docs-out` mirror.
- `tests/test_harnesses/experiments/test_equal_byte_ladder.py`
  - Regression tests for byte estimation, feasible/infeasible selection, point-ID suffixes, config threading, and the plan-only CLI.

## Fixture plan-only dry-run

Command:

```bash
python -m scripts.run_scaling_ladder \
  --family equal-byte-precision \
  --byte-budgets 64KB,256KB,1MB \
  --precision-formats fp16,int8,int4,ternary \
  --widths 32,64,96,128,192 \
  --tolerance 0.05 \
  --plan-only \
  --out outputs/runs/cap3-05-equal-byte \
  --docs-out docs/design/iter-cap3-05-equal-byte-ladder-20260718.json
```

Recipe: CPU; synthetic TwoTower-like surrogate; scratch track; proportional depths; group size 128; tolerance ±5% for this wiring run.

### Plan manifest summary

| budget | format | d_model | context / denoiser | actual bytes | delta | status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 64 KB | fp16 | 32 | 1 / 2 | 68,048 | +3.8% | **feasible** |
| 64 KB | int8 | 32 | 1 / 2 | 37,556 | −42.7% | infeasible |
| 64 KB | int4 | 32 | 1 / 2 | 22,068 | −66.3% | infeasible |
| 64 KB | ternary | 64 | 1 / 2 | 43,800 | −33.2% | infeasible |
| 256 KB | fp16 | 64 | 1 / 2 | 257,808 | −1.7% | **feasible** |
| 256 KB | int8 | 64 | 1 / 2 | 136,344 | −48.0% | infeasible |
| 256 KB | int4 | 96 | 1 / 3 | 205,772 | −21.5% | infeasible |
| 256 KB | ternary | 96 | 1 / 3 | 115,724 | −55.9% | infeasible |
| 1 MB | fp16 | 96 | 1 / 3 | 740,432 | −29.4% | infeasible |
| 1 MB | int8 | 128 | 2 / 4 | 986,528 | −5.9% | infeasible |
| 1 MB | int4 | 128 | 2 / 4 | 519,072 | −50.5% | infeasible |
| 1 MB | ternary | 192 | 3 / 6 | 901,288 | −14.0% | infeasible |

Feasible points: **2 of 12** with this sparse width grid and ±5% tolerance, both FP16 controls. There is no matched cross-format pair in this grid, so it cannot support CAP-H5. The next campaign must densify the width/depth grid and implement native/QAT low-bit arms before training.

### Hypothesis / falsifier

- **CAP-H5:** At one or more byte budgets, a wider low-bit model will improve semantic/execution quality over the narrowest higher-precision model that fits.
- **Falsifier:** Quality remains precision-dominated and widening does not recover margins at equal bytes.

This dry-run does not train, so no hypothesis verdict is claimed. It proves deterministic selection and explicit infeasibility only.

## Honest caveats

- **Plan-only / wiring evidence.** No model was trained; no quality, parse, meaningful-program, or latency measurement is claimed.
- **Modeled bytes.** `actual_bytes` is modeled checkpoint/static storage from a synthetic TwoTower-like module, including per-tensor metadata and alignment. It is not serialized-size reconciliation or measured resident memory.
- **Reference-only low-bit arms.** INT8/INT4/ternary are labeled `post_training_reference`; the runner refuses to train them as native/QAT.
- **Sparse grid.** Only a coarse width sweep was used; a production campaign would widen the grid, sweep depths, and include mixed-precision allocations from CAP3-04.
- **No measured latency.** Equal-measured-latency and equal-training-compute regimes are not executed here.
- **No checkpoint.** `model_card_updated: false`; nothing is promoted or recorded in `docs/MODEL_CARD.md`.
- SLM-94 remains incomplete until the central trained matrix, serialized bytes, measured latency, AgentV evaluation, and quality-cost frontiers exist.

## Verification checklist

- [x] Experiment-family + quantization tests — 104 passed.
- [x] `python -m scripts.run_scaling_ladder --family equal-byte-precision ... --plan-only` — manifest written with 12 points.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `python -m scripts.check_changed --base-ref origin/main --changed-tests-only` — 8 passed.
- [ ] Broad `.githooks/check-changed` — interrupted at the 170-second hard cap; not evidence. Scoped affected suites above passed.
- [x] `git diff --check` — clean.
