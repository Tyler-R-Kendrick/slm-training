# CAP3-05: Width × precision ladder at equal deployed bytes (SLM-94)

**Linear issue:** SLM-94  
**Branch:** `agent/slm-94-cap3-05-equal-byte-ladder`  
**Date:** 2026-07-18  
**Status:** wiring evidence / plan-only dry-run

Evidence: [iter-cap3-05-equal-byte-ladder-20260718.json](iter-cap3-05-equal-byte-ladder-20260718.json).  
Harness: [`src/slm_training/harnesses/experiments/ladder.py`](../../src/slm_training/harnesses/experiments/ladder.py),  
runner: [`scripts/run_scaling_ladder.py`](../../scripts/run_scaling_ladder.py).  
Tests: [`tests/test_harnesses/experiments/test_equal_byte_ladder.py`](../../tests/test_harnesses/experiments/test_equal_byte_ladder.py).

## What changed

Extended the canonical scaling ladder with CAP3-05 equal-byte / precision metadata:

- `src/slm_training/harnesses/experiments/ladder.py`
  - Added optional `byte_budget`, `precision_format`, `actual_bytes`, `budget_delta`, and `status` fields to `LadderPoint` in a backward-compatible way.
  - `point_id` now appends `_b{byte_budget}` and `_p{precision_format}` only when those fields are set; legacy IDs are unchanged.
  - Added `_SyntheticTwoTower` module and `estimate_bytes()` so the planner can call the existing CAP3-01 `build_model_ledger` without requiring a real checkpoint or tokenizer.
  - Added `plan_equal_byte_ladder()` that searches a width/depth grid and picks the candidate closest to each target byte budget within tolerance; infeasible formats are still emitted with a reason.
- `src/slm_training/harnesses/model_build/config.py`
  - Added `byte_budget` field; `model_build_config_for_point` now threads `quant_format`, `byte_budget`, and `use_dynamic_quant` for equal-byte points.
- `scripts/run_scaling_ladder.py`
  - Added `--family equal-byte-precision`.
  - Added `--byte-budgets`, `--precision-formats`, `--plan-only`, `--tolerance`, and `--group-size` arguments.
  - `--plan-only` writes a versioned manifest (`equal_byte_plan_<timestamp>.json`) without training.
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
  --out outputs/runs/cap3-05-equal-byte
```

Recipe: CPU; synthetic TwoTower-like surrogate; scratch track; proportional depths; group size 128; tolerance ±5% for this wiring run.

### Plan manifest summary

| budget | format | d_model | context / denoiser | actual bytes | delta | status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 64 KB | fp16 | 32 | 1 / 2 | 72,240 | +10.2% | infeasible |
| 64 KB | int8 | 32 | 1 / 2 | 41,748 | −36.3% | infeasible |
| 64 KB | int4 | 32 | 1 / 2 | 26,260 | −59.9% | infeasible |
| 64 KB | ternary | 64 | 1 / 2 | 51,576 | −21.3% | infeasible |
| 256 KB | fp16 | 64 | 1 / 2 | 265,584 | +1.3% | **feasible** |
| 256 KB | int8 | 64 | 1 / 2 | 144,120 | −45.0% | infeasible |
| 256 KB | int4 | 96 | 1 / 3 | 219,980 | −16.1% | infeasible |
| 256 KB | ternary | 96 | 1 / 3 | 129,932 | −50.4% | infeasible |
| 1 MB | fp16 | 96 | 1 / 3 | 754,640 | −28.0% | infeasible |
| 1 MB | int8 | 128 | 2 / 4 | 1,012,704 | −3.4% | **feasible** |
| 1 MB | int4 | 128 | 2 / 4 | 545,248 | −48.0% | infeasible |
| 1 MB | ternary | 192 | 3 / 6 | 956,616 | −8.8% | infeasible |

Feasible points: **2 of 12** with this sparse width grid and ±5% tolerance. The two feasible points are the FP16 control at 256 KB (d=64) and the INT8 arm at 1 MB (d=128). The INT8 point uses a 33% wider `d_model` than the closest FP16 candidate at the same budget, which is exactly the capacity-redundancy question CAP3-05 is designed to answer once trained.

### Hypothesis / falsifier

- **CAP-H5:** At one or more byte budgets, a wider low-bit model will improve semantic/execution quality over the narrowest higher-precision model that fits.
- **Falsifier:** Quality remains precision-dominated and widening does not recover margins at equal bytes.

This dry-run does not train, so no hypothesis verdict is claimed. It only proves the planner can produce matched architecture points from whole-model physical bytes.

## Honest caveats

- **Plan-only / wiring evidence.** No model was trained; no quality, parse, meaningful-program, or latency measurement is claimed.
- **Modeled bytes.** `actual_bytes` comes from a synthetic TwoTower-like module run through the CAP3-01 ledger. It is not a measured resident size on a deployed device and does not include runtime activation/KV memory.
- **Sparse grid.** Only a coarse width sweep was used; a production campaign would widen the grid, sweep depths, and include mixed-precision allocations from CAP3-04.
- **No measured latency.** Equal-measured-latency and equal-training-compute regimes are not executed here.
- **No checkpoint.** `model_card_updated: false`; nothing is promoted or recorded in `docs/MODEL_CARD.md`.
- Ship-grade run requires GPU + local E224+ checkpoints + full `--ship-gates`.

## Verification checklist

- [x] `pytest tests/test_harnesses/experiments/test_equal_byte_ladder.py tests/test_harnesses/experiments/test_capacity_ladder.py tests/test_harnesses/experiments/test_ladder_promotion.py -q` — 17 passed.
- [x] `python -m scripts.run_scaling_ladder --family equal-byte-precision ... --plan-only` — manifest written with 12 points.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — passed.
- [x] `git diff --check` — clean.
