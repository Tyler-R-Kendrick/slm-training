# CAP3-03: Equal-storage ternary falsification matrix (2026-07-18)

**Linear issue:** SLM-92  
**Branch:** `agent/slm-92-cap3-03-ternary-falsification`  
**Date:** 2026-07-18  
**Status:** wiring evidence / fixture dry-run

Evidence: [iter-cap3-03-ternary-falsification-20260718.json](iter-cap3-03-ternary-falsification-20260718.json).  
Harness: [`src/slm_training/harnesses/experiments/cap3_03_ternary_falsification.py`](../../src/slm_training/harnesses/experiments/cap3_03_ternary_falsification.py),  
runner: [`scripts/run_cap3_03_ternary_falsification.py`](../../scripts/run_cap3_03_ternary_falsification.py).

## What changed

Added a standalone falsification matrix harness that compares low-bit weight
representations for the local action scorer at matched physical storage:

- `src/slm_training/harnesses/experiments/cap3_03_ternary_falsification.py`
  - `ArmConfig`, `ArmResult`, `MatrixReport`, `MatchedConditions`
  - `make_format` maps format ids to existing `QuantFormat` factories.
  - `evaluate_arm` runs PTQ (plus optional short QAT reconstruction) on a
    `LocalFlatHead`, records teacher-top1 accuracy, action flip rate, KL,
    margin preservation, regret distribution, zero/support rates, symbol
    entropy, and a physical-cost ledger hash.
  - `build_arms` / `run_matrix` create one arm per (format, seed) and assert
    `MatchedConditions` parity across arms so storage and protocol are fair.
  - Reuses SLM-90 `QuantFormat` / `build_model_ledger` and SLM-91
    `calibrate_scales_ptq` / `qat_reconstruct_local_scorer`.
- `scripts/run_cap3_03_ternary_falsification.py`
  - CLI with `--checkpoint`, `--trace-dir`, `--synthetic-traces`, `--formats`,
    `--group-size`, `--samples`, `--seeds`, `--qat-steps`, `--lr`, `--hidden-dim`,
    `--strategy`, `--out-dir`, `--dry-run`.
- `tests/test_harnesses/experiments/test_cap3_03_ternary_falsification.py`
  - Format factory coverage, `make_format`, `build_arms`, `MatchedConditions`
    parity, `evaluate_arm`, `run_matrix` versioned report, mismatched physical-bit
    error handling, and JSON serialization smoke test.

## Fixture matrix

Command:

```bash
python -m scripts.run_cap3_03_ternary_falsification \
  --checkpoint toy \
  --synthetic-traces 256 \
  --samples 64 \
  --formats ternary,learned4zero \
  --out-dir outputs/runs/cap3-03-toy
```

Recipe: CPU; toy `LocalFlatHead`; 256 synthetic `grammar_decision` traces;
`group_size=128`; `hybrid_coverage_margin` sampling; 64 calibration samples;
no QAT.

| arm_id | top1_acc | teacher_top1_acc | flip_rate | KL | margin | mean_regret | cvar90 | zero_rate | support_rate | entropy_bits | physical_bytes | total_bytes |
| ------ | -------- | ---------------- | --------- | -- | ------ | ----------- | ------ | --------- | ------------ | ------------ | -------------- | ----------- |
| ternary_gs128_s0 | 0.5938 | 0.7969 | 0.2031 | 0.0031 | 0.0089 | 0.2031 | 1.0 | 0.8156 | 0.1844 | -0.0000 | 32 | 140 |
| learned4zero_gs128_s0 | 0.5625 | 0.4844 | 0.5156 | 0.0072 | 0.1031 | 0.5156 | 1.0 | 1.0000 | 0.0000 | -0.0000 | 32 | 140 |

## Matched-conditions parity

- Both arms share `checkpoint_id=toy`, `group_size=128`,
  `physical_slot_bits=2`, `sample_count=64`,
  `sampling_strategy=hybrid_coverage_margin`, and the same calibration manifest.
- Physical weight bytes are equal (32 B) and total bytes are equal (140 B),
  confirming the equal-storage constraint is enforced by the ledger.

## Hard gates

- Arms passing matched-conditions parity and evaluation: 2/2
- Mismatched physical-bit arms are flagged as `error` without terminating the
  matrix.
- **PASS** the equal-storage falsification harness is wired end-to-end.

## Honest caveats

- This is a CPU/toy fixture run.  It proves wiring, matched conditions, ledger
  parity, and metric collection only.
- No `--ship-gates`, meaningful-parse metric, or held-out eval was run.
- The toy `LocalFlatHead` has tiny random embeddings; numeric differences between
  formats are not predictive of production model quality.
- No checkpoint was written to the HF bucket.
- Ship-grade falsification requires GPU + local E224+ checkpoints + full
  `--ship-gates` evaluation.

## Verification checklist

- [x] `pytest tests/test_harnesses/experiments/test_cap3_03_ternary_falsification.py` — 9 passed.
- [x] `python -m scripts.run_cap3_03_ternary_falsification ... --dry-run` — describes matrix.
- [x] `python -m scripts.run_cap3_03_ternary_falsification ...` — 2/2 arms OK.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — passed.
- [x] `git diff --check` — passed.
