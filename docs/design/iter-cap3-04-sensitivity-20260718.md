# CAP3-04: Grammar-conditioned quantization sensitivity and mixed-precision allocation (2026-07-18)

**Linear issue:** SLM-93
**Branch:** `agent/slm-93-cap3-04-quant-sensitivity`
**Date:** 2026-07-18
**Status:** wiring evidence / fixture dry-run

Evidence: [iter-cap3-04-sensitivity-20260718.json](iter-cap3-04-sensitivity-20260718.json).
Harness: [`src/slm_training/harnesses/quantization/sensitivity.py`](../../src/slm_training/harnesses/quantization/sensitivity.py),
allocator: [`src/slm_training/harnesses/quantization/allocation.py`](../../src/slm_training/harnesses/quantization/allocation.py).
Runners: [`scripts/profile_quant_sensitivity.py`](../../scripts/profile_quant_sensitivity.py),
[`scripts/allocate_mixed_precision.py`](../../scripts/allocate_mixed_precision.py).

## What changed

Added the CAP3-04 sensitivity + allocation harness:

- `src/slm_training/harnesses/quantization/sensitivity.py`
  - `ParameterGroup` / `GroupingPolicy` for stable, versioned parameter grouping.
  - `GroupFormatPoint` / `SensitivityReport` for versioned evidence.
  - `profile_group_sensitivity` quantizes one group at a time, measures
    local-head task perturbation (flip rate, KL, margin, mean/CVaR regret),
    records physical bytes from the ledger, and restores the baseline.
  - `compute_gradient_proxy` diagnostic squared-gradient proxy per group.
- `src/slm_training/harnesses/quantization/allocation.py`
  - Deterministic multiple-choice knapsack DP over `(group, format)` points.
  - Supports hard byte budget, tail-loss cap, and exclusions.
  - Emits uniform, random, and hand-hybrid baselines in `AllocationManifest`.
- `scripts/profile_quant_sensitivity.py` and `scripts/allocate_mixed_precision.py`
  CLI entry points with dry-run + fixture-run modes.
- `tests/test_harnesses/quantization/test_sensitivity.py` and
  `tests/test_harnesses/quantization/test_allocation.py` covering group
  matching, baseline restoration, excluded groups, budget/tail/exclusion
  constraints, and deterministic baselines.

## Fixture sensitivity matrix

Command:

```bash
python -m scripts.profile_quant_sensitivity \
  --checkpoint toy \
  --synthetic-traces 256 \
  --samples 64 \
  --formats ternary,learned4zero,int4,int8 \
  --out-dir outputs/runs/cap3-04-sensitivity
```

Recipe: CPU; toy `LocalFlatHead`; 256 synthetic `grammar_decision` traces;
`group_size=128`; `hybrid_coverage_margin` sampling; 64 calibration samples.

| group_id | format_id | packed_bytes | total_bytes | flip_rate | KL | margin | mean_regret | cvar90 |
| -------- | --------- | ------------ | ----------- | --------- | -- | ------ | ----------- | ------ |
| local_head/scorer | ternary | 32 | 68 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local_head/scorer | learned4zero | 32 | 68 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local_head/scorer | int4 | 64 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local_head/scorer | int8 | 128 | 164 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local_head/embeddings | ternary | 192 | 396 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local_head/embeddings | learned4zero | 192 | 396 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local_head/embeddings | int4 | 384 | 588 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local_head/embeddings | int8 | 768 | 972 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Other default groups (semantic_input, context_encoder, denoiser, attention, mlp,
latent_projection, norms_and_biases) are either unmatched in the toy model or
explicitly excluded and are recorded with a reason.

## Fixture allocation

Command:

```bash
python -m scripts.allocate_mixed_precision \
  --sensitivity-report outputs/runs/cap3-04-sensitivity/sensitivity_report_cap3-04-20260718223655.json \
  --byte-budget 2048 \
  --out outputs/runs/cap3-04-allocation/allocation.json
```

| group_id | format_id | bytes | mean_regret |
| -------- | --------- | ----- | ----------- |
| local_head/embeddings | int4 | 588 | 0.0000 |
| local_head/scorer | ternary | 68 | 0.0000 |

- Solver status: `optimal`
- Total bytes: 656 / 2048
- Baseline uniform: 688 bytes, cost 0.0
- Baseline random: 1136 bytes, cost 0.0
- Baseline hand_hybrid: 560 bytes, cost 0.21875

## Honest caveats

- CPU/toy fixture wiring only.
- Fisher/Jacobian proxies are diagnostic stubs; no claim they predict direct
  perturbation at scale.
- No measured latency; latency-aware allocation uses modeled kernel registry
  flags and is labeled accordingly.
- The toy `LocalFlatHead` has only scorer + embedding groups, so most default
  policy groups are unmatched; a real `TwoTower` would populate them.
- Ship-grade profiling needs GPU + local E224+ checkpoints + full `--ship-gates`.

## Verification checklist

- [x] `pytest tests/test_harnesses/quantization/test_sensitivity.py tests/test_harnesses/quantization/test_allocation.py -q` — 12 passed.
- [x] `python -m scripts.profile_quant_sensitivity ... --dry-run` — describes matrix.
- [x] `python -m scripts.profile_quant_sensitivity ...` — 8/8 OK points for matched groups.
- [x] `python -m scripts.allocate_mixed_precision ...` — optimal allocation under budget.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — passed.
- [x] `git diff --check` — passed.
