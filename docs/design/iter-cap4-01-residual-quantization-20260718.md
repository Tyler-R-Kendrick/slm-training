# CAP4-01 residual ternary planes — fixture results

**Experiment:** `CAP4-01 residual ternary planes fixture`  
**Artifact:** [`iter-cap4-01-residual-quantization-20260718.json`](iter-cap4-01-residual-quantization-20260718.json)  
**Run date:** 2026-07-18  
**Claim:** Wiring evidence only. No ship gate, no checkpoint promotion.

## What changed

- Added `ResidualTritStack` in `src/slm_training/models/quantization/residual_planes.py`:
  - base `nn.Linear` plus `R` ternary-quantized residual planes,
  - scale modes `geometric_balanced`, `learned_independent`, `learned_monotone`,
  - normalizations `none`, `rms`, `variance_preserving`,
  - sequential plane-fitting helper `fit_planes_sequential`,
  - per-plane diagnostics (`PlaneOutput`) including physical byte cost.
- Extended `build_model_ledger` to cost `*.planes.*.weight` tensors with the
  supplied `residual_plane_format`.
- Added `ResidualTritPlaneHead` to `src/slm_training/models/local_action_head.py`:
  - base action-embedding score + `ResidualTritStack` refinement,
  - preserves the forced-decision shortcut for single-legal-action states.
- Added regression tests in `tests/test_models/test_residual_trit_planes.py`.
- Added fixture script `scripts/run_residual_trit_fixture.py`.

## Recipe

| Field | Value |
| --- | --- |
| Device | CPU |
| Hidden dim | 16 |
| Actions | 8 |
| Train records | 256 |
| Test records | 64 |
| Residual planes `R` | 2 |
| Scale mode | `geometric_balanced` |
| Base training steps | 60 |
| Plane fitting steps | 40 |

## Fixture results

| Variant | Test MSE | Estimated bytes |
| --- | --- | --- |
| FP16 baseline | 2.74 | 432 |
| Ternary direct | 3.90 | 210 |
| Residual trit (base + 2 planes) | 3.41 | 632 |

The residual-trit variant sits between the FP16 baseline and the ternary-direct
scorer on MSE, while the plane weights are costed at ~2 bits per weight. This is
expected behavior for a toy synthetic teacher; it is not a production fidelity
claim.

## Verification

- `py_compile` passes for `residual_planes.py`, `local_action_head.py`, and the
  fixture script.
- `pytest tests/test_models/test_residual_trit_planes.py` passes (13/13).
- `pytest tests/test_models/test_quantization.py tests/test_models/test_local_action_head.py` passes.
- `python -m scripts.verify_version_stamps --check` passes.
- `python -m scripts.repo_policy` passes.
- `.githooks/check-changed` passes.
- `git diff --check` passes.

## Honest caveats

- This is a **wiring fixture** on synthetic data; it does not exercise the
  full OpenUI grammar, `rico_held`, or any ship gate.
- Byte costs are analytical packing estimates, not measured on-device.
- `ResidualTritPlaneHead` uses hash-based action indexing; a real production
  head needs a stable action-code registry to avoid collisions.
- `fit_planes_sequential` is a reference greedy fitting routine, not a full
  end-to-end training loop.
- No checkpoint was written or promoted, so `docs/MODEL_CARD.md` is unchanged.
