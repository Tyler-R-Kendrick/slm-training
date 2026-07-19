# CAP4-02 compiler-floor + runtime-signal adaptive-plane routing — fixture results

**Experiment:** `CAP4-02 adaptive residual-plane routing fixture`  
**Artifact:** [`iter-cap4-02-adaptive-plane-routing-20260718.json`](iter-cap4-02-adaptive-plane-routing-20260718.json)  
**Run date:** 2026-07-18  
**Claim:** Wiring evidence only. No ship gate, no checkpoint promotion.

## What changed

- Added `PlaneScheduleSpec`, `PlaneScheduler`, `AdaptivePlaneRoutingContext`,
  `PlaneRouter`, and `oracle_min_planes` in
  `src/slm_training/models/quantization/adaptive_planes.py`.
- Implemented all eight required schedule modes:
  `uniform_1`, `uniform_max`, `structural_floor`, `floor_plus_entropy`,
  `floor_plus_margin`, `floor_plus_sensitivity`, `floor_plus_learned_router`,
  and offline `oracle_min_planes`.
- Structural floors:
  - `local_action_code`: `ceil(log_3 branch_count)` trits, with forced states
    using zero model planes.
  - `completion_support`: derived from `StateContext.completion_support_size`
    (CAP1-04 posterior effective support).
  - `margin_preservation`: uses the same base-3 floor for geometric planes;
    learned independent scales rely on empirical envelopes.
- Runtime signals use posterior entropy, top-two margin, sensitivity mean, or a
  small learned MLP whose features never include the target action or future
  verification result.
- Extended `ResidualTritPlaneHead.score` with optional `max_planes` and
  `return_diagnostics` so the router can stop at an arbitrary plane prefix.
- Added `StateContext.sensitivity` and `StateContext.completion_support_size`
  for schedule inputs.
- Added regression tests in
  `tests/test_models/test_adaptive_plane_routing.py`.
- Added fixture script `scripts/run_adaptive_plane_fixture.py`.

## Recipe

| Field | Value |
| --- | --- |
| Device | CPU |
| Hidden dim | 16 |
| Max actions | 32 |
| Decision records | 128 |
| Residual planes `R` | 4 |
| Scale mode | `geometric_balanced` |
| Grouping policies | `whole_batch`, `compact` |

## Fixture results

Average planes per decision by schedule (baseline = `uniform_max`):

| Schedule | Avg planes |
| --- | --- |
| `uniform_1` | 0.93 |
| `uniform_max` | 3.72 |
| `structural_floor` | 1.74 |
| `floor_plus_entropy` | 3.14 |
| `floor_plus_margin` | 1.74 |
| `floor_plus_sensitivity` | 2.71 |
| `floor_plus_learned_router` | 1.74 |
| `oracle_min_planes` | 3.72 |

`structural_floor`, `floor_plus_margin`, and `floor_plus_learned_router` all
stop near the local-action code floor on this random head because the preset
thresholds rarely trigger an additional plane.  The router is untrained; its
behavior is present for plumbing only.  These numbers are algorithmic plane
budgets, not wall-clock or energy measurements.

## Verification

- `py_compile` passes for `adaptive_planes.py`, `local_action_head.py`,
  `quantization/__init__.py`, the fixture script, and the regression tests.
- `pytest tests/test_models/test_adaptive_plane_routing.py` passes (19/19).
- `pytest tests/test_models/` passes (335/335).
- `.githooks/check-changed` passes.
- `python -m scripts.repo_policy` passes.
- `python -m scripts.verify_version_stamps --check` passes; bumped
  `model.quantization` from `v1` to `v2`.
- `git diff --check` passes.

## Honest caveats

- This is a **wiring fixture** on synthetic legal-action traces with a randomly
  initialized head; it does not exercise the full OpenUI grammar, `rico_held`,
  or any ship gate.
- Plane counts are algorithmic savings.  The reference implementation scores
  each prefix by re-running the residual stack; a real system needs an
  incremental packed-plane kernel to convert plane savings into wall-clock
  speed-up.
- Energy, wall-clock latency, and bytes-read numbers are not measured
  on-device.
- The learned router is not trained; threshold schedules use preset constants.
- `oracle_min_planes` is an offline diagnostic and is not deployable.
- No checkpoint was written or promoted, so `docs/MODEL_CARD.md` is unchanged.
