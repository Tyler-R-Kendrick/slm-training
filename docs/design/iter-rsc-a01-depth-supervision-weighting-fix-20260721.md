# RSC-A01: repair recursive deep-supervision weighting and fail closed (SLM-237)

Run id: `iter_rsc_a01_depth_supervision_weighting_fix`
Status: **correctness_fix** (no quality claim made from this patch)
Date: 2026-07-21

## What this is

`TwoTowerModel.training_loss`'s `recursive_depth_supervision_weights` auxiliary
loss had a live mathematical defect: the loop bound each per-depth weight
(`w`) from `enumerate(ds_weights[:usable])` but never multiplied `d_loss` by
it. It computed

```
L_aux_current = sum_d(L_d) / sum_d(w_d)          # unweighted mean
```

instead of the intended

```
L_aux_intended = sum_d(w_d * L_d) / sum_d(w_d)   # weighted mean
```

This is a pure correctness fix per SLM-237's own acceptance criteria — **no
OOD quality comparison or checkpoint-promotion claim is made here**. RSC-A02
(whether the final recursion depth belongs in the auxiliary term at all) is
explicitly out of scope for this issue.

## Checked-fixture proof of the historical defect

The already-committed `docs/design/iter-slm138-recursive-denoiser-20260720.md`
recorded, for `recursive_depth_supervision_weights=(0.5, 1.0)`:

- `recursive_depth_loss_0 = 26.03812026977539`
- `recursive_depth_loss_1 = 23.953886032104492`
- `recursive_depth_supervision_loss = 33.3280029296875`

`(L0 + L1) / 1.5 = 33.328002...` — an exact match to the recorded value, and
to the *defective* (unweighted-mean) formula, not the intended weighted one
(which would be `(0.5*L0 + 1.0*L1) / 1.5 = 24.65...`). Re-running the same
fixture (`scripts/run_slm138_recursive_denoiser_fixture.py --mode fixture`)
against this patch now yields the corrected weighted value instead (see
`tests/test_models/test_recursive_denoiser.py::test_fixture_metrics_agree_with_manual_calculation`).

## Six historical failure modes: before (real pre-patch code) vs after (this patch)

Each row below was reproduced against the **actual** production code, not a
hand re-implementation: the "before" column was captured with
`git stash` reverting the three changed files to the tip-of-branch (pre-patch)
state and running `TwoTowerModel.training_loss` directly; the "after" column
is the current working tree. Full data (weights, raw L0/L1, computed
aux values, exception text) is in the sibling JSON's `six_failure_modes` key.

| # | Failure mode | Before (pre-patch) | After (this patch) |
|---|---|---|---|
| 1 | `(0, 1)` — zero should disable a depth | Included L0 anyway: `aux=61.2788` ≠ `L1=29.8113` | `aux == L1` exactly (29.8113) |
| 2 | `(0.5, 1)` vs `(1, 2)` — should be the same normalized mean | Exact 2x scale difference: `40.8526` vs `20.4263` (ratio 2.0) | Identical: both `30.3634` |
| 3 | `(0, 0)` — should be rejected, not silently disabled | Silently skipped: no `recursive_depth_supervision_loss` / `recursive_depth_loss_0` telemetry at all, loss still finite | Raises `ValueError` before any loss/backward: *"...are all zero...use an empty tuple () to explicitly turn the feature off instead."* |
| 4 | Negative weight `(-1, 3)` — should be rejected | Accepted unchecked; historical formula ignores the sign entirely (only the denominator sum changes: `-1+3=2`, `aux=(L0+L1)/2=30.6394`, masquerading as symmetric `(1,1)`) | Raises `ValueError`: *"...is negative...would anti-supervise that depth..."* |
| 5 | Length mismatch `(1.0,)` vs 2 depths — should be rejected | Silently truncated via `min(len(depth_logits), len(ds_weights))`: only depth 0's loss logged, depth 1 dropped without error | Raises `ValueError`: *"...has length 1...but the denoiser produced 2 recursion depth(s)...no truncation or padding..."* |
| 6 | Weights `(0.5, 1.0)` on a `stacked` denoiser (no `recursive_outputs`) — should be rejected | Silently ignored: plain `denoiser(...)` forward ran, no aux term, no error | Raises `ValueError`: *"...does not expose recursive_outputs...leave the tuple empty () to keep the feature off..."* |

Backward compatibility: the empty tuple `()` remains valid and feature-off on
every architecture (`stacked`, `shared_recursive`, HF denoisers, and old
checkpoints) both before and after this patch — confirmed via
`test_empty_tuple_valid_on_every_architecture_no_aux_term`.

## Implementation

- `ValidatedDepthSupervision` + `validate_recursive_depth_supervision(*, weights, num_depths, supports_recursive_outputs, architecture=...)`
  in `src/slm_training/models/twotower.py` — the single canonical validator,
  called unconditionally in `training_loss` right after the denoiser forward
  pass (before the `predict_mask.any()` branch), so an invalid config fails
  before a training step regardless of mask contents.
- Corrected objective: `raw_depth_loss[d]`, `weighted_depth_contribution[d] =
  norm_w[d] * raw_depth_loss[d]` (where `norm_w[d] = w[d] / sum(w)`), and
  `recursive_depth_supervision_loss = sum(weighted_depth_contribution)`.
- Telemetry added/versioned: `recursive_depth_loss_{d}`,
  `recursive_depth_weight_{d}`, `recursive_depth_weighted_contribution_{d}`,
  `recursive_depth_supervision_weight_sum`, `recursive_depth_supervision_loss`,
  `recursive_depth_supervision_enabled` (now always set, so a disabled run is
  distinguishable from a missing-metrics bug).
- `scripts/run_slm138_recursive_denoiser_fixture.py` had its own latent
  instance of failure mode #6: it applied
  `recursive_depth_supervision_weights=(0.5, 1.0)` to the `stacked` arm too,
  which the old code silently ignored. The new fail-closed validator
  correctly rejects that combination, so the fixture now only sets the
  weights for the `shared_recursive` arm.

## Tests

`tests/test_models/test_recursive_denoiser.py` — 26 passed (9 pre-existing +
17 new), covering all 11 required properties from SLM-237 (weighted-mean
identities, validator rejections for negative/NaN/inf/all-zero/length
mismatch/unsupported architecture, empty-tuple backward compatibility,
gradient isolation to positive-weight depths via independently
differentiable synthetic tensors, and fixture/manual-calculation agreement).

```
python -m pytest tests/test_models/test_recursive_denoiser.py -q
# 26 passed
```

## Version bumps

- `model.twotower`: v62 -> v63 (owns `training_loss` and the new validator).
- `model.recursive_denoiser`: v1 -> v2 (owns the fixture script and this test
  file, both edited).

## SLM-277 (RSC-A06 audit) follow-up

The same defect was independently re-flagged as
[SLM-277](https://linear.app/quickdeploy-ai/issue/SLM-277) during the RSC-A06
numeric-schedule audit. That issue's required scope is already covered by this
correction:

- The intended semantics are the normalized weighted mean implemented above.
- The loss term is fixed in `src/slm_training/models/twotower.py`.
- The obsolete `schedule-guard: allow UNUSED_LOOP_WEIGHT ...` suppression
  comment is not present in the corrected code.
- `validate_recursive_depth_supervision` rejects the six historical failure
  modes and is exercised by the tests below.

The remaining dependent work — re-running the matched recursive-depth campaign
under the corrected weighting — is tracked by
[SLM-233](https://linear.app/quickdeploy-ai/issue/SLM-233) (RSC2-01), which
this issue unblocks.

## Non-goals honored

- No choice of whether the final depth belongs in the auxiliary term
  (RSC-A02).
- No recursive architecture redesign.
- No quality campaign or checkpoint promotion.
- No weakening of any repository run/evidence gate.
