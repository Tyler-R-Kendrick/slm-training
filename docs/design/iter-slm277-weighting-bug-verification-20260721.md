# SLM-277 verification: no reproduction on `main` (fleet duplication, not a live defect)

Run id: `iter_slm277_weighting_bug_verification`
Status: **verification_no_reproduction** (no code change made)
Date: 2026-07-21

## What this is

[SLM-277](https://linear.app/quickdeploy-ai/issue/SLM-277) was filed as a
fast-follow finding from [PR #660](https://github.com/Tyler-R-Kendrick/slm-training/pull/660)
(SLM-242 / RSC-A06)'s audit: `TwoTowerModel.training_loss`'s deep-supervision
block sums each per-depth weight `w` into a zero-guard total but never
multiplies it into the per-depth loss term, so every supervised depth
contributes equally regardless of its configured ratio.

That finding is accurate for the code PR #660 audited. It does **not**
reproduce on `main` (commit `91edcae`). This doc records why, so the fix
suggested by SLM-277 is not redundantly re-implemented.

## Why it doesn't reproduce

`main` already carries this exact fix, landed independently via
`ada0081` ("E626-E631 decode-margin lineage + SLM-237/238/239
recursive-denoiser work", PR #625) — see
[`iter-rsc-a01-depth-supervision-weighting-fix-20260721.md`](iter-rsc-a01-depth-supervision-weighting-fix-20260721.md)
for that fix's own record. `training_loss` now computes
`weighted_contribution[d] = norm_w[d] * raw_depth_loss[d]` and sums those
(`src/slm_training/models/twotower.py`, the `SLM-237 (RSC-A01)` block).

PR #660 is stacked on `claude/great-dirac-0wieip` @ `eb60c600`, a branch that
forked **before** `ada0081` landed on `main`. Confirmed directly:

```
git merge-base --is-ancestor ada0081 origin/claude/great-dirac-0wieip
# -> not an ancestor (exit 1)

git show origin/claude/great-dirac-0wieip:src/slm_training/models/twotower.py \
  | grep 'total_w = sum'
# -> total_w = sum(ds_weights[:usable])   (the historical, w-unused formula)
```

So PR #660's audit was correct for the snapshot it ran against, but that
snapshot predates the independent fix that already reached `main` through a
different branch in the same agent fleet. SLM-277 is a valid historical
finding, stale as a live-`main` defect.

## Evidence gathered against `main`

```
python -m pytest -q tests/test_models/test_recursive_denoiser.py
# 108 passed, 1 skipped
```

The pinned regression `test_fixture_metrics_agree_with_manual_calculation`
specifically asserts the recorded loss matches the *weighted* formula and
explicitly rejects the *unweighted* one. Re-running the SLM-138 fixture
(`scripts.run_slm138_recursive_denoiser_fixture._run_fixture()`) directly
confirms the same, with fresh numbers (this run's RNG state differs from the
`iter-rsc-a01` doc's, so the raw losses differ — the formula match is what
matters):

| quantity | value |
|---|---|
| `recursive_depth_loss_0` | 35.52149200439453 |
| `recursive_depth_loss_1` | 32.49333572387695 |
| weights | `(0.5, 1.0)` |
| weighted formula `(0.5*L0 + 1.0*L1) / 1.5` | 33.50272115071615 |
| unweighted (defective) formula `(L0 + L1) / 1.5` | 45.34321848551432 |
| `recursive_depth_supervision_loss` (recorded) | 33.502723693847656 |

The recorded value matches the weighted formula (`rel diff < 1e-5`) and
differs from the defective formula by `~11.84` — conclusive.

## Recommendation (recorded here, acted on via Linear/GitHub, not code)

- **SLM-277**: comment with this evidence; recommend closing as
  already-resolved-by-different-lineage rather than re-implementing the fix.
- **PR #660**: its own body states the bug was "deliberately left unfixed" —
  that framing is now stale relative to `main`. Its numeric/schedule
  validation contribution (`harness_core/schedule_validation.py` +
  `twotower_schedule_policy.py`) is architecturally different from what
  landed on `main` for the same ticket (`twotower_numeric_gates.py`, commit
  `91edcae`) — reconciling those two independent SLM-242 implementations is
  an owner/human call, not automated here.
- **Fleet coordination**: this is a second observed instance (after the
  SLM-242 double-implementation) of independently-forked `claude/great-dirac-*`
  branches re-solving the same ticket without visibility into each other's
  merges. No branch other than this run's own
  (`claude/great-dirac-196url`) was touched.

## Non-goals honored

- No code change on `main` (nothing to fix there).
- No re-run of SLM-233's matched recursive-depth campaign (contingent on a
  real fix landing, which already happened via SLM-237; re-running it is
  SLM-277's own suggested next step once a fix is confirmed, and is a
  separate, larger unit of work).
- No unilateral close/merge action on SLM-277 or PR #660 — recorded as a
  recommendation for the human owner and reflected via a Linear comment only.
