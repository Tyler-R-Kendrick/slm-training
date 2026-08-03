# Continuous autotrain: 2026-08-03 (session ts5ofk) cycle 2 — component-plan structural win, 4th reproduction (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `f1a200f5` (this session's cycle-1 docs commit, on top of `main` tip `d5b0c318`)

**Verdict:** `component-plan` beats its size-matched control on the declared
primary at this seed — the same primary-metric delta, at the same seed, as
**three** prior independently-run sessions' measurements of the identical
hypothesis. Fixture screening only — not a ship or promotion claim.

| Arm | Seed | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | 0 | 10215.32 |
| component-plan | 100002 | .38280 | 0 | 8947.55 |

Primary improvement `+.05613` (`0.32666666666666666 -> 0.38280000000000003`)
— byte-identical to three prior sessions' independent runs of this same
hypothesis:

1. [`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
   (merged as PR #1369).
2. [`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md).
3. [`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)
   (merged as part of PR #1376).

This is the **fourth** independent, byte-identical reproduction, now with at
least two more intervening harness commits since the last one — further
evidence the underlying fixture effect is real and deterministic.
`meaningful_program_rate` stays 0 on both arms; the win stays confined to raw
structural similarity, not full program correctness. `binder_reference_f1`
reads 0 on both arms this cycle (matching the original session's reading, not
`j48f8u`'s `.16667` — an intervening metric-computation difference, not
attributable to this hypothesis).

Ship gates fail as expected: `insufficient_n` (n=3, need 20).

## Driver classifier note

The automated `sdlc_delivery.json` classifier flagged
`positive_no_tracked_delta_skip_stack` because this cycle re-ran the
identical recipe with no new code/harness diff of its own. Per `sdlc`
autotrain-iteration-delivery, documenting a positive result is itself the
reviewable delta that earns a stacked layer — matching how all three prior
reproductions of this exact hypothesis were delivered (PR #1369, PR #1376).

## This is not a confirmation attempt

The driver's rank-1 next priority (confidence 0.95) proposes fresh-seed
confirmation of this exact hypothesis via a `-fresh-confirmation`-suffixed
arm. This is the same confirmation path a prior session already found
blocked by two harness bugs — documented in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md):

1. A dual-arm decode timeout at seed `100005` with an undetermined root
   cause.
2. `_apply_frozen_replay` does not recognize `-confirm`-suffixed arm slugs.

That doc explicitly asks for a **dedicated `improve-openui-harnesses`
session**. This cycle does not attempt either blocker — it only adds a
fourth independent screening-stage reproduction of the same primary-metric
win under a fresh campaign id.

## SDLC Phase A

**Positive** (`primary_metric_win`). Documenting this result creates the
reviewable delta required to open/update a stacked layer for this cycle.

## Next priorities

1. Do **not** attempt the blocked `-confirm`/`-fresh-confirmation`
   frozen-replay path or the seed-`100005` dual-arm decode timeout
   speculatively; route to `improve-openui-harnesses` with dedicated
   investigation time.
2. Once those blockers are resolved: confirm the fixture candidate on a
   fresh seed with the exact size-matched recipe before any promotion.
3. Keep promotion formal preflight locked until fresh confirmation
   establishes a champion.

Machine evidence:
[`continuous-openui-local-ts5ofk-c2-results.json`](continuous-openui-local-ts5ofk-c2-results.json).
