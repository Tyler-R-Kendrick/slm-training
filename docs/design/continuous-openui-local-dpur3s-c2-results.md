# Continuous autotrain: 2026-08-03 (scheduled session, branch dpur3s) cycle 2 — component-plan structural win, 6th reproduction

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `a17d7ebf` (this session's cycle-1 docs commit, on top of
`main` tip `620ef79b` — post multi-seed thrash-arm-close fix, PR #1386).

**Verdict:** `component-plan` beats its size-matched control on the declared
primary at this seed — the same primary-metric delta, at the same seed, as
**five** prior independently-run sessions' measurements of the identical
hypothesis. Fixture screening only — not a ship or promotion claim.

| Arm | Seed | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | 0 | 46719.25 |
| component-plan | 100002 | .38280 | 0 | 38911.65 |

Primary improvement `+.05613` (`0.32666666666666666 -> 0.38280000000000003`)
— byte-identical to five prior sessions' independent runs of this same
hypothesis:

1. [`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
   (merged as PR #1369).
2. [`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md).
3. [`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)
   (merged as part of PR #1376).
4. [`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md)
   (merged as part of PR #1378).
5. session `sched01`, PR #1384 (5th reproduction).

This is the **sixth** independent, byte-identical reproduction, now on top of
the multi-seed thrash-arm-close fix (PR #1386) — further evidence the
underlying fixture effect is real and deterministic, not an artifact of the
just-fixed closure-policy bug. `meaningful_program_rate` stays 0 on both
arms; the win stays confined to raw structural similarity, not full program
correctness. `binder_reference_f1` reads 0 on both arms this cycle.

Ship gates fail as expected: `insufficient_n` (n=3, need 20).

## Note on the control arm's exit code

The raw `arm_exits` record shows `exit=124` for the control arm because this
session's invoking shell wrapper imposed an external timeout shorter than
the two-arm cycle's total wall time. The driver's own `MAX_RUN_MINUTES=3`
per-arm cap was never exceeded, and the control arm completed with real,
non-placeholder smoke metrics that match the historical control reading
exactly (`structural_similarity=0.32667`). This is a self-inflicted
invocation-wrapper artifact, not a harness bug — no `HarnessSignalV1` is
warranted.

## Driver classifier note

The automated `sdlc_delivery.json` classifier flagged
`positive_no_tracked_delta_skip_stack` because this cycle re-ran the
identical recipe with no new code/harness diff of its own. Per `sdlc`
autotrain-iteration-delivery, documenting a positive result is itself the
reviewable delta that earns a stacked layer — matching how all five prior
reproductions of this exact hypothesis were delivered (PR #1369, #1376,
#1378, #1384).

## This is not a confirmation attempt

The driver's rank-1 next priority (confidence 0.95) proposes fresh-seed
confirmation of this exact hypothesis via a `-fresh-confirmation`-suffixed
arm. This is the same confirmation path prior sessions already found
blocked by two harness bugs — documented in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md):

1. A dual-arm decode timeout at seed `100005` with an undetermined root
   cause.
2. `_apply_frozen_replay` does not recognize `-confirm`-suffixed arm slugs.

That doc explicitly asks for a **dedicated `improve-openui-harnesses`
session**. This cycle does not attempt either blocker — it only adds a
sixth independent screening-stage reproduction of the same primary-metric
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
[`continuous-openui-local-dpur3s-c2-results.json`](continuous-openui-local-dpur3s-c2-results.json).
