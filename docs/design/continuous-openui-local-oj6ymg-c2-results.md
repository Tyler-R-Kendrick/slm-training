# Continuous autotrain: 2026-08-05 (scheduled session oj6ymg) cycle 2 — component-plan structural win, 7th reproduction, no new PR (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `6e0a5c7c` (this session's cycle-1 docs commit, on top of `main` tip `bdf143cd`)

**Verdict:** `component-plan` beats its size-matched control on the declared
primary at this seed — the same primary-metric delta, at the same seed, as
**six** prior independently-run and merged sessions' measurements of the
identical hypothesis. Fixture screening only — not a ship or promotion
claim.

| Arm | Seed | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | 0 | 32894.4 |
| component-plan | 100002 | .38280 | 0 | 31507.5 |

Primary improvement `+.05613` (`0.32666666666666666 -> 0.38280000000000003`)
— byte-identical to at least six prior sessions' independent, already-merged
measurements of this same hypothesis:

1. [`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
   (merged as PR #1369).
2. [`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md).
3. [`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)
   (merged as part of PR #1376).
4. [`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md)
   (merged as part of PR #1378).
5. [`continuous-openui-local-peuum8-c4-results.md`](continuous-openui-local-peuum8-c4-results.md).
6. [`continuous-openui-local-sched01-c2-results.md`](continuous-openui-local-sched01-c2-results.md)
   (merged as PR #1384).

`meaningful_program_rate` stays 0 on both arms; the win stays confined to
raw structural similarity, not full program correctness. `binder_reference_f1`
reads 0 on both arms this cycle. Ship gates fail as expected:
`insufficient_n` (n=3, need 20).

## Departure from prior-session precedent: no stack layer this cycle

Prior sessions (`sched01`, `j48f8u`, `ts5ofk`) each opened or extended a
stacked PR for this exact byte-identical reproduction, reasoning that
documenting a positive result is itself the reviewable delta. This session
does **not** do that, for two reasons:

1. The driver's own `sdlc_delivery.json` classifier already marks this cycle
   `stack_layer: false`, `stack_action: positive_no_tracked_delta_skip_stack`
   — no new code or harness delta produced the metric win, only a repeat
   measurement.
2. At cycle start, `origin/main` carried **17 open, unmerged** autotrain PRs
   from sessions in the preceding ~16 hours (see the cycle-1 doc,
   [`continuous-openui-local-oj6ymg-c1-results.md`](continuous-openui-local-oj6ymg-c1-results.md)).
   Opening an eighth PR to restate an already-six-times-merged, byte-identical
   fact adds backlog rather than new information. Per `sdlc`
   autotrain-iteration-delivery, "when uncertain, treat as not positive and
   keep going (docs + local commits only)" — this cycle treats a
   zero-new-information reproduction the same way.

This local doc + JSON satisfies the iron-law documentation requirement.
`MODEL_CARD.md` is intentionally not touched, consistent with how the merged
`sched01`/`j48f8u`/`ts5ofk` PRs handled this same local scratch/fixture
screening-checkpoint class.

## This is not a confirmation attempt

The driver's rank-1 next priority (confidence 0.95) proposes fresh-seed
confirmation of this exact hypothesis via a `-fresh-confirmation`-suffixed
arm. This remains the same confirmation path already found blocked by two
harness bugs in a prior session — documented in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md):

1. A dual-arm decode timeout at seed `100005` with an undetermined root
   cause.
2. `_apply_frozen_replay` does not recognize `-confirm`-suffixed arm slugs.

That doc explicitly asks for a **dedicated `improve-openui-harnesses`
session**. This cycle does not attempt either blocker.

## SDLC Phase A

**Positive** (`primary_metric_win`) but **no stack layer** — see rationale
above. Local commit only.

## Next priorities

1. Do not open a stacked PR for further byte-identical reproductions of this
   hypothesis; route confirmation/promotion attempts to a dedicated
   `improve-openui-harnesses` session that fixes the `-confirm` slug bug and
   the seed-`100005` decode timeout first.
2. Prioritize an `sdlc` Phase B closeout pass over the 17 open autotrain PRs
   before further scheduled sessions add non-positive or duplicate-positive
   layers.
3. Keep promotion formal preflight locked until fresh confirmation
   establishes a champion.

Machine evidence:
[`continuous-openui-local-oj6ymg-c2-results.json`](continuous-openui-local-oj6ymg-c2-results.json).
