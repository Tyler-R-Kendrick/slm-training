# Continuous autotrain: 2026-08-05 (scheduled session oj6ymg) cycle 1 — null delta on knob-rotation arm (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `bdf143cd` (`origin/main` tip at cycle start)

**Verdict:** the size-matched `bounds` (knob-rotation) arm ties its control
exactly on the declared primary at this seed — a null delta, not positive.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 1.0 | .63333 | 4112.84 |
| bounds | 100001 | .05750 | 1.0 | .63333 | 5001.45 |

Primary delta `0.0` — `meaningful_program_rate` stays 0 on both arms. Ship
gates fail as expected: `fixture_insufficient_n` (n=3, need 20).

## Another independent reproduction

This is the same c1 null-delta screening result already reproduced by
sessions `sched01`, `j48f8u`, and `ts5ofk`
([`continuous-openui-local-sched01-c1-results.md`](continuous-openui-local-sched01-c1-results.md),
[`continuous-openui-local-j48f8u-c1-results.md`](continuous-openui-local-j48f8u-c1-results.md),
[`continuous-openui-local-ts5ofk-c1-results.md`](continuous-openui-local-ts5ofk-c1-results.md)).
This cycle does not attempt the blocked `-confirm` fresh-seed path or the
seed-`100005` dual-arm decode timeout — both remain routed to a dedicated
`improve-openui-harnesses` session per
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land as a local commit and the loop's ranked successor priority (the
`component-plan` hypothesis, independently reproduced with a `+0.056`
structural_similarity win across at least five prior sessions) is the next
candidate.

## Repository hygiene note (not part of this cycle's experiment)

At cycle start, `origin/main` carried **17 open PRs** from prior autotrain
scheduled sessions created within the preceding ~16 hours (#1436–#1454,
none merged), several of which (#1437, #1438, #1440, #1441, #1442, #1443)
independently re-fix the same `generate_batch_size` `ExperimentKnobs` schema
gap that commit `bdf143c` (PR #1444) already merged to `main`. This cycle
does not attempt to triage or close that backlog — it is flagged here as a
diagnostic signal for a dedicated `sdlc` Phase B bottom-up closeout pass
(inventory `gh`-equivalent open PRs, drop/close the ones superseded by
`#1444`, and merge or close the rest) rather than continuing to add
non-positive-cycle sprawl on top of it.

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Do not speculatively re-attempt the blocked `-confirm` slug bug or
   seed-`100005` decode timeout; that stays a dedicated harness-repair task.
4. Run a Phase B closeout pass over the 17 open autotrain PRs before more
   scheduled sessions add further non-positive layers.

Machine evidence:
[`continuous-openui-local-oj6ymg-c1-results.json`](continuous-openui-local-oj6ymg-c1-results.json).
