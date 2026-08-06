# Continuous autotrain: 2026-08-05 (scheduled session sched02) cycle 2 — operator-timeout measurement gap (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `bdf143cd` (`origin/main` tip at cycle start)

**Verdict:** not a model result. Both arms of the frozen `canvas`
(component-plan) vs `control` comparison exited `124` because this session's
first supervised-driver invocation was wrapped in an external `timeout 280`
shell guard that raced the two sequential 70s-wall-capped arms — the
research + hypothesize stages already consumed most of the 280s budget, so
the wrapper's kill signal landed mid-run of both arms.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms | exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | 1.0 | 0 | 31359.1 | 124 (metrics captured before kill) |
| canvas (component-plan) | 100002 | — | — | — | — | 124 (no scoreboard) |

The control arm's metrics were already computed and match every prior
session's control reading for this exact recipe/seed
([`continuous-openui-local-sched01-c2-results.md`](continuous-openui-local-sched01-c2-results.md)
and its four prior antecedents). The canvas arm never reached a scoreboard.

## Root cause: operator wrapper, not harness

The repo's own guidance
([`autotrain/references/continuous.md`](../../.claude/skills/autotrain/references/continuous.md))
says every child command already obeys the repository-wide `MAX_RUN_MINUTES`
wall cap internally. Wrapping the supervised driver invocation in a shorter
external `timeout` duplicates and conflicts with that internal cap. This
cycle's fix is procedural: re-invoke the supervised driver without an
external wrapper (or with a wrapper generous enough to exceed
research + hypothesize + `N * experiment_wall_seconds`).

## SDLC Phase A

**Non-positive** (`measurement_incomplete`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land locally and the loop continues by consuming the unacknowledged
`retry_measurement` handoff action, which replays the identical frozen arm
(`frozen_manifest_sha256`
`4abfa37fbabbde6d3da4f7ab49f6c65b9968508c60be8f665226f9d2cf37db0b`) on the
next supervised cycle.

## Next priorities

1. Retry the identical frozen arm with no external timeout wrapper shorter
   than the driver's own per-arm wall cap.
2. Once clean, treat the component-plan vs control `+.05613`
   `structural_similarity` delta as a further independent reproduction
   (fifth+ confirmation across sessions `j48f8u`, `ts5ofk`, `peuum8`,
   `sched01`).
3. Do not attempt the blocked `-confirm`/`-fresh-confirmation` frozen-replay
   path or the seed-`100005` dual-arm decode timeout speculatively; that
   remains routed to a dedicated `improve-openui-harnesses` session.

Machine evidence:
[`continuous-openui-local-sched02-c2-results.json`](continuous-openui-local-sched02-c2-results.json).
