# Continuous autotrain: 2026-08-05 (scheduled session sched02) cycle 4 — decode-timeout capacity ceiling, still open (harness, screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `1a65c9a2` (this session's cycle-3 harness-repair commit, on top of `main` tip `bdf143cd`)

**Verdict:** not a model result, and not a new regression. Both arms trained
cleanly, but every smoke record timed out during ship-gated
`evaluate_model` (`decode_timeout_count=3/3`), with
`compiler_ms_mean` ~34.3–34.5s/record.

| Arm | decode_timeout_count | compiler_ms_mean | exit |
| --- | ---: | ---: | ---: |
| control | 3/3 | 34453.6 | 2 |
| canvas (component-plan) | 3/3 | 34327.2 | 2 |

## Diagnosis (via `improve-openui-harnesses`, not a hasty knob bump)

`compiler_ms` is the mandatory grammar/symbol-table compilation phase span
(`models/decode_stats.py` `ATTRIBUTED_PHASE_FIELDS`) that
[AGENTS.md's non-negotiable decode invariants](../../AGENTS.md) require —
speculation only ranks over forward-calculated symbol tables and always
verifies before commit. It is **not** gated by `--compiler-decode-mode off`
(already set), which only toggles an optional additional search mode
(`eval_runner.py compiler_decode_mode`: `off`/`tree`).

`_fit_screening_decode_timeout_seconds`'s wall-fit ceiling for this arm's
70s screening share is 14.0s (`(70-20-8)/3`). `screening_stage_wall_minutes`
is already pinned at `policy.v1.json`'s max allowed value — equal to the
repository's hard `MAX_RUN_MINUTES=3` cap — so there is **no legal room** to
raise the arm wall budget, and raising `screening_decode_timeout_seconds`
toward 14s would still fall far short of the observed ~34s/record cost.

This reproduces and slightly worsens the same open capacity limitation
first documented in
[`harness.autoresearch.experiment_campaign` v179](../../src/slm_training/resources/versions.json)
(then ~23s/record, motivating the 8→12s bump that itself acknowledged it
was still short). No tracked file governing this cost path changed in this
session — most plausibly this sandbox's CPU is slower or more contended
than prior sessions', not a new code regression.

## SDLC Phase A

**Non-positive**, and the `repair_harness` handoff action is acknowledged
as **blocked** (real, evidenced, structural capacity ceiling — not a
recoverable-in-session harness bug) rather than forced with a speculative
change. Per `sdlc` autotrain-iteration-delivery, no stack layer opens.

## Next priorities

1. Route the underlying `compiler_ms` cost-reduction question (not the
   wall/timeout knobs, which are already at their legal ceiling) to a
   dedicated `improve-openui-harnesses` session with profiling time.
2. Continue the continuous loop by rotating to a different screening
   hypothesis/knob that doesn't depend on this frozen decode-timeout-bound
   arm reproducing cleanly on this host.
3. Do not attempt the blocked `-confirm`/`-fresh-confirmation` frozen-replay
   path or the seed-`100005` dual-arm decode timeout speculatively; both
   remain routed to a dedicated `improve-openui-harnesses` session per
   [`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md).

Machine evidence:
[`continuous-openui-local-sched02-c4-results.json`](continuous-openui-local-sched02-c4-results.json).
