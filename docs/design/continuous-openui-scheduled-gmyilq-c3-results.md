# Autotrain c3 (continuous-openui-scheduled-gmyilq): AgentV fix confirmed, new blocker is decode wall capacity

**Verdict:** infrastructure failure, not scoreable — but progress. Replaying
the frozen c2 control/canvas arms after the `evals.agentv` v10 fix
(commit `9a53ea83c4265629f50bcb490f3abfb142059a8c`) confirms the harness
repair worked: `scripts.evaluate_model --ship-gates` now runs to completion
for both arms and produces a real `gates.json` instead of crashing with
`ERR_MODULE_NOT_FOUND`. The AgentV self-heal false-positive is resolved.

## New blocker

Both arms still fail to produce a *usable* scoreboard — every smoke record
timed out during decode:

| Arm | compiler_ms_mean | decode_timeout_count | need |
| --- | ---: | ---: | --- |
| control | 32,872.9 ms | 3 | 0 |
| canvas | 34,534.8 ms | 3 | 0 |

With `decode_timeout_seconds=12.0` and a real per-record compile+decode cost
of ~33-35s on this CPU-only sandbox, all 3 fixture records exceed the
budget, leaving every quality metric (`parse_rate`,
`structural_similarity`, `meaningful_program_rate`, `binder_reference_f1`,
etc.) null. Lean is `not_applicable:screening`.

This matches a decode-capacity gap a prior session already investigated in
depth on a different loop
([`continuous-openui-scheduled-fe71636-c4-results.md`](continuous-openui-scheduled-fe71636-c4-results.md)):
that session concluded "timeout-value tuning is exhausted" — raising
`decode_timeout_seconds` alone cannot close a roughly 2-3x gap — and left
three untried next levers:

1. Reduce `screening_smoke_n` below 3.
2. Reduce screening training steps (currently 20) to leave more of the
   shared 180s `MAX_RUN_MINUTES` wall budget for eval.
3. Accept that `wf_smoke_v2` screening does not fit `MAX_RUN_MINUTES=3` on
   this sandbox class and treat this loop's screening role as
   diagnostic-only until faster compute is authorized.

This is this loop's (`continuous-openui-scheduled-gmyilq`) **first**
occurrence of the decode-capacity blocker, not a third identical
recurrence, so the loop law's repeated-hard-block rule does not apply yet.
The next cycle tries lever (2) — a genuinely untried knob, not a repeat of
the already-exhausted timeout-bump lever.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `fixture_insufficient_n` on both
arms, `primary_metric_unavailable`). No stack layer. The AgentV fix from c2
stays a local commit on this branch pending a cycle that reaches a usable
scoreboard or an independently-shippable infra-only PR at Phase B closeout.

Machine evidence:
[`continuous-openui-scheduled-gmyilq-c3-results.json`](continuous-openui-scheduled-gmyilq-c3-results.json).
