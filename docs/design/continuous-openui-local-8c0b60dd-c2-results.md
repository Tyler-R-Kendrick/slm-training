# Continuous autotrain: 2026-08-04 (session 8c0b60dd) cycle 2 — frozen replay confirms null delta

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `3911ebce` (cycle 1's harness-repair + docs commits merged)
**Replay of:** [`continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`](continuous-openui-local-8c0b60dd-c1-results.md)

**Verdict:** with the AgentV bootstrap repair in place, the frozen
control/`bounds` arm replays to a **complete** measurement — an honest gate
rejection instead of an infrastructure incomplete. The size-matched arm ties
its control exactly on the primary metric: a null delta, not positive.

| Arm | Seed | structural_similarity | meaningful_program_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 0.0 | .63333 | 5331.89 |
| bounds | 100001 | .05750 | 0.0 | .63333 | 6016.32 |

Primary delta `0.0`. Ship gates fail as expected on the smoke fixture:
`insufficient_n` (n=3, need 20) plus quality-threshold misses
(`meaningful_program_rate`, `structural_similarity`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score` all below gate). This is
fixture wiring evidence, not a ship claim.

## Infrastructure confirmation

This is the concrete evidence that the cycle-1 harness repair
(`26083c6`, `evals.agentv` v7 → v8) is a genuine **proven executable
unblock**: both arms went from `harness_failure`/`missing_scoreboard` in
cycle 1 to a complete `scoreboard.json` + gate verdict here, with zero code
changes to the model/training path.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `fixture_insufficient_n`).
No new stack layer opens for the model result. Per `sdlc`
autotrain-iteration-delivery, the loop continues into the driver's ranked
successor priority.

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
