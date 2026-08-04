# Autotrain c1762: bounded evaluation finalizes typed timeout evidence

**Verdict:** the exact frozen batch-size-one replay remains inconclusive, but
the c1761 harness repair worked at its intended boundary. Both completed c1760
training stages were reused, so no model was retrained and no checkpoint was
created. Unlike c1760/c1761, the candidate wrote both suite scoreboards, all
eight record dispositions, AgentEvals assertions, gates, and an AgentV bundle.
Every candidate record timed out under the cumulative evaluation wall, so no
candidate quality metric is scoreable and ship gates fail.

| Arm | Train work | Params | Smoke | Held-out | AgentV | Decision |
| --- | --- | ---: | --- | --- | --- | --- |
| batch size 1 | cached c1760 checkpoint | 1,608,962 | n=3; 0 complete; 3 typed timeouts; p50 incl. incomplete 4,566.26 ms | n=5; 0 complete; 5 typed timeouts; p50 incl. incomplete 4,540.28 ms | 0/2 assertions; 0 execution errors | infrastructure-incomplete; not scoreable |
| control, batch size 2 | cached c1760 checkpoint | 1,608,962 | n=3; parse 1; meaning .3333; structure .13527; recall .1667; p50 1,325.28 ms | n=5; parse 1; meaning 0; structure .06024; recall .02857; p50 1,361.05 ms | 0/2 assertions; 0 execution errors | complete gate rejection |

The candidate used an injected 40.0 s cumulative evaluation wall. The evaluator
fair-shared the remaining budget across the eight selected records: effective
per-record caps were 4.564-4.565 s for smoke and 4.536-4.540 s for held-out.
This bounded every pathological decode and reserved enough evaluator-local time
to persist canonical suite results. Timeout records are excluded from quality
aggregates; their null metrics are measurement incompleteness, not zero-quality
model evidence.

The remaining e2e defect is outside evaluator-local finalization. The candidate
stage finished its two scoreboards and AgentV work at the outer supervisor edge,
so the campaign outcome is still `stopped` with `stage exceeded wall-time limit`.
The next harness change must subtract a small post-evaluation reserve before
injecting `--evaluation-wall-seconds`, leaving the parent process time to bind
the scoreboard, gates, AgentV summary, outcome, and feedback. This does not
widen any run cap, per-record timeout, or ship gate.

If that reserved-tail replay completes but still yields eight typed timeouts,
the next priority moves from orchestration to the constrained decoder: use the
per-record traces to target compiler witness/numeric-literal expansion. Do not
spend another unchanged scheduling replay and do not weaken constrained decode.

Lean remains `not_applicable:no_champion`: this fixture replay produced no
promotion candidate. The promotion-integrated Lean/formal lane remains required
when a champion exists; absence here is explicit rather than omitted.

Machine evidence:
[`autotrain-cycle-1762-finalized-timeout-matrix.json`](autotrain-cycle-1762-finalized-timeout-matrix.json).
