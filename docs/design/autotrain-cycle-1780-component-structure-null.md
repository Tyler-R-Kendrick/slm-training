# Autotrain c1780: component structure is a quality null

**Verdict:** reject. Joint component-plan/edge supervision and its size-matched
control tie exactly on smoke and held-out quality. The candidate is 6.37%
faster on smoke but 2.38% slower on held-out, the preregistered primary remains
null, and candidate training takes 2.38x as long.

| Arm | Params / train | Smoke | Held-out | Decision |
| --- | --- | --- | --- | --- |
| component structure | 1,913,789; 21 steps; loss 17.25025; 6.46 s | n=3; meaning .3333; structure .17417; binder .6333; p50 1,013.25 ms | n=5; meaning 0; structure .09758; binder .4371; p50 1,091.01 ms | reject |
| matched control | 1,913,789; 21 steps; loss 12.54634; 2.72 s | identical quality; p50 1,082.16 ms | identical quality; p50 1,065.64 ms | baseline |

Both arms parse at 1.0, their AgentV bundles are complete with zero execution
errors, and ship gates fail. This CPU fixture is not ship evidence. Both
explicit no-sync scratch checkpoints are provenance-only and must not be
reused, promoted, synced, or shipped. Lean is `not_applicable:no_champion`:
the promotion-cadence cycle produced no champion, so the fail-closed formal
preflight correctly had no candidate to prove.

The component-structure family is exhausted. The next ranked diagnostic is
`canvas`, testing whether compact active-canvas decoding changes runtime
without lowering parse rate; its runtime result must not be relabeled as a
quality win.

Machine evidence:
[`autotrain-cycle-1780-component-structure-null.json`](autotrain-cycle-1780-component-structure-null.json).
