# Autotrain c1732: component-plan promotion-cadence screen

**Verdict:** the third exactly size-matched component-plan comparison is another
quality null. Control and treatment match on every smoke and held-out quality
metric and every recorded deterministic-work counter. The treatment is 0.46%
slower on smoke and 3.96% faster on held-out, while taking 2.89 times the
training wall. It is rejected and is not checkpoint, promotion, or ship evidence.

## Result matrix

| Arm | Params | Suite | n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,755,764 | smoke | 3 | 1.0 | .8222 | .3333 | .2153 | .1667 | .1667 / 0 | 2,057.09 ms | complete fixture control; gates fail |
| component-plan loss `1.0` | 1,755,764 | smoke | 3 | 1.0 | .8222 | .3333 | .2153 | .1667 | .1667 / 0 | 2,066.56 ms | exact quality null; slower |
| matched control | 1,755,764 | held-out | 5 | 1.0 | .7076 | 0 | .06024 | .02857 | .04444 / 0 | 1,734.45 ms | complete fixture control; gates fail |
| component-plan loss `1.0` | 1,755,764 | held-out | 5 | 1.0 | .7076 | 0 | .06024 | .02857 | .04444 / 0 | 1,665.84 ms | exact quality null; empty meaning |

Both smoke arms also match placeholder fidelity (.7222), reward (.8417), 22
neural forwards, 50,910 unique completion states, 3,536 witness expansions,
and 54,534 parser forks. Candidate training loss is worse (16.0217 vs 12.5424)
and training wall is 8.319 s vs 2.881 s.

## E2E harness signals and repair

| Signal | Observed failure | Repair |
| --- | --- | --- |
| policy drift | c1731's stored v3 `positive=true` overrode its v4 reclassification during successor selection | Re-score the predecessor artifacts under the live climb policy before consuming priorities |
| repeated arm | promotion-cadence screen treated its null component-plan arm as executable again | Exhaust completed non-positive candidates for both screening and promotion-cadence screens |
| queue evidence | efficiency wins did not carry the `quality_held` reason required by the champion queue | Emit the already-proved held parse/meaning evidence with accepted efficiency wins |
| primary mismatch | the promotion role recorded `smoke.structural_similarity` because the CLI same-leaf override crossed suites | Ignore CLI suite overrides in promotion; policy-owned `held_out.structural_similarity` is authoritative |
| Lean presentation | terminal `Lean bands` showed an ambiguous dash although this was not a confirmed-champion promote | Render `not_applicable:no_champion`; actual promote remains fail-closed on proved LeverProof evidence |
| liveness | an old heartbeat changed an intentionally idle between-cycle state to `STALE` | Apply heartbeat staleness only while the driver reports `RUNNING` |
| gate presentation | completed expected gate rejections rendered `Gates —` because only legacy exit-8 outcomes projected the canonical scoreboard | Project any complete, hash-bound AgentV rejection as `fail / complete (gate reject)` |
| causal wording | each arm's standalone gate diagnosis said it “improved locally” although paired metrics were exact ties | Say only that the arm completed and failed gates; require matched-control comparison before improvement language |

The previously reported process gaps—bounded subprocess trees, partial-evidence
rejection, theorem-stop prerequisites, typed action-receipt validation, and live
child heartbeats—were verified already merged in PR #1268 and were not duplicated.

## Honest gate state and next priorities

AgentV completed both arms with zero execution errors. Smoke `n=3` and held-out
`n=5` are below the evidence floor; both suites miss meaningful, structural,
component-recall, AST BEq, and canonical BEq gates, and adversarial, OOD, and
`rico_held` were not run. This is fixture evidence only.

1. Run the distinct, size-matched component-edge hypothesis next; do not spend a
   fourth cycle on unchanged component-plan supervision.
2. Keep `held_out.structural_similarity` locked on promotion cadence and show
   smoke only as a secondary diagnostic suite.
3. Preserve exact parameter matching, deterministic-work counters, both suite
   views, and explicit Lean applicability in every terminal matrix.
4. Run the mathlib-free LeverProof preflight only for an actual confirmed
   champion promote; absence or failure of its locked certificate blocks promotion.

Machine-readable evidence is in
[`autotrain-cycle-1732-component-plan-promotion-screen.json`](autotrain-cycle-1732-component-plan-promotion-screen.json).
