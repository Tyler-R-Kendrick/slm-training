# Autotrain c1748: component-plan promotion-cadence screen

**Verdict:** the exactly size-matched component-plan arm is another quality
null. Control and treatment match on every smoke and held-out quality metric.
The treatment is 3.51% faster on the three-record smoke fixture but 1.60%
slower on held-out, takes 2.19 times the training wall, and ends at higher
loss. It is rejected and is not promotion or ship evidence.

## Result matrix

| Arm | Params | Suite | n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,755,764 | smoke | 3 | 1.0 | .6333 | 0 | .05750 | 0 | 0 / 0 | 1,406.90 ms | complete fixture control; gates fail |
| component-plan loss `1.0` | 1,755,764 | smoke | 3 | 1.0 | .6333 | 0 | .05750 | 0 | 0 / 0 | 1,357.57 ms | exact quality null; fixture latency only |
| matched control | 1,755,764 | held-out | 5 | 1.0 | .4371 | 0 | .04274 | 0 | 0 / 0 | 1,315.32 ms | complete fixture control; gates fail |
| component-plan loss `1.0` | 1,755,764 | held-out | 5 | 1.0 | .4371 | 0 | .04274 | 0 | 0 / 0 | 1,336.36 ms | exact quality null; slower |

Both arms completed every document with zero decode timeouts and zero
unconstrained fallbacks. Candidate training loss is worse (12.7035 vs 9.8156)
and training wall is 6.650 s vs 3.039 s. The two 24-step CPU scratch
checkpoints are local with explicit no-sync and are not reusable champions.

## Signals and next hypothesis

The repeated exact quality tie says component-plan supervision is not reaching
the model-ranked choices that determine output quality at this budget. Its
smoke-only latency movement reverses on held-out and is below the 5% efficiency
floor, so it is noise rather than a promotable signal. Zero meaningful rate,
component recall, AST node/edge F1, AST BEq, and canonical BEq point upstream
of polish: the model is producing legal syntax but not selecting the right
component topology.

1. Prioritize the distinct size-matched component-edge hypothesis, which more
   directly targets topology selection; keep the control and parameter count
   identical.
2. If it is also quality-null, stop rotating auxiliary losses unchanged and
   inspect gradient application, target prevalence, and decode consumption for
   the auxiliary heads before another train.
3. Keep held-out structural similarity as the promotion primary; smoke remains
   a wiring diagnostic and cannot establish a quality or latency win.
4. Keep RL locked until SFT produces non-zero meaningful and component-recall
   signal on complete held-out evaluation.

## Honest gates and formal evidence

AgentV bundles were emitted for both arms. Ship gates fail closed: smoke `n=3`
and held-out `n=5` are below the evidence floor; both suites miss meaningful,
structural, component-recall, AST BEq, and canonical BEq thresholds; and the
adversarial, OOD, and `rico_held` suites were not run. This is fixture evidence
only.

No empirical optimum band or confirmed champion exists in this cycle, so a
Lean promotion certificate is not applicable. The formal gate remains
integrated and fail-closed: an actual promotion still requires locked
expectations, the mathlib-free LeverProof preflight, and a replayed
`metric_certificate/v2`.

Machine-readable evidence is in
[`autotrain-cycle-1748-component-plan-screen.json`](autotrain-cycle-1748-component-plan-screen.json).
