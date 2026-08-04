# Autotrain c1760: batch-size-one measurement incomplete

**Verdict:** inconclusive. The batch-size-one arm trained to its declared 22
steps, but its evaluation exceeded the bounded stage wall after one complete
smoke record and while decoding a second. It has no scoreboard, AgentEvals
JSONL, AgentV result bundle, gate decision, or scoreable quality/latency metric.
The matched control completed smoke and held-out evaluation and failed honest
ship gates, but a control-only result cannot decide the preregistered comparison.

| Arm | Params | Steps | Train loss / wall | Evaluation | Smoke | Held-out | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| batch size 1 | 1,608,962 | 22 | 11.3952 / 2.515 s | interrupted; progress is non-scoreable | 1 processed before interruption; no aggregate metrics | not reached | retry exact frozen measurement |
| control, batch size 2 | 1,608,962 | 22 | 13.8092 / 3.509 s | complete AgentV, 0 execution errors | n=3; parse 1, meaning .3333, structure .13527, recall .1667, p50 1,386.07 ms | n=5; parse 1, meaning 0, structure .06024, recall .02857, p50 1,306.76 ms | gates fail; comparison unavailable |

Both CPU scratch arms used seed 101760, `wf_smoke_v2`, lexer output, strict
symbol-only grammar constraints, and explicit local/no-sync checkpoint policy.
The runtime-only batch-size lever did not change model capacity. Arm execution
was counterbalanced with the candidate first.

The partial `DecodeProgressV1` artifact is explicitly
`measurement_complete=false` and `scoreable=false`. It records one active
smoke record at interruption and cumulative diagnostics for two decode attempts:
247 emitted tokens, 237 neural forwards, 31.35 s total decode time, 22.69 s in
the backbone, 8.14 s in compiler completion, and 24,624 witness-state
expansions. Those values identify a runtime investigation surface; they are not
candidate metrics and do not support a model, lever, or ship claim.

The control AgentV bundle completed 0/2 suites with 0 execution errors because
all suite assertions did not pass. `--ship-gates` also fails for fixture volume,
meaning, structure, recall, AST BEq, canonical BEq, and missing adversarial/OOD/
`rico_held` suites. Lean is `not_applicable:no_champion`: there is no promotion
candidate or formal claim to certify in this incomplete cycle.

Next: replay the exact hash-bound frozen control and batch-size-one manifests.
The replay harness must reuse both completed training checkpoints and rerun only
evaluation. If the identical evaluation fails again, the bounded replay budget
routes the signal to `improve-openui-harnesses/model_build`; do not widen the
timeout, weaken constraints, or substitute partial telemetry for a scoreboard.

Machine evidence:
[`autotrain-cycle-1760-batch1-incomplete.json`](autotrain-cycle-1760-batch1-incomplete.json).
