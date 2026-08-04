# Autotrain c1761: cached batch-size-one replay reproduces timeout

**Verdict:** the exact frozen c1760 replay remains inconclusive and reproduces
the infrastructure failure. Both completed c1760 training stages were reused;
no model was retrained and no new checkpoint was created. The control evaluation
completed, while the batch-size-one evaluation again exceeded the bounded stage
wall on `smoke_button_01` and emitted no scoreboard or AgentV bundle.

| Arm | Train work | Params | Evaluation | Smoke | Held-out | Decision |
| --- | --- | ---: | --- | --- | --- | --- |
| batch size 1 | cached c1760 checkpoint | 1,608,962 | interrupted; non-scoreable | 1 processed before interruption; no aggregate metrics | not reached | repeated infrastructure failure |
| control, batch size 2 | cached c1760 checkpoint | 1,608,962 | complete AgentV, 0 execution errors | n=3; parse 1, meaning .3333, structure .13527, recall .1667, p50 1,382.58 ms | n=5; parse 1, meaning 0, structure .06024, recall .02857, p50 1,370.70 ms | gates fail; comparison unavailable |

The successor manifests preserve the c1760 recipes, seed 101760, frozen
endpoints, gates, and stopping rules while linking to current commit
`8b6ddd099048f6e3bc643fac66896150e78beaaf`. Checkpoint reuse avoided both
22-step training stages. This validates stage-level caching in the continuous
driver and isolates the repeated failure to candidate evaluation.

The candidate `DecodeProgressV1` is still `measurement_complete=false` and
`scoreable=false`. It records 299 emitted tokens, 289 neural forwards, 37.96 s
total decode time, 27.81 s in the backbone, 9.51 s in compiler completion, and
29,616 witness-state expansions. Its trace repeats the same unclosed long
numeric literal seen in c1760. These are harness diagnostics, not quality or
latency metrics.

The control AgentV bundle completed 0/2 suite assertions with 0 execution
errors, and honest ship gates fail. Lean remains
`not_applicable:no_champion`: this is an infrastructure replay, not a promotion
claim. No checkpoint is synced, promoted, reusable outside the frozen replay,
or ship evidence.

Next: preserve the frozen replay lineage and route the exact manifest to
`improve-openui-harnesses/model_build`. Policy v5 now treats one original
failure plus this identical cached reproduction as sufficient to require
repair; another unmodified replay is receipt-blocked until the canonical owner
changes.

Machine evidence:
[`autotrain-cycle-1761-batch1-frozen-replay.json`](autotrain-cycle-1761-batch1-frozen-replay.json).
