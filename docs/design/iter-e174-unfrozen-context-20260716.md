# E174 unfrozen-context control (2026-07-16)

E174 tested whether updating the HF context tower would improve semantic
component grounding. It used the same judged corpus and schema context as
E173, but ran 8 steps with the context tower unfrozen.

| Metric | E173 frozen, 32 steps | E174 unfrozen, 8 steps |
| --- | ---: | ---: |
| final loss | 11.0876 | 39.4253 |
| bounded syntax parse | 1.0000 | 0.0000 |
| bounded meaningful parse | 0.0000 | 0.0000 |
| structural similarity | 0.1542 | 0.3208 |
| p50 latency (ms) | 3896.5 | 29935.68 |

The unfrozen control produced malformed output and a much higher loss. It is
rejected; the context tower remains frozen for the next data/supervision
iteration. Both results are bounded one-record probes, not ship evaluations.

Evidence: [result JSON](iter-e174-unfrozen-context-20260716.json), [train summary](../../outputs/runs/e174-unfrozen-context-8step/train_summary.json), [probe eval](../../outputs/runs/e174-unfrozen-context-probe/eval_smoke.json), and the AgentV JSONL path recorded in the result.
