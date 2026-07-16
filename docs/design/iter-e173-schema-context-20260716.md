# E173 schema-context training control (2026-07-16)

E173 trained the judged corpus for 32 steps with schema and slot-contract
context enabled, then ran a bounded one-record smoke probe using the same
grammar-derived compiler.

| Training / probe metric | E173 |
| --- | ---: |
| train steps | 32 |
| final loss | 11.0876 |
| schema context | enabled |
| probe n | 1 |
| probe meaningful parse | 0.0000 |
| probe syntax parse | 1.0000 |
| probe structural similarity | 0.1542 |
| probe component recall | 0.2500 |
| probe p50 latency (ms) | 3896.5 |

The probe prediction was valid syntax but a single `TextContent` instead of the
hero hierarchy. Schema context alone therefore does not fix semantic component
selection. The corpus audit found 498 judged records with `root = Stack` and a
strong component-role skew; the next training variant should address semantic
coverage or supervision explicitly.

The full three-record invocation was terminated before writing a scoreboard,
so E173 is a bounded diagnostic, not a full smoke or ship result.

Evidence: [result JSON](iter-e173-schema-context-20260716.json), [train summary](../../outputs/runs/e173-schema-context-32step/train_summary.json), [probe eval](../../outputs/runs/e173-probe256/eval_smoke.json), and the AgentV JSONL path recorded in the result.
