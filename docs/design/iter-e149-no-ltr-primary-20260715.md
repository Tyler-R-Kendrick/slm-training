# E149 — No-LTR constrained ablation (2026-07-15)

## Question

Does the constrained decoder recover if both LTR primary and LTR repair are disabled?

## Method

The E147 checkpoint was evaluated on the same one-record normalized smoke case, with the same HF/local-only CPU policy, schema context, slot-contract constrained decoding, skipped exact stream probe, three attempts, and 20-second timeout. Both `grammar_ltr_primary` and `grammar_ltr_repair` were false.

## Result

| Metric | E147 control | E148 no repair | E149 no LTR |
| --- | ---: | ---: | ---: |
| parse rate | 0.0 | 0.0 | 0.0 |
| structural similarity | 0.0 | 0.0 | 0.0 |
| timeout count | 1 | 1 | 1 |
| p50 latency | 20,000.96 ms | 20,001.01 ms | 20,000.93 ms |
| AgentEvals | 0/5 | 0/5 | 0/5 |

The no-LTR run emits more partial tokens (547) but remains unparsable and times out. The evidence rules out a simple LTR-primary/repair toggle as the fix. Next work should inspect the general constrained candidate/probe loop and its interaction with the checkpoint, with bounded tracing retained.
