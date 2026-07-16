# E148 — LTR repair ablation (2026-07-15)

## Question

Is the malformed continuation caused specifically by the LTR repair phase?

## Method

The E147 checkpoint was evaluated on one normalized smoke record with the same CPU/HF/local-only policy, constrained slot decoding, schema context, 20-second timeout, and three attempts. LTR primary remained enabled; LTR repair was disabled.

## Result

| Metric | E147 control | E148 no repair |
| --- | ---: | ---: |
| parse rate | 0.0 | 0.0 |
| structural similarity | 0.0 | 0.0 |
| placeholder validity | 0.0 | 0.0 |
| timeout count | 1 | 1 |
| p50 latency | 20,000.96 ms | 20,001.01 ms |
| AgentEvals | 0/5 | 0/5 |

The first captured choice changes from `root` to whitespace, but the run still times out, emits an unparsable result, and records 64 bounded selection events. Therefore the failure is not isolated to the repair flag. The next experiment should inspect primary LTR candidate construction/commit behavior or compare a constrained decode policy without LTR primary.
