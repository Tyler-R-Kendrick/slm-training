# E150 — Unconstrained checkpoint control (2026-07-15)

## Question

Is the 20-second timeout caused by the checkpoint or by the constrained decoder?

## Method

The E147 checkpoint was evaluated on the same one-record smoke case with the same HF/local-only CPU setup and 20-second timeout, but `grammar_constrained=false`. No LTR or slot-contract constrained path was enabled.

## Result

| Metric | Constrained E149 | Unconstrained E150 |
| --- | ---: | ---: |
| parse rate | 0.0 | 0.0 |
| structural similarity | 0.0 | 0.0724 |
| placeholder validity | 0.0 | 0.3385 |
| timeout count | 1 | 0 |
| p50 latency | 20,000.9 ms | 842.4 ms |
| AgentEvals | 0/5 | 0/5 |

The unconstrained control finishes roughly 24x faster and does not time out, proving the checkpoint is not the sole source of the timeout. Its output remains malformed (`TextContent"Update a content...`), so this does not make an unconstrained model shippable. The next harness experiment should bound or bypass expensive constrained candidate/probe work while preserving hard legality.
