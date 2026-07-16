# E147 — Judge-gated corpus control train (2026-07-15)

## Question

Does training on the source-controlled corpus after G11 judge-gate persistence improve the constrained-decoder result?

## Recipe

The run used the rebuilt `remediated_roots_judged` corpus with 498 records, HF context `HuggingFaceTB/SmolLM2-135M`, CPU, compositional output tokens, batch size 4, seed 0, and 8 steps. LTR primary/repair, schema context, slot-contract context/decoding, and no DESIGN.md context were unchanged from E145. Training telemetry, checkpoint, evaluator output, and AgentEvals JSONL are under `outputs/runs/iter-e147-judge-gated-control-20260715/`.

## Results

| Measurement | E145 | E147 |
| --- | ---: | ---: |
| records | 405 | 498 |
| last loss | 39.0902 | 26.7987 |
| prompt / target tokens | 5,457 / 1,880 | 5,820 / 1,848 |
| parse rate | 0.0 | 0.0 |
| structural similarity | 0.0 | 0.0 |
| placeholder validity | 0.0 | 0.0 |
| timeout count | 1 | 1 |
| p50 latency | 20,000.5 ms | 20,001.0 ms |
| AgentEvals | 0/5 | 0/5 |

The larger rebuilt corpus lowers the short-run loss by `12.2915`, but the diagnostic generated quality remains unchanged and still times out. The improvement is therefore not a ship or data-quality claim.

## Decision

Keep the judge-gated corpus as the canonical future-run input because its verification state is now durable, but do not promote this checkpoint. The next experiment should address the constrained LTR failure or evaluate a semantic judge/data subset against this control.
