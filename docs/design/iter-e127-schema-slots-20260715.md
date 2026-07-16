# E127 schema and slot-contract iteration — 2026-07-15

E127 trains the 405-record judged corpus with schema and slot-contract context,
constrained slot decoding, LTR primary/repair, and compositional output
tokenization. It is a matched diagnostic continuation of E123.

| Suite | n | Parse | Placeholder validity | Normalized fidelity | Reward | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 1 | 0.0 | 0.55 | 0.25 | 0.0 | 9,308 ms |

Training loss was **10.71** after 32 CPU steps. The run emitted complete train
telemetry and an AgentEvals JSONL bundle. The placeholder signals are better
than E123's zero signal, and the matched decode did not use fallback, but parse
and reward remain zero. This is promising diagnostic evidence, not a ship or
promotion result.
