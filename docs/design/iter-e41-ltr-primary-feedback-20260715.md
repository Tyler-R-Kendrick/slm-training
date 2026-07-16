# E41 LTR-primary constrained feedback — 2026-07-15

The E41 lexer-native checkpoint was re-evaluated with grammar LTR decoding as
the primary path and constrained repair enabled.

| suite | n | parse | structural similarity | reward | p50 latency | timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.000 | 0.000 | 0.000 | 5,623 ms | 0 |

The predictions remained malformed binder/state sequences and the parser
reported “parser produced no root element”. LTR-primary reduced latency versus
the original E41 decode, but did not improve quality. This is a negative decode
intervention; the next change must improve the lexer-native training signal.

AgentEvals and telemetry artifacts are under
`outputs/runs/iter-e41-roots-aligned-20260715/feedback_ltr_primary/`.
