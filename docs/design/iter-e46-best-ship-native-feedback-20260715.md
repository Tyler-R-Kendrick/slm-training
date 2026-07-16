# E46 selected-checkpoint native-tokenizer feedback — 2026-07-15

Repeating the E46 selected-checkpoint constrained smoke evaluation without an
output-tokenizer override produced the same failure. This rules out the CLI
override as the primary cause.

| suite | n | parse | structural similarity | reward | p50 latency | timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.000 | 0.000 | 0.000 | 19,144 ms | 0 |

All predictions were the same malformed binder-token sequence and ended in
`false`; the parser produced no root element. AgentEvals was emitted with 0
passed and 5 failed cases. The next change must address the checkpoint’s
output-tokenizer/head and constrained decoder alignment, not only evaluation
flags.

Evidence: `outputs/runs/iter-e46-qualityselect-20260715/feedback_native/`.
