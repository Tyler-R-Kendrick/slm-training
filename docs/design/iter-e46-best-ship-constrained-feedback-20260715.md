# E46 selected-checkpoint constrained feedback — 2026-07-15

The quality-selected E46 step-16 checkpoint was evaluated through the
grammar-constrained decode path after training. This is a negative result and
the authoritative feedback for the next iteration.

| suite | n | parse | structural similarity | reward | p50 latency | timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.000 | 0.000 | 0.000 | 38,589 ms | 0 |

All three predictions were malformed lexer-native binder sequences beginning
with `b0 b5 b13 ...` and ending in `false`; the parser reported “parser
produced no root element”. The checkpoint was trained with grammar constraints,
template fill, and slot context enabled, so this is a tokenizer/output-head or
decode alignment failure, not merely a missing grammar flag.

The run emitted AgentEvals JSONL and the full evaluation artifact under
`outputs/runs/iter-e46-qualityselect-20260715/feedback_constrained/`.

Next intervention: verify checkpoint tokenizer identity and decode output mode
without forcing an incompatible tokenizer, then train/evaluate a matched
output-head configuration. Do not promote this checkpoint despite its
unconstrained smoke score.
