# E47 lexer-native LTR supervision control — 2026-07-15

E47 doubled the E41 lexer-native LTR loss weight to 2.0 while keeping the
Silver+ corpus, tokenizer, context, and strict constrained LTR evaluation
fixed. The run used 256 CPU steps and the remediated smoke suite (n=3).

| path | parse | structural | fidelity | reward | p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict LTR | 0/3 | 0.000 | 0.000 | 0.000 | 5,105 ms |

AgentV recorded 0/5 passed with no execution errors and no unconstrained
fallbacks. Increasing LTR supervision weight alone did not recover structure.

Decision: reject loss-weight-only tuning. The next investigation targets
lexer-native target IDs, DFA transition prefixes, and their training/eval
surface alignment.

Evidence:

- outputs/runs/iter-e47-silver-256-20260715/qx_e47_ltr_supervision/matrix_result.json
- outputs/runs/iter-e47-silver-256-20260715/strict_ltr/e47-silver-256-strict-ltr/eval_smoke.json

This is a scratch, smoke-only result and is not a ship claim.
