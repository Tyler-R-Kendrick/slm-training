# E41 lexer-native root-corpus control — 2026-07-15

E41 was rerun on the source-controlled `remediated_roots` corpus as a matched
control for E46, removing kind-factorized embeddings while retaining the
lexer-native tokenizer, symbol table, slot context, and constrained repair.

| suite | n | parse | structural similarity | placeholder fidelity | reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.000 | 0.000 | 0.000 | 0.000 |

Predictions collapsed to malformed binder/state sequences such as
`b0 b2$s15 b4 b59 b3 b3 b2 b5 true`; the parser reported “parser produced no
root element”. This reproduces the E46 failure without factorized embeddings,
so the next intervention must address lexer-native target/decode alignment or
training signal rather than only the embedding factorization.

Recipe: scratch context, 64 steps, batch size 8, root corpus fingerprint
`f8d714f122ac7f091236fd4e562935758de330534cac146abf30af13d0ac98ce`, E41
configuration, smoke-only diagnostic evaluation. The matrix reported missing
non-smoke suites; this is not a ship evaluation.

Telemetry and AgentEvals artifacts remain under
`outputs/runs/iter-e41-roots-aligned-20260715/qx_e41_symtable/`.
