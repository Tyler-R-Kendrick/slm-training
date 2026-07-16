# E172 generated-schema type constraints (2026-07-16)

E172 evaluated the same E160 lexer checkpoint after the compiler began deriving
candidate categories from generated schema types and Lark terminals. The map
is generic across string, numeric, boolean, null, array, and object slots; it
does not name individual components or punctuation tokens.

| Metric | E171 | E172 |
| --- | ---: | ---: |
| meaningful parse | 0.0000 | 0.0000 |
| syntax parse | 0.3333 | 0.6667 |
| structural similarity | 0.1567 | 0.1297 |
| component type recall | — | 0.2500 |
| compiler fallbacks | 2 | 1 |
| seeded unconstrained fallbacks | 0 | 0 |
| p50 latency (ms) | 8194.70 | 3834.76 |

The type constraint materially improves lexical validity, but the model still
selects the wrong component for two smoke prompts and does not pass the
meaningful-program gate. This is now a data/supervision and semantic candidate
coverage question. More literal compiler patches or more training steps are
not justified by this evidence.

Evidence: [result JSON](iter-e172-schema-types-20260716.json), [eval JSON](../../outputs/runs/e172-schema-types-20260716/eval_smoke.json), and the AgentV JSONL path recorded in the result.
