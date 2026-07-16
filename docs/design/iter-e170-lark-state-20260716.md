# E170 Lark parser-state compiler follow-up (2026-07-16)

E170 reran E169 after replacing source-text call scanning with the active
Lark `InteractiveParser` value stack. This makes schema-property lookup depend
on grammar-reduced call state rather than parentheses, quote, or component
string matching.

| Metric | E169 | E170 |
| --- | ---: | ---: |
| meaningful parse | 0.0000 | 0.0000 |
| syntax parse | 0.3333 | 0.3333 |
| structural similarity | 0.1567 | 0.1567 |
| compiler fallbacks | 2 | 2 |
| seeded unconstrained fallbacks | 0 | 0 |
| p50 latency (ms) | 13713.66 | 6329.78 |

The refactor is behavior-neutral for quality on this checkpoint. The next
implementation step is a reusable generated-AST/schema completion predicate
that distinguishes a complete semantic value from a still-legal partial
expression. Adding another literal-specific candidate exclusion is explicitly
out of scope.

Evidence: [result JSON](iter-e170-lark-state-20260716.json), [eval JSON](../../outputs/runs/e170-lark-state-20260716/eval_smoke.json), and the AgentV JSONL path recorded in the result.
