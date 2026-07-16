# E169 grammar-derived compiler audit (2026-07-16)

E169 re-ran the E160 lexer-native checkpoint on the same smoke set after
removing the compiler's literal-specific punctuation/component filters.

| Metric | E169 |
| --- | ---: |
| meaningful parse | 0.0000 |
| syntax parse | 0.3333 |
| structural similarity | 0.1567 |
| compiler fallbacks | 2 |
| seeded unconstrained fallbacks | 0 |
| timeouts | 0 |
| p50 latency (ms) | 13713.66 |

The result confirms that the grammar constraint is active, but grammar
reachability alone permits valid partial expressions to continue past a
complete component expression. The resulting canvas can be syntactically
incomplete even without an unconstrained fallback. This is a semantic
completion-state problem, not evidence for more training steps.

The next hypothesis is to derive completion boundaries and candidate types from
the generated AST/schema and Lark parser state. New fixes must be expressed as
reusable grammar/schema rules and tested as invariants; exact component names
or punctuation bans are not acceptable.

Evidence: [result JSON](iter-e169-grammar-derived-20260716.json), [eval JSON](../../outputs/runs/e169-grammar-derived-20260716/eval_smoke.json), and the AgentV JSONL path recorded in the result.
