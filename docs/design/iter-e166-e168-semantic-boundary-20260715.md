# E166–E168 semantic boundary and metric separation (2026-07-16)

The evaluator reports both
`syntax_parse_rate` and `meaningful_program_rate`; the existing `parse_rate`
remains the ship-oriented meaningful-program gate for compatibility.

The first draft of this experiment included schema-aware scalar filtering and
a top-level document-boundary rule. Those two compiler rules were rejected as
overfit: they matched punctuation and observed examples rather than a formal
grammar/AST state. They are not part of the accepted compiler design or a
claim about the model.

| Iteration | Syntax parse | Meaningful program | Compiler fallback | Seeded fallback | p50 ms | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E166 | 0.0000 | 0.0000 | 3 | 0 | — | scalar filter exposed incomplete boundary |
| E167 | 0.6667 | 0.0000 | 0 | 0 | 917 | top-level boundary restored valid syntax |
| E168 | 0.6667 | 0.0000 | 0 | 0 | 819 | metric separation persisted |

The metric separation remains valid, but these runs do not establish semantic
correctness. The next compiler iteration must derive all candidate categories
from Lark parser state and generated schema/AST metadata; it must not add
literal-specific exclusions as new failures appear.

Evidence: [result JSON](iter-e166-e168-semantic-boundary-20260715.json) and
the E168 AgentEvals bundle under the run path listed there.
