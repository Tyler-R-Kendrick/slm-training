# E48 structural newline guard — 2026-07-15

After DFA resynchronization, generation reached valid root and binding
structure but emitted repeated newlines inside unfinished lists and literals.
The picker now rejects newline while delimiters or quoted literals remain open.

Focused tests passed: 24. The matched smoke result remained parse 0/3 and
structural similarity moved from 0.3744 to 0.3633, with p50 latency 35.9s.

The guard is retained as a grammar-safety invariant, but it is rejected as a
quality intervention. The next work should address native symbol/literal
representation and model confidence rather than add more newline heuristics.

This is scratch smoke evidence, not a ship claim.
