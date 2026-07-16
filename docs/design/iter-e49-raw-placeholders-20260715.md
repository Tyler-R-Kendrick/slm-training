# E49 raw placeholder representation — 2026-07-15

E49 disabled lexer symbol-table substitution while keeping the judged Silver+
corpus and 256-step scratch recipe matched to E47/E48. This tests whether
repeated native symbols and premature literal boundaries are caused by the
symbol-table representation itself.

The result regressed sharply: parse, structural similarity, fidelity, and
reward were all zero. Predictions became unstructured token streams such as
`root Stack TextContent ...`, with p50 latency 31.8s.

Decision: reject raw placeholder spelling. Keep symbol tables and focus the
next intervention on native `<SYM_*>` transition supervision/decoding.

This is scratch smoke evidence, not a ship claim.
