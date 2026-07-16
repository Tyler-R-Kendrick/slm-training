# E105 required-argument grammar guard (2026-07-15)

E105 added a narrow semantic check at component close-parens: reject
`TextContent()` when the parser reports its required `text` field missing.
This targets the malformed `TextContent()` observed in E104 without rejecting
valid empty `Stack([])` expressions or broad incomplete prefixes.

Focused grammar tests passed (`28`). Re-evaluating the unchanged E104
checkpoint under strict no-fallback decoding produced parse/raw syntax
`0.0/0.0`, structural similarity `0.425` (baseline `0.4208`), contract
precision/recall `1.0/1.0`, placeholder fidelity `1.0`, component recall
`0.25`, and latency `9676.63 ms`. AgentV had 5 failed checks.

Decision: retain the narrow invariant and regression test, but reject E105 as
insufficient for ship. The remaining failure is broader Stack-list generation,
not only zero-argument components.
