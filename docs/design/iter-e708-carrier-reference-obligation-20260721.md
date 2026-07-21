# E708 — carrier-specific root-reference obligation

Date: 2026-07-21  
Status: completed five-suite retained scratch improvement; not ship

E708 narrows E707's useful Rico mechanism to the exact carrier created for one
missing visible slot. The decoder records that carrier's expected section index
and family, waits until the terminal `Stack` is about to close, then floors only
that reference. It abstains if the expected section was not actually emitted and
does not force unrelated orphan components.

The first Rico diagnostic fired before normal root assembly and regressed badly;
it is retained as negative evidence. Gating the obligation to the root-close
boundary recovered the intended carrier without disturbing existing references.
The final full-suite `n=19` replay is quality-identical to E701/E706 outside Rico.
On Rico, contract recall and fidelity improve 0.9583→1.0 versus E701, validity
0.9750→1.0, structure 0.7611→0.7915, node F1 0.7942→0.8419, and reward
0.9875→1.0. Edge F1 moves 0.7757→0.7727. The hardened r4 replay exactly matches
r3 on every tracked non-latency metric.

Retain v180 and the combined schema-value weight 5 recipe for this scratch path.
This is not ship evidence: Rico strict remains 0.0 on the independent
`TextContent.size` semantic-role mismatch, AgentV is 0/5, and the 190-token Rico
p95 exceeds the 160-token canvas. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e708-carrier-reference-obligation-20260721.json).
