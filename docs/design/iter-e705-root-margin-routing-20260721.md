# E705 — root-margin slot routing

Date: 2026-07-21  
Status: completed partial-suite negative; rejected; not ship

E705 combines E704's schema-value weight 5 with semantic-plan root margin 0 to
test whether the missing Rico placeholder can route into another compatible
carrier instead of being omitted. The bounded Rico-only diagnostic evaluated
all three committed records under constrained-only decode with no timeout or
fallback and emitted AgentEvals plus an AgentV SDK bundle.

The arm fails before a full-suite replay is warranted. Strict meaning remains
0.0; fidelity falls 0.8750→0.8333, structure 0.7611→0.5098, and reward
0.9625→0.9170 relative to E704 weight 5. Decode continues into duplicate
TextContent, Tag, Modal, and AccordionItem subtrees, raising p95 latency to
24.6 seconds rather than producing one bounded compatible carrier.

Reject the arm and retain root margin 2. This is partial-suite scratch evidence,
not a ship evaluation; AgentV is 0/1 and the Rico length budget still fails. No
checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e705-root-margin-routing-20260721.json).
