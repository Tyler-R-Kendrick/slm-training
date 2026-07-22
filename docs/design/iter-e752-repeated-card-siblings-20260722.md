# E752 — repeated Card sibling topology

**Date:** 2026-07-22  
**Decision:** retain model v213; no checkpoint promotion  
**Evidence:** [`iter-e752-repeated-card-siblings-20260722.json`](iter-e752-repeated-card-siblings-20260722.json)

E752 replays E751 on the same local CPU checkpoint, immutable symbol-only data,
and lever values. Model v213 removes inline role metadata from component
cardinality and treats repeated families as siblings unless the prompt
explicitly declares an outer-group nesting relation.

The six-deep Card chain disappears. Outputs contain sibling Cards with
TextContent children, component recall remains 1.0, and p95 falls from 6974 to
4634 ms. This is not accepted as complete: the root array closes after two of
five planned Cards because the semantic-plan margin does not reach the lexer
`,` versus `]` decision. Fidelity falls to 0.4545, structure to 0.5025,
strict-v2 stays zero, and AgentV stays 0/1.

The next change extends the existing plan margin across that delimiter boundary
instead of adding a tuning value. No checkpoint was created or synced. All
predictions remain symbol-only with declared template markers and no free-form
target text.
