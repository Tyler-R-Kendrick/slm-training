# E753-E754 — semantic-plan delimiter continuity

**Date:** 2026-07-22  
**Decision:** retain model v214; no checkpoint promotion  
**Evidence:** [`iter-e753-e754-plan-margin-delimiter-20260722.json`](iter-e753-e754-plan-margin-delimiter-20260722.json)

Model v214 carries the existing semantic-plan margin across the lexer
`,` versus `]` decision when sibling component requirements remain. It adds no
new lever or tuning value. The same plan contract now governs both component
selection and the delimiter needed to reach the next component.

E753 used an 8-second per-record diagnostic guard and timed out on one row. It
is documented but is not evidence. E754 changed only that guard to 12 seconds;
the command still used the canonical 110-second interrupt and two-minute total
cap. E754 completed with zero timeouts and zero fallbacks.

Compared with E752, E754 restores all five sibling Cards on the five-Card rows:
parse 1.0, fidelity 0.8788, validity 0.9273, structure 0.8901, component recall
1.0, and reward 0.9007. Strict-v2 remains zero because markers are distributed
unevenly and the three-Card row still emits two Cards. AgentV remains 0/1, so
this is not a ship result. No checkpoint was created or synced. Every output is
limited to grammar/AST symbols, schema enum literals, and declared markers.
