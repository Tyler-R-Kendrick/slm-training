# E755-E757 — repeated marker ownership

**Date:** 2026-07-22  
**Decision:** retain model v217; no checkpoint promotion  
**Evidence:** [`iter-e755-e757-repeated-marker-ownership-20260722.json`](iter-e755-e757-repeated-marker-ownership-20260722.json)

The existing `semantic_plan_margin_decode_weight=2` now owns the complete
marker namespace for each repeated lexer component instance. No new lever or
weight was added; the legacy repeated-slot-specific weight remained zero.

E755 was byte-equivalent to E754 because compiler-tree path ranking did not
invoke the ownership scorer. E756 wired it into that path but incorrectly kept
ownership active after a Card closed, collapsing the parent Stack to one Card.
Both valid negative results are retained. Model v217 scopes ownership to open
component calls and fixes that regression.

E757 completed locally under the 110-second command cap with no timeout or
fallback. RICO n=3 improved from E754 fidelity 0.8788 / structure 0.8901 /
strict-v2 0.0 to fidelity 0.9394 / structure 0.9639 / strict-v2 0.3333. The
three-Card row now has three sibling Cards with its six markers correctly
paired. The two five-Card rows have five correctly paired Cards but omit their
standalone text marker, which is the next accepted-path defect. AgentV remains
0/1, so this is not a ship result. No checkpoint was created or synced. All
outputs contain only grammar/AST symbols, schema enum literals, and declared
template markers.
