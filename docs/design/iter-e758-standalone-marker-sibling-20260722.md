# E758 — standalone marker sibling

**Date:** 2026-07-22  
**Decision:** retain model v218; no checkpoint promotion  
**Evidence:** [`iter-e758-standalone-marker-sibling-20260722.json`](iter-e758-standalone-marker-sibling-20260722.json)

The semantic-plan harness now reserves declared marker namespaces against
reachable structural component capacity. Five planned Cards can own five
independent namespaces; a sixth namespace must receive a direct compatible
sibling. The repeated-instance decoder uses the same namespace grouping
primitive. This fixes the harness invariant without a phrase-specific parser,
UI filtering, duplicated ownership rule, or new lever.

The matched local RICO n=3 replay completed under the 110-second command cap
with no timeout or fallback. Compared with E757, fidelity improved from 0.9394
to 1.0, validity from 0.9637 to 1.0, structure from 0.9639 to 1.0, reward from
0.9188 to 0.9370, and strict-v2 from 0.3333 to 1.0. Both five-Card rows now
contain their standalone `TextContent` sibling and all five correctly paired
Cards. The three-Card row remains exact on structure and markers.

AgentV remains 0/1 because this is a three-record scratch diagnostic, not a
ship evaluation. No checkpoint was created or synced. Every completion uses
only grammar/AST symbols, schema enum literals, and declared template markers;
free-form output strings remain zero.
