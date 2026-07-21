# E672 — schema-owned typed array item

Date: 2026-07-21
Status: completed neutral scratch; rejected; not ship

One capped CPU OOD `n=4` evaluation-only scratch run reused E620's rejected
local checkpoint under the exact E653 policy. It emitted AgentEvals and AgentV
with no timeout or fallback after all 118 compiler tests passed.

E672 allows an active typed-array schema that can reach a missing visible slot
to start its minimal item without requiring an authored semantic-plan owner.
All prediction hashes and authoritative metrics remain identical to E671:
strict v2 is 0.7500, structure 0.7230, and AgentV 0/1.

Reject neutral v128 and restore retained E666 behavior as v129. Direct
inspection shows Carousel's minimal item is `TextContent`, but its real item
schema is an `anyOf` of public `$ref` entries; the slot-reachability predicate
does not resolve those references and still abstains. No checkpoint was
created, synced, or promoted.

Evidence: [JSON](iter-e672-schema-owned-array-20260721.json).
