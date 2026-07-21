# E701 — schema-aware role capacity

Date: 2026-07-21  
Status: completed retained scratch improvement; not ship

E701 fixes the runtime gap exposed after E700. A visible-role binding was
considered satisfied when some planned component could merely reach its carrier,
even when no carrier instance was counted. The retained v171 rule materializes a
replacement only when a directly compatible planned carrier is capacity-exhausted,
abstains when a role has no name-matched direct property (for example enum-guided
Input selection), and preserves ordinary carriers beneath explicitly typed
containers.

The matched v167 control and v171 treatment each evaluated all five committed
suites (`n=19`) with constrained-only decode, no timeout or fallback, and
AgentEvals plus an AgentV SDK bundle. Every tracked quality metric is identical
on Smoke, adversarial, OOD, and Rico. Held-out improves across every tracked
quality metric:

| Held-out `n=5` | v167 control | v171 |
| --- | ---: | ---: |
| strict v2 / coverage | 0.8000 / 1.0000 | 1.0000 / 1.0000 |
| fidelity / validity | 0.9600 / 0.9760 | 1.0000 / 1.0000 |
| structure / component recall | 0.7724 / 0.8433 | 0.8104 / 0.8933 |
| reward | 0.9514 | 0.9658 |
| AST node / edge F1 | 0.8609 / 0.6888 | 0.8831 / 0.7174 |
| timeout / fallback | 0 / 0 | 0 / 0 |

Two of 19 predictions change. `held_out_form_01` gains the missing
`TextContent(:held.form.title)` while preserving the correct Button, Input, and
Callout hint pair. `adv_deep_nest_01` re-coheres the inner title/body namespace
under one Callout and leaves all adversarial aggregate metrics unchanged.

V168 left reachable bindings advisory, v169 over-materialized nested carriers,
and v170 still treated enum-only roles as direct-property capacity failures.
Those arms are rejected. Retain v171, but do not claim ship readiness: AgentV is
0/5, adversarial and Rico strict rates remain below readiness, and Rico's p95
length requirement (190) exceeds the 160-token evaluation canvas. No checkpoint
was created, synced, or promoted; latency movement is not a performance claim.

Evidence: [JSON](iter-e701-schema-aware-role-capacity-20260721.json).
