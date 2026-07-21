# E678 — joint-role provenance correction

Date: 2026-07-21
Status: completed neutral provenance correction; retained; not ship

E676 placed a model-only role-grouping helper in `data.quality`, which is part
of binding-aware meaningful v2's implementation hash. That mechanically
changed the hash without a metric-version change. E678 moves the helper into
TwoTower, restoring v2.3.0's original hash while leaving decode logic intact.

The capped OOD `n=4` replay completed with AgentEvals and AgentV. All four
predictions and every quality metric are identical to E676. The metric hash is
restored from `a65e…f65be` to v2.3.0's `8253…89b55`. Retain v134 as the
provenance-correct E676 implementation, not new quality or ship evidence.
AgentV remains 0/1, and no checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e678-joint-role-provenance-20260721.json).
