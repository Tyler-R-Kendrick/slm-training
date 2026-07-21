# E658 — active positional-property role ownership

Date: 2026-07-21
Status: completed negative; reverted; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy and no unconstrained fallback. It emitted AgentEvals and
AgentV with no timeout or fallback. The treatment passed the full 115-test
compiler suite after preserving the established generic `Input.placeholder`
contract.

E658 gates both raw-slot margins by the active positional property's schema
role. Dashboard changes to `Slice(":ood.dash.status.body", 1)`, reusing an
already-covered slot in `Slice.category` and dropping the required metric.
Meaningful v1 and strict v2 hold at 1.0000 and 0.7500, but fidelity falls
1.0000→0.9500, validity 1.0000→0.9700, structure 0.7692→0.7513, reward
0.9730→0.9580, node F1 0.8222→0.8097, and edge F1 0.7102→0.6964. Reject v113
and restore E653 behavior as v114; soft margins cannot exclude the remaining
schema-role-invalid candidate. No checkpoint was created, synced, or promoted;
AgentV remained 0/1.

Evidence: [JSON](iter-e658-property-role-ownership-20260721.json).
