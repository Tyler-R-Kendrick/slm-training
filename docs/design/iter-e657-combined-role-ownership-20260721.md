# E657 — combined raw-slot role ownership

Date: 2026-07-21
Status: completed negative; reverted; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy and no unconstrained fallback. It emitted AgentEvals and
AgentV with no timeout or fallback.

E657 combines the independently neutral E655 and E656 constraints across both
raw-slot forcing branches. It changes Dashboard's nested metric from a raw
Carousel child to `Slice(":ood.dash.m1.value", 1)`, but the placeholder lands
in `Slice.category`, so strict v2 remains 0.7500 and the same semantic failures
remain. Fidelity, validity, recall, and reward hold, while structure regresses
0.7692→0.7513, node F1 0.8222→0.8097, edge F1 0.7102→0.6964, p50
2827.87→3176.54 ms, and p95 7555.34→8127.30 ms versus the immediately matched
E655 timing. Reject v111 and restore retained E653 behavior as v112. The next
repair needs property-level role compatibility before choosing the nested leaf.
No checkpoint was created, synced, or promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e657-combined-role-ownership-20260721.json).
