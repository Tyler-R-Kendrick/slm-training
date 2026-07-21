# E653 — nested planned-container role ownership

Date: 2026-07-21
Status: completed positive scratch result; retained; not ship

E653 reuses the existing schema-reachability helper: when a planned component
can contain a visible role, the compatible leaf is bound without increasing
top-level family cardinality. This targets E652's detached metric leaves.

Both capped CPU OOD `n=4` runs reused E620's rejected local-only checkpoint and
emitted AgentEvals JSONL plus AgentV bundles. R1 allowed unconstrained fallback
and is excluded from the causal comparison despite observing zero fallbacks.
Matched r2 disabled fallback and also completed without timeout or fallback.

| OOD `n=4` | E650 baseline | E653 r2 |
| --- | ---: | ---: |
| meaningful v1 / strict v2 | 1.0000 / 0.7500 | 1.0000 / 0.7500 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.7355 / 0.8750 | 0.7692 / 0.8750 |
| reward | 0.9790 | 0.9730 |
| AST node / edge F1 | 0.7987 / 0.5798 | 0.8222 / 0.7102 |
| latency p50 / p95 | 2844.86 / 7879.73 ms | 2843.47 / 7045.22 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Dashboard now nests both metric roles under the two planned Cards instead of
emitting detached leaves or empty Cards. Retain v104: structure, both AST F1
metrics, and p95 improve while all hard quality metrics hold. Reward slips by
0.006, and Dashboard still has Callout/Carousel semantic-role mismatches, so
this diagnostic subset is not ship evidence. No checkpoint was created,
synced, or promoted.

Evidence: [authoritative r2 JSON](iter-e653-nested-role-ownership-20260721.json)
and [excluded r1 JSON](iter-e653-nested-role-ownership-r1-20260721.json).
