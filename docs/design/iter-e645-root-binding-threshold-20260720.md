# E645 — semantic root-binding threshold

Date: 2026-07-20
Status: completed neutral; rejected; not ship

E645 tested whether E644's remaining Dashboard failure was caused by an
underweighted existing semantic-plan binding score. Two capped CPU OOD `n=4`
arms raised `semantic_plan_binding_decode_weight` from the E644 value of 1 to
4 and 8. Both reused E620's rejected local-only checkpoint and completed
without timeout, fallback, or AgentV execution error.

| OOD `n=4` | E644 w1 | E645 w4 | E645 w8 |
| --- | ---: | ---: | ---: |
| meaningful v1 / strict v2 | 0.7500 / 0.7500 | 0.7500 / 0.7500 | 0.7500 / 0.7500 |
| fidelity / validity | 0.8500 / 0.9100 | 0.8500 / 0.9100 | 0.8500 / 0.9100 |
| structure / component recall | 0.6056 / 0.6875 | 0.6056 / 0.6875 | 0.6056 / 0.6875 |
| reward | 0.9115 | 0.9115 | 0.9115 |
| AST node / edge F1 | 0.6778 / 0.4464 | 0.6778 / 0.4464 | 0.6778 / 0.4464 |
| latency p50 / p95 | 2963.90 / 13254.53 ms | 3031.05 / 14137.67 ms | 3105.93 / 14446.83 ms |
| timeout / fallback | 0 / 0 | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 | 0/1 |

All four predictions are byte-identical across the three policies. Higher
weights increased internal binding interventions but did not change the final
root, confirming that the verified root projection—not the binding score—is
the next decision boundary. Reject both E645 arms and retain E644 weight 1. The
runs were stamped model.twotower v88; after lineage rebase the retained E644
behavior is v91. No checkpoint was created, synced, or promoted.

Evidence: [w4 JSON](iter-e645-root-binding-w4-20260720.json) and
[w8 JSON](iter-e645-root-binding-w8-20260720.json).
