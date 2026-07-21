# E646 — complete semantic-root reachability

Date: 2026-07-20
Status: completed neutral; reverted; not ship

E646 changed verified semantic-root construction to reference every completed
top-level element exactly once after required plan-family coverage, rather than
only the plan-counted sections. The hypothesis was that Dashboard's extra
slot-bearing TextContent made the planned root verifier reject an orphan.

One capped CPU OOD `n=4` run reused E620's rejected local-only checkpoint with
the exact E644 policy. It completed in 22.5 seconds without timeout or fallback
and emitted AgentEvals JSONL plus an AgentV SDK bundle without execution errors.

| OOD `n=4` | E644 baseline | E646 r1 |
| --- | ---: | ---: |
| meaningful v1 / strict v2 | 0.7500 / 0.7500 | 0.7500 / 0.7500 |
| fidelity / validity | 0.8500 / 0.9100 | 0.8500 / 0.9100 |
| structure / component recall | 0.6056 / 0.6875 | 0.6056 / 0.6875 |
| reward | 0.9115 | 0.9115 |
| AST node / edge F1 | 0.6778 / 0.4464 | 0.6778 / 0.4464 |
| latency p50 / p95 | 2963.90 / 13254.53 ms | 2906.33 / 12959.12 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

All predictions are byte-identical. The deterministic root still abstains, so
the extra-section reachability hypothesis is false. Reject the treatment
stamped v89. After rebasing onto retained E644 v91, the append-only lineage
records E646 as treatment v92 and restoration v93. The next diagnostic must expose the root verifier's exact
abstention reason before changing selection again. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e646-complete-root-reachability-20260720.json).
