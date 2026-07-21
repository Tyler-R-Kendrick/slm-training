# E647 — semantic-root abstention trace

Date: 2026-07-20
Status: completed diagnostic; retained telemetry; not ship

E647 added bounded, behavior-neutral evidence when the deterministic semantic
root verifier rejects a planned closure. The trace records only error type,
short message, token/section/reference counts, and structural frame metadata.

Two capped CPU OOD `n=4` runs reused E620's rejected local-only checkpoint with
the exact E644 policy. Both completed without timeout or fallback, emitted
AgentEvals JSONL plus AgentV SDK bundles, and reproduced all E644 predictions
and quality metrics exactly. Initial r1 proved the signal but emitted 51 records
as section counts changed; r2 deduplicated by stable error signature and emitted
one record.

| OOD `n=4` | E644 baseline | E647 r1 | E647 r2 |
| --- | ---: | ---: | ---: |
| meaningful v1 / strict v2 | 0.7500 / 0.7500 | 0.7500 / 0.7500 | 0.7500 / 0.7500 |
| fidelity / validity | 0.8500 / 0.9100 | 0.8500 / 0.9100 | 0.8500 / 0.9100 |
| structure / component recall | 0.6056 / 0.6875 | 0.6056 / 0.6875 | 0.6056 / 0.6875 |
| reward | 0.9115 | 0.9115 | 0.9115 |
| latency p50 / p95 | 2963.90 / 13254.53 ms | 2822.76 / 12860.35 ms | 2771.18 / 13031.29 ms |
| abstention records | — | 51 | 1 |
| timeout / fallback | 0 / 0 | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 | 0/1 |

The single r2 record occurs at Dashboard position 35 with five completed
sections and five planned references: `ParseError: unknown choice token:
LIT_STR`. The root verifier calls the production choice decoder on the model's
dynamic literal marker protocol, so it abstains before root scoring. Retain the
telemetry stamped v92. After rebasing onto E646 restoration v93, the append-only
lineage records E647 instrumentation v94 and deduplicated telemetry v95. The
next repair must normalize dynamic literal markers for this verification probe
without changing emitted choices. No checkpoint was created, synced, or
promoted.

Evidence: [authoritative r2 JSON](iter-e647-root-abstention-trace-20260720.json)
and [r1 diagnostic JSON](iter-e647-root-abstention-trace-r1-20260720.json).
