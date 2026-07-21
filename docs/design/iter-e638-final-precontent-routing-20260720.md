# E638 — final-boundary pre-content routing

Date: 2026-07-20
Status: completed negative; rejected and reverted; not ship

E638 split E637's pre-content literal score from optional opaque handling and
applied it after the repeated-slot margin. This tested whether the remaining
Auth role mismatch was only an ordering conflict at the final local choice.

No training ran and no checkpoint was created. The clean CPU OOD `n=4` eval
reused E620's rejected local checkpoint (SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`)
with the exact E637 recipe. It completed under the three-minute cap with no
timeout/fallback and emitted AgentEvals plus AgentV evidence.

| OOD `n=4` | E637 r3 | E638 |
| --- | ---: | ---: |
| meaningful v1 / strict v2 | 0.7500 / 0.0000 | 0.5000 / 0.0000 |
| fidelity / validity | 0.6750 / 0.8050 | 0.5083 / 0.7050 |
| structure / component recall | 0.5729 / 0.6250 | 0.3379 / 0.3750 |
| reward | 0.8515 | 0.7850 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.3857 / 0.2625 |
| latency p50 / p95 | 2680.25 / 5833.64 ms | 2938.12 / 16254.73 ms |
| AgentV | 0/1 | 0/1 |

Auth collapsed from the complete Stack/Button/two-Input inventory to
`TextContent(email)`, while the other three records stayed unchanged. This
matches E637 r2 and disproves the local-ordering hypothesis: forcing both
operational literals changes downstream section/root selection.

Reject v71 and restore E637's non-regressing behavior as v72. Do not sync,
promote, or make a ship claim. The next attempt must score Input property roles
together with section retention/root reachability, not impose another local
argmax override.

Evidence: [JSON](iter-e634-final-precontent-routing-20260720.json).
