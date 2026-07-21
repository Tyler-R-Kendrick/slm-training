# E635 — property-compatible slot coverage

Date: 2026-07-20
Status: completed positive mixed; confirmed; default-off; not ship

E634 showed that final-boundary `Input.name` routing alone caused a still-missing
slot to move into optional `Input.type`, after which section/root generation ran
to the canvas cap. E635 composes two public-schema invariants: final pre-content
literal routing and direct coverage slots only at placeholder-annotated active
component properties. Object-property and structural continuation behavior is
unchanged.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. Two clean CPU evaluations reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E633 OOD `n=4` recipe. Both completed under the three-minute cap
with no timeout or fallback and emitted AgentEvals JSONL plus AgentV SDK bundles
without execution errors.

## Measured result

| OOD `n=4` | E633 r3 baseline | E635 r1 | E635 r2 confirmation |
| --- | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.2500 | 0.2500 |
| v2 judgment coverage | 1.0000 | 1.0000 | 1.0000 |
| placeholder fidelity / validity | 0.6750 / 0.8050 | 0.6750 / 0.8050 | 0.6750 / 0.8050 |
| structure / component recall | 0.5729 / 0.6250 | 0.5729 / 0.6250 | 0.5729 / 0.6250 |
| reward | 0.8515 | 0.8515 | 0.8515 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.6357 / 0.5125 | 0.6357 / 0.5125 |
| latency p50 / p95 | 2680.25 / 5833.64 ms | 2227.25 / 6228.65 ms | 2222.19 / 6095.12 ms |
| timeout / fallback | 0 / 0 | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 | 0/1 |

Both E635 runs are byte-identical on all four predictions. Auth becomes
`Stack([Button(create), Input("", name), Input("", email)], "column")`, has
no strict-v2 reason codes, and remains 1.0 on all continuous per-record
metrics. Dashboard, Gallery, and Modal are unchanged. The suite's strict-v2
rate therefore rises from 0/4 to 1/4 without any aggregate continuous-quality
regression. Latency is descriptive only at this sample size.

## Decision

Retain model v73 as a default-off scratch policy. Do not sync, promote, or make
a ship claim: the partial OOD suite and AgentV still fail, while Dashboard,
Gallery, and Modal remain unresolved. The next experiment should target
Modal's property-role mismatch using the same active-property compatibility
principle, without changing the now-correct Auth path.

Evidence: [authoritative r2 JSON](iter-e635-property-compatible-coverage-20260720.json)
and [r1 JSON](iter-e635-property-compatible-coverage-r1-20260720.json).
