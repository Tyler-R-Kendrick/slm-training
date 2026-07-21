# E637 — active Input role routing

Date: 2026-07-20
Status: completed neutral partial; r2 rejected; default-off; not ship

E637 narrows E632's failed global penalty to the active schema boundary. When
a required operational string is immediately followed by a public
`x-openui-placeholder` property, the schema-opaque treatment floors the legal
empty literal above visible slots. This targets `Input.name` without changing
component-family selection or unrelated strings.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. All three clean CPU evaluations
reused E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E631 OOD `n=4` recipe plus
`schema_opaque_decode_weight=4`. Every run completed under the three-minute cap
with no timeout or fallback and emitted AgentEvals JSONL plus an AgentV SDK
bundle without execution errors.

## Measured result

| OOD `n=4` | E631 baseline | E637 r1 | E637 r2 rejected | E637 r3 retained |
| --- | ---: | ---: | ---: | ---: |
| meaningful v1 | 0.7500 | 0.7500 | 0.5000 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fidelity / validity | 0.6750 / 0.8050 | 0.6750 / 0.8050 | 0.5083 / 0.7050 | 0.6750 / 0.8050 |
| structure / component recall | 0.5729 / 0.6250 | 0.5729 / 0.6250 | 0.3379 / 0.3750 | 0.5729 / 0.6250 |
| reward | 0.8515 | 0.8515 | 0.7850 | 0.8515 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.6357 / 0.5125 | 0.3857 / 0.2625 | 0.6357 / 0.5125 |
| latency p50 / p95 | 3025.78 / 6394.56 | 2910.21 / 6059.43 | 3424.65 / 15644.59 | 2680.25 / 5833.64 |
| timeout / fallback | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 | 0/1 | 0/1 |

R1 and the authoritative r3 change only the second Auth Input from
`Input(name-slot, email-slot)` to `Input("", email-slot)`. The first Input still
uses the email slot in `Input.name`, because the later repeated-instance margin
overrides the earlier schema score for that instance. Strict therefore remains
zero and all aggregate quality metrics are unchanged from E631.

R2 tried to restrict that repeated-instance margin to direct content
properties. It collapsed Auth to `TextContent(email)`, reducing meaningful v1
to 0.50 and regressing every continuous aggregate metric. Model v70 restores
v68 behavior; r3 reproduces r1 byte-for-byte on all four predictions.

## Decision

Retain v70 as a default-off, non-regressing partial intervention. Do not sync,
promote, or make a ship claim. Do not further constrain the broad repeated-slot
lever. The next experiment should compose the two scores at the final choice
boundary for only the pre-content schema pattern, or replace the competing
margins with one property-aware target calculation.

Evidence: [authoritative JSON](iter-e633-input-active-role-routing-20260720.json),
[r1 JSON](iter-e633-input-active-role-routing-r1-20260720.json), and
[rejected r2 JSON](iter-e633-input-active-role-routing-r2-20260720.json).
