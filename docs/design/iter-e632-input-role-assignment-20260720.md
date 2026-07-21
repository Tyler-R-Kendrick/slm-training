# E632 — broad non-content string slot penalty

Date: 2026-07-20
Status: completed negative; rejected and reverted; not ship

E631 produced the correct Auth component inventory but placed visible field
slots in both `Input.name` and `Input.placeholder`. E632 tested whether the
existing schema-opaque score could be generalized from optional unconstrained
arguments to every non-content string property. The treatment penalized visible
slots in string properties without `x-openui-placeholder` or an enum.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. The clean CPU evaluation reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E631 OOD `n=4` recipe plus
`schema_opaque_decode_weight=4`. It completed under the three-minute cap with
no timeout or fallback and emitted AgentEvals JSONL plus an AgentV SDK bundle
without execution errors.

## Measured result

| OOD `n=4` | E631 baseline | E632 treatment |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.0000 |
| placeholder fidelity | 0.6750 | 0.5917 |
| placeholder validity | 0.8050 | 0.7550 |
| structural similarity | 0.5729 | 0.4704 |
| component recall | 0.6250 | 0.5000 |
| reward | 0.8515 | 0.8205 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.5524 / 0.3875 |
| latency p50 / p95 | 3025.78 / 6394.56 ms | 3783.53 / 11959.96 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The penalty did prevent the original bad Input-name continuation, but it did so
at the wrong decision boundary. Auth lost both Inputs and became a Stack with a
Button plus raw repeated placeholders, including placeholder spam. All
continuous aggregate quality metrics regressed, strict v2 stayed at zero, and
p95 latency increased by 5.57 seconds. The other three predictions were
unchanged.

## Decision

Reject the treatment and restore E631's optional-empty-only behavior as model
v67; v66 remains in history as negative evidence. Do not sync, promote, or make
a ship claim. A follow-up must preserve Input family selection and intervene
only after an Input is active: assign an operational literal to `Input.name`
and reserve the authored visible slot for `Input.placeholder`, using public
schema roles rather than globally suppressing slot tokens.

Evidence: [JSON](iter-e632-input-role-assignment-20260720.json).
