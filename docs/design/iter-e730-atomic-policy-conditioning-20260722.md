# E730 atomic policy conditioning preservation

**Date:** 2026-07-22  
**Decision:** keep decode safety atomic while preserving checkpoint conditioning  
**Evidence:** [`iter-e730-atomic-policy-conditioning-20260722.json`](iter-e730-atomic-policy-conditioning-20260722.json)

## Finding

The first atomic `strict_compiler_tree` preset coupled decode safety to two
input-conditioning levers. It silently changed E723 from
`schema_in_context=false` and `slot_contract_in_context=false` to both true.
That is not a stricter evaluation of the same checkpoint; it is a different
model input.

The canonical model-build policy now owns only the invariants that define a
strict decode: grammar-constrained left-to-right tree decode, final validation,
constrained slot references, the honest slot contract, and no unconstrained
fallback. Schema, slot-contract, semantic-role, and DESIGN.md conditioning
remain checkpoint/experiment inputs and survive policy normalization unchanged.
Focused config and decode-path tests lock this separation.

## Local smoke evidence

Both completed arms reuse E723's symbol-only checkpoint SHA
`787d2d21d7c29d56637355fd364f16a0d67b1f452fc0f4ce3a7d486b2bd62795`
on the same three-record smoke suite SHA
`9cf9ab46201b79e787dca445035f7d5cda2fa0c835e5d33dc38eca88db900e1c`.
They use local CPU, one attempt, an eight-second record timeout, and a
two-minute cumulative cap. No arm timed out.

| Policy state | Schema / slot context | Parse | Meaning-v1 | Strict-v2 | Fidelity | Structure | Recall | Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v39 regressed preset | on / on | 1.0000 | 0.3333 | 0.0000 | 0.0000 | 0.1353 | 0.1667 | 0.6070 |
| v40 corrected, clean | off / off | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 |

The v39 preset suppresses every slot-owner application and collapses outputs
toward an empty root list. The clean v40 run restores three slot-owner
applications and changes, exactly reproducing E723's accepted quality metrics.
AgentV remains 0/1, strict-v2 remains 0.0, and no checkpoint was created,
uploaded, promoted, or claimed shippable. A completed dirty-tree v40 run is
disclosed in JSON but excluded from accepted evidence.

## Disposition

This fixes the harness and central policy boundary; it does not hide the
regression or reinterpret it as model progress. Future changes to strict decode
behavior remain centralized in `eval_policy.py`, while input-conditioning
experiments remain explicit checkpoint levers. E723 remains the training
baseline for the next local structural-closure experiment.
