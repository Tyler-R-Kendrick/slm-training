# DSH3-01 compiler-owned operator contracts

SLM-369 defines the model-independent identity and evidence layer for later
CAP2 operator work. It does not add an operator registry, executable rewrite,
model token, learned head, corpus, checkpoint, or training claim.

## Contract boundary

`src/slm_training/dsl/operators/contracts.py` owns four immutable schemas:

| Schema | Purpose |
| --- | --- |
| `ast_operator/v1` | Stable operator declaration: domain/codomain, typed argument slots and binding phases, preconditions, declared delta classes, inverse/commutativity/idempotence metadata, locality, and cost. |
| `action_effect/v1` | Typed consumed/produced roles and binders plus scope, cardinality, property, and topology deltas, compiler coverage, and estimated completion cost. |
| `operator_application/v1` | Before/after state and AST digests, bound opaque arguments, exact effect, proof or typed rejection, and compiler/source provenance. |
| Opaque references | Request-local `NodeRef`, `RoleRef`, `IndexRef`, `ValueRef`, `SymbolRef`, and `TemplateRef` surfaces. |

Canonical JSON with sorted set-like fields produces SHA-256 declaration,
effect, and application identities. Argument order remains declarative for an
operator but bound application arguments are keyed by slot. No display text,
token ID, user name, object address, or learned vector is part of identity or
legality.

## Fail-closed rules

- Every argument slot declares one reference kind and one compiler binding
  phase (`request`, `state`, or `application`).
- Preconditions may reference only declared slots.
- Opaque references in one application must share its provenance request ID.
- Successful applications require both after digests, a typed effect, and a
  proof bound to the exact effect fingerprint.
- Rejections cannot claim an after state or effect.
- All state, AST, source-artifact, compiler-result, operator, and effect
  identities are full lowercase SHA-256 digests.

The pack-owned pure apply/dry-run interface remains intentionally deferred to
DSH3-02. Typed reference derivation and permutation invariance beyond the
request-local surface remain DSH3-03.

## Verification

`tests/test_dsl/test_operator_contracts.py` covers canonical declaration
identity, model/display independence, typed slots and binding phases, opaque
request locality, delta-bucket typing, success provenance, proof/effect
binding, deterministic rejection, and cross-request rejection.

This is schema/unit evidence only: no train, eval, benchmark, profile, matrix,
checkpoint, or model-card change was produced.
