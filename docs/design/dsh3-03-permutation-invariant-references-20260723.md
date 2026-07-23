# DSH3-03 permutation-invariant operator references

SLM-371 binds operator arguments to inference-visible compiler facts without
using user-chosen names, object addresses, or semantically meaningful public
ordinals. It extends the DSH3-01 opaque reference surfaces with one immutable,
state- and branch-bound descriptor table.

## Identity layers

1. `persistent_node_fingerprint` hashes canonical structural identity, the
   persistent parent fingerprint, and an opaque branch-local disambiguator.
2. `branch_local_disambiguator` hashes an internal same-structure collision
   index. The index never appears in the reference surface or descriptor.
3. `ReferenceDescriptorV1` exposes only typed compiler facts: reference kind,
   semantic fingerprint, value type, stable fact IDs, and—only for an
   `IndexRef`—the current parent fingerprint/order digest and relative
   position. Fact IDs come from the closed `CompilerFact` enum; arbitrary
   strings are rejected rather than becoming a user-data side channel.
4. `ReferenceTableV1` binds descriptors and optional surface-free runtime
   symbol descriptors to exactly one request, state digest, and branch digest.
5. Opaque IDs are hashes of request/seed/descriptor identity. Allocation sorts
   descriptors before deterministic shuffling, so input/candidate order does
   not assign semantic meaning.

The table serializes alongside runtime-symbol fingerprints and the closed
`RuntimeSymbolRole` enum, not original display surfaces or arbitrary role text.

## Resolution and stable failures

Resolution checks request, state, branch, expected type, unique membership,
and current ordered-parent identity in that order. Stable codes are:

| Code | Meaning |
| --- | --- |
| `ref.cross_request` | Reference belongs to another request. |
| `ref.stale_state` | Table/reference is being used against a changed state. |
| `ref.cross_branch` | Reference is being used on another branch. |
| `ref.type_incompatible` | Reference kind does not match the argument slot. |
| `ref.missing` | Opaque ID is not present in the bound table. |
| `ref.duplicate` | Table contains an ambiguous kind/ID pair. |
| `ref.index_context_required` | Index resolution omitted current parent order. |
| `ref.stale_index` | A conflicting edit changed the ordered parent. |
| `ref.runtime_symbol_missing` | Runtime-symbol descriptor names no table descriptor. |

No stale or cross-branch resolution returns a semantic target, so it cannot
reach the SLM-370 apply path.

## Permutation evidence

`ReferenceTableV1.permuted(seed)` assigns new opaque IDs and candidate order
while preserving descriptors. Unit fixtures show that the original and
permuted references resolve to the same semantic fingerprint and therefore
the same canonical fixture result. Separate tests cover alpha-normalized
structural identity, descriptor-input order invariance, stale/cross-branch/
missing/type/duplicate failures, ordered-parent invalidation, surface-free
runtime-symbol serialization, and JSON round trip.

This is contract/unit evidence only. It adds no production operator, corpus,
train, eval, checkpoint, model card, or CAP2 capability claim.
