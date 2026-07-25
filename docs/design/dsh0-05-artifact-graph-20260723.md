# DSH0-05 derivation DAG and leakage firewall (SLM-349)

**Disposition:** adopt the append-only `artifact_graph_sidecar/v1` and
`root_family_split/v1` contracts. This is a fixture contract, not a data build,
train/eval/benchmark, checkpoint, certificate, or ship claim.

Machine-readable evidence:
[`dsh0-05-artifact-graph-20260723.json`](dsh0-05-artifact-graph-20260723.json).

## Split and lineage boundary

The root family is hashed into `train`, `validation`, or `test` before
expansion. Every descendant must retain the root as its `split_group_id` and
the assigned split. A composition whose parents occupy incompatible splits
fails before it can be written.

`ArtifactGraphStore` writes immutable records below a dataset's
`artifact_graph/records/` sidecar. The artifact content/activity identity is
the filename. Identical reruns are no-ops; a different payload at an existing
identity is a collision. Parent traversal is deterministic and complete.

## Leakage and quarantine

Every candidate is compared across splits for:

- root family;
- any shared source parent;
- exact surface digest;
- alpha-equivalent digest;
- canonical AST digest;
- bounded near-template signature.

A blocking candidate is never silently dropped. Its full proposed artifact,
typed reason codes, and all overlap evidence are persisted under
`artifact_graph/quarantine/`. Unresolved parents are quarantined likewise.
The explain CLI reads the same sidecars and emits either concise text or JSON
for every remaining overlap candidate.

Seven focused fixtures cover deterministic assignment, inheritance, idempotent
append, ancestry, mixed-split composition rejection, multi-reason quarantine,
shared-parent explanation, and clean/quarantine CLI output. The
artifact-contract, integrity, and staged suites bring the integrated total to
32 passing tests.

## Next disposition

The synthesis-plan runner may call this store while materializing staged
artifacts. Existing non-staged dataset builds remain unchanged.
