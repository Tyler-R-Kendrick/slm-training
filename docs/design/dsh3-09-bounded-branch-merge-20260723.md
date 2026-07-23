# DSH3-09 bounded branch merge (SLM-377)

Date: 2026-07-23
Status: implemented repository contract; no train/eval/benchmark run
Scope: CAP2 compiler-owned AST operators

## Decision

Branch merge is a conservative, replayable three-way decision over one verified
operator edge from each of two distinct forks of the same immutable base state.
It is not a general conflict resolver.

The contract returns either:

- a compiler-valid `BranchMergeArtifactV1`, or
- one immutable `BranchMergeConflictV1` with a typed conflict class.

It never selects a branch, edits a target, or resolves a conflict heuristically.

## Accepted boundary

Each `BranchEditV1` binds its exact input node, output node, and successful
`OperatorApplicationV1`. Before merge, the implementation:

1. requires both branch inputs to contain the exact base `OperatorStateV1`;
2. requires distinct branch identities;
3. resolves the pack-owned library for each immutable branch input;
4. replays both recorded applications and compares their exact output states;
5. requires exact compiler effect coverage and at least one typed effect target;
6. maps fork-local opaque effect refs back to their base descriptor
   fingerprints using the deterministic fork lineage transform.

Stale or unmappable refs fail closed. The opaque ref itself is never treated as
semantic identity.

## Auto-merge

The implementation parses base, left, and right through the owning pack and
performs a structural three-way composition:

- identical values remain identical;
- a value changed by only one branch adopts that exact change;
- independently changed dataclass fields, mapping keys, or fixed sequence
  members compose;
- concurrent deletion/modification, incompatible leaf replacement, and
  concurrent sequence-shape changes refuse.

The pack serializes the composed AST and ordinary `OperatorStateV1.from_source`
authority revalidates parse/serialize/canonicalization, schema/static oracle,
scope extraction, and property ordering. A failed authority check returns
`unsupported_effect`; it never emits a state.

Disjoint base targets are eligible. Overlapping targets require symmetric
`commutes_with` declarations and must still compose without a structural
conflict. A declaration is permission to attempt exact composition, not
permission to choose an outcome.

## Typed conflicts

| Type | Conservative trigger |
| --- | --- |
| `same_node_incompatible_edit` | overlapping property/leaf changes or incompatible structural values |
| `delete_modify` | a removal/deletion operator overlaps another effect, or structural deletion races modification |
| `role_cardinality` | overlapping role/cardinality effects |
| `child_order` | overlapping topology effects or concurrent sequence-shape changes |
| `scope_binder` | overlapping scope/binder effects |
| `stale_ref` | arguments/effects are absent from the branch input table or cannot map to the base |
| `unsupported_effect` | wrong base, same branch, unavailable authority, failed replay, non-exact/empty effect, or invalid merged AST |

Conflict resolution is intentionally absent. A later explicit operator/task may
consume the artifact; CAP2 synthesis must not infer a repair silently.

## Replay and provenance

Merge and conflict identities include:

- exact base state ID;
- canonical pair of branch output state IDs;
- canonical pair of application IDs;
- typed target fingerprints for conflicts;
- merged state and deterministic merged-branch digest for successes.

`replay_branch_merge` recomputes branch authority, application replay, effect
lineage, structural composition, pack validation, and the complete decision
identity. Swapping left/right inputs produces the same decision identity.

## Evidence

Focused tests cover:

- valid disjoint composition and exact pack revalidation;
- input-order invariance and deterministic replay;
- mutually declared commuting equal-overlap;
- all requested overlap conflict families;
- stale branch refs;
- non-exact effect refusal;
- provenance-complete deterministic conflict identity.

Repository validation is recorded in SLM-377 and PR evidence. No model,
checkpoint, metric, ship gate, or training-data artifact changed.

## Research lineage

[Yin et al., 2019](https://arxiv.org/abs/1810.13337) motivates treating edits as
explicit structured objects. [Brody et al., 2020](https://arxiv.org/abs/2005.13209)
motivates structural edit representation. This implementation is an adapted
repository contract: neither paper specifies this three-way merge algorithm,
typed conflict taxonomy, opaque-reference lineage, or compiler authority
boundary, and no paper result is reproduced here.
