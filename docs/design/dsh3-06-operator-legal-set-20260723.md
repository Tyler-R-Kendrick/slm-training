# DSH3-06 exact operator legal set

SLM-374 adds a model-independent legality boundary above the pack-owned
operator registry. `OperatorLegalSetV1` binds one canonical operator state,
reference table, registry, and bounded enumeration policy. It exposes only
operator IDs with at least one pack-authorized argument tuple while retaining
explicit uncertainty for any product that was not fully searched.

## Enumeration contract

For each registered declaration, the compiler builds one typed domain per
argument slot from the state-bound DSH3-03 reference table. Candidates are
ordered by semantic descriptor fingerprint, not opaque reference ID or input
order. Declaration order defines the hierarchical Cartesian product.

Each tuple is passed through the registry's ordinary `dry_run` path, so a legal
action has the same pack authority proof and application identity as a real
application. The entry records:

- typed per-slot candidate domains;
- the admitted action and application proof IDs;
- evaluated and total combination counts;
- complete or partial coverage;
- stable rejection-code counts; and
- `SUPPORTED`, `UNSUPPORTED`, or `UNKNOWN`.

Products are consumed lazily and capped independently by
`max_combinations_per_operator`. Complete zero-action products are the only
hard-prunable operators. A truncated product with no witness is `UNKNOWN`; a
truncated product with a witness remains supported but cannot certify that its
action list is complete. Repeated slots without an explicit finite arity are
also `UNKNOWN` and execute no combinations.

## Reserved action surface

Operator actions use the canonical reserved form:

```text
OPERATOR <operator_id> <slot_id>=<ref_kind>:<request_id>:<opaque_id> ...
```

Arguments remain in declaration order and retain their typed opaque compiler
references. Parsing rejects malformed prefixes, identifiers, kinds, reference
surfaces, and duplicate slots. Semantic action identity hashes the operator
declaration plus descriptor fingerprints, so opaque-ID and candidate-order
permutations preserve legal membership even though their request-local wire
surfaces differ.

Ordinary grammar/token actions are copied without filtering or reordering.
Exact singleton force emission is available only when every operator product
is complete and the union of ordinary and operator actions contains exactly
one action. Partial coverage never force-emits and never hard-prunes.

## Evidence and scope

Deterministic tests independently brute-force every tuple in a small complete
domain and compare exact successful application IDs. Additional controls cover
zero-action hard pruning, lazy budget truncation, partial witnessed actions,
unbounded repeated slots, opaque-reference and candidate-order permutation,
strict serialization round trips, ordinary-action preservation, and exact
singleton forcing.

These are compiler contract/unit fixtures. No train, eval, benchmark, matrix,
checkpoint, model-card, ship-gate, or model-quality claim is produced.
