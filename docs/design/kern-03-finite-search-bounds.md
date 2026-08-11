# KERN-03 — Finite completion-search upper bounds (SLM-521)

## Claim

Finite completion search over proved hole domains has explicit upper bounds on
complete assignments and prefix-tree nodes. Bounds are stated against the named
machine/query model `ExpansionVerificationQueryModel` (per-expansion and
per-verification query costs). They are **not** wall-clock latency claims.

## Preconditions (explicit)

1. **Domain completeness** — sizes come from proof-bearing `CompleteDomain` /
   `CompleteDomainEvidenceV1` (KERN-02), never bare `coverageComplete` Booleans.
2. **Finite sizes** — each `d_i` is a non-negative integer domain cardinality.
3. **Algorithm correspondence** — exhaustive prefix-tree enumeration over the
   ordered hole domains (Python fixtures cross-check exact counts).
4. **Cost model** — `ExpansionVerificationQueryModel` query accounting only.

## Lean

- Module: `LeverProofLean.FiniteSearchBounds`
- `completeAssignmentCount` / `P = ∏ d_i` (`completeAssignmentCount_eq`)
- `prefixTreeNodeBound = 1 + Σ_k ∏_{i≤k} d_i`
- Derived coarse bound `≤ 1 + hP` when every `d_i ≥ 1`
  (`prefix_tree_le_one_plus_holes_mul_assignments`)
- Pruning/memoization that only reduces visits cannot raise the bound
  (`pruning_memo_cannot_increase_bound`)
- Sizes via `sizesFromProvedDomains` / `modelFromProvedDomains`

## Python

- Adapter: `slm_training.formal.finite_search_bounds`
- Exhaustive enumeration cross-check + frozen fixtures
  (`resources/formal/finite_search_bounds_fixtures.v1.json`)
- Four-axis export: `export_four_axis_bound_evidence` →
  `RevmathFourAxisAnalysisV1` with `resource_bounds` proved on
  `bound.finite_search.prefix_tree.v1`

## Empirical boundary

Query bounds certify combinatorial search cost under the named model. VSS /
enumerator wall-clock, device scheduling, and cache effects remain empirical
remainders on the `implementation_refinement` axis — not Lean theorems.
