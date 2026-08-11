# KERN-05 — Strict closure progress and finite stabilization (SLM-523)

## Claim

Exact closure is not only monotone (`closePass_subset`): every pass is either a
membership fixed point or removes at least one previously live candidate.
Cumulative strict removals are bounded by the height of the product of finite
subset lattices over hole domains.

## Bounds

| Measure | Formula | Role |
| --- | --- | --- |
| Strict-removal / lattice height | `Σ d_i` | Closure-pass candidate removals |
| Nonempty-invariant height | `Σ (d_i - 1)` | When every hole stays nonempty |
| Assignment search (KERN-03) | `∏ d_i` | Full completion search — **not** closure passes |

These measures are deliberately separated: a witness such as `[1,1,1]` has
`Σ = 3` and `∏ = 1`.

## Preconditions (explicit)

1. Certificate-checked removals only (`removable` = replay-valid UNSUPPORTED).
2. Finite live / hole domains.
3. **Batch fairness/coverage** — every live candidate marked removable is named
   in the batch (`batchCoversRemovable`). Under the current encoding this holds
   definitionally; it remains an explicit obligation for alternate batching.
4. Uniqueness (`List.Nodup` / unique live ids) when charging length decrease
   under the named uniqueness hypothesis.
5. Nonempty invariance (when using `Σ(d_i-1)`) — every hole stays nonempty.

**Not claimed without proof:** maximality or oracle-completeness of a fixed
point; wall-clock latency.

## Lean

- Module: `LeverProofLean.ExactClosure` (KERN-05 section)
- `strict_progress_or_fixed_point` / `not_fixed_implies_length_decreases`
- `totalRemovedAcross_le_live` / `totalRemovedAcrossHoles_le_strictRemovalBound`
- `removals_le_nonempty_invariant`
- `closure_complexity_separated_from_assignment_search`
- Fixtures: `fixture_two_three_stabilization_bounds`,
  `fixture_flat_total_removed_le_live`

## Python

- Adapter: `slm_training.formal.closure_stabilization`
- Frozen traces: `resources/formal/closure_stabilization_fixtures.v1.json`
- Mutation tests: duplicate live bookkeeping; non-fixed pass with no progress
- Bound AST: `bound.closure.strict_removals.v1` (`Σ d_i`); live cardinality
  remains `bound.closure.live_upper.v1` (aligned with KERN-05 survivor law)

## HARN-07 alignment

Quantitative-bound extraction for `closure_live_upper` cites KERN-05 strict
progress / stabilization (no longer “KERN-05 may still be backlog”). Extraction
still reports the registered live-upper cardinality; stabilization sum bounds
are available via the strict-removals AST for proof-mining follow-ons.

## Empirical boundary

Bounds certify combinatorial removal counts under certificate-checked exact
closure. VSS wall-clock, device scheduling, and solver oracle completeness
remain empirical remainders — not Lean theorems.
