# SLM-130 / EFS3-05: Wire canonical AST deduplication and measure valid semantic-mode coverage

**Claim class:** wiring / fixture only  
**Run date:** 2026-07-19  
**Machine-readable result:** [`iter-slm130-canonical-ast-dedup-20260719.json`](iter-slm130-canonical-ast-dedup-20260719.json)

This iteration implements the SLM-130 canonical-AST deduplication harness. No
X22 or compiler-tree checkpoint was trained, no GPU was used, no frozen
real-world candidate pools were evaluated, and no ship-gate claim is made.

## What landed

- `src/slm_training/harnesses/experiments/canonical_ast_dedup.py`
  - Frozen dataclasses: `CanonicalAstFingerprintV1`, `AbstractModeSignatureV1`,
    `CandidateEquivalenceGroupV1`, `DiversityCoverageReportV1`.
  - `RepresentativePolicy` enum: `first`, `best_generator_score`,
    `best_selector_score`, `deterministic_lexicographic`.
  - `build_canonical_ast_fingerprint()` using the D2 canonicalizer and the
    existing component-multiset `ast_fingerprint`.
  - `build_abstract_mode_signature()` — diagnostic coarse signature only,
    never hard-equivalence authority.
  - `group_candidates_by_canonical_ast()` with full provenance, multiplicity,
    generation ranks, and disagreement flags.
  - `unique_slot_truncation()` — select up to `k` finalists with at most one
    slot per canonical AST group, refilling from already-scored candidates.
  - `compute_diversity_coverage()` and `dedup_arms_for_pool()` for the five
    preregistered arms (raw, exact-output, terminal-canonical, unique-slot,
    abstract-mode-spread).
- `src/slm_training/dsl/grammar/fastpath/lattice_search.py`
  - `TrajectoryCandidate` now carries an optional `canonical_fingerprint`.
  - `group_trajectory_candidates()` and `select_unique_slot_truncation()`
    adapters that reuse the shared grouping utility.
- `scripts/run_canonical_ast_dedup.py`
  - `--fixture` CLI that exercises the five dedup arms on a tiny synthetic pool.
- Tests under `tests/test_harnesses/experiments/test_canonical_ast_dedup.py`
  and `tests/test_scripts/test_run_canonical_ast_dedup.py`.
- Registry entries: `harness.experiments` bumped to v23 and a new
  `harness.experiments.canonical_ast_dedup` v1 component.

## Fingerprint and equivalence definitions

`CanonicalAstFingerprintV1` is the hard-equivalence key. It hashes the D2
canonical form (`dsl.canonicalize.canonical_fingerprint`) and records the
existing component-multiset `ast_fingerprint` as a secondary structural signal.
No learned model or heuristic may declare two distinct ASTs equivalent.

`AbstractModeSignatureV1` is a diagnostic coverage signal. It collapses literal
payloads while preserving component topology, binding kind, and placeholder role.
Every normalization rule is listed; collisions against canonical fingerprints
and semantic reports must be audited before it can be used as anything other
than a spread diagnostic.

`CandidateEquivalenceGroupV1` persists:

- fingerprint/version and optional abstract-mode signature;
- all member candidate IDs, generator scores, selector scores, hard levels, and
  semantic-report hashes;
- selected representative and representative policy;
- multiplicity, first/last generation rank;
- hard and semantic disagreement flags.

Any hard/semantic disagreement inside a canonical AST group is treated as a
bug/environmental signal, not averaged.

## Runtime integration

The primary integration seam is the shared grouping utility. The lattice-search
module exposes two adapters:

- `group_trajectory_candidates(candidates, dsl=..., policy=...)` returns
  `CandidateEquivalenceGroupV1` records for a `TrajectoryCandidate` pool.
- `select_unique_slot_truncation(candidates, k, ...)` returns up to `k`
  finalists with at most one per canonical AST group.

Both default to `deterministic_lexicographic` representative selection:

```text
CONTRACT_SATISFIED > VALID > UNKNOWN > INVALID
  then higher generator_score
  then higher selector_score
  then lower generation_rank
  then lexicographic candidate_id
```

This preserves stronger hard/contract evidence; a duplicate with a higher raw
score does not silently replace a candidate with stronger contract evidence.

## Fixture results

Key numbers from [`iter-slm130-canonical-ast-dedup-20260719.json`](iter-slm130-canonical-ast-dedup-20260719.json):

| Arm | pool | raw_valid | unique_canonical_ast | duplicate_multiplicity | semantic_pass@K |
| --- | --- | --- | --- | --- | --- |
| A_raw_no_dedup | 5 | 4 | 4 | 1 | 0.40 |
| B_exact_output_dedup | 4 | 3 | 4 | 0 | 0.25 |
| C_terminal_canonical_ast | 5 | 4 | 4 | 1 | 0.40 |
| D_unique_slot_truncation | 4 | 3 | 4 | 0 | 0.25 |
| E_abstract_mode_spread | 5 | 4 | 4 | 1 | 0.40 |

The fixture contains one alpha-equivalent duplicate pair (`c0_base` /
`c1_alpha`), one semantically distinct program, one invalid candidate, and one
unknown-verdict candidate. The duplicate pair is correctly collapsed into a
single canonical AST group; the invalid candidate is never merged with valid
groups; the unique-slot truncation arm refills from already-scored groups
without extra generation.

## Honest verdict

**`no_safe_direction` / wiring-only.** The harness compiles, the schemas are
stable, and the grouping/coverage math behaves as designed on a toy corpus. The
fixture is too small and too artificial to establish whether canonical AST
deduplication improves within-prompt valid semantic-mode coverage on real
candidate pools. A production claim would require:

- Frozen X22 and compiler-tree candidate pools from durable checkpoints;
- At least three seeds/checkpoints and the full `K ∈ {1,4,8}` grid;
- Binding-aware meaningful v2 and independent/AgentV labels for every finalist;
- Paired per-prompt comparisons with preregistered minimum-effect sizes;
- Zero unsafe canonical merges and a published collision audit;
- Measured latency/memory grouping overhead.

Until then this is wiring and a reusable harness, not a ship result.
