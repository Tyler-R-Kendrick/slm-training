# Proof-carrying goal support — finite structural laws (PGS-F01 / SLM-506)

Lean module: [`src/leverproof_lean/LeverProofLean/GoalSupport.lean`](../../src/leverproof_lean/LeverProofLean/GoalSupport.lean)

Python mapping: [`src/slm_training/formal/goal_support_mapping.py`](../../src/slm_training/formal/goal_support_mapping.py)

Golden fixtures: [`src/slm_training/resources/formal/goal_support_golden_cases.v1.json`](../../src/slm_training/resources/formal/goal_support_golden_cases.v1.json)

## Scope (proved)

Finite-set structural laws that Python goal-support **actually claims**:

| Area | Python owner | Lean definitions |
| --- | --- | --- |
| Legal action set + four partitions | `GoalDomainActionPartitionsV1` | `Partitions`, `WellFormedPartitions` |
| Certified removal authority | `_assign_partition` unsupported + hard replay | `certifiedRemovable`, `certifiedRemovalSet` |
| Domain-adequacy classification | `_classify_domain_adequacy` | `classifyDomainAdequacy` |
| Obstruction hitting / subset-minimal | `_hits_all`, `_subset_minimal_deletion_core` | `hitsAll`, `subsetMinimalExact` |
| Required-constraint witness | `GoalSupportResultV1.witness_terminal_evidence_digest` | `isWitness`, `TerminalWitness` |
| Certified singleton survivors | `goal_support_certified_prune` / live filter | `applyCertifiedRemoval` |

## Assumptions (explicit)

1. **Well-formed partitions** (`WellFormedPartitions`): four buckets are pairwise disjoint, canonically sorted in Python validators, and their union equals `legal`.
2. **Certified removal** is keyed off the **unsupported bucket** plus per-action evidence with `replayOk ∧ hardProfile`; unknown/unobserved buckets never intersect unsupported under (1).
3. **Domain adequacy** follows the closed truth table in `domain_adequacy_classification_table()`; `coverage_unknown` is the catch-all.
4. **Obstruction cores**: `subsetMinimalExact` proves **subset-minimal deletion**, not global minimum cardinality. Python may emit `minimum_cardinality_exact` or `sound_overapprox` — Lean exposes `sound_overapprox_not_claimed_subset_minimal` to block over-claiming.
5. **Witness stability** is over required-constraint **ids** only; removing non-required ids preserves `isWitness`.
6. **Singleton survivor** is list-level: one certified survivor implies no alternate live choice.

## Theorem cores

| # | Lean theorem | Statement (finite) |
| --- | --- | --- |
| 1 | `unknown_not_in_certified_removal` / `unobserved_not_in_certified_removal` | Unknown/unobserved bucket members ∉ `certifiedRemovalSet` |
| 2 | `certified_removal_subset_unsupported_evidence` + `certified_removal_has_unsupported_evidence` | Certified removal ⊆ unsupported bucket; each removed action has replay-valid hard unsupported evidence |
| 3 | `wellformed_partitions_cover_legal` + `buckets_pairwise_disjoint` | Four partitions disjoint and cover `legal` under constructor |
| 4 | `classification_exhaustive` + `classification_mutually_exclusive_under_wellformed` | Classifier tags mutually exclusive (key cases) and exhaustive |
| 5 | `recorded_core_hits_every_failure` | Hitting set intersects every terminal failure set |
| 6 | `subset_minimal_removing_atom_breaks_hitting` | Subset-minimal: deleting any core atom breaks hitting |
| 7 | `witness_survives_nonrequired_removal` | Witness for `required` implies witness for any subset |
| 8 | `adding_candidates_preserves_witness` | Expanding `legal` does not falsify recorded witness |
| 9 | `certified_live_singleton_no_learned_choice` | One certified survivor ⇒ all survivors equal that token |

Bridge to VSS exact closure: `certified_removal_respects_exact_closure_unknown/supported`.

## Proof / non-proof boundary

**Proved (structural):** partition laws, certified-removal authority, classifier table, hitting/subset-minimal cores (mode-aware), witness subset monotonicity, singleton survivor uniqueness.

**Not proved:** OpenUI verifier correctness, NL/prompt meaning, evaluator semantic accuracy, global OpenUI satisfiability, oracle completeness, empirical decode quality, minimum-cardinality optimality when Python selects `sound_overapprox` or subset-minimal deletion overapprox.

## Validation

```bash
make -C src/leverproof_lean test
pytest -q tests/test_formal/test_goal_support_mapping.py
```

Axiom audit: `#print axioms` entries in [`src/leverproof_lean/Test/Proofs.lean`](../../src/leverproof_lean/Test/Proofs.lean) — no `sorryAx`; core structural theorems depend only on standard Lean axioms (`propext`, `Quot.sound` where list membership appears).

## Python drift guard

`tests/test_formal/test_goal_support_mapping.py` serializes finite inputs (`goal_support_formal_mapping/v1`) and compares evaluator output to golden fixtures for every classifier case plus exact/subset-minimal obstruction examples. Semantic drift in `_classify_domain_adequacy` or obstruction deletion fails CI without touching ship gates.
