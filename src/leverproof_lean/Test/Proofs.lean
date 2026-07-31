import LeverProofLean

open LeverProofLean

#print axioms sampleInterval_bounds
#print axioms sampleMean_denominator_positive
#print axioms Interval.add_contains
#print axioms Interval.mul_contains
#print axioms Interval.pow_contains
#print axioms Interval.inv_contains
#print axioms evalPiecewise_uses_declared_box
#print axioms selectBest_member
#print axioms selectBest_minimizes_qualityFailures
#print axioms globallyPreferred_iff
#print axioms check_requires_valid_evidence
#print axioms checked_selection_comes_from_derived_candidates
#print axioms checked_selection_minimizes_qualityFailures
#print axioms checked_selection_is_globally_preferred
#print axioms evalMetricProgram_sound
#print axioms classifySample_below_iff
#print axioms classifySample_above_iff
#print axioms classifySample_within_iff
#print axioms countPositions_total
#print axioms inBand_has_no_violations
#print axioms relation_inBand_iff

-- Core formal claims (Mathlib-free)
#print axioms Forest.close_monotone
#print axioms Forest.close_idempotent
#print axioms Forest.close_history_preserved
#print axioms Forest.close_never_adds_live
#print axioms Forest.lossy_history_counterexample
#print axioms Trace.valid_step_sound
#print axioms Trace.valid_trace_all_steps
#print axioms Trace.close_step_valid
#print axioms StructuralMetrics.recall_mono
#print axioms StructuralMetrics.structural_similarity_mono
#print axioms StructuralMetrics.mean_mono
#print axioms StructuralMetrics.extra_component_can_reduce_similarity
#print axioms ExactClosure.closePass_subset
#print axioms ExactClosure.supported_not_removable
#print axioms ExactClosure.unknown_not_removable
#print axioms ExactClosure.failed_replay_not_removable
#print axioms ExactClosure.iterate_subset
#print axioms ExactClosure.certified_bottom_of_all_removed
#print axioms ExactClosure.honest_when_no_accepted_unsupported
#print axioms DecodeInvariants.singleton_bypasses_ranker
#print axioms DecodeInvariants.empty_domain_is_dead_end
#print axioms DecodeInvariants.ranked_token_is_legal
#print axioms DecodeInvariants.unconstrained_is_illegal
#print axioms DecodeInvariants.unconstrained_fallback_is_illegal
#print axioms EcosystemTier.core_modules_are_core
#print axioms EcosystemTier.ecosystem_modules_are_ecosystem
#print axioms EcosystemTier.core_ecosystem_names_disjoint
#print axioms EcosystemTier.core_success_ignores_library_size
#print axioms EcosystemTier.library_growth_preserves_core_success
#print axioms EcosystemTier.all_true_core_succeeds

#guard digestValid (String.ofList (List.replicate 64 'a'))
#guard !(digestValid (String.ofList (List.replicate 63 'a')))
#guard Workload.valid ⟨1, 9⟩
#guard !(Workload.valid ⟨0, 0⟩)
