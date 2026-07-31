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

-- Constrained diffusion × topology (ADR claim cores)
#print axioms ConstrainedDiffusion.gamma_only_tightens
#print axioms ConstrainedDiffusion.production_subset_static
#print axioms ConstrainedDiffusion.production_is_gamma_filter
#print axioms ConstrainedDiffusion.e1_parity_components
#print axioms ConstrainedDiffusion.no_silent_unknown_to_unsupported
#print axioms ConstrainedDiffusion.e1_parity_transitive
#print axioms ConstrainedDiffusion.rankInsideLegal_length_guard
#print axioms ConstrainedDiffusion.rank_single_legal
#print axioms ConstrainedDiffusion.rank_single_illegal
#print axioms ConstrainedDiffusion.rank_skips_illegal_higher_score
#print axioms ConstrainedDiffusion.legalScored_only_from_true
#print axioms ConstrainedDiffusion.soft_legality_forbidden
#print axioms ConstrainedDiffusion.honest_overapprox_preserves_admit
#print axioms ConstrainedDiffusion.forced_macro_no_literals
#print axioms ConstrainedDiffusion.forced_macro_append
#print axioms ConstrainedDiffusion.incomplete_cannot_be_forced_edge
#print axioms ConstrainedDiffusion.singleton_forwards_optimum_zero
#print axioms ConstrainedDiffusion.singleton_forwards_optimum_is_minimum
#print axioms ConstrainedDiffusion.e9_honest_not_aot
#print axioms ConstrainedDiffusion.never_put_six_before_one
#print axioms ConstrainedDiffusion.circuit_has_max_rank
#print axioms ConstrainedDiffusion.executor_has_min_rank
#print axioms ConstrainedDiffusion.forest_verified_cannot_add
#print axioms ConstrainedDiffusion.predicate_before_circuit
#print axioms ConstrainedDiffusion.freeze_topology_preserves_executor_pass
#print axioms ConstrainedDiffusion.artifact_not_sole_executor
#print axioms ConstrainedDiffusion.score_keys_subset
#print axioms ConstrainedDiffusion.singleton_requires_zero_residual_work
#print axioms ConstrainedDiffusion.soft_token_collision_example
#print axioms ConstrainedDiffusion.restart_contraction_factor_le_one

#guard digestValid (String.ofList (List.replicate 64 'a'))
#guard !(digestValid (String.ofList (List.replicate 63 'a')))
#guard Workload.valid ⟨1, 9⟩
#guard !(Workload.valid ⟨0, 0⟩)

-- Executable residual ranking guards
#guard ConstrainedDiffusion.rankInsideLegal [1, 9, 2] [true, false, true] == some 2
#guard ConstrainedDiffusion.rankInsideLegal [1, 9] [false, false] == none
#guard ConstrainedDiffusion.ForwardsOptimum
  { domain := { status := .complete, candidates := [[0]] }, forcedToken := some 0 } == 0
