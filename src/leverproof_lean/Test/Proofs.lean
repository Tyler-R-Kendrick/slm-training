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
#print axioms ExactClosure.finite_search_rejects_partial
#print axioms ExactClosure.finite_search_rejects_stale_state
#print axioms ExactClosure.singleton_survivor_requires_proved_domain
#print axioms ExactClosure.strict_progress_or_fixed_point
#print axioms ExactClosure.not_fixed_implies_length_decreases
#print axioms ExactClosure.totalRemovedAcross_le_live
#print axioms ExactClosure.totalRemovedAcrossHoles_le_strictRemovalBound
#print axioms ExactClosure.removals_le_nonempty_invariant
#print axioms ExactClosure.closure_complexity_separated_from_assignment_search
#print axioms ExactClosure.fixture_two_three_stabilization_bounds
#print axioms CompleteDomain.forged_coverage_complete_never_authorizes
#print axioms CompleteDomain.proved_members_eq_legal
#print axioms CompleteDomain.stale_state_rejects
#print axioms CompleteDomain.singleton_is_legal
#print axioms CompleteDomain.singleton_unique_legal
#print axioms CompleteDomain.partial_never_authorizes_pruning
#print axioms FiniteSearchBounds.completeAssignmentCount_eq
#print axioms FiniteSearchBounds.prefix_tree_le_one_plus_holes_mul_assignments
#print axioms FiniteSearchBounds.pruning_memo_cannot_increase_bound
#print axioms FiniteSearchBounds.fixture_two_three_assignment_count
#print axioms FiniteSearchBounds.fixture_empty_holes
#print axioms FiniteSearchBounds.fixture_zero_domain
#print axioms BlackBoxUnsupportedLowerBound.black_box_unsupported_query_lower_bound
#print axioms BlackBoxUnsupportedLowerBound.early_stop_distinguishes
#print axioms BlackBoxUnsupportedLowerBound.worst_case_bound_eq_P
#print axioms BlackBoxUnsupportedLowerBound.fixture_two_three_P
#print axioms EventTrace.traceCost_append
#print axioms EventTrace.cost_nonneg
#print axioms EventTrace.decodeUnitWork_traceCost_eq_sum
#print axioms EventTrace.physical_bound_from_hypothesis
#print axioms EventTrace.wallClockUpperBound_append
#print axioms EventTrace.fixture_decode_unit_work_sum
#print axioms DecodeInvariants.singleton_bypasses_ranker
#print axioms DecodeInvariants.forged_coverage_complete_never_bypasses
#print axioms DecodeInvariants.coverage_complete_flag_not_singleton
#print axioms DecodeInvariants.legacy_telemetry_never_proved
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

-- Advisory residual plane (non-vacuous scorer control + membership)
-- Every theorem in AdvisoryResidual is listed for axiom-certification parity.
#print axioms AdvisoryResidual.singleton_is_zero_work
#print axioms AdvisoryResidual.incomplete_not_scored
#print axioms AdvisoryResidual.stale_digest_rejects_example
#print axioms AdvisoryResidual.legal_mismatch_rejects_example
#print axioms AdvisoryResidual.decode_off_zero_apps_non_singleton
#print axioms AdvisoryResidual.decode_off_zero_apps_singleton
#print axioms AdvisoryResidual.filterLegal_subset
#print axioms AdvisoryResidual.filterLegal_drops_illegal
#print axioms AdvisoryResidual.filterLegal_example
#print axioms AdvisoryResidual.scored_keys_are_filter_example
#print axioms AdvisoryResidual.scored_decode_off_zero_apps_example
#print axioms AdvisoryResidual.reconstruct_encode_example
#print axioms AdvisoryResidual.reconstruct_encode_singleton
#print axioms AdvisoryResidual.zipRolesOnto_preserves_nodes
#print axioms AdvisoryResidual.role_shuffle_preserves_membership
#print axioms AdvisoryResidual.role_shuffle_example
#print axioms AdvisoryResidual.role_shuffle_preserves_role_multiset_example
#print axioms AdvisoryResidual.golden_degrees
#print axioms AdvisoryResidual.golden_S_column_masses_from_B
#print axioms AdvisoryResidual.soft_token_collision
#print axioms AdvisoryResidual.contraction_factor_lt_one

-- Goal support finite structural laws (PGS-F01 / SLM-506)
#print axioms GoalSupport.unknown_not_certified_removable
#print axioms GoalSupport.unobserved_not_certified_removable
#print axioms GoalSupport.supported_not_certified_removable
#print axioms GoalSupport.certified_removal_subset_unsupported_evidence
#print axioms GoalSupport.certified_removal_has_unsupported_evidence
#print axioms GoalSupport.unknown_not_in_certified_removal
#print axioms GoalSupport.unobserved_not_in_certified_removal
#print axioms GoalSupport.WellFormedPartitions.wellformed_partitions_cover_legal
#print axioms GoalSupport.WellFormedPartitions.buckets_pairwise_disjoint
#print axioms GoalSupport.classify_adequate
#print axioms GoalSupport.classify_regret
#print axioms GoalSupport.classify_unresolved
#print axioms GoalSupport.classify_inadequate
#print axioms GoalSupport.classify_partial_coverage_unknown
#print axioms GoalSupport.classification_exhaustive
#print axioms GoalSupport.classification_mutually_exclusive_under_wellformed
#print axioms GoalSupport.recorded_core_hits_every_failure
#print axioms GoalSupport.subset_minimal_removing_atom_breaks_hitting
#print axioms GoalSupport.sound_overapprox_not_claimed_subset_minimal
#print axioms GoalSupport.witness_survives_nonrequired_removal
#print axioms GoalSupport.adding_candidates_preserves_witness
#print axioms GoalSupport.certified_live_singleton_no_learned_choice
#print axioms GoalSupport.certified_live_singleton_forces_unique_survivor
#print axioms GoalSupport.certified_removal_respects_exact_closure_unknown
#print axioms GoalSupport.certified_removal_respects_exact_closure_supported

-- Solver judgment (KERN-01 / SLM-516)
#print axioms Judgment.invalid_never_authorizes
#print axioms Judgment.unknown_never_authorizes
#print axioms Judgment.unchecked_never_authorizes
#print axioms Judgment.timeout_never_refuted
#print axioms Judgment.skipped_replay_never_refuted
#print axioms Judgment.unsupported_capability_never_refuted
#print axioms Judgment.incomplete_coverage_never_refuted
#print axioms Judgment.missing_tool_never_refuted
#print axioms Judgment.timeout_never_authorizes_removal
#print axioms Judgment.skipped_replay_never_authorizes_removal
#print axioms Judgment.classification_exhaustive
#print axioms Judgment.classified_payload_matches
#print axioms Judgment.exact_closure_unknown_never_authorizes_removal

#guard digestValid (String.ofList (List.replicate 64 'a'))
#guard !(digestValid (String.ofList (List.replicate 63 'a')))
#guard Workload.valid ⟨1, 9⟩
#guard !(Workload.valid ⟨0, 0⟩)

-- Executable residual ranking guards
#guard ConstrainedDiffusion.rankInsideLegal [1, 9, 2] [true, false, true] == some 2
#guard ConstrainedDiffusion.rankInsideLegal [1, 9] [false, false] == none
#guard ConstrainedDiffusion.ForwardsOptimum
  { domain := { status := .complete, candidates := [[0]] }, forcedToken := some 0 } == 0

-- Advisory residual executable guards (bridged to golden_vectors.v1.json)
#guard AdvisoryResidual.filterLegal [0, 1, 2] [0, 9, 1, 7] == [0, 1]
#guard AdvisoryResidual.decodeOffZeroApps [0, 1, 2] [0, 9] == true
#guard AdvisoryResidual.decodeOffZeroApps [0] [0, 1] == true
#guard AdvisoryResidual.colDegrees AdvisoryResidual.goldenB == [2, 2]
#guard AdvisoryResidual.rowDegrees AdvisoryResidual.goldenB == [2, 1, 1]
#guard AdvisoryResidual.softToken [1, 0, 0] [7, 7, 3] == AdvisoryResidual.softToken [0, 1, 0] [7, 7, 3]
#guard AdvisoryResidual.goldenSColumnSumNums.all (· = AdvisoryResidual.sColumnSumTarget AdvisoryResidual.goldenB) == true
#guard AdvisoryResidual.membership
  (AdvisoryResidual.shuffleRoles
    [{ role := 1, node := 10 }, { role := 2, node := 20 }, { role := 3, node := 30 }] 1) ==
  [10, 20, 30]
#guard (AdvisoryResidual.shuffleRoles
  [{ role := 1, node := 10 }, { role := 2, node := 20 }, { role := 3, node := 30 }] 1).map
  (·.role) == [2, 3, 1]
#guard AdvisoryResidual.reconstruct [0, 1]
  (AdvisoryResidual.encodeIncidence
    [{ factorId := 0, nodes := [1, 2] }, { factorId := 1, nodes := [2, 3] }]) ==
  [{ factorId := 0, nodes := [1, 2] }, { factorId := 1, nodes := [2, 3] }]

-- Goal support executable guards (bridged to goal_support_golden_cases.v1.json)
#guard GoalSupport.classifyDomainAdequacy
  { legal := [1, 2, 3], supported := [2], unsupported := [1], unknown := [], unobserved := [3] }
  { selected := 2, hardProfile := true, capApplied := false,
    domainComplete := true,
    allUnsupportedReplayValid := true, obstruction := { present := false } } ==
  .adequateSelectedSupported
#guard GoalSupport.classifyDomainAdequacy
  { legal := [1, 2], supported := [1], unsupported := [2], unknown := [], unobserved := [] }
  { selected := 2, hardProfile := true, capApplied := false,
    domainComplete := true,
    allUnsupportedReplayValid := true, obstruction := { present := false } } ==
  .selectionRegret
#guard GoalSupport.classifyDomainAdequacy
  { legal := [1, 2], supported := [], unsupported := [1, 2], unknown := [], unobserved := [] }
  { selected := 1, hardProfile := true, capApplied := false,
    domainComplete := false,
    allUnsupportedReplayValid := true, obstruction := { present := false } } ==
  .coverageUnknown
#guard GoalSupport.hitsAllB [10, 11]
  [{ terminal := 0, atoms := [10, 12] }, { terminal := 1, atoms := [11, 13] }] == true
#guard GoalSupport.subsetMinimalExactB [10, 11]
  [{ terminal := 0, atoms := [10] }, { terminal := 1, atoms := [11] }] == true

-- Solver judgment executable guards (bridged to judgment_truth_table.v1.json)
#guard Judgment.classifyOutcome
  { malformedInput := false, payloadMismatch := false, timedOut := true,
    skippedReplay := false, unsupportedCapability := false, incompleteCoverage := false,
    missingTool := false, vssVerdict := some .unsupported, hasWitnessDigest := false,
    hasCounterexampleDigest := false, exhausted := true, replayChecked := true,
    replayOk := true } == .unknown
#guard Judgment.classifyOutcome
  { malformedInput := false, payloadMismatch := false, timedOut := false,
    skippedReplay := true, unsupportedCapability := false, incompleteCoverage := false,
    missingTool := false, vssVerdict := some .unsupported, hasWitnessDigest := false,
    hasCounterexampleDigest := false, exhausted := true, replayChecked := true,
    replayOk := true } == .unknown
#guard Judgment.authorizesSemanticConclusion
  (Judgment.classify
    { malformedInput := false, payloadMismatch := false, timedOut := false,
      skippedReplay := false, unsupportedCapability := false, incompleteCoverage := false,
      missingTool := false, vssVerdict := some .supported, hasWitnessDigest := true,
      hasCounterexampleDigest := false, exhausted := false, replayChecked := true,
      replayOk := true }
    true) == true
#guard !Judgment.authorizesRemoval
  (Judgment.classify
    { malformedInput := false, payloadMismatch := false, timedOut := true,
      skippedReplay := false, unsupportedCapability := false, incompleteCoverage := false,
      missingTool := false, vssVerdict := some .unsupported, hasWitnessDigest := false,
      hasCounterexampleDigest := false, exhausted := true, replayChecked := true,
      replayOk := true }
    true)

-- Finite search bounds (KERN-03)
#guard FiniteSearchBounds.completeAssignmentCount [2, 3] = 6
#guard FiniteSearchBounds.prefixTreeNodeBound [2, 3] = 9
#guard FiniteSearchBounds.completeAssignmentCount ([] : List Nat) = 1
#guard FiniteSearchBounds.prefixTreeNodeBound ([] : List Nat) = 1
#guard FiniteSearchBounds.completeAssignmentCount [2, 0, 4] = 0

-- Black-box UNSUPPORTED lower bound (KERN-04)
#guard (BlackBoxUnsupportedLowerBound.modelFromDomainSizes [2, 3]).assignmentCount = 6
#guard (BlackBoxUnsupportedLowerBound.modelFromDomainSizes ([] : List Nat)).assignmentCount = 1
#guard BlackBoxUnsupportedLowerBound.worstCaseQueryLowerBound ⟨6⟩ = 6
#guard BlackBoxUnsupportedLowerBound.escapeLeavesBlackBox .symbolicConstraint = true

-- Event traces / machine cost models (KERN-06)
#guard EventTrace.decodeUnitWorkCostOfCounts
  { tokenize := 2, grammarTransition := 3, solverExpansion := 5,
    certificateCheck := 7, neuralForward := 11 } = 28
#guard EventTrace.traceCost EventTrace.decodeUnitWorkModel
  ([.neuralForward 2] ++ [.solverExpansion 3]) = 5
#guard EventTrace.wallClockUpperBound EventTrace.decodeUnitWorkModel ⟨10⟩
  [.neuralForward 3] = 30
