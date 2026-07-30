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

#guard digestValid (String.ofList (List.replicate 64 'a'))
#guard !(digestValid (String.ofList (List.replicate 63 'a')))
#guard Workload.valid ⟨1, 9⟩
#guard !(Workload.valid ⟨0, 0⟩)
