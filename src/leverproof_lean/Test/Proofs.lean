import LeverProofLean

open LeverProofLean

#print axioms sampleInterval_bounds
#print axioms sampleMean_denominator_positive
#print axioms evalPiecewise_uses_declared_box
#print axioms selectBest_member
#print axioms selectBest_minimizes_qualityFailures
#print axioms check_requires_valid_evidence
#print axioms checked_selection_comes_from_derived_candidates
#print axioms checked_selection_minimizes_qualityFailures
#print axioms countPositions_total
#print axioms inBand_has_no_violations

#guard digestValid (String.ofList (List.replicate 64 'a'))
#guard !(digestValid (String.ofList (List.replicate 63 'a')))
#guard Workload.valid ⟨1, 9⟩
#guard !(Workload.valid ⟨0, 0⟩)
