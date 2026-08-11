/-
  KERN-10 / SLM-538: resource-bound + trigger parity mirrors.

  Case ids match `resource_bound_trigger_parity_fixtures.v1.json`.
  Numeric #guard / theorems must stay bit-identical with Python expect.
-/

import LeverProofLean.FiniteSearchBounds
import LeverProofLean.BlackBoxUnsupportedLowerBound
import LeverProofLean.ExactClosure
import LeverProofLean.EventTrace

namespace LeverProofLean.ResourceBoundParity

open LeverProofLean
open FiniteSearchBounds
open BlackBoxUnsupportedLowerBound
open EventTrace

/-- Frozen fixture schema id (must match Python SCHEMA). -/
def paritySchema : String := "resource_bound_trigger_parity/v1"

#guard paritySchema = "resource_bound_trigger_parity/v1"

/-- KERN-10 case `fs_empty`. -/
def caseId_fs_empty : String := "fs_empty"
theorem parity_fs_empty_assignments :
    completeAssignmentCount ([] : List Nat) = 1 := by
  decide
theorem parity_fs_empty_prefix :
    prefixTreeNodeBound ([] : List Nat) = 1 := by
  decide
#guard completeAssignmentCount ([] : List Nat) = 1
#guard prefixTreeNodeBound ([] : List Nat) = 1

/-- KERN-10 case `fs_one`. -/
def caseId_fs_one : String := "fs_one"
theorem parity_fs_one_assignments :
    completeAssignmentCount [1] = 1 := by
  decide
theorem parity_fs_one_prefix :
    prefixTreeNodeBound [1] = 2 := by
  decide
#guard completeAssignmentCount [1] = 1
#guard prefixTreeNodeBound [1] = 2

/-- KERN-10 case `fs_two_three`. -/
def caseId_fs_two_three : String := "fs_two_three"
theorem parity_fs_two_three_assignments :
    completeAssignmentCount [2, 3] = 6 := by
  decide
theorem parity_fs_two_three_prefix :
    prefixTreeNodeBound [2, 3] = 9 := by
  decide
#guard completeAssignmentCount [2, 3] = 6
#guard prefixTreeNodeBound [2, 3] = 9

/-- KERN-10 case `fs_zero_mid`. -/
def caseId_fs_zero_mid : String := "fs_zero_mid"
theorem parity_fs_zero_mid_assignments :
    completeAssignmentCount [2, 0, 4] = 0 := by
  decide
theorem parity_fs_zero_mid_prefix :
    prefixTreeNodeBound [2, 0, 4] = 3 := by
  decide
#guard completeAssignmentCount [2, 0, 4] = 0
#guard prefixTreeNodeBound [2, 0, 4] = 3

/-- KERN-10 case `fs_many_singletons`. -/
def caseId_fs_many_singletons : String := "fs_many_singletons"
theorem parity_fs_many_singletons_assignments :
    completeAssignmentCount [1, 1, 1] = 1 := by
  decide
theorem parity_fs_many_singletons_prefix :
    prefixTreeNodeBound [1, 1, 1] = 4 := by
  decide
#guard completeAssignmentCount [1, 1, 1] = 1
#guard prefixTreeNodeBound [1, 1, 1] = 4

/-- KERN-10 case `fs_many_duplicates`. -/
def caseId_fs_many_duplicates : String := "fs_many_duplicates"
theorem parity_fs_many_duplicates_assignments :
    completeAssignmentCount [2, 2, 2] = 8 := by
  decide
theorem parity_fs_many_duplicates_prefix :
    prefixTreeNodeBound [2, 2, 2] = 15 := by
  decide
#guard completeAssignmentCount [2, 2, 2] = 8
#guard prefixTreeNodeBound [2, 2, 2] = 15

/-- KERN-10 case `fs_huge_product`.
  Python owns u64 overflow classification; Lean binds the exact product identity. -/
def caseId_fs_huge_product : String := "fs_huge_product"
theorem parity_fs_huge_product_product :
    (1048576 : Nat) * 1048576 = 1099511627776 := by
  decide
theorem parity_fs_huge_product_assignments :
    completeAssignmentCount [1048576, 1048576] = 1099511627776 := by
  decide
#guard (1048576 : Nat) * 1048576 = 1099511627776


/-- KERN-10 case `bb_empty`. -/
def caseId_bb_empty : String := "bb_empty"
theorem parity_bb_empty :
    (modelFromDomainSizes ([] : List Nat)).assignmentCount = 1 ∧
      worstCaseQueryLowerBound (modelFromDomainSizes ([] : List Nat)) = 1 := by
  decide
#guard (modelFromDomainSizes ([] : List Nat)).assignmentCount = 1

/-- KERN-10 case `bb_two_three`. -/
def caseId_bb_two_three : String := "bb_two_three"
theorem parity_bb_two_three :
    (modelFromDomainSizes [2, 3]).assignmentCount = 6 ∧
      worstCaseQueryLowerBound (modelFromDomainSizes [2, 3]) = 6 := by
  decide
#guard (modelFromDomainSizes [2, 3]).assignmentCount = 6

/-- KERN-10 case `bb_zero_domain`. -/
def caseId_bb_zero_domain : String := "bb_zero_domain"
theorem parity_bb_zero_domain :
    (modelFromDomainSizes [0]).assignmentCount = 0 ∧
      worstCaseQueryLowerBound (modelFromDomainSizes [0]) = 0 := by
  decide
#guard (modelFromDomainSizes [0]).assignmentCount = 0

/-- KERN-10 case `cl_empty`. -/
def caseId_cl_empty : String := "cl_empty"
theorem parity_cl_empty :
    ExactClosure.strictRemovalUpperBound ([] : List Nat) = 0 ∧
      ExactClosure.nonemptyInvariantUpperBound ([] : List Nat) = 0 ∧
      ExactClosure.assignmentSearchBound ([] : List Nat) = 1 := by
  decide
#guard ExactClosure.strictRemovalUpperBound ([] : List Nat) = 0

/-- KERN-10 case `cl_one`. -/
def caseId_cl_one : String := "cl_one"
theorem parity_cl_one :
    ExactClosure.strictRemovalUpperBound [1] = 1 ∧
      ExactClosure.nonemptyInvariantUpperBound [1] = 0 ∧
      ExactClosure.assignmentSearchBound [1] = 1 := by
  decide
#guard ExactClosure.strictRemovalUpperBound [1] = 1

/-- KERN-10 case `cl_two_three`. -/
def caseId_cl_two_three : String := "cl_two_three"
theorem parity_cl_two_three :
    ExactClosure.strictRemovalUpperBound [2, 3] = 5 ∧
      ExactClosure.nonemptyInvariantUpperBound [2, 3] = 3 ∧
      ExactClosure.assignmentSearchBound [2, 3] = 6 := by
  decide
#guard ExactClosure.strictRemovalUpperBound [2, 3] = 5

/-- KERN-10 case `et_unit_28`. -/
def caseId_et_unit_28 : String := "et_unit_28"
theorem parity_et_unit_28 :
    decodeUnitWorkCostOfCounts { tokenize := 2, grammarTransition := 3, solverExpansion := 5, certificateCheck := 7, neuralForward := 11 } = 28 := by
  decide
#guard decodeUnitWorkCostOfCounts { tokenize := 2, grammarTransition := 3, solverExpansion := 5, certificateCheck := 7, neuralForward := 11 } = 28

/-- KERN-10 case `et_zero`. -/
def caseId_et_zero : String := "et_zero"
theorem parity_et_zero :
    decodeUnitWorkCostOfCounts { tokenize := 0, grammarTransition := 0, solverExpansion := 0, certificateCheck := 0, neuralForward := 0 } = 0 := by
  decide
#guard decodeUnitWorkCostOfCounts { tokenize := 0, grammarTransition := 0, solverExpansion := 0, certificateCheck := 0, neuralForward := 0 } = 0

/-- KERN-10 case `et_forward_only_three`. -/
def caseId_et_forward_only_three : String := "et_forward_only_three"
theorem parity_et_forward_only_three :
    decodeUnitWorkCostOfCounts { tokenize := 0, grammarTransition := 0, solverExpansion := 0, certificateCheck := 0, neuralForward := 3 } = 3 := by
  decide
#guard decodeUnitWorkCostOfCounts { tokenize := 0, grammarTransition := 0, solverExpansion := 0, certificateCheck := 0, neuralForward := 3 } = 3

/-- KERN-10 case `cv_finite_decidable_empty` (computability label: finite_decidable). -/
def caseId_cv_finite_decidable_empty : String := "cv_finite_decidable_empty"
theorem parity_cv_finite_decidable_empty :
    completeAssignmentCount ([] : List Nat) = 1 ∧
      prefixTreeNodeBound ([] : List Nat) = 1 := by
  decide

end LeverProofLean.ResourceBoundParity
