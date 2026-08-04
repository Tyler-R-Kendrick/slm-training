/-
  VSS exact-closure theory (certificate-checked domain reduction).

  Axiomatizes the VSS1-01 fixed-point operator as a pure membership model:
  * only replay-valid UNSUPPORTED candidates may be removed;
  * SUPPORTED / UNKNOWN / failed-replay never shrink a domain;
  * one pass is monotone (live set only shrinks);
  * the multi-pass fixpoint is a monotone iteration of passes;
  * certified bottom means every value of some hole was certificate-removed.

  Soft scores never appear. This is the formal safety layer for exact closure;
  it does not prove oracle completeness or empirical solver quality.
-/

import LeverProofLean.ListSet

namespace LeverProofLean.ExactClosure

open LeverProofLean

abbrev Candidate := Nat
abbrev HoleId := Nat

/-- Support oracle verdict (VSS0 tri-state). -/
inductive Verdict where
  | supported
  | unsupported
  | unknown
  deriving DecidableEq, Repr, BEq

/-- One oracle answer for a single (hole, value) query. -/
structure QueryResult where
  hole : HoleId
  candidate : Candidate
  verdict : Verdict
  /-- Replay succeeded against the pass-start state. Only meaningful for unsupported. -/
  replayOk : Bool
  deriving Repr, BEq

/-- Destructive removal is allowed only for replay-valid UNSUPPORTED. -/
def removable (r : QueryResult) : Bool :=
  match r.verdict with
  | .unsupported => r.replayOk
  | _ => false

/-- Whether a candidate is removable according to a batch of results. -/
def isRemovable (results : List QueryResult) (c : Candidate) : Bool :=
  results.any fun r => decide (r.candidate = c) && removable r

/--
  One exact-closure pass over a live domain: drop only certificate-backed
  unsupported candidates. Pass-consistency is modeled by treating `results`
  as computed against a single frozen pre-pass state.
-/
def closePass (live : List Candidate) (results : List QueryResult) : List Candidate :=
  live.filter fun c => !(isRemovable results c)

theorem closePass_subset (live : List Candidate) (results : List QueryResult) :
    Subset (closePass live results) live := by
  intro c hc
  exact (List.mem_filter.mp hc).1

theorem closePass_never_adds (live : List Candidate) (results : List QueryResult)
    (c : Candidate) (hc : c ∈ closePass live results) : c ∈ live :=
  closePass_subset live results c hc

/-- SUPPORTED answers never authorize removal. -/
theorem supported_not_removable (r : QueryResult)
    (h : r.verdict = Verdict.supported) : removable r = false := by
  simp [removable, h]

/-- UNKNOWN answers never authorize removal. -/
theorem unknown_not_removable (r : QueryResult)
    (h : r.verdict = Verdict.unknown) : removable r = false := by
  simp [removable, h]

/-- Failed certificate replay never authorizes removal. -/
theorem failed_replay_not_removable (r : QueryResult)
    (h : r.replayOk = false) : removable r = false := by
  cases hv : r.verdict <;> simp [removable, hv, h]

/-- A candidate that no result marks removable survives the pass. -/
theorem survivor_if_not_removable
    (live : List Candidate) (results : List QueryResult) (c : Candidate)
    (hin : c ∈ live) (hnot : isRemovable results c = false) :
    c ∈ closePass live results := by
  simp only [closePass, List.mem_filter, hin, hnot, Bool.not_false, and_self]

/-- A candidate marked removable leaves the live set. -/
theorem removable_leaves
    (live : List Candidate) (results : List QueryResult) (c : Candidate)
    (hrem : isRemovable results c = true) :
    c ∉ closePass live results := by
  intro hc
  have := (List.mem_filter.mp hc).2
  simp [hrem] at this

/-- Multi-pass iteration: fold closePass over a list of result batches. -/
def iterate (live : List Candidate) : List (List QueryResult) → List Candidate
  | [] => live
  | batch :: rest => iterate (closePass live batch) rest

theorem iterate_subset (live : List Candidate) (batches : List (List QueryResult)) :
    Subset (iterate live batches) live := by
  induction batches generalizing live with
  | nil =>
      intro c hc
      exact hc
  | cons batch rest ih =>
      intro c hc
      have hmid := ih (closePass live batch) c hc
      exact closePass_subset live batch c hmid

/-- Fixed point: a pass that removes nothing leaves the live set unchanged. -/
def isFixedPoint (live : List Candidate) (results : List QueryResult) : Prop :=
  SetEq (closePass live results) live

theorem fixed_point_keeps_all
    (live : List Candidate) (results : List QueryResult)
    (h : ∀ c ∈ live, isRemovable results c = false) :
    isFixedPoint live results := by
  constructor
  · exact closePass_subset live results
  · intro c hc
    exact survivor_if_not_removable live results c hc (h c hc)

/--
  Certified bottom for a single hole: every value of that hole was removed under
  an accepted certificate. Modeled as "no live survivors remain".
-/
def certifiedBottom (live : List Candidate) : Prop :=
  live = []

private theorem filter_eq_nil_of_all_false
    (xs : List Candidate) (p : Candidate → Bool)
    (h : ∀ c ∈ xs, p c = false) :
    xs.filter p = [] := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
      have hhead := h head (by simp)
      have htail : ∀ c ∈ tail, p c = false := by
        intro c hc
        exact h c (by simp [hc])
      simp [List.filter, hhead, ih htail]

theorem certified_bottom_of_all_removed
    (live : List Candidate) (results : List QueryResult)
    (h : ∀ c ∈ live, isRemovable results c = true) :
    certifiedBottom (closePass live results) := by
  unfold certifiedBottom closePass
  -- filter keeps when !isRemovable; all isRemovable=true ⇒ keep predicate is false
  apply filter_eq_nil_of_all_false
  intro c hc
  simp [h c hc]

/-- Empty result list removes nothing. -/
theorem empty_results_fixed_point (live : List Candidate) :
    isFixedPoint live [] := by
  apply fixed_point_keeps_all
  intro c _hc
  simp [isRemovable]

/-- A single non-removable result cannot shrink the domain. -/
theorem nonremovable_result_fixed_point
    (live : List Candidate) (r : QueryResult)
    (hr : removable r = false) :
    isFixedPoint live [r] := by
  apply fixed_point_keeps_all
  intro c _hc
  simp [isRemovable, hr]

/--
  Honesty composition: if every individual result is non-removable, the batch
  is a fixed point. Proved by induction on the result list.
-/
theorem honest_when_no_accepted_unsupported
    (live : List Candidate) (results : List QueryResult)
    (h : ∀ r ∈ results, removable r = false) :
    isFixedPoint live results := by
  apply fixed_point_keeps_all
  intro c _hc
  induction results with
  | nil =>
      simp [isRemovable]
  | cons r rest ih =>
      have hr : removable r = false := h r (by simp)
      have hrest : ∀ r' ∈ rest, removable r' = false := by
        intro r' hr'
        exact h r' (List.mem_cons_of_mem r hr')
      have ih' := ih hrest
      -- isRemovable (r :: rest) c = ((r.candidate == c) && removable r) || isRemovable rest c
      simp only [isRemovable, List.any_cons, hr, Bool.and_false, Bool.false_or]
      exact ih'

end LeverProofLean.ExactClosure
