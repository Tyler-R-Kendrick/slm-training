/-
  Certified-closure trace validity.

  A JSON-serializable step is valid when its removed set is the previous removed
  set union the certified removals, and history is prefix-extended. A multi-step
  trace is valid when every step is valid and adjacent states agree.

  Lean proves what acceptance implies. Certificate replay itself is an external
  precondition (the production exact-closure checker must establish it first).
-/

import LeverProofLean.Forest
import LeverProofLean.ListSet

namespace LeverProofLean.Trace

open LeverProofLean
open LeverProofLean.Forest

structure Step where
  beforeRemoved : List Candidate
  afterRemoved : List Candidate
  certified : List Candidate
  beforeHistory : List Event
  afterHistory : List Event

/-- Boolean subset check (decidable; no Finset). -/
def subsetb (xs ys : List Candidate) : Bool :=
  xs.all fun x => decide (x ∈ ys)

def setEqb (xs ys : List Candidate) : Bool :=
  subsetb xs ys && subsetb ys xs

theorem subsetb_sound (xs ys : List Candidate) (h : subsetb xs ys = true) :
    Subset xs ys := by
  intro x hx
  have hall := List.all_eq_true.mp h
  have := hall x hx
  exact of_decide_eq_true this

theorem setEqb_sound (xs ys : List Candidate) (h : setEqb xs ys = true) :
    SetEq xs ys := by
  simp only [setEqb, Bool.and_eq_true] at h
  exact ⟨subsetb_sound xs ys h.1, subsetb_sound ys xs h.2⟩

/-- Prefix check that only needs DecidableEq on Event (no LawfulBEq). -/
def isPrefixB : List Event → List Event → Bool
  | [], _ => true
  | _ :: _, [] => false
  | a :: as, b :: bs => decide (a = b) && isPrefixB as bs

theorem isPrefixB_sound (xs ys : List Event) (h : isPrefixB xs ys = true) :
    xs <+: ys := by
  induction xs generalizing ys with
  | nil =>
      exact List.nil_prefix
  | cons a as ih =>
      match ys with
      | [] =>
          simp [isPrefixB] at h
      | b :: bs =>
          simp only [isPrefixB, Bool.and_eq_true, decide_eq_true_eq] at h
          obtain ⟨heq, hrest⟩ := h
          subst b
          exact (List.cons_prefix_cons).mpr ⟨rfl, ih bs hrest⟩

/-- Boolean step contract (mirrors the Python `check_formal_trace` checker). -/
def validStep (step : Step) : Bool :=
  setEqb step.afterRemoved (union step.beforeRemoved step.certified) &&
    isPrefixB step.beforeHistory step.afterHistory

/-- Multi-step trace contract with adjacent-state agreement. -/
def validTrace : List Step → Bool
  | [] => true
  | [step] => validStep step
  | step₁ :: step₂ :: rest =>
      validStep step₁ &&
        setEqb step₁.afterRemoved step₂.beforeRemoved &&
        decide (step₁.afterHistory = step₂.beforeHistory) &&
        validTrace (step₂ :: rest)

theorem valid_step_sound (step : Step) (h : validStep step = true) :
    SetEq step.afterRemoved (union step.beforeRemoved step.certified) ∧
      step.beforeHistory <+: step.afterHistory := by
  simp only [validStep, Bool.and_eq_true] at h
  exact ⟨setEqb_sound _ _ h.1, isPrefixB_sound _ _ h.2⟩

theorem valid_trace_all_steps (steps : List Step) (h : validTrace steps = true) :
    ∀ step ∈ steps,
      SetEq step.afterRemoved (union step.beforeRemoved step.certified) ∧
        step.beforeHistory <+: step.afterHistory := by
  induction steps with
  | nil =>
      intro step hmem
      cases hmem
  | cons head tail ih =>
      match tail with
      | [] =>
          intro step hmem
          have hmem' : step = head := by
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
            exact hmem
          subst step
          exact valid_step_sound head (by simpa [validTrace] using h)
      | next :: rest =>
          have hparts :
              validStep head = true ∧
                setEqb head.afterRemoved next.beforeRemoved = true ∧
                head.afterHistory = next.beforeHistory ∧
                validTrace (next :: rest) = true := by
            simp only [validTrace, Bool.and_eq_true, decide_eq_true_eq] at h
            exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩
          intro step hmem
          have hmem' : step = head ∨ step ∈ next :: rest := by
            simpa [List.mem_cons] using hmem
          cases hmem' with
          | inl heq =>
              subst step
              exact valid_step_sound head hparts.1
          | inr htail =>
              exact ih hparts.2.2.2 step htail

/--
  A single certified close step projects to a valid formal-trace proposition when
  histories and removed sets are read from the forest state.
-/
theorem close_step_valid (state : State) (certified : List Candidate) :
    let after := close state certified
    let step : Step := {
      beforeRemoved := state.removed
      afterRemoved := after.removed
      certified := certifiedLive state certified
      beforeHistory := state.history
      afterHistory := after.history
    }
    SetEq step.afterRemoved (union step.beforeRemoved step.certified) ∧
      step.beforeHistory <+: step.afterHistory := by
  refine And.intro ?_ (close_history_preserved state certified)
  constructor
  · intro c hc
    -- after.removed = union state.removed (certifiedLive ...)
    have := (mem_close_removed state certified c).mp (by
      simpa [close, certifiedLive] using hc)
    cases this with
    | inl hr => exact (mem_union_iff c state.removed (certifiedLive state certified)).mpr (Or.inl hr)
    | inr h =>
        exact (mem_union_iff c state.removed (certifiedLive state certified)).mpr
          (Or.inr ((mem_inter_iff c certified (live state)).mpr h))
  · intro c hc
    have hc' := (mem_union_iff c state.removed (certifiedLive state certified)).mp hc
    apply (mem_close_removed state certified c).mpr
    cases hc' with
    | inl hr => exact Or.inl hr
    | inr hcl =>
        exact Or.inr ((mem_inter_iff c certified (live state)).mp hcl)

end LeverProofLean.Trace
