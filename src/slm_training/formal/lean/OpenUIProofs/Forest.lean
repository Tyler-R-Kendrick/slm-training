import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Defs
import Mathlib.Tactic

namespace OpenUIProofs.Forest

abbrev Candidate := ℕ

inductive Event where
  | deduction (candidate : Candidate)
  | decision (candidate : Candidate)
  | rollback (candidate : Candidate)
  deriving DecidableEq, Repr

structure State where
  initial : Finset Candidate
  removed : Finset Candidate
  history : List Event
  removed_valid : removed ⊆ initial

noncomputable def close (state : State) (certified : Finset Candidate) : State where
  initial := state.initial
  removed := state.removed ∪ (certified ∩ state.initial)
  history := state.history ++ (certified ∩ state.initial).toList.map Event.deduction
  removed_valid := by
    intro candidate h
    simp only [Finset.mem_union, Finset.mem_inter] at h
    rcases h with h | h
    · exact state.removed_valid h
    · exact h.2

def live (state : State) : Finset Candidate :=
  state.initial \ state.removed

theorem close_monotone (state : State) (a b : Finset Candidate) (hab : a ⊆ b) :
    (close state a).removed ⊆ (close state b).removed := by
  intro candidate h
  simp only [close, Finset.mem_union, Finset.mem_inter] at h ⊢
  rcases h with h | ⟨ha, hi⟩
  · exact Or.inl h
  · exact Or.inr ⟨hab ha, hi⟩

theorem close_idempotent (state : State) (certified : Finset Candidate) :
    (close (close state certified) certified).removed =
      (close state certified).removed := by
  ext candidate
  simp [close]

theorem close_history_preserved (state : State) (certified : Finset Candidate) :
    state.history <+: (close state certified).history := by
  change state.history <+: state.history ++ _
  exact List.prefix_append _ _

theorem close_never_adds_live (state : State) (certified : Finset Candidate) :
    live (close state certified) ⊆ live state := by
  intro candidate h
  simp [live, close] at h ⊢
  exact ⟨h.1, h.2.1⟩

structure DecisionFrame where
  preLive : Finset Candidate
  chosen : Candidate

def rollback (frame : DecisionFrame) (irreversible : Finset Candidate) :
    Finset Candidate :=
  frame.preLive \ irreversible

theorem rollback_retains_only_predecision
    (frame : DecisionFrame) (irreversible : Finset Candidate) :
    rollback frame irreversible ⊆ frame.preLive := by
  intro candidate h
  exact (Finset.mem_sdiff.mp h).1

theorem rollback_retains_irreversible_removals
    (frame : DecisionFrame) (irreversible : Finset Candidate) :
    Disjoint (rollback frame irreversible) irreversible := by
  rw [Finset.disjoint_left]
  intro candidate hrollback hirreversible
  exact (Finset.mem_sdiff.mp hrollback).2 hirreversible

theorem ranker_permutation_preserves_membership
    (before after : List Candidate) (h : before.Perm after) :
    before.toFinset = after.toFinset := by
  induction h with
  | nil => rfl
  | cons candidate _ ih => simp [ih]
  | swap a b tail => simp [Finset.insert_comm]
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

def lossyHistory (_state : State) : List Event :=
  [Event.rollback 0]

theorem lossy_history_counterexample :
    let state : State := {
      initial := {0, 1}
      removed := ∅
      history := [Event.decision 1]
      removed_valid := by simp
    }
    ¬ state.history <+: lossyHistory state := by
  decide

end OpenUIProofs.Forest
