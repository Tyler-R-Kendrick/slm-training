/-
  Completion-forest certified-closure algebra.

  Models the irreversible, certificate-backed removal of candidates from a finite
  completion forest / support domain. History is append-only. Soft rankers never
  appear in the state; they only permute live candidates outside this theory.

  Scope: structural claims about `close` (monotonicity, idempotence, history
  preservation, live-set shrinkage) and rollback's irreversible partition.
  Not a claim about empirical quality or decode latency.
-/

import LeverProofLean.ListSet

namespace LeverProofLean.Forest

open LeverProofLean

abbrev Candidate := Nat

/-- Events recorded by forest closure and search. -/
inductive Event where
  | deduction (candidate : Candidate)
  | decision (candidate : Candidate)
  | rollback (candidate : Candidate)
  deriving DecidableEq, Repr, BEq

/-- Finite forest state: initial domain, irreversible removals, append-only history. -/
structure State where
  initial : List Candidate
  removed : List Candidate
  history : List Event
  removed_valid : Subset removed initial

/-- Live candidates are initial members not yet removed. -/
def live (state : State) : List Candidate :=
  diff state.initial state.removed

/-- Certified candidates that may still be removed (live members of `certified`). -/
def certifiedLive (state : State) (certified : List Candidate) : List Candidate :=
  inter certified (live state)

/--
  Apply a certified-removal set.

  Only candidates already in the initial domain and currently live may leave.
  Every accepted removal is recorded as a deduction event.
-/
def close (state : State) (certified : List Candidate) : State :=
  let removedNow := certifiedLive state certified
  { initial := state.initial
    removed := union state.removed removedNow
    history := state.history ++ removedNow.map Event.deduction
    removed_valid := by
      intro candidate h
      have hmem := (mem_union_iff candidate state.removed removedNow).mp h
      cases hmem with
      | inl hr => exact state.removed_valid candidate hr
      | inr hc =>
          have hc' := (mem_inter_iff candidate certified (live state)).mp hc
          exact (mem_diff_iff candidate state.initial state.removed).mp hc'.2 |>.1 }

/-- Membership view of `close` on the removed set. -/
theorem mem_close_removed (state : State) (certified : List Candidate) (c : Candidate) :
    c ∈ (close state certified).removed ↔
      c ∈ state.removed ∨ (c ∈ certified ∧ c ∈ live state) := by
  simp only [close, certifiedLive, mem_union_iff, mem_inter_iff]

/-- Closing with a larger certified set never revives a removal (monotonicity). -/
theorem close_monotone (state : State) (a b : List Candidate)
    (hab : Subset a b) :
    Subset (close state a).removed (close state b).removed := by
  intro c hc
  have hc' := (mem_close_removed state a c).mp hc
  apply (mem_close_removed state b c).mpr
  cases hc' with
  | inl hr => exact Or.inl hr
  | inr h => exact Or.inr ⟨hab c h.1, h.2⟩

/-- Closing twice with the same certified set is a no-op on removals (idempotence). -/
theorem close_idempotent (state : State) (certified : List Candidate) :
    SetEq (close (close state certified) certified).removed
      (close state certified).removed := by
  constructor
  · intro c hc
    have hc' := (mem_close_removed (close state certified) certified c).mp hc
    cases hc' with
    | inl hr => exact hr
    | inr h =>
        -- Second close claims c is live after first close, so c was not removed first.
        -- But c ∈ certified and c ∈ live after implies, if c ∈ live before, first close
        -- would have removed it. Derive contradiction from live-after membership.
        have hAfterLive := h.2
        have hNotRemAfter : c ∉ (close state certified).removed :=
          (mem_diff_iff c (close state certified).initial
            (close state certified).removed).mp hAfterLive |>.2
        -- Show c ∈ live state, then first close removes c, contradicting hNotRemAfter.
        have hinit : c ∈ state.initial := by
          simpa [close] using
            (mem_diff_iff c (close state certified).initial
              (close state certified).removed).mp hAfterLive |>.1
        have hNotRemBefore : c ∉ state.removed := by
          intro hrb
          exact hNotRemAfter ((mem_close_removed state certified c).mpr (Or.inl hrb))
        have hLiveBefore : c ∈ live state :=
          (mem_diff_iff c state.initial state.removed).mpr ⟨hinit, hNotRemBefore⟩
        have hRemAfter : c ∈ (close state certified).removed :=
          (mem_close_removed state certified c).mpr (Or.inr ⟨h.1, hLiveBefore⟩)
        exact False.elim (hNotRemAfter hRemAfter)
  · intro c hc
    exact (mem_close_removed (close state certified) certified c).mpr (Or.inl hc)

/-- History is always extended as a prefix (append-only). -/
theorem close_history_preserved (state : State) (certified : List Candidate) :
    state.history <+: (close state certified).history := by
  change state.history <+: state.history ++ _
  exact List.prefix_append state.history _

/-- Closing never adds a live candidate. -/
theorem close_never_adds_live (state : State) (certified : List Candidate) :
    Subset (live (close state certified)) (live state) := by
  intro c hc
  have hc' := (mem_diff_iff c (close state certified).initial
    (close state certified).removed).mp hc
  have hinit : c ∈ state.initial := by simpa [close] using hc'.1
  have hNotRemAfter := hc'.2
  have hNotRemBefore : c ∉ state.removed := by
    intro hrb
    exact hNotRemAfter ((mem_close_removed state certified c).mpr (Or.inl hrb))
  exact (mem_diff_iff c state.initial state.removed).mpr ⟨hinit, hNotRemBefore⟩

/-- Decision frame for reversible branching over a pre-decision live set. -/
structure DecisionFrame where
  preLive : List Candidate
  chosen : Candidate

/-- Rollback restores the pre-decision live set minus irreversible removals. -/
def rollback (frame : DecisionFrame) (irreversible : List Candidate) : List Candidate :=
  diff frame.preLive irreversible

theorem rollback_retains_only_predecision
    (frame : DecisionFrame) (irreversible : List Candidate) :
    Subset (rollback frame irreversible) frame.preLive :=
  diff_subset frame.preLive irreversible

theorem rollback_retains_irreversible_removals
    (frame : DecisionFrame) (irreversible : List Candidate) (c : Candidate)
    (hc : c ∈ rollback frame irreversible) : c ∉ irreversible :=
  diff_disjoint frame.preLive irreversible c hc

/--
  A ranker may only reorder live candidates. Membership (hard legality) is
  preserved under pairwise swap of the presentation list — the generator of
  finite permutations used by the controller's ranker check.
-/
theorem ranker_swap_preserves_membership (a b : Candidate) (tail : List Candidate) :
    SetEq (a :: b :: tail) (b :: a :: tail) := by
  constructor
  · intro x hx
    simp only [List.mem_cons] at hx ⊢
    rcases hx with hx | hx | hx
    · exact Or.inr (Or.inl hx)
    · exact Or.inl hx
    · exact Or.inr (Or.inr hx)
  · intro x hx
    simp only [List.mem_cons] at hx ⊢
    rcases hx with hx | hx | hx
    · exact Or.inr (Or.inl hx)
    · exact Or.inl hx
    · exact Or.inr (Or.inr hx)

theorem ranker_cons_preserves_membership (c : Candidate) (xs ys : List Candidate)
    (h : SetEq xs ys) : SetEq (c :: xs) (c :: ys) := by
  constructor
  · intro x hx
    simp only [List.mem_cons] at hx ⊢
    cases hx with
    | inl heq => exact Or.inl heq
    | inr hmem => exact Or.inr (h.1 x hmem)
  · intro x hx
    simp only [List.mem_cons] at hx ⊢
    cases hx with
    | inl heq => exact Or.inl heq
    | inr hmem => exact Or.inr (h.2 x hmem)

/-- Lossy reconstruction that ignores stored history (counterexample seed). -/
def lossyHistory (_state : State) : List Event :=
  [Event.rollback 0]

/--
  Reconstructing history from only the current live set can drop prior decisions.
  This rejects lossy history designs before training.
-/
private def lossyCounterexampleState : State :=
  { initial := [0, 1]
    removed := []
    history := [Event.decision 1]
    removed_valid := by
      intro x hx
      cases hx }

theorem lossy_history_counterexample :
    ¬ lossyCounterexampleState.history <+: lossyHistory lossyCounterexampleState := by
  rintro ⟨t, ht⟩
  -- ht : [decision 1] ++ t = [rollback 0]
  cases t with
  | nil =>
      -- [Event.decision 1] = [Event.rollback 0]
      injection ht with hhead _
      cases hhead
  | cons head tail =>
      -- left length ≥ 2, right length = 1
      have hlen :
          (lossyCounterexampleState.history ++ head :: tail).length =
            (lossyHistory lossyCounterexampleState).length := by
        rw [ht]
      simp [lossyCounterexampleState, lossyHistory] at hlen

end LeverProofLean.Forest
