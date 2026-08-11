/-
  Black-box lower bound for certifying UNSUPPORTED (KERN-04 / SLM-522).

  Query model: over a finite complete domain of ``P`` assignments, a solver
  learns only one Boolean validity answer per complete-assignment query.

  Model-relative claim (not a universal solver lower bound): any deterministic
  sound refuter that concludes UNSUPPORTED must query all ``P`` assignments in
  the worst case. Stopping early leaves an unqueried index that distinguishes
  the all-invalid world from a one-valid world consistent with the transcript.

  Escapes (symbolic constraints, certificates, shared residual states, structure)
  leave this model — they are not counterexamples to the theorem inside it.
-/

import LeverProofLean.FiniteSearchBounds

namespace LeverProofLean.BlackBoxUnsupportedLowerBound

open LeverProofLean

/--
  Named black-box verifier-query model.

  One Boolean validity answer per complete assignment. Not a wall-clock model
  and not a claim about solvers that share intermediate state or certificates.
-/
structure BlackBoxVerifierQueryModel where
  /-- Complete-assignment count ``P``. -/
  assignmentCount : Nat
  deriving Repr, BEq, DecidableEq

/-- Unit-cost query accounting: cost equals number of assignment queries. -/
def queryCost (_m : BlackBoxVerifierQueryModel) (queries : Nat) : Nat :=
  queries

/-- Exported worst-case lower bound under this model: ``P`` queries. -/
def worstCaseQueryLowerBound (m : BlackBoxVerifierQueryModel) : Nat :=
  m.assignmentCount

/-- Build the model from proved domain sizes (KERN-02/03): ``P = ∏ d_i``. -/
def modelFromDomainSizes (sizes : List Nat) : BlackBoxVerifierQueryModel :=
  { assignmentCount := FiniteSearchBounds.completeAssignmentCount sizes }

/-- All-invalid oracle (every complete assignment is invalid). -/
def allInvalid : Nat → Bool := fun _ => false

/-- One-valid oracle: only index ``i`` is valid. -/
def oneValid (i : Nat) : Nat → Bool := fun j => decide (j = i)

/-- Two oracles agree on every queried index. -/
def oraclesAgreeOn (ω₁ ω₂ : Nat → Bool) (queried : List Nat) : Prop :=
  ∀ j ∈ queried, ω₁ j = ω₂ j

/--
  Soundness of an UNSUPPORTED / refuted claim for oracle ``ω`` on domain size
  ``P``: every complete assignment must be invalid.
-/
def soundUnsupportedClaim (P : Nat) (ω : Nat → Bool) : Prop :=
  ∀ k, k < P → ω k = false

theorem allInvalid_false (k : Nat) : allInvalid k = false :=
  rfl

theorem oneValid_at_self (i : Nat) : oneValid i i = true := by
  simp [oneValid]

theorem oneValid_ne (i j : Nat) (h : j ≠ i) : oneValid i j = false := by
  simp [oneValid, h]

/--
  Gap lemma: if index ``i`` was never queried, the all-invalid and one-valid-at-``i``
  oracles produce identical Boolean answers on the queried set.
-/
theorem unqueried_oracles_agree
    (queried : List Nat) (i : Nat) (hmiss : i ∉ queried) :
    oraclesAgreeOn allInvalid (oneValid i) queried := by
  intro j hj
  have hne : j ≠ i := by
    intro heq
    exact hmiss (heq ▸ hj)
  simp [allInvalid, oneValid, hne]

/-- The one-valid world is not an all-invalid world when ``i < P``. -/
theorem oneValid_not_sound (P i : Nat) (hi : i < P) :
    ¬ soundUnsupportedClaim P (oneValid i) := by
  intro hs
  have := hs i hi
  simp [oneValid] at this

/--
  Early-stop distinguisher: a partial query set missing ``i`` cannot soundly
  certify UNSUPPORTED for every oracle consistent with all-false answers,
  because ``oneValid i`` agrees on the transcript but is not all-invalid.
-/
theorem early_stop_distinguishes
    (P i : Nat) (queried : List Nat)
    (hi : i < P) (hmiss : i ∉ queried) :
    oraclesAgreeOn allInvalid (oneValid i) queried ∧
      ¬ soundUnsupportedClaim P (oneValid i) :=
  ⟨unqueried_oracles_agree queried i hmiss, oneValid_not_sound P i hi⟩

/--
  Coverage corollary (model-relative): if every oracle that agrees with
  ``allInvalid`` on ``queried`` admits a sound UNSUPPORTED claim, then every
  index ``i < P`` must appear in ``queried``.
-/
theorem sound_unsupported_requires_full_coverage
    (P : Nat) (queried : List Nat)
    (hsound :
      ∀ ω, oraclesAgreeOn allInvalid ω queried → soundUnsupportedClaim P ω)
    (i : Nat) (hi : i < P) : i ∈ queried :=
  match (inferInstance : Decidable (i ∈ queried)) with
  | isTrue hmem => hmem
  | isFalse hmiss =>
      False.elim ((early_stop_distinguishes P i queried hi hmiss).2
        (hsound (oneValid i) (early_stop_distinguishes P i queried hi hmiss).1))

/--
  Main model-relative lower-bound claim: sound UNSUPPORTED certification under
  the black-box Boolean-query model requires querying every assignment index.

  The numeric export ``worstCaseQueryLowerBound`` is ``P``. Duplicate-free query
  lists that cover ``{0..P-1}`` therefore have length at least ``P``; Python
  finite models check that combinatorial fact exhaustively for small ``P``.

  Narrow scope: **not** a universal solver lower bound.
-/
theorem black_box_unsupported_query_lower_bound
    (P : Nat) (queried : List Nat)
    (hsound :
      ∀ ω, oraclesAgreeOn allInvalid ω queried → soundUnsupportedClaim P ω) :
    ∀ i, i < P → i ∈ queried :=
  fun i hi => sound_unsupported_requires_full_coverage P queried hsound i hi

/-- Convenience: exported numeric lower bound equals ``P``. -/
theorem worst_case_bound_eq_P (m : BlackBoxVerifierQueryModel) :
    worstCaseQueryLowerBound m = m.assignmentCount :=
  rfl

/-- Fixture: ``P = 6`` from domain sizes ``[2, 3]``. -/
theorem fixture_two_three_P :
    (modelFromDomainSizes [2, 3]).assignmentCount = 6 := by
  decide

theorem fixture_empty_domain_P_one :
    (modelFromDomainSizes ([] : List Nat)).assignmentCount = 1 := by
  decide

/-- Concrete small-P coverage forces ``P`` queries when the query list is exactly
    ``List.range P`` (canonical exhaustive schedule). -/
theorem fixture_full_range_queries_eq_P (P : Nat) :
    (List.range P).length = worstCaseQueryLowerBound ⟨P⟩ := by
  simp [worstCaseQueryLowerBound, List.length_range]

/-! ### Escape classes (outside the black-box model)

  These document optimization classes that **evade the assumption**, not the
  theorem. Inside ``BlackBoxVerifierQueryModel`` the lower bound stands; a
  solver that leaves the model is simply not covered by the claim.
-/

/-- Named escape classes that leave the black-box Boolean-query assumption. -/
inductive EscapeClass where
  | symbolicConstraint
  | certificate
  | sharedResidualState
  | structure
  deriving Repr, BEq, DecidableEq

/-- Escapes are outside the model — the lower bound is not claimed for them. -/
def escapeLeavesBlackBox (_e : EscapeClass) : Bool :=
  true

theorem escapes_leave_black_box_model (e : EscapeClass) :
    escapeLeavesBlackBox e = true :=
  rfl

def symbolicConstraintEscapeNote : String :=
  "symbolic constraints query regions, not black-box assignment Booleans"

def certificateEscapeNote : String :=
  "certificates carry proof structure beyond one Bool per assignment"

def sharedResidualEscapeNote : String :=
  "shared residual states correlate answers across assignments"

def structureEscapeNote : String :=
  "structure/prefix pruning rejects many leaves without leaf Boolean queries"

theorem escape_notes_nonempty :
    symbolicConstraintEscapeNote.length > 0 ∧
      certificateEscapeNote.length > 0 ∧
      sharedResidualEscapeNote.length > 0 ∧
      structureEscapeNote.length > 0 := by
  decide

end LeverProofLean.BlackBoxUnsupportedLowerBound
