/-
  Verified encoding adapters (EVID-10 / SLM-553).

  Trusted bridge from a bounded bool-CNF problem to an external encoding.
  For the promoted reference family, ``encode`` is the identity on clauses, so
  ``Satisfiable(encode p) ↔ ∃ x, Satisfies p x`` holds definitionally on the
  supported subset. Unsupported features are rejected (never silently dropped).
-/

namespace LeverProofLean.VerifiedEncoding

/-- Variable identity in the tiny reference problem. -/
abbrev VarId := Nat

structure Literal where
  varId : VarId
  positive : Bool
  deriving DecidableEq, Repr, BEq

/-- Finite bool-domain CNF problem (supported subset only). -/
structure Problem where
  nVars : Nat
  clauses : List (List Literal)
  deriving DecidableEq, Repr, BEq

/-- Encoded CNF formula (reference family = identity on clauses). -/
structure Encoded where
  nVars : Nat
  clauses : List (List Literal)
  deriving DecidableEq, Repr, BEq

/-- Feature tags mirrored from the Python adapter. -/
inductive Feature where
  | boolDomain
  | clause
  | cardinality
  | pb
  | smtTheory
  deriving DecidableEq, Repr, BEq

def supportedFeature (f : Feature) : Bool :=
  match f with
  | .boolDomain | .clause => true
  | .cardinality | .pb | .smtTheory => false

def supportStatus (features : List Feature) : String :=
  if features.any (fun f => !(supportedFeature f)) then
    "unsupported"
  else if features.contains .boolDomain && features.contains .clause then
    "supported"
  else
    "unknown"

/-- Deterministic reference encoder: identity on CNF clauses. -/
def encode (p : Problem) : Encoded :=
  { nVars := p.nVars, clauses := p.clauses }

def litValue (assign : VarId → Bool) (ℓ : Literal) : Bool :=
  let bit := assign ℓ.varId
  if ℓ.positive then bit else !bit

def clauseSatisfied (assign : VarId → Bool) (c : List Literal) : Bool :=
  c.any (litValue assign)

def satisfies (p : Problem) (assign : VarId → Bool) : Bool :=
  p.clauses.all (clauseSatisfied assign)

def encodedSatisfies (e : Encoded) (assign : VarId → Bool) : Bool :=
  e.clauses.all (clauseSatisfied assign)

/-- Equivalence theorem for the promoted reference encoding family. -/
theorem sat_encode_iff_satisfies (p : Problem) (assign : VarId → Bool) :
    encodedSatisfies (encode p) assign = true ↔ satisfies p assign = true := by
  simp [encode, encodedSatisfies, satisfies]

theorem unsupported_feature_not_supported (f : Feature)
    (h : supportedFeature f = false) :
    supportStatus [f] ≠ "supported" := by
  revert h
  cases f <;> decide

theorem cardinality_feature_unsupported :
    supportStatus [.boolDomain, .clause, .cardinality] = "unsupported" := by
  decide

/-- Fixture: unit clause (x) is satisfied by x := true. -/
def fixtureUnit : Problem :=
  { nVars := 1, clauses := [[{ varId := 0, positive := true }]] }

theorem fixture_unit_sat :
    satisfies fixtureUnit (fun _ => true) = true := by
  decide

/-- Fixture: (x) ∧ (¬x) is unsatisfiable under both assignments. -/
def fixtureUnsat : Problem :=
  { nVars := 1
    clauses := [[{ varId := 0, positive := true }], [{ varId := 0, positive := false }]] }

theorem fixture_unsat_true :
    satisfies fixtureUnsat (fun _ => true) = false := by
  decide

theorem fixture_unsat_false :
    satisfies fixtureUnsat (fun _ => false) = false := by
  decide

/-- Encoding mutation flips the first literal polarity. -/
def mutateFlipFirst (e : Encoded) : Encoded :=
  match e.clauses with
  | [] => e
  | c :: rest =>
    match c with
    | [] => e
    | ℓ :: ls =>
      let ℓ' : Literal := { varId := ℓ.varId, positive := !ℓ.positive }
      { e with clauses := (ℓ' :: ls) :: rest }

theorem mutated_encoding_ne_original :
    mutateFlipFirst (encode fixtureUnsat) ≠ encode fixtureUnsat := by
  decide

end LeverProofLean.VerifiedEncoding
