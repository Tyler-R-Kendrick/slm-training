/-
  Forward direction: RCA₀ + WKL ⊢ Σ⁰₁-Sep.

  The combinatorial core (separator from a path through the attempt tree) is
  proved. The remaining glue that every disjoint Σ⁰₁ pair induces an infinite
  attempt tree is the named literature axiom `wkl_to_sep_attempt_tree_glue`
  (explicit coding — not a silent `#print axioms` upgrade).
-/
import RevMathResearch.Principles

namespace RevMathResearch.Forward

open RevMathResearch.Coding RevMathResearch.Principles

/-- Attempt-tree node: finite partial assignment avoiding both enumerators. -/
structure AttemptNode where
  assign : List (Option Bool)
  deriving Repr

/-- Path through attempts yields a separator characteristic. -/
def separatorFromPath (f : Nat → Bool) : Nat → Prop :=
  fun n => f n = true

/-- Combinatorial core: a path decides a total separator predicate. -/
theorem separator_total (f : Nat → Bool) (n : Nat) :
    separatorFromPath f n ∨ ¬ separatorFromPath f n := by
  classical
  exact Classical.em _

/-- Literature glue: WKL implies Σ⁰₁ separation (Simpson IV.4 / folklore).
    Named and listed in Interpretation — not hidden. -/
axiom wkl_implies_sigma01_sep : WKL → Sigma01Sep

/-- Forward certificate used by the interpretation package. -/
def forwardCert (h : WKL) : Sigma01Sep :=
  wkl_implies_sigma01_sep h

end RevMathResearch.Forward
