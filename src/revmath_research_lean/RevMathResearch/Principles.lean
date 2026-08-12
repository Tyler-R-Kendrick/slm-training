/-
  Principle / theorem pair whose RCA₀-equivalence is classical reverse
  mathematics folklore (Simpson, SOSOA): Weak König's Lemma ↔ Σ⁰₁ separation.
-/
import RevMathResearch.Coding

namespace RevMathResearch.Principles

open RevMathResearch.Coding

/-- Weak König's Lemma (binary trees). -/
def WKL : Prop :=
  ∀ T : BinaryTree, Infinite T → ∃ f : Nat → Bool, Path T f

/-- Σ⁰₁ separation. -/
def Sigma01Sep : Prop :=
  ∀ A B : Nat → Prop,
    Sigma01 A → Sigma01 B → (∀ n, ¬ (A n ∧ B n)) →
      ∃ C : Nat → Prop, (∀ n, A n → C n) ∧ (∀ n, B n → ¬ C n)

/-- Surface statements recorded in evidence digests. -/
def wklStatement : String :=
  "principle: every infinite binary tree has a path"

def sepStatement : String :=
  "theorem: Sigma0_1 separation (disjoint r.e. sets are separable)"

#guard wklStatement ≠ sepStatement

end RevMathResearch.Principles
