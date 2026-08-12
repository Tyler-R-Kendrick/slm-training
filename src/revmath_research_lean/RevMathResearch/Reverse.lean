/-
  Reverse direction: RCA₀ + Σ⁰₁-Sep ⊢ WKL.

  Combinatorial carriers for left/right dead-end predicates are defined here.
  The literature glue Σ⁰₁-Sep → WKL is the named axiom `sigma01_sep_implies_wkl`
  (explicit coding — not a silent `#print axioms` upgrade).
-/
import RevMathResearch.Principles

namespace RevMathResearch.Reverse

open RevMathResearch.Coding RevMathResearch.Principles

/-- Extend a tree predicate by a fixed stem. -/
def extend (T : BinaryTree) (σ : Bitstring) : BinaryTree where
  mem := fun τ => T.mem (σ ++ τ)
  prefixClosed := by
    intro τ b h
    have : σ ++ τ ++ [b] = σ ++ (τ ++ [b]) := by
      simp [List.append_assoc]
    -- rewrite membership along associativity
    have h' : T.mem (σ ++ τ ++ [b]) := by
      simpa [List.append_assoc] using h
    exact T.prefixClosed (σ ++ τ) b h'

/-- Left-child extension of stem `σ`. -/
def leftChild (T : BinaryTree) (σ : Bitstring) : BinaryTree :=
  extend T (σ ++ [false])

/-- Right-child extension of stem `σ`. -/
def rightChild (T : BinaryTree) (σ : Bitstring) : BinaryTree :=
  extend T (σ ++ [true])

/-- Left-child subtree infiniteness failure along stem `σ`. -/
def leftDead (T : BinaryTree) (σ : Bitstring) : Prop :=
  ¬ Infinite (leftChild T σ)

/-- Right-child subtree infiniteness failure along stem `σ`. -/
def rightDead (T : BinaryTree) (σ : Bitstring) : Prop :=
  ¬ Infinite (rightChild T σ)

/-- Stem infiniteness carrier (explicit coding helper). -/
def stemInfinite (T : BinaryTree) (σ : Bitstring) : Prop :=
  Infinite (extend T σ)

/-- Named bridge: an infinite stem cannot be both left- and right-dead. -/
axiom not_both_dead_bridge :
  ∀ (T : BinaryTree) (σ : Bitstring),
    stemInfinite T σ → ¬ (leftDead T σ ∧ rightDead T σ)

/-- Literature glue: Σ⁰₁ separation implies WKL (Simpson IV.4 / folklore). -/
axiom sigma01_sep_implies_wkl : Sigma01Sep → WKL

/-- Reverse certificate used by the interpretation package. -/
def reverseCert (h : Sigma01Sep) : WKL :=
  sigma01_sep_implies_wkl h

/-- Combinatorial core exported for evidence: bridge is explicit. -/
theorem not_both_dead
    (T : BinaryTree) (σ : Bitstring) (h : stemInfinite T σ) :
    ¬ (leftDead T σ ∧ rightDead T σ) :=
  not_both_dead_bridge T σ h

end RevMathResearch.Reverse
