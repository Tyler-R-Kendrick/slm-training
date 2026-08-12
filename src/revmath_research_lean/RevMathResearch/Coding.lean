/-
  Explicit coding layer for the RESEARCH-03 Big-Five pilot.

  This module is the interpretation surface: SOA-facing names are bound to
  concrete Lean carriers. Classification must cite `codingId` — never
  `#print axioms` alone (HARN-08 / KERN-12).
-/
namespace RevMathResearch.Coding

/-- Finite binary strings (Cantor-space prefixes). -/
abbrev Bitstring := List Bool

/-- A (possibly non-decidable) binary tree as a prefix-closed predicate. -/
structure BinaryTree where
  mem : Bitstring → Prop
  prefixClosed : ∀ σ b, mem (σ ++ [b]) → mem σ

/-- The tree has a node at every finite length. -/
def Infinite (T : BinaryTree) : Prop :=
  ∀ n : Nat, ∃ σ : Bitstring, σ.length = n ∧ T.mem σ

/-- An infinite path through `T`. -/
def Path (T : BinaryTree) (f : Nat → Bool) : Prop :=
  ∀ n : Nat, T.mem ((List.range n).map f)

/-- Σ⁰₁ (r.e.) predicate via a witness enumerator `e n s`. -/
def Sigma01 (P : Nat → Prop) : Prop :=
  ∃ e : Nat → Nat → Bool, ∀ n, P n ↔ ∃ s, e n s = true

/-- Stable coding id consumed by the Python interpretation package. -/
def codingId : String := "coding.wkl_sigma01_sep.v1"

/-- Intended subsystem label (research-only; never production authority). -/
def targetSubsystem : String := "WKL0"

/-- Base theory id for the pilot interpretation. -/
def baseTheoryId : String := "RCA0"

#guard codingId = "coding.wkl_sigma01_sep.v1"
#guard targetSubsystem = "WKL0"
#guard baseTheoryId = "RCA0"

end RevMathResearch.Coding
