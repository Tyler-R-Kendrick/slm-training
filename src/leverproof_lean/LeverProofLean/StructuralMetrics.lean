/-
  Structural metric algebra.

  Models the monotone components used by structural similarity proxies:
  overlap/recall ratios and a fixed 7:3 weighted combination of Jaccard and
  depth similarity. Scores are natural numbers (basis points or raw counts);
  ordering is ordinary Nat.le — no rationals, no Mathlib.

  Scope: algebraic direction of the declared proxy under explicit input
  inequalities. Not a claim that the proxy matches human quality or ship gates.
-/

namespace LeverProofLean.StructuralMetrics

/-- Overlap-over-gold rate as a cross-multiplied comparison pair. -/
structure Rate where
  numerator : Nat
  denominator : Nat
  deriving Repr, BEq, Inhabited

def Rate.le (a b : Rate) : Prop :=
  a.numerator * b.denominator ≤ b.numerator * a.denominator

def overlapRatio (overlap gold : Nat) : Rate :=
  ⟨overlap, gold⟩

/--
  Structural similarity in Nat arithmetic with fixed 7:3 weights.
  Inputs are assumed on a common nonnegative scale (e.g. basis points).
-/
def structuralSimilarity (jaccard depthSimilarity : Nat) : Nat :=
  (7 * jaccard + 3 * depthSimilarity) / 10

/-- Mean of a nonempty Nat list (integer division). -/
def mean (values : List Nat) : Nat :=
  match values with
  | [] => 0
  | _ => values.sum / values.length

/-- Pointwise inequality of equal-length lists (stdlib has no Forall₂ here). -/
inductive PointwiseLe : List Nat → List Nat → Prop where
  | nil : PointwiseLe [] []
  | cons {x y : Nat} {xs ys : List Nat} :
      x ≤ y → PointwiseLe xs ys → PointwiseLe (x :: xs) (y :: ys)

theorem recall_mono {a b gold : Nat} (hab : a ≤ b) :
    (overlapRatio a gold).le (overlapRatio b gold) := by
  unfold overlapRatio Rate.le
  exact Nat.mul_le_mul_right gold hab

theorem structural_similarity_mono {j₁ j₂ d₁ d₂ : Nat}
    (hj : j₁ ≤ j₂) (hd : d₁ ≤ d₂) :
    structuralSimilarity j₁ d₁ ≤ structuralSimilarity j₂ d₂ := by
  unfold structuralSimilarity
  exact Nat.div_le_div_right
    (Nat.add_le_add (Nat.mul_le_mul_left 7 hj) (Nat.mul_le_mul_left 3 hd))

theorem sum_le_sum_of_pointwise
    {xs ys : List Nat} (h : PointwiseLe xs ys) :
    xs.sum ≤ ys.sum := by
  induction h with
  | nil => simp
  | cons hxy _ ih =>
      simp only [List.sum_cons]
      exact Nat.add_le_add hxy ih

theorem pointwise_length {xs ys : List Nat} (h : PointwiseLe xs ys) :
    xs.length = ys.length := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]

theorem mean_mono {xs ys : List Nat}
    (hvalues : PointwiseLe xs ys)
    (hnonempty : xs ≠ []) :
    mean xs ≤ mean ys := by
  match xs, ys, hvalues, hnonempty with
  | [], _, h, hne =>
      cases h
      exact (hne rfl).elim
  | x :: xs', y :: ys', h, _ =>
      cases h with
      | cons hxy hrest =>
          have hsum :=
            sum_le_sum_of_pointwise (PointwiseLe.cons hxy hrest)
          have hlen :=
            pointwise_length (PointwiseLe.cons hxy hrest)
          simp only [mean]
          -- both nonempty: sum / length
          simpa [hlen] using Nat.div_le_div_right hsum

/--
  Extra unmatched structure can lower the structural-similarity proxy even when
  depth similarity is perfect. Guards against treating the proxy as free.
-/
theorem extra_component_can_reduce_similarity :
    structuralSimilarity 50 100 < structuralSimilarity 100 100 := by
  decide

end LeverProofLean.StructuralMetrics
