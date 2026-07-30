import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace OpenUIProofs.Metrics

def overlapRatio (overlap gold : ℕ) : ℚ :=
  (overlap : ℚ) / (gold : ℚ)

def structuralSimilarity (jaccard depthSimilarity : ℚ) : ℚ :=
  (7 / 10) * jaccard + (3 / 10) * depthSimilarity

def mean (values : List ℚ) : ℚ :=
  values.sum / values.length

theorem recall_mono {a b gold : ℕ} (hgold : 0 < gold) (hab : a ≤ b) :
    overlapRatio a gold ≤ overlapRatio b gold := by
  unfold overlapRatio
  apply div_le_div_of_nonneg_right
  · exact_mod_cast hab
  · positivity

theorem structural_similarity_mono {j₁ j₂ d₁ d₂ : ℚ}
    (hj : j₁ ≤ j₂) (hd : d₁ ≤ d₂) :
    structuralSimilarity j₁ d₁ ≤ structuralSimilarity j₂ d₂ := by
  unfold structuralSimilarity
  linarith

theorem sum_le_sum_of_forall₂ {xs ys : List ℚ}
    (h : List.Forall₂ (· ≤ ·) xs ys) : xs.sum ≤ ys.sum := by
  induction h with
  | nil => simp
  | cons hxy _ ih =>
      simp only [List.sum_cons]
      exact add_le_add hxy ih

theorem mean_mono {xs ys : List ℚ}
    (hvalues : List.Forall₂ (· ≤ ·) xs ys)
    (hnonempty : xs ≠ []) :
    mean xs ≤ mean ys := by
  unfold mean
  have hlength := hvalues.length_eq
  rw [← hlength]
  apply div_le_div_of_nonneg_right
  · exact sum_le_sum_of_forall₂ hvalues
  · have hpositive : (0 : ℚ) < xs.length := by
      exact_mod_cast List.length_pos_of_ne_nil hnonempty
    exact hpositive.le

theorem extra_component_can_reduce_similarity :
    structuralSimilarity 1 1 > structuralSimilarity (1 / 2) 1 := by
  norm_num [structuralSimilarity]

end OpenUIProofs.Metrics
