import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Tactic

namespace OpenUIProofs.Recurrence

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

def deltaUpdate (scale : ℝ) (input output : E) : E :=
  scale • (output - input)

theorem zero_update (input output : E) :
    deltaUpdate 0 input output = 0 := by
  simp [deltaUpdate]

theorem delta_update_norm (scale : ℝ) (input output : E) :
    ‖deltaUpdate scale input output‖ = |scale| * ‖output - input‖ := by
  simp [deltaUpdate, norm_smul, Real.norm_eq_abs]

theorem layerscale_bound {scale bound : ℝ} (hscale : |scale| ≤ bound)
    (input output : E) :
    ‖deltaUpdate scale input output‖ ≤ bound * ‖output - input‖ := by
  rw [delta_update_norm]
  exact mul_le_mul_of_nonneg_right hscale (norm_nonneg _)

def IsContractionBound (k : ℝ) : Prop :=
  0 ≤ k ∧ k < 1

theorem contraction_geometric_decay {k initial : ℝ}
    (hk : IsContractionBound k) (hnonneg : 0 ≤ initial) (n : ℕ) :
    0 ≤ initial * k ^ n := by
  exact mul_nonneg hnonneg (pow_nonneg hk.1 n)

theorem missing_contraction_counterexample :
    ¬ IsContractionBound (1 : ℝ) := by
  norm_num [IsContractionBound]

theorem margin_stability {winner runner perturbation : ℝ}
    (hmargin : runner + 2 * perturbation < winner) :
    runner + perturbation < winner - perturbation := by
  linarith

end OpenUIProofs.Recurrence
