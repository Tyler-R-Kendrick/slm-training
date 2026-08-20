/-
  Screening sample-size range (eval-side n for the climb loop).

  Three exact Nat-arithmetic quantities consumed through the bound_ast/v1
  registry (`bound.screening_n.decidability_lower.v1`,
  `bound.screening_n.budget_upper.v1`):

  * `signTestDecidabilityFloor alphaNum alphaDen maxN` — the least per-arm n
    (searched upward from 1, capped at `maxN`) at which the paired two-sided
    sign test can reject at `alpha = alphaNum/alphaDen` at all. The test's
    minimum two-sided p at n pairs is `2/2^n`, so rejection is attainable iff
    `2*alphaDen ≤ alphaNum*2^n`. This is attainability combinatorics over the
    sign-arrangement space, not an empirical power claim.

  * `screeningBudgetUpperBound armWall trainFloor overhead minDecode` — the
    most records whose decode fits the arm wall beside the train floor and
    eval overhead: `⌊(armWall - trainFloor - overhead) / minDecode⌋` with
    saturating Nat subtraction. Inputs are integral second budgets declared
    by the caller; the arithmetic here is proved, the budgets are not.

  * `screeningRangeFeasible nMin nBudget nSuite` — the range composition:
    n must clear the floor and fit both ceilings (budget and suite volume).
    `screeningRangeFeasible_fits_budget` shows the smallest sufficient choice
    `nMin` then fits the arm wall.

  Neither quantity is a quality, generalization, or wall-clock claim about a
  trained model; the wall-budget bound schedules compute already measured by
  the caller.
-/

namespace LeverProofLean.ScreeningSampleSize

/-- Attainability predicate: at `n` paired records the sign test's minimum
    two-sided p (`2/2^n`) is at most `alphaNum/alphaDen`. -/
def signTestDecidable (alphaNum alphaDen n : Nat) : Prop :=
  2 * alphaDen ≤ alphaNum * 2 ^ n

/-- Search upward from `start` for the least `n ≤ maxN` satisfying
    `signTestDecidable`; `fuel` bounds the remaining attempts. -/
def signTestFloorFrom (alphaNum alphaDen maxN start : Nat) :
    (fuel : Nat) → Option Nat
  | 0 => none
  | fuel + 1 =>
    if maxN < start then none
    else if 2 * alphaDen ≤ alphaNum * 2 ^ start then some start
    else signTestFloorFrom alphaNum alphaDen maxN (start + 1) fuel

/-- The least per-arm n in `[1, maxN]` whose sign-test p-floor is at most
    alpha, if one exists. -/
def signTestDecidabilityFloor (alphaNum alphaDen maxN : Nat) : Option Nat :=
  signTestFloorFrom alphaNum alphaDen maxN 1 maxN

/-- Soundness: a returned floor really attains rejection at alpha. -/
theorem signTestFloorFrom_sound
    (alphaNum alphaDen maxN start fuel n : Nat)
    (h : signTestFloorFrom alphaNum alphaDen maxN start fuel = some n) :
    2 * alphaDen ≤ alphaNum * 2 ^ n := by
  induction fuel generalizing start with
  | zero => contradiction
  | succ fuel ih =>
    rw [signTestFloorFrom] at h
    split at h
    · contradiction
    · split at h
      · next hdec =>
          obtain rfl := Option.some.inj h
          exact hdec
      · exact ih (start + 1) h

/-- Minimality: no smaller n at or above the search start attains rejection. -/
theorem signTestFloorFrom_minimal
    (alphaNum alphaDen maxN start fuel n : Nat)
    (h : signTestFloorFrom alphaNum alphaDen maxN start fuel = some n) :
    ∀ m, start ≤ m → m < n → ¬ 2 * alphaDen ≤ alphaNum * 2 ^ m := by
  induction fuel generalizing start with
  | zero => contradiction
  | succ fuel ih =>
    rw [signTestFloorFrom] at h
    split at h
    · contradiction
    · split at h
      · next _ =>
          obtain rfl := Option.some.inj h
          intro m hm hlt hcon
          omega
      · next hdec =>
          intro m hm hlt
          by_cases heq : m = start
          · subst heq
            exact hdec
          · exact ih (start + 1) h m (by omega) hlt

/-- Soundness of the canonical floor entry point (search starts at n=1). -/
theorem signTestDecidabilityFloor_sound (alphaNum alphaDen maxN n : Nat)
    (h : signTestDecidabilityFloor alphaNum alphaDen maxN = some n) :
    2 * alphaDen ≤ alphaNum * 2 ^ n :=
  signTestFloorFrom_sound alphaNum alphaDen maxN 1 maxN n h

/-- Minimality of the canonical floor entry point. -/
theorem signTestDecidabilityFloor_minimal (alphaNum alphaDen maxN n : Nat)
    (h : signTestFloorFrom alphaNum alphaDen maxN 1 maxN = some n) :
    ∀ m, 1 ≤ m → m < n → ¬ 2 * alphaDen ≤ alphaNum * 2 ^ m :=
  signTestFloorFrom_minimal alphaNum alphaDen maxN 1 maxN n h

/-- Records whose decode fits the arm wall beside the train floor and eval
    overhead (saturating Nat subtraction; division by a zero decode floor
    yields zero). -/
def screeningBudgetUpperBound (armWall trainFloor overhead minDecode : Nat) :
    Nat :=
  (armWall - trainFloor - overhead) / minDecode

/-- Any n at or under the budget ceiling really fits the arm wall. -/
theorem screeningBudgetUpperBound_fits
    (armWall trainFloor overhead minDecode n : Nat)
    (hwall : trainFloor + overhead ≤ armWall)
    (hn : n ≤ screeningBudgetUpperBound armWall trainFloor overhead minDecode) :
    n * minDecode + trainFloor + overhead ≤ armWall := by
  unfold screeningBudgetUpperBound at hn
  have h1 : n * minDecode ≤ armWall - trainFloor - overhead := by
    calc n * minDecode
        ≤ (armWall - trainFloor - overhead) / minDecode * minDecode :=
          Nat.mul_le_mul_right minDecode hn
      _ ≤ armWall - trainFloor - overhead := Nat.div_mul_le_self _ _
  omega

/-- The certified range is non-empty iff the floor clears both ceilings —
    arm-wall budget and screening-suite volume. -/
def screeningRangeFeasible (nMin nBudgetCeiling nSuiteCeiling : Nat) : Prop :=
  nMin ≤ min nBudgetCeiling nSuiteCeiling

/-- The smallest sufficient choice (`nMin`) of a feasible range fits the arm
    wall. -/
theorem screeningRangeFeasible_fits_budget
    (armWall trainFloor overhead minDecode nMin nSuite : Nat)
    (hwall : trainFloor + overhead ≤ armWall)
    (h : screeningRangeFeasible nMin
      (screeningBudgetUpperBound armWall trainFloor overhead minDecode)
      nSuite) :
    nMin * minDecode + trainFloor + overhead ≤ armWall :=
  screeningBudgetUpperBound_fits armWall trainFloor overhead minDecode nMin
    hwall (Nat.le_trans h (Nat.min_le_left _ _))

-- Registry parity anchors (mirrored by bound_ast_parity_fixtures.v1.json).
#guard signTestDecidabilityFloor 1 20 64 = some 6
#guard signTestDecidabilityFloor 1 20 5 = none
#guard signTestDecidabilityFloor 1 4 64 = some 3
#guard screeningBudgetUpperBound 70 20 8 14 = 3
#guard screeningBudgetUpperBound 70 20 8 2 = 21
#guard screeningBudgetUpperBound 10 20 8 2 = 0

end LeverProofLean.ScreeningSampleSize
