import Lean.Elab.Tactic.Omega

namespace LeverProofLean

structure Ratio where
  numerator : Nat
  denominator : Nat
deriving Repr, BEq, Inhabited

namespace Ratio

def valid (q : Ratio) : Bool := q.denominator != 0

def Le (x y : Ratio) : Prop :=
  x.numerator * y.denominator ≤ y.numerator * x.denominator

instance : DecidableRel Ratio.Le := by
  intro x y
  unfold Le
  infer_instance

def leb (x y : Ratio) : Bool := decide (x.Le y)

def eqb (x y : Ratio) : Bool :=
  x.numerator * y.denominator == y.numerator * x.denominator

def add (x y : Ratio) : Ratio :=
  ⟨x.numerator * y.denominator + y.numerator * x.denominator,
   x.denominator * y.denominator⟩

def mul (x y : Ratio) : Ratio :=
  ⟨x.numerator * y.numerator, x.denominator * y.denominator⟩

def inv? (x : Ratio) : Option Ratio :=
  if x.numerator = 0 then none else some ⟨x.denominator, x.numerator⟩

def ofNat (n : Nat) : Ratio := ⟨n, 1⟩

def pow (x : Ratio) : Nat → Ratio
  | 0 => ofNat 1
  | n + 1 => x.mul (pow x n)

end Ratio

structure Interval where
  lower : Ratio
  upper : Ratio
deriving Repr, BEq, Inhabited

namespace Interval

def valid (i : Interval) : Bool :=
  i.lower.valid && i.upper.valid && i.lower.leb i.upper

def ofNat (n : Nat) : Interval := ⟨Ratio.ofNat n, Ratio.ofNat n⟩

def add (x y : Interval) : Interval :=
  ⟨x.lower.add y.lower, x.upper.add y.upper⟩

def mul (x y : Interval) : Interval :=
  ⟨x.lower.mul y.lower, x.upper.mul y.upper⟩

def pow (x : Interval) (power : Nat) : Interval :=
  ⟨x.lower.pow power, x.upper.pow power⟩

def inv? (x : Interval) : Option Interval := do
  let low ← x.upper.inv?
  let high ← x.lower.inv?
  pure ⟨low, high⟩

end Interval

def minimumFrom : Nat → List Nat → Nat
  | current, [] => current
  | current, x :: rest => minimumFrom (min current x) rest

def maximumFrom : Nat → List Nat → Nat
  | current, [] => current
  | current, x :: rest => maximumFrom (max current x) rest

def sampleInterval : List Nat → Option Interval
  | [] => none
  | x :: rest =>
      some ⟨Ratio.ofNat (minimumFrom x rest),
            Ratio.ofNat (maximumFrom x rest)⟩

def sampleMean (xs : List Nat) : Option Ratio :=
  if xs.isEmpty then none else some ⟨xs.sum, xs.length⟩

theorem minimumFrom_le_seed (seed : Nat) (xs : List Nat) :
    minimumFrom seed xs ≤ seed := by
  induction xs generalizing seed with
  | nil => simp [minimumFrom]
  | cons x xs ih =>
      simp only [minimumFrom]
      exact Nat.le_trans (ih (min seed x)) (Nat.min_le_left seed x)

theorem minimumFrom_le_member (seed x : Nat) (xs : List Nat)
    (h : x ∈ xs) : minimumFrom seed xs ≤ x := by
  induction xs generalizing seed with
  | nil => simp at h
  | cons head tail ih =>
      simp only [minimumFrom]
      simp only [List.mem_cons] at h
      cases h with
      | inl hhead =>
          subst head
          exact Nat.le_trans
            (minimumFrom_le_seed (min seed x) tail)
            (Nat.min_le_right seed x)
      | inr htail => exact ih (min seed head) htail

theorem maximumFrom_ge_seed (seed : Nat) (xs : List Nat) :
    seed ≤ maximumFrom seed xs := by
  induction xs generalizing seed with
  | nil => simp [maximumFrom]
  | cons x xs ih =>
      simp only [maximumFrom]
      exact Nat.le_trans (Nat.le_max_left seed x) (ih (max seed x))

theorem maximumFrom_ge_member (seed x : Nat) (xs : List Nat)
    (h : x ∈ xs) : x ≤ maximumFrom seed xs := by
  induction xs generalizing seed with
  | nil => simp at h
  | cons head tail ih =>
      simp only [maximumFrom]
      simp only [List.mem_cons] at h
      cases h with
      | inl hhead =>
          subst head
          exact Nat.le_trans
            (Nat.le_max_right seed x)
            (maximumFrom_ge_seed (max seed x) tail)
      | inr htail => exact ih (max seed head) htail

theorem sampleInterval_bounds (xs : List Nat) (i : Interval)
    (hsample : sampleInterval xs = some i) (x : Nat) (hx : x ∈ xs) :
    i.lower.Le (Ratio.ofNat x) ∧ (Ratio.ofNat x).Le i.upper := by
  cases xs with
  | nil => simp [sampleInterval] at hsample
  | cons seed rest =>
      simp only [sampleInterval, Option.some.injEq] at hsample
      subst i
      simp only [Ratio.Le, Ratio.ofNat, Nat.mul_one]
      constructor
      · simp only [List.mem_cons] at hx
        cases hx with
        | inl h => subst x; exact minimumFrom_le_seed seed rest
        | inr h => exact minimumFrom_le_member seed x rest h
      · simp only [List.mem_cons] at hx
        cases hx with
        | inl h => subst x; exact maximumFrom_ge_seed seed rest
        | inr h => exact maximumFrom_ge_member seed x rest h

theorem sampleMean_denominator_positive (xs : List Nat) (q : Ratio)
    (hmean : sampleMean xs = some q) : 0 < q.denominator := by
  cases xs with
  | nil => simp [sampleMean] at hmean
  | cons x xs =>
      simp only [sampleMean, List.isEmpty, Bool.false_eq_true, ↓reduceIte,
        Option.some.injEq] at hmean
      subst q
      simp

end LeverProofLean
