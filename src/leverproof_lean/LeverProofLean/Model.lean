import LeverProofLean.Interval

namespace LeverProofLean

inductive Model where
  | constant (value : Ratio)
  | input
  | add (left right : Model)
  | multiply (left right : Model)
  | power (base : Model) (power : Nat)
  | inversePower (base : Model) (power : Nat)
deriving Repr, BEq, Inhabited

def Model.eval (model : Model) (input : Interval) : Option Interval :=
  match model with
  | .constant value => some ⟨value, value⟩
  | .input => some input
  | .add left right => do
      let leftValue ← left.eval input
      let rightValue ← right.eval input
      pure (leftValue.add rightValue)
  | .multiply left right => do
      let leftValue ← left.eval input
      let rightValue ← right.eval input
      pure (leftValue.mul rightValue)
  | .power base exponent => do
      let value ← base.eval input
      pure (value.pow exponent)
  | .inversePower base exponent => do
      let value ← base.eval input
      (value.pow exponent).inv?

structure BoxCase where
  lower : Nat
  upper : Nat
  model : Model
deriving Repr, BEq, Inhabited

def BoxCase.contains (box : BoxCase) (input : Nat) : Bool :=
  decide (box.lower ≤ input ∧ input ≤ box.upper)

def evalPiecewise : List BoxCase → Nat → Option Interval
  | [], _ => none
  | box :: rest, input =>
      if box.contains input
      then box.model.eval (Interval.ofNat input)
      else evalPiecewise rest input

theorem evalPiecewise_uses_declared_box
    (cases : List BoxCase) (input : Nat) (result : Interval)
    (heval : evalPiecewise cases input = some result) :
    ∃ box ∈ cases,
      box.contains input = true ∧
      box.model.eval (Interval.ofNat input) = some result := by
  induction cases with
  | nil => simp [evalPiecewise] at heval
  | cons box rest ih =>
      simp only [evalPiecewise] at heval
      split at heval
      · next hbox =>
          exact ⟨box, by simp, hbox, heval⟩
      · next hbox =>
          obtain ⟨chosen, hin, hcontains, hmodel⟩ := ih heval
          exact ⟨chosen, by simp [hin], hcontains, hmodel⟩

end LeverProofLean
