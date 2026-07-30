import LeverProofLean.Interval
import Lean.Elab.Tactic.Omega

namespace LeverProofLean

structure Workload where
  coldRequests : Nat
  warmRequests : Nat
deriving Repr, BEq, Inhabited

def Workload.valid (workload : Workload) : Bool :=
  workload.coldRequests + workload.warmRequests != 0

structure RawCandidate where
  id : String
  artifactSha256 : String
  coldLatencyNs : List Nat
  warmLatencyNs : List Nat
  inputUnits : List Nat
  passes : List Nat
  energyUj : List Nat
  costMicroUsd : List Nat
  successes : Nat
  qualityFailures : Nat
  trainableParameters : Nat
deriving Repr, BEq, Inhabited

structure DerivedCandidate where
  id : String
  artifactSha256 : String
  expectedLatencyNs : Interval
  inputMeanUnits : Ratio
  passMean : Ratio
  successRate : Ratio
  energyMeanUj : Ratio
  costMeanMicroUsd : Ratio
  qualityFailures : Nat
  trainableParameters : Nat
deriving Repr, BEq, Inhabited

def weightedInterval
    (workload : Workload) (cold warm : Interval) : Interval :=
  let total := workload.coldRequests + workload.warmRequests
  ⟨⟨workload.coldRequests * cold.lower.numerator +
      workload.warmRequests * warm.lower.numerator, total⟩,
   ⟨workload.coldRequests * cold.upper.numerator +
      workload.warmRequests * warm.upper.numerator, total⟩⟩

def deriveCandidate
    (workload : Workload) (raw : RawCandidate) : Option DerivedCandidate := do
  if !workload.valid then none else pure ()
  let cold ← sampleInterval raw.coldLatencyNs
  let warm ← sampleInterval raw.warmLatencyNs
  let inputs ← sampleMean raw.inputUnits
  let passCount ← sampleMean raw.passes
  let energy ← sampleMean raw.energyUj
  let cost ← sampleMean raw.costMicroUsd
  if raw.trainableParameters = 0 then none else
  pure {
    id := raw.id
    artifactSha256 := raw.artifactSha256
    expectedLatencyNs := weightedInterval workload cold warm
    inputMeanUnits := inputs
    passMean := passCount
    successRate := ⟨raw.successes,
      raw.coldLatencyNs.length + raw.warmLatencyNs.length⟩
    energyMeanUj := energy
    costMeanMicroUsd := cost
    qualityFailures := raw.qualityFailures
    trainableParameters := raw.trainableParameters
  }

def deriveCandidates
    (workload : Workload) : List RawCandidate → Option (List DerivedCandidate)
  | [] => some []
  | raw :: rest => do
      let candidate ← deriveCandidate workload raw
      let candidates ← deriveCandidates workload rest
      pure (candidate :: candidates)

def ratioLt (left right : Ratio) : Bool :=
  left.leb right && !left.eqb right

def secondaryPrefer (left right : DerivedCandidate) : Bool :=
  if !left.successRate.eqb right.successRate then
    ratioLt right.successRate left.successRate
  else if left.trainableParameters != right.trainableParameters then
    left.trainableParameters < right.trainableParameters
  else if !left.expectedLatencyNs.upper.eqb right.expectedLatencyNs.upper then
    ratioLt left.expectedLatencyNs.upper right.expectedLatencyNs.upper
  else if !left.passMean.eqb right.passMean then
    ratioLt left.passMean right.passMean
  else if !left.energyMeanUj.eqb right.energyMeanUj then
    ratioLt left.energyMeanUj right.energyMeanUj
  else if !left.costMeanMicroUsd.eqb right.costMeanMicroUsd then
    ratioLt left.costMeanMicroUsd right.costMeanMicroUsd
  else
    left.id ≤ right.id

def preferCandidate (left right : DerivedCandidate) : Bool :=
  if left.qualityFailures != right.qualityFailures
  then left.qualityFailures < right.qualityFailures
  else secondaryPrefer left right

def chooseBetter
    (left right : DerivedCandidate) : DerivedCandidate :=
  if preferCandidate left right then left else right

def selectFrom
    (current : DerivedCandidate) : List DerivedCandidate → DerivedCandidate
  | [] => current
  | candidate :: rest => selectFrom (chooseBetter current candidate) rest

def selectBest : List DerivedCandidate → Option DerivedCandidate
  | [] => none
  | candidate :: rest => some (selectFrom candidate rest)

theorem chooseBetter_member (left right : DerivedCandidate) :
    chooseBetter left right = left ∨ chooseBetter left right = right := by
  cases h : preferCandidate left right <;> simp [chooseBetter, h]

theorem preferCandidate_quality_le (left right : DerivedCandidate)
    (hprefer : preferCandidate left right = true) :
    left.qualityFailures ≤ right.qualityFailures := by
  by_cases heq : left.qualityFailures = right.qualityFailures
  · omega
  · simp only [preferCandidate, heq, bne_iff_ne, ne_eq] at hprefer
    exact Nat.le_of_lt (of_decide_eq_true hprefer)

theorem not_preferCandidate_quality_ge (left right : DerivedCandidate)
    (hprefer : preferCandidate left right = false) :
    right.qualityFailures ≤ left.qualityFailures := by
  by_cases hle : right.qualityFailures ≤ left.qualityFailures
  · exact hle
  · have hlt : left.qualityFailures < right.qualityFailures :=
      Nat.lt_of_not_ge hle
    have hne : left.qualityFailures ≠ right.qualityFailures :=
      Nat.ne_of_lt hlt
    simp [preferCandidate, hne, hlt] at hprefer

theorem chooseBetter_quality_le (left right : DerivedCandidate) :
    (chooseBetter left right).qualityFailures ≤ left.qualityFailures ∧
    (chooseBetter left right).qualityFailures ≤ right.qualityFailures := by
  cases h : preferCandidate left right with
  | false =>
      simp only [chooseBetter, h, Bool.false_eq_true, ↓reduceIte]
      exact ⟨not_preferCandidate_quality_ge left right h, Nat.le_refl _⟩
  | true =>
      simp only [chooseBetter, h, ↓reduceIte]
      exact ⟨Nat.le_refl _, preferCandidate_quality_le left right h⟩

theorem selectFrom_member (current : DerivedCandidate)
    (rest : List DerivedCandidate) :
    selectFrom current rest = current ∨ selectFrom current rest ∈ rest := by
  induction rest generalizing current with
  | nil => simp [selectFrom]
  | cons candidate rest ih =>
      simp only [selectFrom]
      rcases ih (chooseBetter current candidate) with h | h
      · rw [h]
        rcases chooseBetter_member current candidate with hchoice | hchoice
        · exact Or.inl hchoice
        · exact Or.inr (by simp [hchoice])
      · exact Or.inr (by simp [h])

theorem selectFrom_quality_le_current (current : DerivedCandidate)
    (rest : List DerivedCandidate) :
    (selectFrom current rest).qualityFailures ≤ current.qualityFailures := by
  induction rest generalizing current with
  | nil => simp [selectFrom]
  | cons candidate rest ih =>
      simp only [selectFrom]
      exact Nat.le_trans
        (ih (chooseBetter current candidate))
        (chooseBetter_quality_le current candidate).1

theorem selectFrom_quality_le_member (current candidate : DerivedCandidate)
    (rest : List DerivedCandidate) (hin : candidate ∈ rest) :
    (selectFrom current rest).qualityFailures ≤ candidate.qualityFailures := by
  induction rest generalizing current with
  | nil => simp at hin
  | cons head rest ih =>
      simp only [selectFrom]
      simp only [List.mem_cons] at hin
      rcases hin with rfl | hin
      · exact Nat.le_trans
          (selectFrom_quality_le_current (chooseBetter current candidate) rest)
          (chooseBetter_quality_le current candidate).2
      · exact ih (chooseBetter current head) hin

theorem selectBest_member (candidates : List DerivedCandidate)
    (selected : DerivedCandidate) (hselected : selectBest candidates = some selected) :
    selected ∈ candidates := by
  cases candidates with
  | nil => simp [selectBest] at hselected
  | cons candidate rest =>
      simp only [selectBest, Option.some.injEq] at hselected
      subst selected
      rcases selectFrom_member candidate rest with h | h
      · simp [h]
      · simp [h]

theorem selectBest_minimizes_qualityFailures
    (candidates : List DerivedCandidate) (selected : DerivedCandidate)
    (hselected : selectBest candidates = some selected) :
    ∀ candidate ∈ candidates,
      selected.qualityFailures ≤ candidate.qualityFailures := by
  cases candidates with
  | nil => simp [selectBest] at hselected
  | cons head rest =>
      simp only [selectBest, Option.some.injEq] at hselected
      intro candidate hin
      simp only [List.mem_cons] at hin
      rcases hin with rfl | hin
      · rw [← hselected]
        exact selectFrom_quality_le_current candidate rest
      · rw [← hselected]
        exact selectFrom_quality_le_member head candidate rest hin

end LeverProofLean
