import LeverProofLean.Interval

namespace LeverProofLean

structure NamedInterval where
  name : String
  value : Interval
deriving Repr, BEq, Inhabited

structure NamedRatio where
  name : String
  value : Ratio
deriving Repr, BEq, Inhabited

inductive MetricInstruction where
  | constant (value : Ratio)
  | variable (name : String)
  | add
  | multiply
  | power (exponent : Nat)
  | inverse
deriving Repr, BEq, Inhabited

def lookupInterval (name : String) : List NamedInterval → Option Interval
  | [] => none
  | dependency :: rest =>
      if dependency.name = name then some dependency.value
      else lookupInterval name rest

def lookupRatio (name : String) : List NamedRatio → Option Ratio
  | [] => none
  | dependency :: rest =>
      if dependency.name = name then some dependency.value
      else lookupRatio name rest

def evalMetricProgram
    (dependencies : List NamedInterval) :
    List MetricInstruction → List Interval → Option Interval
  | [], [result] => some result
  | [], _ => none
  | .constant value :: rest, stack =>
      evalMetricProgram dependencies rest (⟨value, value⟩ :: stack)
  | .variable name :: rest, stack => do
      let value ← lookupInterval name dependencies
      evalMetricProgram dependencies rest (value :: stack)
  | .add :: rest, right :: left :: stack =>
      evalMetricProgram dependencies rest (left.add right :: stack)
  | .add :: _, _ => none
  | .multiply :: rest, right :: left :: stack =>
      evalMetricProgram dependencies rest (left.mul right :: stack)
  | .multiply :: _, _ => none
  | .power exponent :: rest, value :: stack =>
      evalMetricProgram dependencies rest (value.pow exponent :: stack)
  | .power _ :: _, _ => none
  | .inverse :: rest, value :: stack => do
      let inverse ← value.inv?
      evalMetricProgram dependencies rest (inverse :: stack)
  | .inverse :: _, _ => none

def evalMetricValue
    (dependencies : List NamedRatio) :
    List MetricInstruction → List Ratio → Option Ratio
  | [], [result] => some result
  | [], _ => none
  | .constant value :: rest, stack =>
      evalMetricValue dependencies rest (value :: stack)
  | .variable name :: rest, stack => do
      let value ← lookupRatio name dependencies
      evalMetricValue dependencies rest (value :: stack)
  | .add :: rest, right :: left :: stack =>
      evalMetricValue dependencies rest (left.add right :: stack)
  | .add :: _, _ => none
  | .multiply :: rest, right :: left :: stack =>
      evalMetricValue dependencies rest (left.mul right :: stack)
  | .multiply :: _, _ => none
  | .power exponent :: rest, value :: stack =>
      evalMetricValue dependencies rest (value.pow exponent :: stack)
  | .power _ :: _, _ => none
  | .inverse :: rest, value :: stack => do
      let inverse ← value.inv?
      evalMetricValue dependencies rest (inverse :: stack)
  | .inverse :: _, _ => none

def EnvironmentContained
    (intervals : List NamedInterval) (values : List NamedRatio) : Prop :=
  ∀ name interval value,
    lookupInterval name intervals = some interval →
    lookupRatio name values = some value →
    interval.Contains value

inductive StackContained : List Interval → List Ratio → Prop
  | nil : StackContained [] []
  | cons : interval.Contains value →
      StackContained intervals values →
      StackContained (interval :: intervals) (value :: values)

theorem evalMetricProgram_sound
    (intervals : List NamedInterval) (values : List NamedRatio)
    (instructions : List MetricInstruction)
    (intervalStack : List Interval) (valueStack : List Ratio)
    (result : Interval) (value : Ratio)
    (henvironment : EnvironmentContained intervals values)
    (hstack : StackContained intervalStack valueStack)
    (hinterval :
      evalMetricProgram intervals instructions intervalStack = some result)
    (hvalue : evalMetricValue values instructions valueStack = some value) :
    result.Contains value := by
  induction instructions generalizing intervalStack valueStack result value with
  | nil =>
      cases intervalStack with
      | nil => simp [evalMetricProgram] at hinterval
      | cons interval rest =>
          cases rest with
          | cons _ _ => simp [evalMetricProgram] at hinterval
          | nil =>
              simp [evalMetricProgram] at hinterval
              subst result
              cases valueStack with
              | nil => simp [evalMetricValue] at hvalue
              | cons point rest =>
                  cases rest with
                  | cons _ _ => simp [evalMetricValue] at hvalue
                  | nil =>
                      simp [evalMetricValue] at hvalue
                      subst value
                      cases hstack with
                      | cons contained _ => exact contained
  | cons instruction rest ih =>
      cases instruction with
      | constant constant =>
          refine ih (⟨constant, constant⟩ :: intervalStack)
            (constant :: valueStack) result value ?_ ?_ ?_
          · exact StackContained.cons (by
              simp [Interval.Contains, Ratio.Le]) hstack
          · exact hinterval
          · exact hvalue
      | «variable» name =>
          cases hi : lookupInterval name intervals with
          | none => simp [evalMetricProgram, hi] at hinterval
          | some interval =>
              cases hv : lookupRatio name values with
              | none => simp [evalMetricValue, hv] at hvalue
              | some point =>
                  refine ih (interval :: intervalStack) (point :: valueStack)
                    result value ?_ ?_ ?_
                  · exact StackContained.cons
                      (henvironment name interval point hi hv) hstack
                  · simpa [evalMetricProgram, hi] using hinterval
                  · simpa [evalMetricValue, hv] using hvalue
      | add =>
          cases intervalStack with
          | nil => simp [evalMetricProgram] at hinterval
          | cons right remaining =>
              cases remaining with
              | nil => simp [evalMetricProgram] at hinterval
              | cons left tail =>
                  cases hstack with
                  | cons hright hremaining =>
                      cases hremaining with
                      | cons hleft htail =>
                          refine ih (left.add right :: tail)
                            (_ :: _) result value ?_ hinterval hvalue
                          exact StackContained.cons
                            (Interval.add_contains left right _ _ hleft hright)
                            htail
      | multiply =>
          cases intervalStack with
          | nil => simp [evalMetricProgram] at hinterval
          | cons right remaining =>
              cases remaining with
              | nil => simp [evalMetricProgram] at hinterval
              | cons left tail =>
                  cases hstack with
                  | cons hright hremaining =>
                      cases hremaining with
                      | cons hleft htail =>
                          refine ih (left.mul right :: tail)
                            (_ :: _) result value ?_ hinterval hvalue
                          exact StackContained.cons
                            (Interval.mul_contains left right _ _ hleft hright)
                            htail
      | power exponent =>
          cases intervalStack with
          | nil => simp [evalMetricProgram] at hinterval
          | cons interval tail =>
              cases hstack with
              | cons contained htail =>
                  refine ih (interval.pow exponent :: tail)
                    (_ :: _) result value ?_ hinterval hvalue
                  exact StackContained.cons
                    (Interval.pow_contains interval _ exponent contained) htail
      | inverse =>
          cases intervalStack with
          | nil => simp [evalMetricProgram] at hinterval
          | cons interval tail =>
              cases hstack with
              | cons contained htail =>
                  rename_i points point
                  cases hi : interval.inv? with
                  | none => simp [evalMetricProgram, hi] at hinterval
                  | some inverse =>
                      cases hv : point.inv? with
                      | none => simp [evalMetricValue, hv] at hvalue
                      | some reciprocal =>
                          refine ih (inverse :: tail) (reciprocal :: points)
                            result value ?_ ?_ ?_
                          · exact StackContained.cons
                              (Interval.inv_contains interval inverse _ reciprocal
                                hi hv contained) htail
                          · simpa [evalMetricProgram, hi] using hinterval
                          · simpa [evalMetricValue, hv] using hvalue

inductive BandPosition where
  | below
  | within
  | above
deriving Repr, BEq, Inhabited

inductive BandRelation where
  | inBand
  | below
  | above
  | mixed
deriving Repr, BEq, Inhabited

inductive MetricAuthority where
  | theorem
  | assumptionBacked
deriving Repr, BEq, Inhabited

structure BandCounts where
  below : Nat := 0
  within : Nat := 0
  above : Nat := 0
deriving Repr, BEq, Inhabited

def classifySample (expected : Interval) (sample : Nat) : BandPosition :=
  let value := Ratio.ofNat sample
  if value.leb expected.lower && !value.eqb expected.lower then .below
  else if expected.upper.leb value && !expected.upper.eqb value then .above
  else .within

theorem classifySample_below_iff (expected : Interval) (sample : Nat) :
    classifySample expected sample = .below ↔
      ((Ratio.ofNat sample).leb expected.lower &&
        !(Ratio.ofNat sample).eqb expected.lower) = true := by
  simp only [classifySample]
  by_cases hbelow :
      ((Ratio.ofNat sample).leb expected.lower &&
        !(Ratio.ofNat sample).eqb expected.lower) = true
  · simp [hbelow]
  · by_cases habove :
        (expected.upper.leb (Ratio.ofNat sample) &&
          !expected.upper.eqb (Ratio.ofNat sample)) = true
    · simp [hbelow, habove]
    · simp [hbelow, habove]

theorem classifySample_above_iff (expected : Interval) (sample : Nat) :
    classifySample expected sample = .above ↔
      ((Ratio.ofNat sample).leb expected.lower &&
        !(Ratio.ofNat sample).eqb expected.lower) = false ∧
      (expected.upper.leb (Ratio.ofNat sample) &&
        !expected.upper.eqb (Ratio.ofNat sample)) = true := by
  simp only [classifySample]
  by_cases hbelow :
      ((Ratio.ofNat sample).leb expected.lower &&
        !(Ratio.ofNat sample).eqb expected.lower) = true
  · simp [hbelow]
  · by_cases habove :
        (expected.upper.leb (Ratio.ofNat sample) &&
          !expected.upper.eqb (Ratio.ofNat sample)) = true
    · simp [hbelow, habove]
    · simp [hbelow, habove]

theorem classifySample_within_iff (expected : Interval) (sample : Nat) :
    classifySample expected sample = .within ↔
      ((Ratio.ofNat sample).leb expected.lower &&
        !(Ratio.ofNat sample).eqb expected.lower) = false ∧
      (expected.upper.leb (Ratio.ofNat sample) &&
        !expected.upper.eqb (Ratio.ofNat sample)) = false := by
  simp only [classifySample]
  by_cases hbelow :
      ((Ratio.ofNat sample).leb expected.lower &&
        !(Ratio.ofNat sample).eqb expected.lower) = true
  · simp [hbelow]
  · by_cases habove :
        (expected.upper.leb (Ratio.ofNat sample) &&
          !expected.upper.eqb (Ratio.ofNat sample)) = true
    · simp [hbelow, habove]
    · simp [hbelow, habove]

def BandCounts.add (counts : BandCounts) : BandPosition → BandCounts
  | .below => { counts with below := counts.below + 1 }
  | .within => { counts with within := counts.within + 1 }
  | .above => { counts with above := counts.above + 1 }

def countPositions (expected : Interval) : List Nat → BandCounts
  | [] => {}
  | sample :: rest => (countPositions expected rest).add (classifySample expected sample)

def BandCounts.relation (counts : BandCounts) : BandRelation :=
  if counts.below = 0 && counts.above = 0 then .inBand
  else if counts.within = 0 && counts.above = 0 then .below
  else if counts.within = 0 && counts.below = 0 then .above
  else .mixed

structure MetricExpectation where
  metricId : String
  unit : String
  authority : MetricAuthority
  dependencies : List NamedInterval
  program : List MetricInstruction
  observations : List Nat
deriving Repr, BEq, Inhabited

structure BandAssessment where
  metricId : String
  unit : String
  authority : MetricAuthority
  expected : Interval
  observed : Interval
  counts : BandCounts
  relation : BandRelation
deriving Repr, BEq, Inhabited

def assessExpectation (expectation : MetricExpectation) : Option BandAssessment := do
  if expectation.metricId.isEmpty || expectation.unit.isEmpty then none else pure ()
  if expectation.dependencies.any (fun dependency =>
      dependency.name.isEmpty || !dependency.value.valid) then none else pure ()
  let expected ← evalMetricProgram expectation.dependencies expectation.program []
  if !expected.valid then none else pure ()
  let observed ← sampleInterval expectation.observations
  let counts := countPositions expected expectation.observations
  pure {
    metricId := expectation.metricId
    unit := expectation.unit
    authority := expectation.authority
    expected
    observed
    counts
    relation := counts.relation
  }

def assessExpectations : List MetricExpectation → Option (List BandAssessment)
  | [] => some []
  | expectation :: rest => do
      let assessment ← assessExpectation expectation
      let assessments ← assessExpectations rest
      pure (assessment :: assessments)

theorem BandCounts_add_total (counts : BandCounts) (position : BandPosition) :
    (counts.add position).below + (counts.add position).within +
      (counts.add position).above =
    counts.below + counts.within + counts.above + 1 := by
  cases position <;> simp [BandCounts.add] <;> omega

theorem countPositions_total (expected : Interval) (samples : List Nat) :
    let counts := countPositions expected samples
    counts.below + counts.within + counts.above = samples.length := by
  induction samples with
  | nil => simp [countPositions]
  | cons sample rest ih =>
      simp only [countPositions, List.length_cons]
      rw [BandCounts_add_total, ih]

theorem inBand_has_no_violations (counts : BandCounts)
    (h : counts.relation = .inBand) :
    counts.below = 0 ∧ counts.above = 0 := by
  by_cases hzero : counts.below = 0 ∧ counts.above = 0
  · exact hzero
  · have hnot : counts.relation ≠ .inBand := by
      unfold BandCounts.relation
      split
      · next hcondition =>
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hcondition
          exact (hzero hcondition).elim
      · split
        · simp
        · split <;> simp
    exact (hnot h).elim

theorem relation_inBand_iff (counts : BandCounts) :
    counts.relation = .inBand ↔ counts.below = 0 ∧ counts.above = 0 := by
  constructor
  · exact inBand_has_no_violations counts
  · intro h
    simp [BandCounts.relation, h]

end LeverProofLean
