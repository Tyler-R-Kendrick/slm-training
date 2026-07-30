import LeverProofLean.Interval

namespace LeverProofLean

structure NamedInterval where
  name : String
  value : Interval
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

end LeverProofLean
