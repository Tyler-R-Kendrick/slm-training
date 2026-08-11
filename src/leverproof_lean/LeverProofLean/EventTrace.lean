/-
  Explicit execution-event traces and named machine cost models (KERN-06 / SLM-524).

  Abstract costs are Nat-valued folds over Event traces under a parameterized
  ``MachineModel.cost``. Named built-in models mirror counters the repo already
  records (DecodeStats / train-loop phase owners). No theorem equates abstract
  cost with physical wall-clock latency unless an explicit external throughput
  or latency hypothesis is supplied.
-/

namespace LeverProofLean.EventTrace

/--
  Atomic work kinds justified by current telemetry owners.

  DecodeStats owns tokenize / grammar / solver / certificate / neural-forward
  counters. Train-loop phase timers own backward / optimizer. Byte lex counters
  may witness memory reads; CUDA / host kernel launches are not yet counted in
  DecodeStats and stay unobserved under decode projection.
-/
inductive Event where
  | tokenize (tokens : Nat)
  | grammarTransition (steps : Nat)
  | solverExpansion (nodes : Nat)
  | certificateCheck (checks : Nat)
  | neuralForward (forwards : Nat)
  | neuralBackward (passes : Nat)
  | optimizerStep (steps : Nat)
  | memoryRead (bytes : Nat)
  | memoryWrite (bytes : Nat)
  | kernelLaunch (launches : Nat)
  deriving Repr, BEq, DecidableEq

/-- Finite execution trace (list of events). -/
abbrev Trace := List Event

/--
  Parameterized machine / cost model.

  ``cost`` maps each event to an abstract Nat cost. ``id`` / ``version`` are
  persisted in four-axis evidence so model identity is never ambient.
-/
structure MachineModel where
  id : String
  version : Nat
  cost : Event → Nat

/-- Fold abstract cost over a trace. -/
def traceCost (m : MachineModel) : Trace → Nat
  | [] => 0
  | e :: es => m.cost e + traceCost m es

/-- Empty-trace cost is zero. -/
theorem traceCost_nil (m : MachineModel) : traceCost m [] = 0 :=
  rfl

/-- Nat costs are nonnegative (definitional; stated for the export surface). -/
theorem cost_nonneg (m : MachineModel) (e : Event) : ∃ n, m.cost e = n :=
  ⟨m.cost e, rfl⟩

theorem traceCost_nonneg (m : MachineModel) (t : Trace) : ∃ n, traceCost m t = n :=
  ⟨traceCost m t, rfl⟩

/-- Composition / additivity of abstract cost under concatenation. -/
theorem traceCost_append (m : MachineModel) (t₁ t₂ : Trace) :
    traceCost m (t₁ ++ t₂) = traceCost m t₁ + traceCost m t₂ := by
  induction t₁ with
  | nil =>
      simp [traceCost]
  | cons e es ih =>
      simp [traceCost, Nat.add_assoc, ih]

/-- Singleton cost equals the model cost of that event. -/
theorem traceCost_singleton (m : MachineModel) (e : Event) :
    traceCost m [e] = m.cost e := by
  simp [traceCost]

/-! ### Named built-in models (repo-recorded metrics) -/

/-- Unit decode-work cost: payload counts for decode-owned events; train / memory / launch = 0. -/
def decodeUnitWorkCost : Event → Nat
  | .tokenize n => n
  | .grammarTransition n => n
  | .solverExpansion n => n
  | .certificateCheck n => n
  | .neuralForward n => n
  | .neuralBackward _ => 0
  | .optimizerStep _ => 0
  | .memoryRead _ => 0
  | .memoryWrite _ => 0
  | .kernelLaunch _ => 0

/-- Named model: DecodeUnitWorkModel v1 (DecodeStats counter projection). -/
def decodeUnitWorkModel : MachineModel where
  id := "DecodeUnitWorkModel"
  version := 1
  cost := decodeUnitWorkCost

/-- Solver expansion + certificate verification only (KERN-03 family bridge). -/
def solverQueryCost : Event → Nat
  | .solverExpansion n => n
  | .certificateCheck n => n
  | _ => 0

def solverQueryModel : MachineModel where
  id := "SolverQueryModel"
  version := 1
  cost := solverQueryCost

/-- Neural forward events only (I2 zero-neural-work surface). -/
def neuralForwardOnlyCost : Event → Nat
  | .neuralForward n => n
  | _ => 0

def neuralForwardOnlyModel : MachineModel where
  id := "NeuralForwardOnlyModel"
  version := 1
  cost := neuralForwardOnlyCost

/-- Train step events (forward + backward + optimizer); decode grammar/solver = 0. -/
def trainStepCost : Event → Nat
  | .neuralForward n => n
  | .neuralBackward n => n
  | .optimizerStep n => n
  | _ => 0

def trainStepModel : MachineModel where
  id := "TrainStepModel"
  version := 1
  cost := trainStepCost

/-- Compact count vector for the five DecodeUnitWorkModel axes. -/
structure DecodeUnitWorkCounts where
  tokenize : Nat
  grammarTransition : Nat
  solverExpansion : Nat
  certificateCheck : Nat
  neuralForward : Nat
  deriving Repr, BEq

/-- Encode counts as a five-event trace (lossless for these axes). -/
def decodeUnitWorkTrace (c : DecodeUnitWorkCounts) : Trace :=
  [
    .tokenize c.tokenize,
    .grammarTransition c.grammarTransition,
    .solverExpansion c.solverExpansion,
    .certificateCheck c.certificateCheck,
    .neuralForward c.neuralForward
  ]

/-- Closed-form cost under DecodeUnitWorkModel. -/
def decodeUnitWorkCostOfCounts (c : DecodeUnitWorkCounts) : Nat :=
  c.tokenize + c.grammarTransition + c.solverExpansion +
    c.certificateCheck + c.neuralForward

theorem decodeUnitWork_traceCost_eq_sum (c : DecodeUnitWorkCounts) :
    traceCost decodeUnitWorkModel (decodeUnitWorkTrace c) =
      decodeUnitWorkCostOfCounts c := by
  simp [decodeUnitWorkTrace, decodeUnitWorkCostOfCounts, decodeUnitWorkModel,
    decodeUnitWorkCost, traceCost, Nat.add_assoc]

/-! ### Conditional wall-clock transfer (external assumptions only) -/

/--
  External throughput / latency hypothesis shape.

  ``latencyPerCostUnit`` is **not** derived inside Lean — callers supply it.
  Physical latency claims require discharging ``PhysicalLatencyHypothesis``.
-/
structure ThroughputAssumption where
  latencyPerCostUnit : Nat
  deriving Repr, BEq

/-- Abstract upper shape: cost × assumed latency-per-unit. -/
def wallClockUpperBound (m : MachineModel) (a : ThroughputAssumption) (t : Trace) : Nat :=
  traceCost m t * a.latencyPerCostUnit

theorem wallClockUpperBound_eq_mul
    (m : MachineModel) (a : ThroughputAssumption) (t : Trace) :
    wallClockUpperBound m a t = traceCost m t * a.latencyPerCostUnit :=
  rfl

/--
  External hypothesis: observed physical nanoseconds are ≤ the abstract bound
  under the supplied throughput assumption. This library does **not** prove the
  hypothesis; it only transfers consequences once the hypothesis is given.
-/
def PhysicalLatencyHypothesis
    (m : MachineModel) (a : ThroughputAssumption) (t : Trace) (physicalNs : Nat) : Prop :=
  physicalNs ≤ wallClockUpperBound m a t

theorem physical_bound_from_hypothesis
    (m : MachineModel) (a : ThroughputAssumption) (t : Trace) (physicalNs : Nat)
    (h : PhysicalLatencyHypothesis m a t physicalNs) :
    physicalNs ≤ wallClockUpperBound m a t :=
  h

/-- Composition of wall-clock upper shapes under the same assumption. -/
theorem wallClockUpperBound_append
    (m : MachineModel) (a : ThroughputAssumption) (t₁ t₂ : Trace) :
    wallClockUpperBound m a (t₁ ++ t₂) =
      wallClockUpperBound m a t₁ + wallClockUpperBound m a t₂ := by
  simp [wallClockUpperBound, traceCost_append, Nat.add_mul]

/-- Fixture: unit-work sum for counts (2,3,5,7,11) = 28. -/
theorem fixture_decode_unit_work_sum :
    decodeUnitWorkCostOfCounts
      { tokenize := 2, grammarTransition := 3, solverExpansion := 5,
        certificateCheck := 7, neuralForward := 11 } = 28 := by
  decide

theorem fixture_traceCost_append_unit :
    traceCost decodeUnitWorkModel
        ([.neuralForward 2] ++ [.solverExpansion 3]) = 5 := by
  simp [traceCost, decodeUnitWorkModel, decodeUnitWorkCost]

theorem fixture_wall_clock_shape :
    wallClockUpperBound decodeUnitWorkModel ⟨10⟩ [.neuralForward 3] = 30 := by
  simp [wallClockUpperBound, traceCost, decodeUnitWorkModel, decodeUnitWorkCost]

end LeverProofLean.EventTrace
