/-
  Mechanism triggers, no-effect implications, and corpus-local dominance
  (KERN-08 / SLM-531).

  Theorem boundary for reverse-experimental admission:
  if enabling a mechanism changes an output, a declared necessary trigger must
  have occurred; therefore proven trigger absence implies no output effect on
  the locked corpus. Dominance adds equal outputs plus treatment cost ≥
  baseline cost under a named KERN-06 machine model.

  Corpus-local only — never generalize to unseen inputs. Unknown activation
  never becomes no-effect. Counterexamples show mechanism classes with no safe
  trigger theorem.
-/

import LeverProofLean.EventTrace

namespace LeverProofLean.MechanismTrigger

open LeverProofLean.EventTrace

/-- Opaque fixture output identity (equality only; not a semantic claim). -/
abbrev OutputId := Nat

/-- Abstract cost under a named ``MachineModel`` (KERN-06). -/
abbrev Cost := Nat

/-- Finite fixture / input identifier. -/
abbrev InputId := String

/--
  Trigger evidence for one fixture under one mechanism.

  ``unknown`` is incomplete evidence: it can never discharge a no-effect or
  dominance certificate.
-/
inductive TriggerEvidence where
  | absent
  | present
  | unknown
  deriving Repr, BEq, DecidableEq

/-- Mechanism identity + whether a safe necessary-trigger theorem is available. -/
structure Mechanism where
  id : String
  hasSafeTriggerTheorem : Bool
  deriving Repr, BEq

/-- One locked-corpus observation: baseline vs enabled under a named cost model. -/
structure Observation where
  inputId : InputId
  baselineOutput : OutputId
  enabledOutput : OutputId
  baselineCost : Cost
  enabledCost : Cost
  trigger : TriggerEvidence
  deriving Repr, BEq

/-- Named cost-model identity persisted on certificates (KERN-06). -/
structure NamedCostModel where
  id : String
  version : Nat
  deriving Repr, BEq

def NamedCostModel.tag (m : NamedCostModel) : String :=
  m.id ++ ".v" ++ toString m.version

def neuralForwardOnly : NamedCostModel :=
  { id := "NeuralForwardOnlyModel", version := 1 }

def decodeUnitWork : NamedCostModel :=
  { id := "DecodeUnitWorkModel", version := 1 }

/-- Complete trigger evidence (not ``unknown``). -/
def triggerEvidenceComplete (t : TriggerEvidence) : Prop :=
  t ≠ .unknown

def triggerEvidenceCompleteBool (t : TriggerEvidence) : Bool :=
  !(t == .unknown)

/-- Output effect on one observation. -/
def outputsDiffer (o : Observation) : Prop :=
  o.enabledOutput ≠ o.baselineOutput

def outputsEqual (o : Observation) : Prop :=
  o.enabledOutput = o.baselineOutput

/-- Necessary-trigger law (statement form): effect ⇒ trigger present. -/
def NecessaryTrigger (o : Observation) : Prop :=
  outputsDiffer o → o.trigger = .present

/-- Contrapositive no-effect form: trigger absent ⇒ outputs equal. -/
def NoEffectFromAbsent (o : Observation) : Prop :=
  o.trigger = .absent → outputsEqual o

/-- Classical helper: ¬(a ≠ b) → a = b on Nat. -/
theorem nat_eq_of_not_ne {a b : Nat} (h : ¬ a ≠ b) : a = b :=
  Classical.not_not.mp h

/-- Effect ⇒ present, plus absent, yields equal outputs (contrapositive discharge). -/
theorem no_effect_of_necessary_and_absent
    (o : Observation) (hn : NecessaryTrigger o) (ha : o.trigger = .absent) :
    outputsEqual o := by
  by_cases hd : outputsDiffer o
  · have hp := hn hd
    simp [ha] at hp
  · exact nat_eq_of_not_ne hd

theorem necessary_implies_no_effect_contrapositive (o : Observation)
    (hn : NecessaryTrigger o) : NoEffectFromAbsent o :=
  fun ha => no_effect_of_necessary_and_absent o hn ha
/-- Corpus-local dominance: equal outputs and treatment cost ≥ baseline. -/
def dominatesLocally (o : Observation) : Prop :=
  outputsEqual o ∧ o.enabledCost ≥ o.baselineCost

/-! ### Certificate structures (autoresearch admission; corpus-local) -/

/--
  No-effect certificate over a locked finite corpus.

  Requires **complete** trigger evidence on every row, a discharged necessary-
  trigger law per row, and proven absence on every row. Does **not** claim
  anything about inputs outside ``corpus``.
-/
structure NoEffectCertificate where
  mechanismId : String
  costModel : NamedCostModel
  corpusId : String
  corpus : List Observation
  evidenceComplete :
    ∀ o, o ∈ corpus → triggerEvidenceComplete o.trigger
  necessary :
    ∀ o, o ∈ corpus → NecessaryTrigger o
  allAbsent :
    ∀ o, o ∈ corpus → o.trigger = .absent

theorem noEffectCertificate_outputs_equal
    (c : NoEffectCertificate) (o : Observation) (hin : o ∈ c.corpus) :
    outputsEqual o :=
  no_effect_of_necessary_and_absent o (c.necessary o hin) (c.allAbsent o hin)

/-- Unknown rows are rejected by the complete-evidence obligation. -/
theorem unknown_not_in_no_effect_corpus
    (c : NoEffectCertificate) (o : Observation)
    (hin : o ∈ c.corpus) (hu : o.trigger = .unknown) : False := by
  have hc := c.evidenceComplete o hin
  simp [triggerEvidenceComplete, hu] at hc

/--
  Dominance certificate: no-effect on the locked corpus plus weakly worse
  treatment cost under the named model (enabled ≥ baseline).
-/
structure DominanceCertificate where
  noEffect : NoEffectCertificate
  costDominated :
    ∀ o, o ∈ noEffect.corpus → o.enabledCost ≥ o.baselineCost

theorem dominanceCertificate_dominates
    (d : DominanceCertificate) (o : Observation)
    (hin : o ∈ d.noEffect.corpus) : dominatesLocally o :=
  ⟨noEffectCertificate_outputs_equal d.noEffect o hin, d.costDominated o hin⟩

/-! ### Supported mechanism classes (necessary condition justified) -/

/-- Singleton-bypass instrumentation: bypass can change output only on singleton. -/
structure SingletonBypassRun where
  inputId : InputId
  isSingleton : Bool
  baselineOutput : OutputId
  enabledOutput : OutputId
  baselineCost : Cost
  enabledCost : Cost
  /-- Well-formed: non-singleton ⇒ enabled equals baseline (bypass cannot fire). -/
  wf : ¬ isSingleton = true → enabledOutput = baselineOutput

def SingletonBypassRun.trigger (r : SingletonBypassRun) : TriggerEvidence :=
  if r.isSingleton then .present else .absent

def SingletonBypassRun.toObservation (r : SingletonBypassRun) : Observation :=
  { inputId := r.inputId
    baselineOutput := r.baselineOutput
    enabledOutput := r.enabledOutput
    baselineCost := r.baselineCost
    enabledCost := r.enabledCost
    trigger := r.trigger }

theorem singleton_bypass_output_change_implies_trigger
    (r : SingletonBypassRun) :
    r.enabledOutput ≠ r.baselineOutput → r.isSingleton = true := by
  intro h
  by_cases hs : r.isSingleton = true
  · exact hs
  · exact False.elim (h (r.wf hs))

theorem singleton_bypass_necessary (r : SingletonBypassRun) :
    NecessaryTrigger r.toObservation := by
  intro hd
  have hne : r.enabledOutput ≠ r.baselineOutput := by
    simpa [SingletonBypassRun.toObservation, outputsDiffer] using hd
  have hs := singleton_bypass_output_change_implies_trigger r hne
  simp [SingletonBypassRun.toObservation, SingletonBypassRun.trigger, hs]
theorem singleton_bypass_absent_no_effect (r : SingletonBypassRun)
    (ha : r.trigger = .absent) : r.enabledOutput = r.baselineOutput := by
  have hn := singleton_bypass_necessary r
  exact no_effect_of_necessary_and_absent r.toObservation hn (by
    simpa [SingletonBypassRun.toObservation] using ha)

/-- Closure-removal: removals can change live set only when removable nonempty. -/
structure ClosureRemovalRun where
  inputId : InputId
  removableNonempty : Bool
  baselineOutput : OutputId
  enabledOutput : OutputId
  baselineCost : Cost
  enabledCost : Cost
  wf : ¬ removableNonempty = true → enabledOutput = baselineOutput

def ClosureRemovalRun.trigger (r : ClosureRemovalRun) : TriggerEvidence :=
  if r.removableNonempty then .present else .absent

def ClosureRemovalRun.toObservation (r : ClosureRemovalRun) : Observation :=
  { inputId := r.inputId
    baselineOutput := r.baselineOutput
    enabledOutput := r.enabledOutput
    baselineCost := r.baselineCost
    enabledCost := r.enabledCost
    trigger := r.trigger }

theorem closure_removal_output_change_implies_trigger
    (r : ClosureRemovalRun) :
    r.enabledOutput ≠ r.baselineOutput → r.removableNonempty = true := by
  intro h
  by_cases hs : r.removableNonempty = true
  · exact hs
  · exact False.elim (h (r.wf hs))

theorem closure_removal_necessary (r : ClosureRemovalRun) :
    NecessaryTrigger r.toObservation := by
  intro hd
  have hne : r.enabledOutput ≠ r.baselineOutput := by
    simpa [ClosureRemovalRun.toObservation, outputsDiffer] using hd
  have hs := closure_removal_output_change_implies_trigger r hne
  simp [ClosureRemovalRun.toObservation, ClosureRemovalRun.trigger, hs]

/-- Cache reuse: output path can diverge from cold path only on a cache hit. -/
structure CacheReuseRun where
  inputId : InputId
  cacheHit : Bool
  baselineOutput : OutputId
  enabledOutput : OutputId
  baselineCost : Cost
  enabledCost : Cost
  wf : ¬ cacheHit = true → enabledOutput = baselineOutput

def CacheReuseRun.trigger (r : CacheReuseRun) : TriggerEvidence :=
  if r.cacheHit then .present else .absent

def CacheReuseRun.toObservation (r : CacheReuseRun) : Observation :=
  { inputId := r.inputId
    baselineOutput := r.baselineOutput
    enabledOutput := r.enabledOutput
    baselineCost := r.baselineCost
    enabledCost := r.enabledCost
    trigger := r.trigger }

theorem cache_reuse_output_change_implies_trigger
    (r : CacheReuseRun) :
    r.enabledOutput ≠ r.baselineOutput → r.cacheHit = true := by
  intro h
  by_cases hs : r.cacheHit = true
  · exact hs
  · exact False.elim (h (r.wf hs))

theorem cache_reuse_necessary (r : CacheReuseRun) :
    NecessaryTrigger r.toObservation := by
  intro hd
  have hne : r.enabledOutput ≠ r.baselineOutput := by
    simpa [CacheReuseRun.toObservation, outputsDiffer] using hd
  have hs := cache_reuse_output_change_implies_trigger r hne
  simp [CacheReuseRun.toObservation, CacheReuseRun.trigger, hs]

/-- Forced span commit: can change only when a forced span is nonempty. -/
structure ForcedSpanRun where
  inputId : InputId
  forcedNonempty : Bool
  baselineOutput : OutputId
  enabledOutput : OutputId
  baselineCost : Cost
  enabledCost : Cost
  wf : ¬ forcedNonempty = true → enabledOutput = baselineOutput

def ForcedSpanRun.trigger (r : ForcedSpanRun) : TriggerEvidence :=
  if r.forcedNonempty then .present else .absent

def ForcedSpanRun.toObservation (r : ForcedSpanRun) : Observation :=
  { inputId := r.inputId
    baselineOutput := r.baselineOutput
    enabledOutput := r.enabledOutput
    baselineCost := r.baselineCost
    enabledCost := r.enabledCost
    trigger := r.trigger }

theorem forced_span_output_change_implies_trigger
    (r : ForcedSpanRun) :
    r.enabledOutput ≠ r.baselineOutput → r.forcedNonempty = true := by
  intro h
  by_cases hs : r.forcedNonempty = true
  · exact hs
  · exact False.elim (h (r.wf hs))

theorem forced_span_necessary (r : ForcedSpanRun) :
    NecessaryTrigger r.toObservation := by
  intro hd
  have hne : r.enabledOutput ≠ r.baselineOutput := by
    simpa [ForcedSpanRun.toObservation, outputsDiffer] using hd
  have hs := forced_span_output_change_implies_trigger r hne
  simp [ForcedSpanRun.toObservation, ForcedSpanRun.trigger, hs]

/-! ### Counterexamples: no safe trigger theorem / unknown never no-effect -/

/--
  Unconstrained learned rerank: outputs may differ with no declared necessary
  trigger. There is no ``wf`` relating a trigger bit to output equality.
-/
structure UnconstrainedRerankRun where
  inputId : InputId
  baselineOutput : OutputId
  enabledOutput : OutputId
  baselineCost : Cost
  enabledCost : Cost

/-- Exists a change with no safe necessary-trigger hypothesis available. -/
theorem unconstrained_rerank_no_safe_trigger_theorem :
    ∃ r : UnconstrainedRerankRun, r.enabledOutput ≠ r.baselineOutput :=
  ⟨{ inputId := "counterexample/unconstrained_rerank"
     baselineOutput := 0
     enabledOutput := 1
     baselineCost := 1
     enabledCost := 1 }, by decide⟩

/-- Mechanism registry flag: unconstrained rerank has no safe theorem. -/
def unconstrainedRerankMechanism : Mechanism :=
  { id := "unconstrained_rerank", hasSafeTriggerTheorem := false }

def singletonBypassMechanism : Mechanism :=
  { id := "singleton_bypass", hasSafeTriggerTheorem := true }

def closureRemovalMechanism : Mechanism :=
  { id := "closure_removal", hasSafeTriggerTheorem := true }

def cacheReuseMechanism : Mechanism :=
  { id := "cache_reuse", hasSafeTriggerTheorem := true }

def forcedSpanMechanism : Mechanism :=
  { id := "forced_span", hasSafeTriggerTheorem := true }

/--
  Unknown activation cannot discharge no-effect: a corpus containing
  ``unknown`` cannot satisfy ``evidenceComplete``.
-/
theorem unknown_activation_blocks_no_effect_certificate
    (mechId corpusId : String) (model : NamedCostModel)
    (o : Observation) (hu : o.trigger = .unknown)
    (rest : List Observation) :
    ¬ ∃ c : NoEffectCertificate,
        c.mechanismId = mechId ∧ c.corpusId = corpusId ∧ c.costModel = model ∧
        c.corpus = o :: rest := by
  intro ⟨c, _, _, _, hcorp⟩
  have hin : o ∈ c.corpus := by simp [hcorp]
  exact unknown_not_in_no_effect_corpus c o hin hu

/-! ### Executable fixtures (#guard / Python parity) -/

def fixtureSingletonAbsent : SingletonBypassRun where
  inputId := "fixture/singleton_absent"
  isSingleton := false
  baselineOutput := 42
  enabledOutput := 42
  baselineCost := 3
  enabledCost := 3
  wf := by intro; rfl

def fixtureClosureAbsent : ClosureRemovalRun where
  inputId := "fixture/closure_absent"
  removableNonempty := false
  baselineOutput := 7
  enabledOutput := 7
  baselineCost := 5
  enabledCost := 5
  wf := by intro; rfl

def fixtureCacheAbsent : CacheReuseRun where
  inputId := "fixture/cache_absent"
  cacheHit := false
  baselineOutput := 11
  enabledOutput := 11
  baselineCost := 9
  enabledCost := 9
  wf := by intro; rfl

def fixtureForcedAbsent : ForcedSpanRun where
  inputId := "fixture/forced_absent"
  forcedNonempty := false
  baselineOutput := 13
  enabledOutput := 13
  baselineCost := 4
  enabledCost := 4
  wf := by intro; rfl

/-- Dominated no-effect row: equal outputs, enabled cost ≥ baseline. -/
def fixtureDominatedAbsent : Observation where
  inputId := "fixture/dominated_absent"
  baselineOutput := 1
  enabledOutput := 1
  baselineCost := 2
  enabledCost := 5
  trigger := .absent

theorem fixture_singleton_absent_no_effect :
    fixtureSingletonAbsent.enabledOutput = fixtureSingletonAbsent.baselineOutput := by
  decide

theorem fixture_closure_absent_no_effect :
    fixtureClosureAbsent.enabledOutput = fixtureClosureAbsent.baselineOutput := by
  decide

theorem fixture_cache_absent_no_effect :
    fixtureCacheAbsent.enabledOutput = fixtureCacheAbsent.baselineOutput := by
  decide

theorem fixture_forced_absent_no_effect :
    fixtureForcedAbsent.enabledOutput = fixtureForcedAbsent.baselineOutput := by
  decide

theorem fixture_dominated_locally :
    dominatesLocally fixtureDominatedAbsent :=
  ⟨rfl, by decide⟩

/-- Present trigger may (and here does) change output — not a no-effect row. -/
def fixtureSingletonPresent : SingletonBypassRun where
  inputId := "fixture/singleton_present"
  isSingleton := true
  baselineOutput := 1
  enabledOutput := 99
  baselineCost := 4
  enabledCost := 0
  wf := by intro h; exact absurd rfl h

theorem fixture_singleton_present_change_requires_trigger :
    fixtureSingletonPresent.enabledOutput ≠ fixtureSingletonPresent.baselineOutput →
      fixtureSingletonPresent.isSingleton = true :=
  singleton_bypass_output_change_implies_trigger fixtureSingletonPresent

end LeverProofLean.MechanismTrigger
