/-
  Proof-carrying goal support — finite structural laws (PGS-F01 / SLM-506).

  Models only what Python goal-support actually claims:
  action partitions, certified removal authority, domain-adequacy classification,
  obstruction hitting sets, required-constraint witnesses, and certified singleton
  survivors. No NL meaning, evaluator correctness, model accuracy, or global
  OpenUI satisfiability.
-/

import LeverProofLean.ListSet
import LeverProofLean.ExactClosure

namespace LeverProofLean.GoalSupport

open LeverProofLean

abbrev Action := Nat
abbrev Atom := Nat
abbrev Terminal := Nat
abbrev Constraint := Nat

/-! ### Partitions -/

inductive Partition where
  | supported
  | unsupported
  | unknown
  | unobserved
deriving DecidableEq, Repr, BEq

structure Partitions where
  legal : List Action
  supported : List Action
  unsupported : List Action
  unknown : List Action
  unobserved : List Action
deriving Repr, BEq

def Partitions.cover (p : Partitions) : List Action :=
  p.supported ++ p.unsupported ++ p.unknown ++ p.unobserved

/-- Constructor assumption: four buckets are pairwise disjoint and cover `legal`. -/
structure WellFormedPartitions where
  parts : Partitions
  cover_legal : SetEq (Partitions.cover parts) parts.legal
  supported_subset : Subset parts.supported parts.legal
  unsupported_subset : Subset parts.unsupported parts.legal
  unknown_subset : Subset parts.unknown parts.legal
  unobserved_subset : Subset parts.unobserved parts.legal
  supported_disjoint_unsupported :
    ∀ a, a ∈ parts.supported → a ∉ parts.unsupported
  supported_disjoint_unknown :
    ∀ a, a ∈ parts.supported → a ∉ parts.unknown
  supported_disjoint_unobserved :
    ∀ a, a ∈ parts.supported → a ∉ parts.unobserved
  unsupported_disjoint_unknown :
    ∀ a, a ∈ parts.unsupported → a ∉ parts.unknown
  unsupported_disjoint_unobserved :
    ∀ a, a ∈ parts.unsupported → a ∉ parts.unobserved
  unknown_disjoint_unobserved :
    ∀ a, a ∈ parts.unknown → a ∉ parts.unobserved

namespace WellFormedPartitions

theorem cover_mem (wf : WellFormedPartitions) (a : Action) :
    a ∈ Partitions.cover wf.parts ↔ a ∈ wf.parts.legal := by
  constructor
  · intro h
    exact wf.cover_legal.1 a h
  · intro h
    exact wf.cover_legal.2 a h

theorem wellformed_partitions_cover_legal (wf : WellFormedPartitions) :
    SetEq (Partitions.cover wf.parts) wf.parts.legal :=
  wf.cover_legal

theorem buckets_pairwise_disjoint (wf : WellFormedPartitions) (a : Action) :
    (a ∈ wf.parts.supported →
      a ∉ wf.parts.unsupported ∧ a ∉ wf.parts.unknown ∧ a ∉ wf.parts.unobserved) ∧
    (a ∈ wf.parts.unsupported →
      a ∉ wf.parts.supported ∧ a ∉ wf.parts.unknown ∧ a ∉ wf.parts.unobserved) ∧
    (a ∈ wf.parts.unknown →
      a ∉ wf.parts.supported ∧ a ∉ wf.parts.unsupported ∧ a ∉ wf.parts.unobserved) ∧
    (a ∈ wf.parts.unobserved →
      a ∉ wf.parts.supported ∧ a ∉ wf.parts.unsupported ∧ a ∉ wf.parts.unknown) := by
  constructor
  · intro hs
    exact ⟨wf.supported_disjoint_unsupported a hs,
      wf.supported_disjoint_unknown a hs,
      wf.supported_disjoint_unobserved a hs⟩
  constructor
  · intro hu
    exact ⟨fun h => wf.supported_disjoint_unsupported a h hu,
      wf.unsupported_disjoint_unknown a hu,
      wf.unsupported_disjoint_unobserved a hu⟩
  constructor
  · intro hk
    exact ⟨fun h => wf.supported_disjoint_unknown a h hk,
      fun h => wf.unsupported_disjoint_unknown a h hk,
      wf.unknown_disjoint_unobserved a hk⟩
  · intro ho
    exact ⟨fun h => wf.supported_disjoint_unobserved a h ho,
      fun h => wf.unsupported_disjoint_unobserved a h ho,
      fun h => wf.unknown_disjoint_unobserved a h ho⟩

end WellFormedPartitions

/-! ### Certified removal -/

structure ActionEvidence where
  action : Action
  partition : Partition
  replayOk : Bool
  hardProfile : Bool
deriving Repr, BEq

/-- Certified removal authority mirrors ``_assign_partition`` unsupported branch. -/
def certifiedRemovable (e : ActionEvidence) : Bool :=
  match e.partition with
  | .unsupported => e.replayOk && e.hardProfile
  | _ => false

/-- Certified removal actions live in the unsupported bucket with replay-valid hard evidence. -/
def certifiedRemovalSet (p : Partitions) (evidence : List ActionEvidence) : List Action :=
  p.unsupported.filter fun a =>
    evidence.any fun e => e.action = a && certifiedRemovable e

def unsupportedEvidenceActions (evidence : List ActionEvidence) : List Action :=
  (evidence.filter fun e => e.partition = .unsupported).map (·.action)

theorem unknown_not_certified_removable (e : ActionEvidence)
    (h : e.partition = .unknown) : certifiedRemovable e = false := by
  simp [certifiedRemovable, h]

theorem unobserved_not_certified_removable (e : ActionEvidence)
    (h : e.partition = .unobserved) : certifiedRemovable e = false := by
  simp [certifiedRemovable, h]

theorem supported_not_certified_removable (e : ActionEvidence)
    (h : e.partition = .supported) : certifiedRemovable e = false := by
  simp [certifiedRemovable, h]

theorem certified_removal_subset_unsupported_evidence
    (p : Partitions) (evidence : List ActionEvidence) :
    Subset (certifiedRemovalSet p evidence) p.unsupported := by
  intro a ha
  exact (List.mem_filter.mp ha).1

theorem certified_removal_has_unsupported_evidence
    (p : Partitions) (evidence : List ActionEvidence) (a : Action)
    (h : a ∈ certifiedRemovalSet p evidence) :
    ∃ e, e ∈ evidence ∧ e.action = a ∧ e.partition = .unsupported ∧ certifiedRemovable e := by
  simp [certifiedRemovalSet, List.mem_filter, List.any_eq_true] at h
  obtain ⟨_, ⟨e, hev, hact, hrem⟩⟩ := h
  cases hp : e.partition <;> simp [certifiedRemovable, hp] at hrem
  exact ⟨e, And.intro hev (And.intro hact (And.intro hp (by simp [certifiedRemovable, hp, hrem])))⟩

theorem unknown_not_in_certified_removal
    (wf : WellFormedPartitions) (evidence : List ActionEvidence) (a : Action)
    (h : a ∈ wf.parts.unknown) :
    a ∉ certifiedRemovalSet wf.parts evidence := by
  intro ha
  have hsub := certified_removal_subset_unsupported_evidence wf.parts evidence a ha
  have hdisj := (WellFormedPartitions.buckets_pairwise_disjoint wf a).2.2.1 h
  exact hdisj.2.1 hsub

theorem unobserved_not_in_certified_removal
    (wf : WellFormedPartitions) (evidence : List ActionEvidence) (a : Action)
    (h : a ∈ wf.parts.unobserved) :
    a ∉ certifiedRemovalSet wf.parts evidence := by
  intro ha
  have hsub := certified_removal_subset_unsupported_evidence wf.parts evidence a ha
  have hdisj := (WellFormedPartitions.buckets_pairwise_disjoint wf a).2.2.2 h
  exact hdisj.2.1 hsub

/-! ### Domain-adequacy classification -/

inductive DomainAdequacy where
  | adequateSelectedSupported
  | selectionRegret
  | selectionUnresolved
  | domainInadequateUnderBounds
  | coverageUnknown
deriving DecidableEq, Repr, BEq

structure ObstructionSummary where
  present : Bool
deriving Repr, BEq

structure ClassificationInputs where
  selected : Action
  hardProfile : Bool
  capApplied : Bool
  domainComplete : Bool
  allUnsupportedReplayValid : Bool
  obstruction : ObstructionSummary
deriving Repr, BEq

def classifyDomainAdequacy (p : Partitions) (inp : ClassificationInputs) :
    DomainAdequacy :=
  if !p.supported.isEmpty then
    if inp.selected ∈ p.supported then
      .adequateSelectedSupported
    else if inp.selected ∈ p.unsupported then
      .selectionRegret
    else if inp.selected ∈ p.unknown ∨ inp.selected ∈ p.unobserved then
      .selectionUnresolved
    else
      .coverageUnknown
  else if p.unknown.isEmpty ∧ p.unobserved.isEmpty ∧ !p.unsupported.isEmpty ∧
      inp.allUnsupportedReplayValid ∧ inp.hardProfile ∧ !inp.capApplied ∧
      inp.domainComplete then
    .domainInadequateUnderBounds
  else
    .coverageUnknown

def caseAdequateSelectedSupported (p : Partitions) (inp : ClassificationInputs) : Prop :=
  !p.supported.isEmpty ∧ inp.selected ∈ p.supported

def caseSelectionRegret (p : Partitions) (inp : ClassificationInputs) : Prop :=
  !p.supported.isEmpty ∧ inp.selected ∈ p.unsupported

def caseSelectionUnresolved (p : Partitions) (inp : ClassificationInputs) : Prop :=
  !p.supported.isEmpty ∧
    (inp.selected ∈ p.unknown ∨ inp.selected ∈ p.unobserved)

def caseDomainInadequate (p : Partitions) (inp : ClassificationInputs) : Prop :=
  p.supported.isEmpty ∧ p.unknown.isEmpty ∧ p.unobserved.isEmpty ∧
    !p.unsupported.isEmpty ∧ inp.allUnsupportedReplayValid ∧ inp.hardProfile ∧
    !inp.capApplied ∧ inp.domainComplete

theorem classify_adequate (p : Partitions) (inp : ClassificationInputs)
    (h : caseAdequateSelectedSupported p inp) :
    classifyDomainAdequacy p inp = .adequateSelectedSupported := by
  unfold classifyDomainAdequacy caseAdequateSelectedSupported at *
  simp [h.1, h.2]

theorem classify_regret (p : Partitions) (inp : ClassificationInputs)
    (hs : !p.supported.isEmpty)
    (hsel : inp.selected ∈ p.unsupported)
    (hnot : inp.selected ∉ p.supported) :
    classifyDomainAdequacy p inp = .selectionRegret := by
  unfold classifyDomainAdequacy
  simp [hs, hsel, hnot]

theorem classify_unresolved (p : Partitions) (inp : ClassificationInputs)
    (hs : !p.supported.isEmpty)
    (hun : inp.selected ∈ p.unknown ∨ inp.selected ∈ p.unobserved)
    (hnots : inp.selected ∉ p.supported)
    (hnotu : inp.selected ∉ p.unsupported) :
    classifyDomainAdequacy p inp = .selectionUnresolved := by
  unfold classifyDomainAdequacy
  simp [hs, hun, hnots, hnotu]

theorem classify_inadequate (p : Partitions) (inp : ClassificationInputs)
    (h : caseDomainInadequate p inp) :
    classifyDomainAdequacy p inp = .domainInadequateUnderBounds := by
  unfold classifyDomainAdequacy caseDomainInadequate at *
  simp [h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2]

theorem classify_partial_coverage_unknown (p : Partitions) (inp : ClassificationInputs)
    (hnoSupport : p.supported.isEmpty)
    (hpartial : inp.domainComplete = false) :
    classifyDomainAdequacy p inp = .coverageUnknown := by
  unfold classifyDomainAdequacy
  simp [hnoSupport, hpartial]

theorem classification_exhaustive (p : Partitions) (inp : ClassificationInputs) :
    classifyDomainAdequacy p inp = .adequateSelectedSupported ∨
    classifyDomainAdequacy p inp = .selectionRegret ∨
    classifyDomainAdequacy p inp = .selectionUnresolved ∨
    classifyDomainAdequacy p inp = .domainInadequateUnderBounds ∨
    classifyDomainAdequacy p inp = .coverageUnknown := by
  match classifyDomainAdequacy p inp with
  | .adequateSelectedSupported => simp
  | .selectionRegret => simp
  | .selectionUnresolved => simp
  | .domainInadequateUnderBounds => simp
  | .coverageUnknown => simp

theorem classification_mutually_exclusive_under_wellformed
    (wf : WellFormedPartitions) (inp : ClassificationInputs)
    (hsel : inp.selected ∈ wf.parts.legal) :
    (caseAdequateSelectedSupported wf.parts inp →
      classifyDomainAdequacy wf.parts inp ≠ .selectionRegret) ∧
    (caseSelectionRegret wf.parts inp →
      classifyDomainAdequacy wf.parts inp ≠ .adequateSelectedSupported) ∧
    (caseSelectionUnresolved wf.parts inp →
      classifyDomainAdequacy wf.parts inp ≠ .selectionRegret) := by
  constructor
  · intro hadeq hcontra
    have := classify_adequate wf.parts inp hadeq
    simp [this] at hcontra
  constructor
  · intro hreg hcontra
    have hs := hreg.1
    have hsel := hreg.2
    have hnot : inp.selected ∉ wf.parts.supported := by
      intro hs'
      exact wf.supported_disjoint_unsupported inp.selected hs' hsel
    have := classify_regret wf.parts inp hs hsel hnot
    simp [this] at hcontra
  · intro hun hcontra
    have hs := hun.1
    have hsel := hun.2
    have hnotu : inp.selected ∉ wf.parts.unsupported := by
      rcases hsel with hsel | hsel
      · intro hu
        exact wf.unsupported_disjoint_unknown inp.selected hu hsel
      · intro hu
        exact wf.unsupported_disjoint_unobserved inp.selected hu hsel
    have hnots : inp.selected ∉ wf.parts.supported := by
      rcases hsel with hsel | hsel
      · intro hs'
        exact wf.supported_disjoint_unknown inp.selected hs' hsel
      · intro hs'
        exact wf.supported_disjoint_unobserved inp.selected hs' hsel
    have := classify_unresolved wf.parts inp hs hsel hnots hnotu
    simp [this] at hcontra

/-! ### Obstruction / hitting sets -/

structure TerminalFailureSet where
  terminal : Terminal
  atoms : List Atom
deriving Repr, BEq

inductive ObstructionCoreMode where
  | minimumCardinalityExact
  | subsetMinimalExact
  | soundOverapprox
deriving DecidableEq, Repr, BEq

def hitsTerminal (core : List Atom) (t : TerminalFailureSet) : Prop :=
  (inter core t.atoms) ≠ []

def hitsTerminalB (core : List Atom) (t : TerminalFailureSet) : Bool :=
  decide ((inter core t.atoms) ≠ [])

def hitsAllB (core : List Atom) (failures : List TerminalFailureSet) : Bool :=
  failures.all fun t => hitsTerminalB core t

def subsetMinimalExactB (core : List Atom) (failures : List TerminalFailureSet) : Bool :=
  hitsAllB core failures &&
    !core.any fun a =>
      hitsAllB (core.filter (· ≠ a)) failures

def hitsAll (core : List Atom) (failures : List TerminalFailureSet) : Prop :=
  ∀ t, t ∈ failures → hitsTerminal core t

def subsetMinimalExact (core : List Atom) (failures : List TerminalFailureSet) : Prop :=
  hitsAll core failures ∧
  ∀ a, a ∈ core → hitsAll (core.filter (· ≠ a)) failures → False

theorem recorded_core_hits_every_failure
    (core : List Atom) (failures : List TerminalFailureSet)
    (h : hitsAll core failures) (t : TerminalFailureSet) (ht : t ∈ failures) :
    hitsTerminal core t :=
  h t ht

theorem subset_minimal_removing_atom_breaks_hitting
    (core : List Atom) (failures : List TerminalFailureSet) (a : Atom)
    (h : subsetMinimalExact core failures) (ha : a ∈ core) :
    ¬ hitsAll (core.filter (· ≠ a)) failures := by
  intro hhit
  exact h.2 a ha hhit

theorem sound_overapprox_not_claimed_subset_minimal
    (mode : ObstructionCoreMode) (h : mode = .soundOverapprox) :
    mode ≠ .subsetMinimalExact := by
  cases h
  decide

/-! ### Required-constraint witnesses -/

structure TerminalWitness where
  terminal : Terminal
  satisfied : List Constraint
deriving Repr, BEq

def isWitness (w : TerminalWitness) (required : List Constraint) : Prop :=
  Subset required w.satisfied

theorem witness_survives_nonrequired_removal
    (w : TerminalWitness) (required reduced : List Constraint)
    (hreq : Subset required w.satisfied) (hred : Subset reduced required) :
    isWitness w reduced := by
  unfold isWitness at *
  exact Subset.trans hred hreq

structure LegalContext where
  legal : List Action
  witness : TerminalWitness
  required : List Constraint
deriving Repr, BEq

def recordedWitnessOk (ctx : LegalContext) : Prop :=
  isWitness ctx.witness ctx.required

theorem adding_candidates_preserves_witness
    (ctx : LegalContext) (extra : List Action)
    (h : recordedWitnessOk ctx) :
    recordedWitnessOk { ctx with legal := ctx.legal ++ extra } := by
  exact h

/-! ### Certified singleton survivors -/

def applyCertifiedRemoval (live : List Action) (removed : List Action) : List Action :=
  live.filter fun a => a ∉ removed

theorem certified_live_singleton_no_learned_choice
    (live removed : List Action) (token : Action)
    (hsingle : applyCertifiedRemoval live removed = [token]) :
    ∀ other, other ∈ applyCertifiedRemoval live removed → other = token := by
  intro other ho
  rw [hsingle] at ho
  exact List.mem_singleton.mp ho

theorem certified_live_singleton_forces_unique_survivor
    (live removed : List Action) (token : Action)
    (hsingle : applyCertifiedRemoval live removed = [token]) :
    (applyCertifiedRemoval live removed).length = 1 := by
  simp [hsingle]

/-! ### Bridge to exact-closure removable law -/

theorem certified_removal_respects_exact_closure_unknown
    (r : ExactClosure.QueryResult)
    (h : r.verdict = ExactClosure.Verdict.unknown) :
    ExactClosure.removable r = false :=
  ExactClosure.unknown_not_removable r h

theorem certified_removal_respects_exact_closure_supported
    (r : ExactClosure.QueryResult)
    (h : r.verdict = ExactClosure.Verdict.supported) :
    ExactClosure.removable r = false :=
  ExactClosure.supported_not_removable r h

end LeverProofLean.GoalSupport
