/-
  Fixture-level runtime proof-trace refinement (KERN-11 / SLM-539).

  Canonical INTEG-01 ``canonical_proof_trace/v1`` compact events are checked
  against abstract decode/search owners (KERN-01 Judgment, KERN-02
  CompleteDomain, KERN-06 EventTrace, DecodeInvariants commit rules).

  The runtime itself is untrusted — only its sealed compact trace is admitted.
  Scope is explicitly a bounded fixture subset; this module does **not** claim
  full PyTorch semantics or physical latency equivalence.
-/

import LeverProofLean.CompleteDomain
import LeverProofLean.DecodeInvariants
import LeverProofLean.EventTrace
import LeverProofLean.Judgment

namespace LeverProofLean.ProofTraceRefinement

open LeverProofLean
open LeverProofLean.CompleteDomain
open LeverProofLean.DecodeInvariants
open LeverProofLean.EventTrace
open LeverProofLean.Judgment

/-- Canonical INTEG-01 field vocabulary. -/
inductive CanonicalField where
  | stateInputIdentity
  | legalDomain
  | rankerModelInvocation
  | chosenAction
  | verifierSolverCertificate
  | cacheMechanism
  | stateTransition
  | observedWork
  deriving DecidableEq, Repr, BEq

/-- Judgment tags carried on verifier events (KERN-01 outcomes). -/
inductive JudgmentTag where
  | witnessed
  | refuted
  | unknown
  | invalid
  | none
  deriving DecidableEq, Repr, BEq

/-- Compact proof-trace event (Lean mirror of INTEG-01 refs used by the checker). -/
structure CompactEvent where
  ordinal : Nat
  field : CanonicalField
  observed : Bool
  identityDigest : String
  stateDigest : String
  domainDigest : String
  domainStatus : String
  domainSize : Nat
  tokenId : Option Nat
  judgment : JudgmentTag
  removedCount : Nat
  forwardsCount : Nat
  neuralForward : Nat
  abstractCost : Nat
  deriving Repr, BEq

/-- Checker cursor over abstract state. -/
structure AbstractState where
  identityDigest : String
  currentStateDigest : String
  domainDigest : String
  domainStatus : String
  domainSize : Nat
  declaredForwards : Nat
  declaredCost : Nat
  sawRankerInvocation : Bool
  sawObservedWork : Bool
  deriving Repr, BEq

/-- Explicit fixture-subset assumptions (KERN-11 contract). -/
structure FixtureAssumptions where
  /-- Domains with status ``complete`` must carry a non-empty digest. -/
  completeRequiresDigest : Bool := true
  /-- Bare coverageComplete Boolean never authorizes (KERN-02). -/
  legacyCoverageNeverAuthorizes : Bool := true
  /-- Event ordinals must be dense ``0 .. n-1`` in order. -/
  denseOrdinals : Bool := true
  /-- Identity digests on events must match the sealed envelope identity. -/
  identityMustMatch : Bool := true
  /-- Declared neural_forward cost requires a ranker_model_invocation event. -/
  costForwardsRequireInvocation : Bool := true
  /-- Unknown / invalid judgments never authorize removal (KERN-01). -/
  unknownNeverRemoves : Bool := true
  deriving Repr, BEq

def defaultAssumptions : FixtureAssumptions := {}

/-- Supported fixture subset ids (frozen Python fixtures). -/
def supportedFixtureIds : List String :=
  ["bundle_stats_mechanism"]

def fixtureSubsetScoped : Bool := true
def claimsFullPytorchSemantics : Bool := false
def claimsPhysicalLatencyEquivalence : Bool := false

#guard fixtureSubsetScoped = true
#guard claimsFullPytorchSemantics = false
#guard claimsPhysicalLatencyEquivalence = false

/-- Domain status string is the complete-domain tag. -/
def statusIsComplete (s : String) : Bool :=
  decide (s = "complete")

/-- Fabricated completeness: ``complete`` without a digest (never authorizes). -/
def fabricatedCompleteness (e : CompactEvent) : Bool :=
  e.field == .legalDomain && e.observed &&
    statusIsComplete e.domainStatus && decide (e.domainDigest = "")

/-- Fabricated refutation: removal under unknown/invalid judgment. -/
def fabricatedRefutation (e : CompactEvent) : Bool :=
  e.field == .verifierSolverCertificate && e.observed &&
    e.removedCount > 0 &&
    (e.judgment == .unknown || e.judgment == .invalid || e.judgment == .none)

/-- Illegal commit: choose a token against an empty proved-complete domain. -/
def illegalCommit (s : AbstractState) (e : CompactEvent) : Bool :=
  e.field == .chosenAction && e.observed &&
    statusIsComplete s.domainStatus && decide (s.domainDigest ≠ "") &&
    decide (s.domainSize = 0)

/-- Stale-state domain: legal_domain digest changes while state fingerprint is
    still the previous digest (domain must rebind with a state transition). -/
def staleStateDomain (s : AbstractState) (e : CompactEvent) : Bool :=
  e.field == .legalDomain && e.observed &&
    decide (s.domainDigest ≠ "") &&
    decide (e.domainDigest ≠ s.domainDigest) &&
    decide (s.currentStateDigest ≠ "") &&
    decide (e.stateDigest = s.currentStateDigest ∨ e.stateDigest = "")

/-- Mismatched identity digest vs sealed envelope. -/
def mismatchedIdentity (envelopeId : String) (e : CompactEvent) : Bool :=
  e.observed && decide (e.identityDigest ≠ "") &&
    decide (e.identityDigest ≠ envelopeId)

/-- Ordinals must equal their list index (dense, ordered). -/
def ordinalsOk : Nat → List CompactEvent → Bool
  | _, [] => true
  | i, e :: rest => decide (e.ordinal = i) && ordinalsOk (i + 1) rest

/-- Apply one observed event; ``none`` means reject. -/
def step
    (assumptions : FixtureAssumptions) (envelopeId : String)
    (s : AbstractState) (e : CompactEvent) : Option AbstractState :=
  if !e.observed then
    some s
  else if assumptions.identityMustMatch && mismatchedIdentity envelopeId e then
    none
  else if fabricatedCompleteness e then
    none
  else if fabricatedRefutation e then
    none
  else if illegalCommit s e then
    none
  else if staleStateDomain s e then
    none
  else
    match e.field with
    | .stateInputIdentity =>
        if decide (e.identityDigest ≠ envelopeId) &&
            decide (e.identityDigest ≠ "") then
          none
        else
          some { s with identityDigest := envelopeId }
    | .legalDomain =>
        some {
          s with
            domainDigest := e.domainDigest
            domainStatus := e.domainStatus
            domainSize := e.domainSize
            currentStateDigest :=
              if decide (e.stateDigest ≠ "") then e.stateDigest
              else s.currentStateDigest
        }
    | .stateTransition =>
        some { s with currentStateDigest := e.stateDigest }
    | .chosenAction =>
        -- Ranked / bypass commit under a non-empty complete domain is admitted
        -- when token is present; empty complete domains already rejected above.
        some s
    | .verifierSolverCertificate =>
        -- Witnessed certificates may remove; unknown/invalid already rejected.
        some s
    | .rankerModelInvocation =>
        some {
          s with
            declaredForwards := e.forwardsCount
            sawRankerInvocation := true
        }
    | .observedWork =>
        some {
          s with
            declaredCost := e.abstractCost
            declaredForwards :=
              max s.declaredForwards e.neuralForward
            sawObservedWork := true
        }
    | .cacheMechanism =>
        some s

/-- Fold the checker over a compact event list. -/
def replay
    (assumptions : FixtureAssumptions) (envelopeId : String)
    (events : List CompactEvent) : Option AbstractState :=
  if assumptions.denseOrdinals && !(ordinalsOk 0 events) then
    none
  else
    let init : AbstractState := {
      identityDigest := envelopeId
      currentStateDigest := ""
      domainDigest := ""
      domainStatus := ""
      domainSize := 0
      declaredForwards := 0
      declaredCost := 0
      sawRankerInvocation := false
      sawObservedWork := false
    }
    events.foldl
      (fun acc e =>
        match acc with
        | none => none
        | some s => step assumptions envelopeId s e)
      (some init)

/-- Final cost/forwards consistency under fixture assumptions. -/
def finalizeOk
    (assumptions : FixtureAssumptions)
    (s : AbstractState)
    (expectedCost : Nat)
    (expectedForwards : Nat) : Bool :=
  decide (s.declaredCost = expectedCost) &&
    decide (s.declaredForwards = expectedForwards) &&
    (if assumptions.costForwardsRequireInvocation && expectedForwards > 0 then
      s.sawRankerInvocation && s.sawObservedWork
     else true)

/-- Full refinement check: replay + finalize. -/
def checkTrace
    (assumptions : FixtureAssumptions) (envelopeId : String)
    (events : List CompactEvent)
    (expectedCost : Nat) (expectedForwards : Nat) : Bool :=
  match replay assumptions envelopeId events with
  | none => false
  | some s => finalizeOk assumptions s expectedCost expectedForwards

/-- Abstract cost parity with KERN-06 DecodeUnitWorkModel. -/
theorem cost_matches_decode_unit_work
    (c : DecodeUnitWorkCounts)
    (h : decodeUnitWorkCostOfCounts c = c.tokenize + c.grammarTransition +
          c.solverExpansion + c.certificateCheck + c.neuralForward) :
    decodeUnitWorkCostOfCounts c =
      c.tokenize + c.grammarTransition + c.solverExpansion +
        c.certificateCheck + c.neuralForward :=
  h

/-- Legacy coverage flag never authorizes completeness (KERN-02 reuse). -/
theorem legacy_coverage_never_authorizes_refinement
    (flag : LegacyCoverageFlag) :
    legacyFlagAuthorizesCompleteness flag = false :=
  forged_coverage_complete_never_authorizes flag

/-- Unknown judgment never authorizes removal (KERN-01 reuse). -/
theorem unknown_judgment_never_authorizes_removal
    (s : Judgment.SourceInput) (checked : Bool)
    (h : Judgment.classifyOutcome s = .unknown) :
    Judgment.authorizesRemoval (Judgment.classify s checked) = false := by
  simp [Judgment.authorizesRemoval, Judgment.classify, h]

/-- Illegal empty-domain commit is rejected by ``illegalCommit``. -/
theorem illegal_empty_complete_commit_detected :
    illegalCommit
      { identityDigest := "id"
        currentStateDigest := "fp0"
        domainDigest := "dom"
        domainStatus := "complete"
        domainSize := 0
        declaredForwards := 0
        declaredCost := 0
        sawRankerInvocation := false
        sawObservedWork := false }
      { ordinal := 0
        field := .chosenAction
        observed := true
        identityDigest := "id"
        stateDigest := "fp0"
        domainDigest := "dom"
        domainStatus := "complete"
        domainSize := 0
        tokenId := some 3
        judgment := .none
        removedCount := 0
        forwardsCount := 0
        neuralForward := 0
        abstractCost := 0 } = true := by
  decide

/-- Fabricated completeness without digest is detected. -/
theorem fabricated_completeness_detected :
    fabricatedCompleteness
      { ordinal := 0
        field := .legalDomain
        observed := true
        identityDigest := "id"
        stateDigest := ""
        domainDigest := ""
        domainStatus := "complete"
        domainSize := 4
        tokenId := none
        judgment := .none
        removedCount := 0
        forwardsCount := 0
        neuralForward := 0
        abstractCost := 0 } = true := by
  decide

/-- Fabricated refutation (removal under unknown) is detected. -/
theorem fabricated_refutation_detected :
    fabricatedRefutation
      { ordinal := 0
        field := .verifierSolverCertificate
        observed := true
        identityDigest := "id"
        stateDigest := "fp0"
        domainDigest := "dom"
        domainStatus := "complete"
        domainSize := 4
        tokenId := none
        judgment := .unknown
        removedCount := 1
        forwardsCount := 0
        neuralForward := 0
        abstractCost := 0 } = true := by
  decide

/-- Reordered ordinals fail ``ordinalsOk``. -/
theorem reordered_ordinals_rejected :
    ordinalsOk 0
      ([
        { ordinal := 1
          field := .stateInputIdentity
          observed := true
          identityDigest := "id"
          stateDigest := ""
          domainDigest := ""
          domainStatus := ""
          domainSize := 0
          tokenId := none
          judgment := .none
          removedCount := 0
          forwardsCount := 0
          neuralForward := 0
          abstractCost := 0 },
        { ordinal := 0
          field := .legalDomain
          observed := true
          identityDigest := "id"
          stateDigest := ""
          domainDigest := "dom"
          domainStatus := "complete"
          domainSize := 4
          tokenId := none
          judgment := .none
          removedCount := 0
          forwardsCount := 0
          neuralForward := 0
          abstractCost := 0 }
      ] : List CompactEvent) = false := by
  decide

/-- Omitted model forwards: cost claims forwards but no invocation event. -/
theorem omitted_forwards_finalize_fails :
    finalizeOk defaultAssumptions
      { identityDigest := "id"
        currentStateDigest := "fp0"
        domainDigest := "dom"
        domainStatus := "complete"
        domainSize := 4
        declaredForwards := 11
        declaredCost := 28
        sawRankerInvocation := false
        sawObservedWork := true }
      28 11 = false := by
  decide

/-- Mismatched identity digest is rejected by ``step``. -/
theorem mismatched_identity_step_none :
    step defaultAssumptions "envelope-id"
      { identityDigest := "envelope-id"
        currentStateDigest := ""
        domainDigest := ""
        domainStatus := ""
        domainSize := 0
        declaredForwards := 0
        declaredCost := 0
        sawRankerInvocation := false
        sawObservedWork := false }
      { ordinal := 0
        field := .stateInputIdentity
        observed := true
        identityDigest := "other-id"
        stateDigest := ""
        domainDigest := ""
        domainStatus := ""
        domainSize := 0
        tokenId := none
        judgment := .none
        removedCount := 0
        forwardsCount := 0
        neuralForward := 0
        abstractCost := 0 } = none := by
  decide

/-! ### Frozen fixture: ``bundle_stats_mechanism`` (INTEG-01) -/

def fixtureEnvelopeId : String :=
  "integ01-fixture"

/-- Compact events for the supported INTEG-01 frozen fixture subset. -/
def fixtureBundleStatsMechanism : List CompactEvent :=
  [
    { ordinal := 0, field := .stateInputIdentity, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := ""
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := none, judgment := .none, removedCount := 0
      forwardsCount := 0, neuralForward := 0, abstractCost := 0 },
    { ordinal := 1, field := .legalDomain, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := ""
      domainDigest := "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
      domainStatus := "complete", domainSize := 4
      tokenId := none, judgment := .none, removedCount := 0
      forwardsCount := 0, neuralForward := 0, abstractCost := 0 },
    { ordinal := 2, field := .stateTransition, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := "fp0"
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := none, judgment := .none, removedCount := 0
      forwardsCount := 0, neuralForward := 0, abstractCost := 0 },
    { ordinal := 3, field := .chosenAction, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := "fp0"
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := some 3, judgment := .none, removedCount := 0
      forwardsCount := 0, neuralForward := 0, abstractCost := 0 },
    { ordinal := 4, field := .verifierSolverCertificate, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := "fp0"
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := none, judgment := .witnessed, removedCount := 0
      forwardsCount := 0, neuralForward := 0, abstractCost := 0 },
    { ordinal := 5, field := .cacheMechanism, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := ""
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := none, judgment := .none, removedCount := 0
      forwardsCount := 0, neuralForward := 0, abstractCost := 0 },
    { ordinal := 6, field := .rankerModelInvocation, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := ""
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := none, judgment := .none, removedCount := 0
      forwardsCount := 11, neuralForward := 0, abstractCost := 0 },
    { ordinal := 7, field := .observedWork, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := ""
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := none, judgment := .none, removedCount := 0
      forwardsCount := 0, neuralForward := 11, abstractCost := 28 },
    { ordinal := 8, field := .cacheMechanism, observed := true
      identityDigest := fixtureEnvelopeId, stateDigest := ""
      domainDigest := "", domainStatus := "", domainSize := 0
      tokenId := none, judgment := .none, removedCount := 0
      forwardsCount := 0, neuralForward := 0, abstractCost := 0 }
  ]

theorem fixture_bundle_stats_mechanism_cost :
    decodeUnitWorkCostOfCounts
      { tokenize := 2, grammarTransition := 3, solverExpansion := 5
        certificateCheck := 7, neuralForward := 11 } = 28 := by
  decide

/-- Valid frozen runtime fixture replays successfully under abstract semantics. -/
theorem fixture_bundle_stats_mechanism_refines :
    checkTrace defaultAssumptions fixtureEnvelopeId
      fixtureBundleStatsMechanism 28 11 = true := by
  decide

/-- Injected illegal empty-domain commit fails the checker. -/
theorem injected_illegal_commit_fails :
    checkTrace defaultAssumptions fixtureEnvelopeId
      ([
        { ordinal := 0, field := .legalDomain, observed := true
          identityDigest := fixtureEnvelopeId, stateDigest := ""
          domainDigest := "dom", domainStatus := "complete", domainSize := 0
          tokenId := none, judgment := .none, removedCount := 0
          forwardsCount := 0, neuralForward := 0, abstractCost := 0 },
        { ordinal := 1, field := .chosenAction, observed := true
          identityDigest := fixtureEnvelopeId, stateDigest := ""
          domainDigest := "", domainStatus := "", domainSize := 0
          tokenId := some 3, judgment := .none, removedCount := 0
          forwardsCount := 0, neuralForward := 0, abstractCost := 0 }
      ] : List CompactEvent) 0 0 = false := by
  decide

/-- Scope disclaimer: theorems are fixture-subset only. -/
theorem scope_is_fixture_subset :
    fixtureSubsetScoped = true ∧
      claimsFullPytorchSemantics = false ∧
      claimsPhysicalLatencyEquivalence = false := by
  decide

end LeverProofLean.ProofTraceRefinement
