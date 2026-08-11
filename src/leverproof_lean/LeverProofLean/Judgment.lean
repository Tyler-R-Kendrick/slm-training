/-
  Total solver judgment (KERN-01 / SLM-516).

  One four-outcome judgment ``witnessed | refuted | unknown | invalid`` with typed
  payloads and explicit authority rules. Theorem-preflight lifecycle status stays
  outside this module — it classifies solver/VSS/closure evidence only.

  Fail-closed: timeout, skipped replay, unsupported capability, incomplete
  coverage, and missing tool map to ``unknown``; malformed input and payload
  mismatch map to ``invalid`` — never ``refuted``.
-/

import LeverProofLean.ExactClosure

namespace LeverProofLean.Judgment

open LeverProofLean

/-! ### Outcomes and typed payloads -/

inductive Outcome where
  | witnessed
  | refuted
  | unknown
  | invalid
deriving DecidableEq, Repr, BEq

inductive UnknownReason where
  | timeout
  | skippedReplay
  | unsupportedCapability
  | incompleteCoverage
  | missingTool
  | inconclusive
deriving DecidableEq, Repr, BEq

inductive InvalidReason where
  | malformedInput
  | payloadMismatch
deriving DecidableEq, Repr, BEq

structure WitnessPayload where
  witnessDigest : String
  source : String
deriving Repr, BEq

structure RefutationPayload where
  refutationDigest : String
  source : String
deriving Repr, BEq

inductive Payload where
  | witness (p : WitnessPayload)
  | refutation (p : RefutationPayload)
  | unknownReason (r : UnknownReason)
  | invalidReason (r : InvalidReason)
deriving Repr, BEq

structure Judgment where
  outcome : Outcome
  payload : Payload
  checked : Bool
deriving Repr, BEq

/-! ### Classification inputs (mirrors Python ``ClassificationInputsV1``) -/

inductive VssVerdict where
  | supported
  | unsupported
  | unknown
deriving DecidableEq, Repr, BEq

structure SourceInput where
  malformedInput : Bool
  payloadMismatch : Bool
  timedOut : Bool
  skippedReplay : Bool
  unsupportedCapability : Bool
  incompleteCoverage : Bool
  missingTool : Bool
  vssVerdict : Option VssVerdict
  hasWitnessDigest : Bool
  hasCounterexampleDigest : Bool
  /-- Telemetry only (EVID-09). -/
  exhausted : Bool
  /-- Telemetry only (EVID-09). -/
  replayChecked : Bool
  /-- Telemetry only (EVID-09). -/
  replayOk : Bool
  /-- Authority-critical checked refutation evidence. -/
  refutationEvidence : Option ExactClosure.CheckedRefutationEvidence
  expectedBinding : ExactClosure.BindingIds
deriving Repr, BEq

def classifyUnknownReason (s : SourceInput) : UnknownReason :=
  if s.timedOut then .timeout
  else if s.skippedReplay then .skippedReplay
  else if s.unsupportedCapability then .unsupportedCapability
  else if s.incompleteCoverage then .incompleteCoverage
  else if s.missingTool then .missingTool
  else .inconclusive

def classifyInvalidReason (s : SourceInput) : InvalidReason :=
  if s.malformedInput then .malformedInput else .payloadMismatch

def classifyOutcome (s : SourceInput) : Outcome :=
  if s.malformedInput || s.payloadMismatch then
    .invalid
  else if s.timedOut || s.skippedReplay || s.unsupportedCapability ||
      s.incompleteCoverage || s.missingTool then
    .unknown
  else
    match s.vssVerdict with
    | none => .unknown
    | some .supported =>
      if s.hasWitnessDigest && s.replayChecked && s.replayOk then
        .witnessed
      else
        .unknown
    | some .unsupported =>
      if s.hasWitnessDigest then
        .invalid
      else if ExactClosure.evidenceAuthorizesRemoval s.refutationEvidence s.expectedBinding then
        .refuted
      else
        .unknown
    | some .unknown => .unknown

def classifyPayload (s : SourceInput) (o : Outcome) : Payload :=
  match o with
  | .witnessed =>
    .witness { witnessDigest := "checked", source := "vss_certificate" }
  | .refuted =>
    .refutation {
      refutationDigest :=
        match s.refutationEvidence with
        | some e => e.evidenceDigest
        | none => "unchecked"
      source :=
        match s.refutationEvidence with
        | some e =>
          match e.kind with
          | .exactReplay => "exact_semantic_replay"
          | .checkedCertificate => "checked_certificate"
        | none => "vss_exhaustion"
    }
  | .unknown => .unknownReason (classifyUnknownReason s)
  | .invalid => .invalidReason (classifyInvalidReason s)

def classify (s : SourceInput) (checked : Bool) : Judgment :=
  let o := classifyOutcome s
  { outcome := o, payload := classifyPayload s o, checked := checked }

/-! ### Authority rules -/

def authorizesSemanticConclusion (j : Judgment) : Bool :=
  j.checked && (j.outcome = .witnessed || j.outcome = .refuted)

def authorizesRemoval (j : Judgment) : Bool :=
  j.checked && j.outcome = .refuted &&
    match j.payload with
    | .refutation _ => true
    | _ => false

def payloadMatchesOutcome (j : Judgment) : Bool :=
  match j.outcome, j.payload with
  | .witnessed, .witness _ => true
  | .refuted, .refutation _ => true
  | .unknown, .unknownReason _ => true
  | .invalid, .invalidReason _ => true
  | _, _ => false

/-! ### Theorems -/

theorem invalid_never_authorizes (s : SourceInput) (checked : Bool)
    (h : classifyOutcome s = .invalid) :
    authorizesSemanticConclusion (classify s checked) = false := by
  simp [authorizesSemanticConclusion, classify, h]

theorem unknown_never_authorizes (s : SourceInput) (checked : Bool)
    (h : classifyOutcome s = .unknown) :
    authorizesSemanticConclusion (classify s checked) = false := by
  simp [authorizesSemanticConclusion, classify, h]

theorem unchecked_never_authorizes (s : SourceInput)
    (_h : classifyOutcome s = .witnessed ∨ classifyOutcome s = .refuted) :
    authorizesSemanticConclusion (classify s false) = false := by
  simp [authorizesSemanticConclusion, classify]

theorem timeout_never_refuted (s : SourceInput) (h : s.timedOut = true) :
    classifyOutcome s ≠ .refuted := by
  cases hmal : (s.malformedInput || s.payloadMismatch)
  · have : classifyOutcome s = .unknown := by
      simp [classifyOutcome, hmal, h]
    simp [this]
  · have : classifyOutcome s = .invalid := by
      simp [classifyOutcome, hmal]
    simp [this]

theorem skipped_replay_never_refuted (s : SourceInput) (h : s.skippedReplay = true) :
    classifyOutcome s ≠ .refuted := by
  cases hmal : (s.malformedInput || s.payloadMismatch)
  · have : classifyOutcome s = .unknown := by
      simp [classifyOutcome, hmal, h]
    simp [this]
  · have : classifyOutcome s = .invalid := by
      simp [classifyOutcome, hmal]
    simp [this]

theorem unsupported_capability_never_refuted (s : SourceInput)
    (h : s.unsupportedCapability = true) :
    classifyOutcome s ≠ .refuted := by
  cases hmal : (s.malformedInput || s.payloadMismatch)
  · have : classifyOutcome s = .unknown := by
      simp [classifyOutcome, hmal, h]
    simp [this]
  · have : classifyOutcome s = .invalid := by
      simp [classifyOutcome, hmal]
    simp [this]

theorem incomplete_coverage_never_refuted (s : SourceInput)
    (h : s.incompleteCoverage = true) :
    classifyOutcome s ≠ .refuted := by
  cases hmal : (s.malformedInput || s.payloadMismatch)
  · have : classifyOutcome s = .unknown := by
      simp [classifyOutcome, hmal, h]
    simp [this]
  · have : classifyOutcome s = .invalid := by
      simp [classifyOutcome, hmal]
    simp [this]

theorem missing_tool_never_refuted (s : SourceInput) (h : s.missingTool = true) :
    classifyOutcome s ≠ .refuted := by
  cases hmal : (s.malformedInput || s.payloadMismatch)
  · have : classifyOutcome s = .unknown := by
      simp [classifyOutcome, hmal, h]
    simp [this]
  · have : classifyOutcome s = .invalid := by
      simp [classifyOutcome, hmal]
    simp [this]

theorem timeout_never_authorizes_removal (s : SourceInput) (checked : Bool)
    (h : s.timedOut = true) :
    authorizesRemoval (classify s checked) = false := by
  have hout := timeout_never_refuted s h
  simp [authorizesRemoval, classify, hout]

theorem skipped_replay_never_authorizes_removal (s : SourceInput) (checked : Bool)
    (h : s.skippedReplay = true) :
    authorizesRemoval (classify s checked) = false := by
  have hout := skipped_replay_never_refuted s h
  simp [authorizesRemoval, classify, hout]

theorem classification_exhaustive (s : SourceInput) :
    classifyOutcome s = .witnessed ∨
    classifyOutcome s = .refuted ∨
    classifyOutcome s = .unknown ∨
    classifyOutcome s = .invalid := by
  match classifyOutcome s with
  | .witnessed => simp
  | .refuted => simp
  | .unknown => simp
  | .invalid => simp

theorem classified_payload_matches (s : SourceInput) (checked : Bool) :
    payloadMatchesOutcome (classify s checked) = true := by
  simp [classify, payloadMatchesOutcome, classifyPayload]
  cases classifyOutcome s <;> simp

/-! ### Bridge to exact-closure removable law -/

def sourceFromQueryResult (r : ExactClosure.QueryResult)
    (timedOut skippedReplay unsupportedCapability incompleteCoverage missingTool : Bool)
    (hasWitness hasCounter : Bool) (exhausted replayChecked replayOk : Bool) :
    SourceInput :=
  { malformedInput := false
    payloadMismatch := false
    timedOut := timedOut
    skippedReplay := skippedReplay
    unsupportedCapability := unsupportedCapability
    incompleteCoverage := incompleteCoverage
    missingTool := missingTool
    vssVerdict := some (match r.verdict with
      | .supported => .supported
      | .unsupported => .unsupported
      | .unknown => .unknown)
    hasWitnessDigest := hasWitness
    hasCounterexampleDigest := hasCounter
    exhausted := exhausted
    replayChecked := replayChecked
    replayOk := replayOk
    refutationEvidence :=
      if timedOut || skippedReplay || unsupportedCapability || incompleteCoverage ||
          missingTool then none
      else r.refutationEvidence
    expectedBinding := r.expectedBinding }

theorem unknown_query_never_refuted_when_replay_skipped
    (r : ExactClosure.QueryResult) :
    classifyOutcome
      (sourceFromQueryResult r false true false false false false false false true false) ≠
      .refuted := by
  exact skipped_replay_never_refuted _ rfl


theorem forged_exhaustion_flags_never_refuted
    (exhausted replayChecked replayOk : Bool)
    (binding : ExactClosure.BindingIds) :
    classifyOutcome
      { malformedInput := false, payloadMismatch := false, timedOut := false,
        skippedReplay := false, unsupportedCapability := false, incompleteCoverage := false,
        missingTool := false, vssVerdict := some .unsupported, hasWitnessDigest := false,
        hasCounterexampleDigest := false, exhausted := exhausted,
        replayChecked := replayChecked, replayOk := replayOk,
        refutationEvidence := none, expectedBinding := binding } ≠ .refuted := by
  simp [classifyOutcome, ExactClosure.evidenceAuthorizesRemoval]

theorem exact_closure_unknown_never_authorizes_removal
    (r : ExactClosure.QueryResult)
    (h : r.verdict = ExactClosure.Verdict.unknown) :
    authorizesRemoval (classify
      (sourceFromQueryResult r false false false false false false false false false r.replayOk)
      true) = false := by
  have hout :
      classifyOutcome
        (sourceFromQueryResult r false false false false false false false false false
          r.replayOk) = .unknown := by
    simp [classifyOutcome, sourceFromQueryResult, h]
  simp [authorizesRemoval, classify, hout]

end LeverProofLean.Judgment
