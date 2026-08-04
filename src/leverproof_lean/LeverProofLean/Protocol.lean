import Lean.Data.Json
import LeverProofLean.Band
import LeverProofLean.Certificate

namespace LeverProofLean

open Lean

structure SourceWire where
  evidence_sha256 : String
  feature_flags_sha256 : String
  campaign_manifest_sha256 : Option String
deriving Repr, BEq, FromJson, ToJson

structure WorkloadWire where
  cold_requests : Nat
  warm_requests : Nat
deriving Repr, BEq, FromJson, ToJson

def WorkloadWire.toCore (wire : WorkloadWire) : Workload :=
  ⟨wire.cold_requests, wire.warm_requests⟩

structure RawCandidateWire where
  id : String
  hardware : String
  lever_snapshot_sha256 : String
  cold_ns : List Nat
  warm_ns : List Nat
  input_units : List Nat
  passes : List Nat
  successes : Nat
  quality_failures : Nat
  trainable_params : Nat
  energy_uj : List Nat
  cost_micro_usd : List Nat
deriving Repr, BEq, FromJson, ToJson

def RawCandidateWire.toCore (wire : RawCandidateWire) : RawCandidate := {
  id := wire.id
  artifactSha256 := wire.lever_snapshot_sha256
  coldLatencyNs := wire.cold_ns
  warmLatencyNs := wire.warm_ns
  inputUnits := wire.input_units
  passes := wire.passes
  energyUj := wire.energy_uj
  costMicroUsd := wire.cost_micro_usd
  successes := wire.successes
  qualityFailures := wire.quality_failures
  trainableParameters := wire.trainable_params
}

structure EvidenceWire where
  schema : String
  run_id : String
  source : SourceWire
  workload : WorkloadWire
  candidates : List RawCandidateWire
deriving Repr, BEq, FromJson, ToJson

def EvidenceWire.toCore (wire : EvidenceWire) : Except String Evidence := do
  if wire.schema != evidenceSchema then
    throw s!"schema: expected {evidenceSchema}"
  let first ← match wire.candidates.head? with
    | none => throw "candidates: expected at least one"
    | some candidate => pure candidate
  if first.hardware.isEmpty then
    throw "candidates[0].hardware: expected non-empty string"
  if wire.candidates.any (·.hardware != first.hardware) then
    throw "candidates: all candidates must use the same hardware"
  pure {
    runId := wire.run_id
    sourceSha256 := wire.source.evidence_sha256
    featureFlagsSha256 := wire.source.feature_flags_sha256
    campaignManifestSha256 := wire.source.campaign_manifest_sha256
    hardware := first.hardware
    workload := wire.workload.toCore
    candidates := wire.candidates.map RawCandidateWire.toCore
  }

structure RatioWire where
  numerator : Nat
  denominator : Nat
deriving Repr, BEq, FromJson, ToJson

def RatioWire.ofCore (ratio : Ratio) : RatioWire :=
  ⟨ratio.numerator, ratio.denominator⟩

structure IntervalWire where
  lower : RatioWire
  upper : RatioWire
deriving Repr, BEq, FromJson, ToJson

def IntervalWire.ofCore (interval : Interval) : IntervalWire :=
  ⟨RatioWire.ofCore interval.lower, RatioWire.ofCore interval.upper⟩

def RatioWire.toCore (ratio : RatioWire) : Ratio :=
  ⟨ratio.numerator, ratio.denominator⟩

def IntervalWire.toCore (interval : IntervalWire) : Interval :=
  ⟨interval.lower.toCore, interval.upper.toCore⟩

structure SourceWireV2 where
  evidence_sha256 : String
  feature_flags_sha256 : String
  campaign_manifest_sha256 : Option String
  metric_expectations_sha256 : String
deriving Repr, BEq, FromJson, ToJson

structure NamedIntervalWire where
  name : String
  value : IntervalWire
deriving Repr, BEq, FromJson, ToJson

def NamedIntervalWire.toCore (wire : NamedIntervalWire) : NamedInterval :=
  ⟨wire.name, wire.value.toCore⟩

structure MetricInstructionWire where
  op : String
  value : Option RatioWire := none
  name : Option String := none
  exponent : Option Nat := none
deriving Repr, BEq, FromJson, ToJson

def MetricInstructionWire.toCore
    (wire : MetricInstructionWire) : Except String MetricInstruction :=
  match wire.op with
  | "constant" =>
      match wire.value with
      | some value =>
          if value.denominator = 0 then
            throw "constant denominator must be positive"
          else pure (.constant value.toCore)
      | none => throw "constant instruction requires value"
  | "variable" =>
      match wire.name with
      | some name =>
          if name.isEmpty then throw "variable instruction requires non-empty name"
          else pure (.variable name)
      | none => throw "variable instruction requires name"
  | "add" => pure .add
  | "multiply" => pure .multiply
  | "power" =>
      match wire.exponent with
      | some exponent => pure (.power exponent)
      | none => throw "power instruction requires exponent"
  | "inverse" => pure .inverse
  | other => throw s!"unsupported metric instruction {other}"

structure MetricExpectationWire where
  metric_id : String
  unit : String
  authority : String
  dependencies : List NamedIntervalWire
  program : List MetricInstructionWire
  observations : List Nat
deriving Repr, BEq, FromJson, ToJson

def MetricExpectationWire.toCore
    (wire : MetricExpectationWire) : Except String MetricExpectation := do
  let authority ← match wire.authority with
    | "theorem" => pure MetricAuthority.theorem
    | "assumption_backed" => pure MetricAuthority.assumptionBacked
    | other => throw s!"unsupported metric authority {other}"
  let names := wire.dependencies.map (·.name)
  if names.eraseDups.length != names.length then
    throw s!"metric {wire.metric_id} has duplicate dependencies"
  let program ← wire.program.mapM MetricInstructionWire.toCore
  pure {
    metricId := wire.metric_id
    unit := wire.unit
    authority
    dependencies := wire.dependencies.map NamedIntervalWire.toCore
    program
    observations := wire.observations
  }

structure EvidenceWireV2 where
  schema : String
  run_id : String
  source : SourceWireV2
  workload : WorkloadWire
  candidates : List RawCandidateWire
  expectations : List MetricExpectationWire
deriving Repr, BEq, FromJson, ToJson

def EvidenceWireV2.toCore (wire : EvidenceWireV2) :
    Except String (Evidence × List MetricExpectation) := do
  if wire.schema != "metric_evidence/v2" then
    throw "schema: expected metric_evidence/v2"
  if !digestValid wire.source.metric_expectations_sha256 then
    throw "source.metric_expectations_sha256: expected lowercase SHA-256"
  if wire.expectations.isEmpty then
    throw "expectations: expected at least one metric"
  let metricIds := wire.expectations.map (·.metric_id)
  if metricIds.eraseDups.length != metricIds.length then
    throw "expectations: metric ids must be unique"
  let v1 : EvidenceWire := {
    schema := evidenceSchema
    run_id := wire.run_id
    source := {
      evidence_sha256 := wire.source.evidence_sha256
      feature_flags_sha256 := wire.source.feature_flags_sha256
      campaign_manifest_sha256 := wire.source.campaign_manifest_sha256
    }
    workload := wire.workload
    candidates := wire.candidates
  }
  let evidence ← v1.toCore
  let expectations ← wire.expectations.mapM MetricExpectationWire.toCore
  pure (evidence, expectations)

def MetricAuthority.toWire : MetricAuthority → String
  | .theorem => "theorem"
  | .assumptionBacked => "assumption_backed"

def BandRelation.toWire : BandRelation → String
  | .inBand => "in_band"
  | .below => "below"
  | .above => "above"
  | .mixed => "mixed"

structure BandCountsWire where
  below : Nat
  within : Nat
  above : Nat
deriving Repr, BEq, FromJson, ToJson

structure BandAssessmentWire where
  metric_id : String
  unit : String
  authority : String
  expected : IntervalWire
  observed : IntervalWire
  counts : BandCountsWire
  relation : String
deriving Repr, BEq, FromJson, ToJson

def BandAssessmentWire.ofCore (assessment : BandAssessment) : BandAssessmentWire := {
  metric_id := assessment.metricId
  unit := assessment.unit
  authority := assessment.authority.toWire
  expected := IntervalWire.ofCore assessment.expected
  observed := IntervalWire.ofCore assessment.observed
  counts := ⟨assessment.counts.below, assessment.counts.within,
    assessment.counts.above⟩
  relation := assessment.relation.toWire
}

structure DerivedCandidateWire where
  id : String
  lever_snapshot_sha256 : String
  expected_latency_ns : IntervalWire
  input_mean_units : RatioWire
  pass_mean : RatioWire
  success_rate : RatioWire
  energy_mean_uj : RatioWire
  cost_mean_micro_usd : RatioWire
  quality_failures : Nat
  trainable_params : Nat
deriving Repr, BEq, FromJson, ToJson

def DerivedCandidateWire.ofCore
    (candidate : DerivedCandidate) : DerivedCandidateWire := {
  id := candidate.id
  lever_snapshot_sha256 := candidate.artifactSha256
  expected_latency_ns := IntervalWire.ofCore candidate.expectedLatencyNs
  input_mean_units := RatioWire.ofCore candidate.inputMeanUnits
  pass_mean := RatioWire.ofCore candidate.passMean
  success_rate := RatioWire.ofCore candidate.successRate
  energy_mean_uj := RatioWire.ofCore candidate.energyMeanUj
  cost_mean_micro_usd := RatioWire.ofCore candidate.costMeanMicroUsd
  quality_failures := candidate.qualityFailures
  trainable_params := candidate.trainableParameters
}

structure CertificateWire where
  schema : String
  checker : String
  verified : Bool
  assurance : String
  run_id : String
  evidence_sha256 : String
  feature_flags_sha256 : String
  campaign_manifest_sha256 : Option String
  hardware : String
  workload : WorkloadWire
  selected_candidate : String
  candidates : List DerivedCandidateWire
  selection_policy : List String
  trusted_boundary : List String
deriving Repr, BEq, FromJson, ToJson

def selectionPolicy : List String := [
  "quality_failures:min",
  "success_rate:max",
  "trainable_params:min",
  "expected_latency_upper_ns:min",
  "pass_mean:min",
  "energy_mean_uj:min",
  "cost_mean_micro_usd:min",
  "candidate_id:lexicographic"
]

def trustedBoundary : List String := [
  "measurement_truth",
  "json_adapter",
  "sha256_implementation",
  "filesystem",
  "lean_runtime",
  "operating_system"
]

def CertificateWire.ofCore (certificate : Certificate) : CertificateWire := {
  schema := certificateSchema
  checker := checkerId
  verified := true
  assurance := "observed_raw_samples"
  run_id := certificate.runId
  evidence_sha256 := certificate.sourceSha256
  feature_flags_sha256 := certificate.featureFlagsSha256
  campaign_manifest_sha256 := certificate.campaignManifestSha256
  hardware := certificate.hardware
  workload := ⟨certificate.workload.coldRequests,
    certificate.workload.warmRequests⟩
  selected_candidate := certificate.selected.id
  candidates := certificate.candidates.map DerivedCandidateWire.ofCore
  selection_policy := selectionPolicy
  trusted_boundary := trustedBoundary
}

structure CertificateWireV2 where
  schema : String
  checker : String
  verified : Bool
  assurance : String
  run_id : String
  evidence_sha256 : String
  feature_flags_sha256 : String
  campaign_manifest_sha256 : Option String
  metric_expectations_sha256 : String
  hardware : String
  workload : WorkloadWire
  selected_candidate : String
  candidates : List DerivedCandidateWire
  assessments : List BandAssessmentWire
  selection_policy : List String
  trusted_boundary : List String
deriving Repr, BEq, FromJson, ToJson

def computeCertificateV2
    (wire : EvidenceWireV2) : Except String CertificateWireV2 := do
  let (evidence, expectations) ← wire.toCore
  let certificate ← match check evidence with
    | none => throw "Lean checker rejected evidence"
    | some certificate => pure certificate
  let assessments ← match assessExpectations expectations with
    | none => throw "Lean checker rejected metric expectations"
    | some assessments => pure assessments
  pure {
    schema := "metric_certificate/v2"
    checker := "leverproof-lean/v2"
    verified := true
    assurance := "calculated_bands_and_observed_raw_samples"
    run_id := certificate.runId
    evidence_sha256 := certificate.sourceSha256
    feature_flags_sha256 := certificate.featureFlagsSha256
    campaign_manifest_sha256 := certificate.campaignManifestSha256
    metric_expectations_sha256 := wire.source.metric_expectations_sha256
    hardware := certificate.hardware
    workload := ⟨certificate.workload.coldRequests,
      certificate.workload.warmRequests⟩
    selected_candidate := certificate.selected.id
    candidates := certificate.candidates.map DerivedCandidateWire.ofCore
    assessments := assessments.map BandAssessmentWire.ofCore
    selection_policy := selectionPolicy
    trusted_boundary := trustedBoundary
  }

def computeCertificate (wire : EvidenceWire) : Except String CertificateWire := do
  let evidence ← wire.toCore
  match check evidence with
  | none => throw "Lean checker rejected evidence"
  | some certificate => pure (CertificateWire.ofCore certificate)

def computeCertificateJson (json : Json) : Except String Json := do
  let schemaJson ← json.getObjVal? "schema"
  let schema ← fromJson? schemaJson
  if schema = evidenceSchema then
    let wire : EvidenceWire ← fromJson? json
    pure (toJson (← computeCertificate wire))
  else if schema = "metric_evidence/v2" then
    let wire : EvidenceWireV2 ← fromJson? json
    pure (toJson (← computeCertificateV2 wire))
  else
    throw s!"unsupported evidence schema {schema}"

end LeverProofLean
