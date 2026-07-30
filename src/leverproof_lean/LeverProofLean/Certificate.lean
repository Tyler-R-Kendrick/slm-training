import LeverProofLean.Resource

namespace LeverProofLean

def evidenceSchema := "metric_evidence/v1"
def certificateSchema := "metric_certificate/v1"
def checkerId := "leverproof-lean/v1"

private def lowerHexDigits : List Char :=
  "0123456789abcdef".toList

def digestValid (digest : String) : Bool :=
  digest.length == 64 && digest.toList.all lowerHexDigits.contains

structure Evidence where
  runId : String
  sourceSha256 : String
  featureFlagsSha256 : String
  campaignManifestSha256 : Option String
  hardware : String
  workload : Workload
  candidates : List RawCandidate
deriving Repr, BEq, Inhabited

def RawCandidate.valid (candidate : RawCandidate) : Bool :=
  !candidate.id.isEmpty &&
  digestValid candidate.artifactSha256 &&
  decide (candidate.successes ≤
    candidate.coldLatencyNs.length + candidate.warmLatencyNs.length)

def Evidence.valid (evidence : Evidence) : Bool :=
  !evidence.runId.isEmpty &&
  digestValid evidence.sourceSha256 &&
  digestValid evidence.featureFlagsSha256 &&
  evidence.campaignManifestSha256.all digestValid &&
  !evidence.hardware.isEmpty &&
  evidence.workload.valid &&
  !evidence.candidates.isEmpty &&
  evidence.candidates.all RawCandidate.valid

structure Certificate where
  runId : String
  sourceSha256 : String
  featureFlagsSha256 : String
  campaignManifestSha256 : Option String
  hardware : String
  workload : Workload
  candidates : List DerivedCandidate
  selected : DerivedCandidate
deriving Repr, BEq, Inhabited

def check (evidence : Evidence) : Option Certificate := do
  if evidence.valid then pure () else none
  let candidates ← deriveCandidates evidence.workload evidence.candidates
  let selected ← selectBest candidates
  if globallyPreferred selected candidates then pure () else none
  pure {
    runId := evidence.runId
    sourceSha256 := evidence.sourceSha256
    featureFlagsSha256 := evidence.featureFlagsSha256
    campaignManifestSha256 := evidence.campaignManifestSha256
    hardware := evidence.hardware
    workload := evidence.workload
    candidates
    selected
  }

theorem check_requires_valid_evidence
    (evidence : Evidence) (certificate : Certificate)
    (hcheck : check evidence = some certificate) :
    evidence.valid = true := by
  simp only [check] at hcheck
  split at hcheck
  · next h => exact h
  · simp at hcheck

theorem checked_selection_comes_from_derived_candidates
    (evidence : Evidence) (certificate : Certificate)
    (hcheck : check evidence = some certificate) :
    certificate.selected ∈ certificate.candidates := by
  simp only [check] at hcheck
  split at hcheck
  · next _ =>
      cases hderive : deriveCandidates evidence.workload evidence.candidates with
      | none => simp [hderive] at hcheck
      | some candidates =>
          cases hselected : selectBest candidates with
          | none => simp [hderive, hselected] at hcheck
          | some selected =>
              simp [hderive, hselected] at hcheck
              rcases hcheck with ⟨_, rfl⟩
              exact selectBest_member candidates selected hselected
  · simp at hcheck

theorem checked_selection_minimizes_qualityFailures
    (evidence : Evidence) (certificate : Certificate)
    (hcheck : check evidence = some certificate) :
    ∀ candidate ∈ certificate.candidates,
      certificate.selected.qualityFailures ≤ candidate.qualityFailures := by
  simp only [check] at hcheck
  split at hcheck
  · next _ =>
      cases hderive : deriveCandidates evidence.workload evidence.candidates with
      | none => simp [hderive] at hcheck
      | some candidates =>
          cases hselected : selectBest candidates with
          | none => simp [hderive, hselected] at hcheck
          | some selected =>
              simp [hderive, hselected] at hcheck
              rcases hcheck with ⟨_, rfl⟩
              exact selectBest_minimizes_qualityFailures
                candidates selected hselected
  · simp at hcheck

theorem checked_selection_is_globally_preferred
    (evidence : Evidence) (certificate : Certificate)
    (hcheck : check evidence = some certificate) :
    ∀ candidate ∈ certificate.candidates,
      preferCandidate certificate.selected candidate = true := by
  simp only [check] at hcheck
  split at hcheck
  · next _ =>
      cases hderive : deriveCandidates evidence.workload evidence.candidates with
      | none => simp [hderive] at hcheck
      | some candidates =>
          cases hselected : selectBest candidates with
          | none => simp [hderive, hselected] at hcheck
          | some selected =>
              simp [hderive, hselected] at hcheck
              rcases hcheck with ⟨hpreferred, rfl⟩
              exact (globallyPreferred_iff selected candidates).mp hpreferred
  · simp at hcheck

end LeverProofLean
