"""Autonomous, evidence-grounded OpenUI training research harness."""

from slm_training.autoresearch.rl_gate import assert_rl_ready, assess_rl_readiness
from slm_training.autoresearch.experiment_campaign import (
    CampaignDeviationV1,
    CampaignResultV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
    validate_result_claim,
)
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    Diagnosis,
    EvidenceSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    HypothesisFeedback,
    HypothesizerBenchmarkReport,
    HypothesisMatrix,
    ResearchRequest,
    ResearcherRun,
    RLReadinessReport,
    ResearchSource,
)
from slm_training.autoresearch.storage import CampaignStore

__all__ = [
    "CampaignSpec",
    "CampaignDeviationV1",
    "CampaignResultV1",
    "CampaignStore",
    "Diagnosis",
    "EvidenceSnapshot",
    "ExperimentOutcome",
    "ExperimentCampaignV1",
    "ExperimentSpec",
    "HypothesisFeedback",
    "HypothesizerBenchmarkReport",
    "HypothesisMatrix",
    "ResearchRequest",
    "ResearcherRun",
    "RLReadinessReport",
    "ResearchSource",
    "assert_rl_ready",
    "assess_rl_readiness",
    "campaign_manifest_sha256",
    "validate_result_claim",
]
