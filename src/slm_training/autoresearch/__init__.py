"""Autonomous, evidence-grounded OpenUI training research harness."""

from slm_training.autoresearch.rl_gate import assert_rl_ready, assess_rl_readiness
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    Diagnosis,
    EvidenceSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    ResearchRequest,
    ResearcherRun,
    RLReadinessReport,
    ResearchSource,
)
from slm_training.autoresearch.storage import CampaignStore

__all__ = [
    "CampaignSpec",
    "CampaignStore",
    "Diagnosis",
    "EvidenceSnapshot",
    "ExperimentOutcome",
    "ExperimentSpec",
    "ResearchRequest",
    "ResearcherRun",
    "RLReadinessReport",
    "ResearchSource",
    "assert_rl_ready",
    "assess_rl_readiness",
]
