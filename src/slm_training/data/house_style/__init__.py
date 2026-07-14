"""Deterministic house-style defaults and target resolution."""

from slm_training.data.house_style.policy import (
    DEFAULT_HOUSE_STYLE,
    CandidateScore,
    HouseStylePolicy,
    HouseStyleResolution,
    resolve_target,
)

__all__ = [
    "DEFAULT_HOUSE_STYLE",
    "CandidateScore",
    "HouseStylePolicy",
    "HouseStyleResolution",
    "resolve_target",
]
