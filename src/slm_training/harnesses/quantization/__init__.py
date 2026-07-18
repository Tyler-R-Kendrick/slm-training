"""Quantization calibration, sensitivity, and allocation harnesses (CAP3)."""

from __future__ import annotations

from slm_training.harnesses.quantization.allocation import (
    AllocationChoice,
    AllocationManifest,
    allocate_mixed_precision,
)
from slm_training.harnesses.quantization.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    PRIMARY_STRATEGIES,
    CalibrationCorpusManifest,
    CalibrationSample,
    build_calibration_corpus,
    calibrate_scales_ptq,
    load_grammar_decision_traces,
    qat_reconstruct_local_scorer,
    run_calibration,
)
from slm_training.harnesses.quantization.sensitivity import (
    GroupFormatPoint,
    GroupingPolicy,
    ParameterGroup,
    SensitivityReport,
    compute_gradient_proxy,
    default_grouping_policy,
    profile_group_sensitivity,
)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "PRIMARY_STRATEGIES",
    "CalibrationCorpusManifest",
    "CalibrationSample",
    "build_calibration_corpus",
    "calibrate_scales_ptq",
    "load_grammar_decision_traces",
    "qat_reconstruct_local_scorer",
    "run_calibration",
    "ParameterGroup",
    "GroupingPolicy",
    "GroupFormatPoint",
    "SensitivityReport",
    "compute_gradient_proxy",
    "default_grouping_policy",
    "profile_group_sensitivity",
    "AllocationChoice",
    "AllocationManifest",
    "allocate_mixed_precision",
]
