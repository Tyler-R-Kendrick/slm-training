"""LDI4-02 representation-analysis: SAE decision-state diagnostics (SLM-136).

Only the torch-free schema (``spec``) is re-exported here so the package imports without
torch; ``sae`` and ``interventions`` require torch and are imported directly.
"""

from slm_training.harnesses.representations.spec import (
    SCHEMA_VERSION,
    CaptureManifest,
    CaptureRow,
    FeatureSelectionError,
    SAEArm,
    SAEConfig,
    matched_sae_arms,
    select_features_train_only,
)

__all__ = [
    "SCHEMA_VERSION",
    "CaptureManifest",
    "CaptureRow",
    "FeatureSelectionError",
    "SAEArm",
    "SAEConfig",
    "matched_sae_arms",
    "select_features_train_only",
]
