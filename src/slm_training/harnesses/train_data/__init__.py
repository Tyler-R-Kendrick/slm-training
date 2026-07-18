"""Training-data harness: produce versioned train corpora."""

from slm_training.harnesses.train_data.pipeline import (
    PROFILES,
    TrainDataConfig,
    build_train_data,
    resolve_profile,
)

__all__ = ["PROFILES", "TrainDataConfig", "build_train_data", "resolve_profile"]
