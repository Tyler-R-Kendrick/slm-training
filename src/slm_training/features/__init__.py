"""Product feature flags via OpenFeature (distinct from ML matrix experiments)."""

from slm_training.features.defaults import PRODUCT_FLAG_DEFAULTS
from slm_training.features.keys import PRODUCT_FLAG_KEYS
from slm_training.features.runtime import FeatureRuntime

__all__ = [
    "FeatureRuntime",
    "PRODUCT_FLAG_DEFAULTS",
    "PRODUCT_FLAG_KEYS",
]
