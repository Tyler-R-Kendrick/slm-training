"""Semantic-contrast corpus builder (SPV2-01)."""

from slm_training.data.semantic_contrast.builder import (
    BUILDER_VERSION,
    PROGRAM_FAMILY,
    SemanticContrastBuilder,
)
from slm_training.data.semantic_contrast.schema import (
    ContrastFamily,
    ContrastList,
    ContrastPair,
    ContrastRole,
    ContrastSeverity,
    CorpusSplit,
    FamilyMetrics,
    SemanticContrastRecord,
)
from slm_training.data.semantic_contrast.transforms import (
    TransformCandidate,
    generate_transforms,
)

__all__ = [
    "BUILDER_VERSION",
    "PROGRAM_FAMILY",
    "SemanticContrastBuilder",
    "ContrastFamily",
    "ContrastList",
    "ContrastPair",
    "ContrastRole",
    "ContrastSeverity",
    "CorpusSplit",
    "FamilyMetrics",
    "SemanticContrastRecord",
    "TransformCandidate",
    "generate_transforms",
]
