"""Semantic-contrast corpus builder (SPV2-01)."""

from slm_training.data.semantic_contrast.builder import (
    BUILDER_VERSION,
    PROGRAM_FAMILY,
    SemanticContrastBuilder,
)
from slm_training.data.semantic_contrast.metamorphic import (
    MetamorphicCase,
    MetamorphicFamily,
    generate_alpha_rename_case,
    generate_ast_rewrite_equivalence_case,
    generate_prompt_paraphrase_case,
    generate_prompt_single_fact_edit_case,
    generate_reorder_case,
    root_family_for,
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
    "MetamorphicCase",
    "MetamorphicFamily",
    "generate_alpha_rename_case",
    "generate_ast_rewrite_equivalence_case",
    "generate_prompt_paraphrase_case",
    "generate_prompt_single_fact_edit_case",
    "generate_reorder_case",
    "root_family_for",
]
