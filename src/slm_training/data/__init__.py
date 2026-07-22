"""Dataset adapters (RICO, leakage helpers, structure scrubbing, mixtures)."""

from slm_training.data.contract import (
    GenerationRequest,
    canonical_slot_contract,
    is_canonical_template_marker,
    normalize_example_record,
)
from slm_training.data.dedup import (
    apply_fuzzy_dedup,
    apply_semantic_cluster_cap,
    cluster_exposure_stats,
)
from slm_training.data.leakage import (
    find_leakage,
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
    load_train_fingerprints,
    normalize_openui_structure,
)
from slm_training.data.mixture import (
    MixtureManifest,
    load_mixture_manifest,
    sample_mixture_batch,
)
from slm_training.data.splits import (
    ClusteredSplit,
    cluster_by_structure,
    clustered_train_val_split,
    structure_fingerprint,
)
from slm_training.data.structure import (
    STYLE_STRING_TOKENS,
    is_style_token,
    strip_style_literals,
)

__all__ = [
    "ClusteredSplit",
    "GenerationRequest",
    "MixtureManifest",
    "STYLE_STRING_TOKENS",
    "apply_fuzzy_dedup",
    "apply_semantic_cluster_cap",
    "canonical_slot_contract",
    "is_canonical_template_marker",
    "cluster_by_structure",
    "cluster_exposure_stats",
    "clustered_train_val_split",
    "find_leakage",
    "fingerprint_openui",
    "fingerprint_openui_structure",
    "fingerprint_pair",
    "fingerprint_prompt",
    "is_style_token",
    "load_mixture_manifest",
    "load_train_fingerprints",
    "normalize_example_record",
    "normalize_openui_structure",
    "sample_mixture_batch",
    "strip_style_literals",
    "structure_fingerprint",
]
