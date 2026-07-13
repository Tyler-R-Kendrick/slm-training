"""Dataset adapters (RICO, leakage helpers, structure scrubbing)."""

from slm_training.data.contract import (
    GenerationRequest,
    canonical_slot_contract,
    normalize_example_record,
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
    "STYLE_STRING_TOKENS",
    "canonical_slot_contract",
    "cluster_by_structure",
    "clustered_train_val_split",
    "find_leakage",
    "fingerprint_openui",
    "fingerprint_openui_structure",
    "fingerprint_pair",
    "fingerprint_prompt",
    "is_style_token",
    "load_train_fingerprints",
    "normalize_example_record",
    "normalize_openui_structure",
    "strip_style_literals",
    "structure_fingerprint",
]
