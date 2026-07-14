"""Language-contract coverage corpus (family ``language_contract``).

Exhaustive, independently-testable coverage of the pinned OpenUI v0.2.x surface:
one minimal *positive* per grammar production / lexical form / component (+ its
positional props), plus *negatives* that each fail a specific verifier gate
(G0 lexical / G1 grammar / G2 schema / G3 references / G4 dataflow).

See ``docs/design/language-contract-corpus.md``.
"""

from slm_training.data.language_contract.corpus import (
    LANGUAGE_CONTRACT_FAMILY,
    build_corpus,
    coverage_report,
    iter_negatives,
    iter_positives,
)

__all__ = [
    "LANGUAGE_CONTRACT_FAMILY",
    "build_corpus",
    "coverage_report",
    "iter_negatives",
    "iter_positives",
]
