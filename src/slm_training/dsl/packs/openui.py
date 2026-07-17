"""OpenUI as the first DSL pack instance (F1, SLM-34).

Pure wiring: every member points at the long-standing OpenUI owner module.
No behavior moves; the pack makes the implicit contract explicit so F2+ DSLs
implement the same shape.
"""

from __future__ import annotations

from slm_training.dsl.pack import DslPack, PlaceholderPolicy, ScopeRules


def build_openui_pack() -> DslPack:
    from slm_training.dsl.canonicalize import canonical_fingerprint, canonicalize
    from slm_training.dsl.language_contract import contract_id
    from slm_training.dsl.parser import validate
    from slm_training.dsl.placeholders import (
        extract_placeholders,
        is_placeholder,
        merge_placeholders,
    )

    return DslPack(
        id="openui",
        grammar="openui",
        canonicalize=canonicalize,
        canonical_fingerprint=canonical_fingerprint,
        # The official @openuidev/lang-core parse+serialize round-trip is the
        # oracle; syntax legality is never delegated to the model.
        validity_oracle=validate,
        scope_rules=ScopeRules(
            # B/C-track representations supported by the production codec /
            # DSL-native tokenizer (C1: verifier-enforced relative refs).
            bind_encodings=("absolute", "relative"),
            reference_legality="dsl.production_codec (ParseError on illegal delta)",
            scope_families_provider=(
                "slm_training.harnesses.train_data.scope_corpus:scope_families"
            ),
        ),
        placeholder_policy=PlaceholderPolicy(
            is_placeholder=is_placeholder,
            extract=extract_placeholders,
            merge=merge_placeholders,
        ),
        contract_id=contract_id,
        corpus_generator_provider=(
            "slm_training.harnesses.train_data.scope_corpus:build_scope_corpus"
        ),
        extras={
            "production_codec": "slm_training.dsl.production_codec:encode_openui",
            "emptiness_probe": "slm_training.evals.emptiness_probe:evaluate_emptiness",
        },
    )
