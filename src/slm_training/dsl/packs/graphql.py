"""GraphQL as the second DSL pack instance (F2, SLM-43).

The proof that the F1 contract generalizes: graphql-js is the validity
oracle, the introspection schema is the scope/symbol table (every selected
field must exist on its parent type — the strongest real-world test of the
C1/C2 externalized-scope levers, since OpenUI 0.2.x has little binding
surface), and variables (``$name``) are the routed-content channel.
"""

from __future__ import annotations

import hashlib
import json

from slm_training.dsl.pack import DslPack, PlaceholderPolicy, ScopeRules


def _canonicalize(source: str) -> str:
    from slm_training.dsl.grammar.backends.graphql_js import ParseError, invoke_bridge

    result = invoke_bridge({"op": "canonicalize", "source": source})
    if not result.get("ok"):
        raise ParseError("; ".join(result.get("errors", ["parse error"])))
    return str(result["canonical"])


def _canonical_fingerprint(source: str) -> str:
    return hashlib.sha256(_canonicalize(source).encode("utf-8")).hexdigest()


def _validity_oracle(source: str):
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("graphql").validate(source)


def _contract_id() -> str:
    """Content-derived language-surface hash: graphql version + demo schema.

    Offline by construction (reads package.json + the SDL file; no bridge),
    so dataset stamping works on hosts without node_modules installed.
    """
    from slm_training.dsl.grammar.backends.graphql_js import (
        BRIDGE_DIR,
        DEFAULT_SCHEMA_PATH,
    )

    package = json.loads((BRIDGE_DIR / "package.json").read_text(encoding="utf-8"))
    version = str(package.get("dependencies", {}).get("graphql", "unknown"))
    digest = hashlib.sha256()
    digest.update(version.encode("utf-8"))
    digest.update(DEFAULT_SCHEMA_PATH.read_bytes())
    return f"graphql-js-{version}-{digest.hexdigest()[:12]}"


def _is_variable(value: str) -> bool:
    return value.startswith("$") and len(value) > 1


def _extract_variables(source: str) -> list[str]:
    from slm_training.dsl.grammar.backends.graphql_js import extract_variables

    return extract_variables(source)


def _merge_variables(*groups) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return merged


def build_graphql_pack() -> DslPack:
    return DslPack(
        id="graphql",
        grammar="graphql",
        canonicalize=_canonicalize,
        canonical_fingerprint=_canonical_fingerprint,
        validity_oracle=_validity_oracle,
        scope_rules=ScopeRules(
            bind_encodings=("schema-symbol",),
            reference_legality=(
                "graphql-js validate: every selected field must exist on its "
                "parent schema type (src/apps/graphql_bridge)"
            ),
            scope_families_provider=None,
        ),
        placeholder_policy=PlaceholderPolicy(
            is_placeholder=_is_variable,
            extract=_extract_variables,
            merge=_merge_variables,
        ),
        contract_id=_contract_id,
        corpus_generator_provider=(
            "slm_training.harnesses.train_data.graphql_corpus:build_graphql_corpus"
        ),
        extras={
            "schema_symbols": (
                "slm_training.dsl.grammar.backends.graphql_js:schema_symbols"
            ),
        },
    )
