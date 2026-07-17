"""F2 (SLM-43): GraphQL as a second DslPack instance.

Fits the canonical :class:`slm_training.dsl.pack.DslPack` slot contract
(landed on main via F1/#290): graphql-js is the validity oracle, the
introspection schema is the scope/symbol table (every selected field must
exist on its parent type — the strongest real-world exercise of the
externalized-scope principle, since OpenUI 0.2.x has little binding surface),
``print(parse(x))`` is the canonical form, and GraphQL variables (``$name``)
are the routed-content channel.

The grammar backend + Node sidecar live in
``dsl/grammar/backends/graphql_js.py`` and ``src/apps/graphql_bridge``.
"""

from __future__ import annotations

import re
from typing import Any

from slm_training.dsl.grammar.backends import get_backend
from slm_training.dsl.pack import DslPack, PlaceholderPolicy

_VARIABLE_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def _graphql_slot_contract(
    source: str, *, declared: Any = None
) -> tuple[str, ...]:
    """Ordered variable inventory ($name) — GraphQL's routed-content slots."""
    seen: list[str] = []
    for match in _VARIABLE_RE.finditer(source or ""):
        if match.group(0) not in seen:
            seen.append(match.group(0))
    for extra in declared or ():
        if extra not in seen:
            seen.append(extra)
    return tuple(seen)


def _graphql_canonicalize(source: str) -> str:
    from slm_training.dsl.grammar.backends.graphql_js import invoke_bridge
    from slm_training.dsl.lang_core import ParseError

    result = invoke_bridge({"op": "canonicalize", "source": source})
    if not result.get("ok"):
        raise ParseError("; ".join(result.get("errors", ["parse error"])))
    return str(result["canonical"])


def _graphql_oracle(record: Any, context: Any = None) -> Any:  # noqa: ARG001
    """Validity verdict: parse + full schema validation via graphql-js."""
    source = record if isinstance(record, str) else getattr(record, "openui", "")
    return get_backend("graphql").validate(source)


def build_graphql_pack() -> DslPack:
    return DslPack(
        pack_id="graphql",
        backend=get_backend("graphql"),
        placeholder_policy=PlaceholderPolicy(
            placeholder_re=_VARIABLE_RE,
            content_props=frozenset(),
            slot_contract=_graphql_slot_contract,
            is_placeholder=lambda value: bool(_VARIABLE_RE.fullmatch(value)),
            extract=lambda source: list(_graphql_slot_contract(source)),
        ),
        # graphql-js proves well-formedness + schema validity, never behavior.
        reward_label="well_formed_not_behavioral",
        canonicalize=_graphql_canonicalize,
        oracle=_graphql_oracle,
        corpus_generator=(
            lambda **kw: __import__(
                "slm_training.harnesses.train_data.graphql_corpus",
                fromlist=["build_graphql_corpus"],
            ).build_graphql_corpus(**kw)
        ),
    )


__all__ = ["build_graphql_pack"]
