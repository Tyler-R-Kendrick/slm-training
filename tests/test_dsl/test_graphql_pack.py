"""F2 (SLM-43): GraphQL DSL pack — the contract's second instance.

Fits the canonical DslPack slot contract (main's F1/#290): the backend +
graphql-js oracle register even without the Node bridge; oracle/canonicalize
calls need the sidecar.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar.backends.graphql_js import bridge_available
from slm_training.dsl.pack import get_pack, list_packs

needs_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="graphql bridge unavailable (cd src/apps/graphql_bridge && npm ci)",
)

GOOD_QUERY = "query { posts(limit: 3) { title author { name } } }"
BAD_FIELD_QUERY = "query { posts { title nonexistentField } }"
VARIABLE_QUERY = "query($id: ID!) { post(id: $id) { title body } }"


def test_graphql_pack_registers_without_bridge() -> None:
    # Registration + slots are offline; only oracle/canonicalize need Node.
    assert "graphql" in list_packs()
    pack = get_pack("graphql")
    assert pack.pack_id == "graphql"
    assert pack.reward_label == "well_formed_not_behavioral"
    # The routed-content channel: GraphQL variables.
    policy = pack.placeholder_policy
    assert policy.is_placeholder("$id") and not policy.is_placeholder("id")
    assert policy.slot_contract(VARIABLE_QUERY) == ("$id",)
    assert policy.extract(VARIABLE_QUERY) == ["$id"]
    # Oracle + canonicalizer slots are filled; require() does not raise.
    assert pack.require("oracle") is not None
    assert pack.require("canonicalize") is not None


@needs_bridge
def test_graphql_oracle_enforces_schema_scope() -> None:
    pack = get_pack("graphql")
    program = pack.require("oracle")(GOOD_QUERY)
    assert program.serialized
    # The schema is the symbol table: a field absent from the parent type is a
    # scope violation the oracle must reject.
    with pytest.raises(Exception, match="nonexistentField"):
        pack.require("oracle")(BAD_FIELD_QUERY)


@needs_bridge
def test_graphql_canonicalizer_idempotent() -> None:
    pack = get_pack("graphql")
    canonicalize = pack.require("canonicalize")
    canonical = canonicalize(GOOD_QUERY)
    assert canonicalize(canonical) == canonical
    reflowed = GOOD_QUERY.replace(" { ", " {\n").replace(" } ", "\n}\n")
    assert canonicalize(reflowed) == canonical


@needs_bridge
def test_graphql_backend_stream_check_classifies_prefixes() -> None:
    from slm_training.dsl.grammar.backends import get_backend

    backend = get_backend("graphql")
    complete = backend.stream_check(GOOD_QUERY)
    assert complete.ok and complete.has_root
    truncated = backend.stream_check("query { posts { title")
    assert not truncated.ok and truncated.incomplete
    assert not truncated.hard_error
    garbage = backend.stream_check("query }{ ???")
    assert not garbage.ok and not garbage.incomplete


@needs_bridge
def test_graphql_generator_outputs_pass_the_packs_own_oracle() -> None:
    pack = get_pack("graphql")
    build = pack.require("corpus_generator")
    records = build(root_id="f2-fixture", depth=1)
    assert records, "generator produced no records"
    oracle = pack.require("oracle")
    for record in records:
        program = oracle(record.openui)
        assert program.serialized
    root_fields = {r.meta["root_field"] for r in records}
    assert {"posts", "post", "authors", "author"} <= root_fields


@needs_bridge
def test_graphql_schema_symbols_expose_the_symbol_table() -> None:
    from slm_training.dsl.grammar.backends.graphql_js import schema_symbols

    symbols = schema_symbols()
    assert {"Query", "Post", "Author", "Comment"} <= set(symbols)
    assert "title" in symbols["Post"] and "name" in symbols["Author"]
