"""F2 (SLM-43): GraphQL DSL pack — the contract's second instance."""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar.backends.graphql_js import bridge_available
from slm_training.dsl.pack import available_packs, get_pack

needs_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="graphql bridge unavailable (cd src/apps/graphql_bridge && npm ci)",
)

GOOD_QUERY = "query { posts(limit: 3) { title author { name } } }"
BAD_FIELD_QUERY = "query { posts { title nonexistentField } }"
VARIABLE_QUERY = "query($id: ID!) { post(id: $id) { title body } }"


def test_graphql_pack_registers_without_bridge() -> None:
    # Registration, resolution, contract id, and the placeholder policy are
    # offline; only oracle/canonicalizer calls need the Node sidecar.
    assert "graphql" in available_packs()
    pack = get_pack("graphql")
    assert pack.grammar == "graphql"
    cid = pack.contract_id()
    assert cid.startswith("graphql-js-") and cid == pack.contract_id()
    policy = pack.placeholder_policy
    assert policy.is_placeholder("$id") and not policy.is_placeholder("id")
    assert policy.extract(VARIABLE_QUERY) == ["$id"]
    assert policy.merge(["$a"], ["$b", "$a"]) == ["$a", "$b"]
    assert pack.scope_rules.bind_encodings == ("schema-symbol",)
    assert "graphql-js validate" in pack.scope_rules.reference_legality


@needs_bridge
def test_graphql_oracle_enforces_schema_scope() -> None:
    pack = get_pack("graphql")
    program = pack.validity_oracle(GOOD_QUERY)
    assert program.serialized
    # The schema is the symbol table: a field that does not exist on the
    # parent type is a scope violation the oracle must reject.
    with pytest.raises(Exception, match="nonexistentField"):
        pack.validity_oracle(BAD_FIELD_QUERY)


@needs_bridge
def test_graphql_canonicalizer_idempotent() -> None:
    pack = get_pack("graphql")
    canonical = pack.canonicalize(GOOD_QUERY)
    assert pack.canonicalize(canonical) == canonical
    assert pack.canonical_fingerprint(GOOD_QUERY) == pack.canonical_fingerprint(
        canonical
    )
    # Whitespace-insensitive normal form.
    reflowed = GOOD_QUERY.replace(" { ", " {\n").replace(" } ", "\n}\n")
    assert pack.canonical_fingerprint(reflowed) == pack.canonical_fingerprint(
        GOOD_QUERY
    )


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
    build = pack.corpus_generator()
    records = build(root_id="f2-fixture", depth=1)
    assert records, "generator produced no records"
    # Typed generation: every record validates against the schema, and the
    # canonical form is stable.
    for record in records:
        program = pack.validity_oracle(record.openui)
        assert program.serialized
        canonical = pack.canonicalize(record.openui)
        assert pack.canonicalize(canonical) == canonical
    root_fields = {r.meta["root_field"] for r in records}
    assert {"posts", "post", "authors", "author"} <= root_fields


@needs_bridge
def test_graphql_schema_symbols_expose_the_symbol_table() -> None:
    from slm_training.dsl.grammar.backends.graphql_js import schema_symbols

    symbols = schema_symbols()
    assert {"Query", "Post", "Author", "Comment"} <= set(symbols)
    assert "title" in symbols["Post"] and "name" in symbols["Author"]
