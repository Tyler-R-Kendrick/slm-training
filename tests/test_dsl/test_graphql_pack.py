"""F2 (SLM-43): GraphQL pack — schema-aware oracle and symbol table."""

from __future__ import annotations

import pytest

from slm_training.dsl.packs import get_pack

_pack = get_pack("graphql")
pytestmark = pytest.mark.skipif(
    not _pack.backend().available(),
    reason="graphql-core not installed (optional graphql extra)",
)

VALID = "query { product(id: \"p1\") { name price inStock } }"


def test_schema_is_the_symbol_table() -> None:
    names = get_pack("graphql").backend().component_names()
    # Types from the fixture schema, not a hardcoded list.
    assert {"Product", "Order", "Customer", "Category"} <= names
    schema = get_pack("graphql").backend().library_schema()
    assert "price" in schema["Product"]
    assert "reviews" in schema["Product"]


def test_oracle_is_schema_aware_not_just_syntactic() -> None:
    pack = get_pack("graphql")
    pack.validity_oracle(VALID, "document")
    # Syntactically fine, but the field does not exist on Product -> reject.
    with pytest.raises(ValueError, match="bogusField|schema validation"):
        pack.validity_oracle(
            'query { product(id: "p1") { bogusField } }', "document"
        )
    # Wrong argument name -> reject.
    with pytest.raises(ValueError):
        pack.validity_oracle(
            'query { product(wrongArg: "p1") { name } }', "document"
        )
    # Malformed syntax -> reject.
    with pytest.raises(ValueError):
        pack.validity_oracle("query { product(id: ", "document")


def test_canonicalizer_is_a_real_normal_form() -> None:
    pack = get_pack("graphql")
    messy = 'query{product(id:"p1"){name    price\ninStock}}'
    canonical = pack.canonicalize(messy)
    assert pack.canonicalize(canonical) == canonical
    assert pack.canonical_equal(messy, VALID)  # same query, different spacing


def test_generated_corpus_is_schema_valid_and_deterministic() -> None:
    pack = get_pack("graphql")
    records = pack.corpus_generator(8, 5)
    assert [r.openui for r in records] == [
        r.openui for r in pack.corpus_generator(8, 5)
    ]
    for record in records:
        pack.validity_oracle(record.openui, "document")
