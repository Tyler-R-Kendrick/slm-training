"""GraphQL DSL pack (F2 / SLM-43): schema-as-symbol-table.

The introspection schema supplies scope: which fields exist on which type,
which arguments each field takes. The validity oracle is graphql-core's
`validate` against a committed SDL fixture, so "valid" means schema-correct,
not merely well-formed — the strongest real-world test of scope-aware
generation in the program (OpenUI 0.2.x has little binding surface).

Canonicalizer is a genuine normal form here (unlike the toy/arith identity
canonicalizers): `print_ast(parse(x))` re-emits GraphQL in the reference
implementation's canonical whitespace/formatting, and is idempotent.
"""

from __future__ import annotations

import random

from slm_training.dsl.packs.types import DSLPack, PlaceholderPolicy
from slm_training.dsl.schema import ExampleRecord

# Query templates over the shop fixture schema. Each is schema-valid; the
# generator only varies leaf field selections and argument literals so every
# emitted query passes the oracle by construction.
_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "Fetch product {id} with its price and stock.",
        'query {{ product(id: "{id}") {{ name price inStock }} }}',
    ),
    (
        "List the first {n} products in {category} with names and prices.",
        "query {{ products(first: {n}, category: {category}) "
        "{{ name price category }} }}",
    ),
    (
        "Get order {id}: total and the names of its items.",
        'query {{ order(id: "{id}") {{ total items {{ name price }} }} }}',
    ),
    (
        "Show the current customer's display name and first {n} orders.",
        "query {{ me {{ displayName orders(first: {n}) {{ id total }} }} }}",
    ),
    (
        "Get product {id} with its first {n} reviews (rating and author name).",
        'query {{ product(id: "{id}") {{ name reviews(first: {n}) '
        "{{ rating author {{ displayName }} }} }} }}",
    ),
)
_CATEGORIES = ("ELECTRONICS", "GROCERY", "CLOTHING", "TOYS")


def _canonicalize(source: str) -> str:
    from graphql import parse, print_ast

    return print_ast(parse(source)).strip()


def _canonical_equal(a: str, b: str) -> bool:
    return _canonicalize(a) == _canonicalize(b)


def _validity_oracle(source: str, output_kind: str = "document") -> object:
    if output_kind != "document":
        raise ValueError("graphql pack validates query documents only")
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("graphql").validate(source)


def _scope_check(source: str):
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("graphql").stream_check(source)


def _corpus_generator(count: int, seed: int) -> list[ExampleRecord]:
    rng = random.Random(seed)
    records: list[ExampleRecord] = []
    for index in range(count):
        prompt_tpl, query_tpl = _TEMPLATES[index % len(_TEMPLATES)]
        values = {
            "id": f"p{rng.randint(1, 999):03d}",
            "n": rng.randint(1, 10),
            "category": rng.choice(_CATEGORIES),
        }
        prompt = prompt_tpl.format(**values)
        query = _canonicalize(query_tpl.format(**values))
        # Fail-closed self-check: never emit a schema-invalid training row.
        _validity_oracle(query, "document")
        records.append(
            ExampleRecord(
                id=f"gql_{seed}_{index}",
                prompt=prompt,
                openui=query,
                placeholders=[],
                meta={"task": "generation"},
            )
        )
    return records


def build_pack() -> DSLPack:
    return DSLPack(
        id="graphql",
        description="GraphQL query DSL — graphql-core schema-aware oracle "
        "over a fixture introspection schema (F2)",
        grammar="graphql",
        canonicalize=_canonicalize,
        canonical_equal=_canonical_equal,
        validity_oracle=_validity_oracle,
        corpus_generator=_corpus_generator,
        scope_check=_scope_check,
        placeholders=PlaceholderPolicy(
            is_placeholder=lambda value: False,
            extract=lambda source: [],
        ),
        notes=(
            "oracle = graphql-core validate against a committed SDL fixture; "
            "'valid' means schema-correct (field/arg existence), not just "
            "well-formed",
            "canonicalizer = print_ast(parse(x)) reference normal form "
            "(idempotent)",
            "graphql-js byte parity is a non-goal; requires the optional "
            "graphql-core dependency (available() gates tests)",
        ),
    )
