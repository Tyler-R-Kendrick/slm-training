"""Toy-layout DSL pack — the second instance proving the contract shape.

Deliberately minimal: identity canonicalizer (stated), backend validate as
the oracle, and a tiny deterministic template generator. Its job is to keep
the pack contract honest — anything OpenUI-shaped that creeps into the
contract breaks this pack's tests first.
"""

from __future__ import annotations

import random

from slm_training.dsl.packs.types import DSLPack, PlaceholderPolicy
from slm_training.dsl.schema import ExampleRecord

_TEMPLATES = (
    'root = row(title, action)\ntitle = text("{a}")\naction = button("{b}")',
    'root = col(hero)\nhero = row(title, body)\ntitle = text("{a}")\nbody = text("{b}")',
    'root = row(items)\nitems = [text("{a}"), button("{b}")]',
)
_SLOTS = (":hero.title", ":hero.body", ":cta.label", ":page.blurb")


def _validity_oracle(source: str, output_kind: str = "document") -> object:
    if output_kind != "document":
        raise ValueError("toy-layout pack validates documents only")
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("toy-layout").validate(source)


def _canonicalize(source: str) -> str:
    # Identity normal form (stated in notes): validated, stripped source.
    _validity_oracle(source, "document")
    return source.strip()


def _canonical_equal(a: str, b: str) -> bool:
    return _canonicalize(a) == _canonicalize(b)


def _scope_check(source: str):
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("toy-layout").stream_check(source)


def _corpus_generator(count: int, seed: int) -> list[ExampleRecord]:
    rng = random.Random(seed)
    records: list[ExampleRecord] = []
    for index in range(count):
        template = _TEMPLATES[index % len(_TEMPLATES)]
        a, b = rng.sample(_SLOTS, 2)
        source = template.format(a=a, b=b)
        _validity_oracle(source, "document")
        records.append(
            ExampleRecord(
                id=f"toy_{seed}_{index}",
                prompt=f"Lay out toy screen {index} with {a} and {b}.",
                openui=source,
                placeholders=[a, b],
            )
        )
    return records


def _is_placeholder(value: str) -> bool:
    from slm_training.dsl.placeholders import is_placeholder

    return is_placeholder(value)


def _extract(source: str) -> list[str]:
    from slm_training.dsl.placeholders import extract_placeholders

    return list(extract_placeholders(source))


def build_pack() -> DSLPack:
    return DSLPack(
        id="toy-layout",
        description="Toy layout DSL — minimal second pack instance",
        grammar="toy-layout",
        canonicalize=_canonicalize,
        canonical_equal=_canonical_equal,
        validity_oracle=_validity_oracle,
        corpus_generator=_corpus_generator,
        scope_check=_scope_check,
        placeholders=PlaceholderPolicy(
            is_placeholder=_is_placeholder,
            extract=_extract,
        ),
        notes=(
            "identity canonicalizer — no codec round-trip normal form",
            "document kind only; no fragment oracle",
        ),
    )
