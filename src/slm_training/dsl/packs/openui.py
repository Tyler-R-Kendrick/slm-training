"""OpenUI as the first DSL pack instance (F1 / SLM-34)."""

from __future__ import annotations

from slm_training.dsl.packs.types import DSLPack, PlaceholderPolicy
from slm_training.dsl.schema import ExampleRecord


def _canonicalize(source: str) -> str:
    from slm_training.dsl.canonicalize import canonicalize

    return canonicalize(source)


def _canonical_equal(a: str, b: str) -> bool:
    from slm_training.dsl.canonicalize import canonical_equal

    return canonical_equal(a, b)


def _validity_oracle(source: str, output_kind: str = "document") -> object:
    from slm_training.dsl.parser import validate_output

    return validate_output(source, output_kind)


def _scope_check(source: str):
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("openui").stream_check(source)


def _corpus_generator(count: int, seed: int) -> list[ExampleRecord]:
    """Coverage-guided typed generation (progspec) projected to records.

    Prompts here are deterministic single-sentence stand-ins — the full
    prompt-synthesis pipeline stays in `harnesses/train_data`; the pack
    generator's contract is valid, typed, reproducible programs.
    """
    from slm_training.data.progspec.generate import generate_program_specs
    from slm_training.data.progspec.schema import emit_record
    from slm_training.data.contract import canonicalize_example_template_markers
    from slm_training.dsl.analysis.templatize import assert_role_safe_output, templatize
    from slm_training.dsl.language_contract import assert_symbol_only_output

    result = generate_program_specs(count, seed=seed)
    records: list[ExampleRecord] = []
    for index, spec in enumerate(result.programs):
        templated = templatize(spec.canonical_openui)
        record = canonicalize_example_template_markers(
            emit_record(
                spec,
                prompt=f"Generate the canonical layout for program {index}.",
                task="generation",
                openui=templated.source,
                meta={"templatized_literals": len(templated.replacements)},
            )
        )
        assert_symbol_only_output(record.openui, output_kind=record.target_kind)
        assert_role_safe_output(record.openui, output_kind=record.target_kind)
        records.append(record)
    return records


def _is_placeholder(value: str) -> bool:
    from slm_training.dsl.placeholders import is_placeholder

    return is_placeholder(value)


def _extract(source: str) -> list[str]:
    from slm_training.dsl.placeholders import extract_placeholders

    return list(extract_placeholders(source))


def build_pack() -> DSLPack:
    from slm_training.dsl.placeholders import CONTENT_PROPS

    return DSLPack(
        id="openui",
        description="OpenUI layout DSL — hybrid lang-core/Lark backend, "
        "codec round-trip canonical form, layered G0-G12 oracle upstream",
        grammar="openui",
        canonicalize=_canonicalize,
        canonical_equal=_canonical_equal,
        validity_oracle=_validity_oracle,
        corpus_generator=_corpus_generator,
        scope_check=_scope_check,
        placeholders=PlaceholderPolicy(
            is_placeholder=_is_placeholder,
            extract=_extract,
            content_props=frozenset(CONTENT_PROPS),
        ),
        notes=(
            "validity_oracle is the syntactic layer; the semantic G0-G12 "
            "stack lives in data/verify and consumes the same backend",
        ),
    )
