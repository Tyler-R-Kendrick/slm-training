"""Scope-graded family invariants: identity, canonical pairs, repairs, typed maps."""

from __future__ import annotations

import pytest

from slm_training.data.corrupt import build_scoped_corruptions
from slm_training.dsl.parser import ParseError, validate, validate_output
from slm_training.harnesses.train_data.pipeline import _normalize_record
from slm_training.harnesses.train_data.scope_corpus import (
    ScopeCorpusConfig,
    build_scope_corpus,
    decanonicalize_variants,
    scope_families,
)

SOURCE = (
    'root = Stack([hero, cta], "column")\n'
    'hero = TextContent(":hero.title")\n'
    'cta = Button(":cta.label", true)'
)


@pytest.fixture(scope="module")
def corpus():
    records, pairs = build_scope_corpus(
        root_id="p1",
        openui=SOURCE,
        split_group_id="p1",
        program_family_id="fam:p1",
        lineage_id="p1",
        config=ScopeCorpusConfig(),
    )
    return records, pairs


def test_all_family_groups_emitted(corpus) -> None:
    records, _ = corpus
    families = {record.source for record in records}
    for prefix in ("scope_identity_", "scope_canonical_", "scope_repair_"):
        assert any(name.startswith(prefix) for name in families), prefix
    assert "lexical_typed_map" in families
    assert families <= set(scope_families())


def test_identity_rows_echo_their_prompt_input(corpus) -> None:
    records, _ = corpus
    identity = [r for r in records if r.source.startswith("scope_identity")]
    assert identity
    for record in identity:
        embedded = record.prompt.split("---INPUT---\n", 1)[1]
        assert embedded == record.openui


def test_identity_rows_survive_normalization_byte_identical(corpus) -> None:
    records, _ = corpus
    for record in records:
        if not record.source.startswith("scope_identity"):
            continue
        normalized = _normalize_record(record)
        assert normalized.openui == record.openui, record.source


def test_canonical_rows_rewrite_to_canonical_form(corpus) -> None:
    records, _ = corpus
    canonical_rows = [r for r in records if r.source.startswith("scope_canonical")]
    assert canonical_rows
    for record in canonical_rows:
        embedded = record.prompt.split("---INPUT---\n", 1)[1]
        assert embedded != record.openui
        if record.source == "scope_canonical_document":
            program = validate(embedded)
            assert (program.serialized or embedded.strip()) == record.openui


def test_canonical_pairs_share_prompts_with_identity_twins(corpus) -> None:
    records, pairs = corpus
    assert pairs
    prompts = {
        r.prompt: r for r in records if r.source.startswith("scope_canonical")
    }
    for pair in pairs:
        assert pair.chosen != pair.rejected
        record = prompts.get(pair.prompt)
        assert record is not None
        assert record.openui == pair.chosen


def test_decanonicalize_variants_round_trip() -> None:
    canonical = validate(SOURCE).serialized or SOURCE
    variants = decanonicalize_variants(canonical)
    assert variants
    for name, variant in variants:
        assert variant != canonical
        program = validate(variant)
        assert (program.serialized or variant.strip()) == canonical, name


def test_scoped_repairs_are_fail_closed(corpus) -> None:
    records, _ = corpus
    repairs = [r for r in records if r.source.startswith("scope_repair")]
    assert repairs
    for record in repairs:
        broken = record.meta["repair"]["broken"]
        kind = record.target_kind
        category = record.target_category
        validate_output(record.openui, kind, category)  # clean side is valid
        with pytest.raises(ParseError):
            validate_output(broken, kind, category)  # broken side must fail


def test_build_scoped_corruptions_rejects_invalid_clean_fragment() -> None:
    with pytest.raises(ParseError):
        build_scoped_corruptions('TextContent(":a"', "expression")


def test_scoped_corruption_boolean_typo() -> None:
    cases = build_scoped_corruptions("true", "lexical", category="boolean")
    assert cases
    assert cases[0].broken_text != "true"
    with pytest.raises(ParseError):
        validate_output(cases[0].broken_text, "lexical", "boolean")


def test_typed_rows_render_from_ast_terminals(corpus) -> None:
    records, _ = corpus
    typed = [r for r in records if r.source == "lexical_typed_map"]
    assert typed
    rendered = {r.meta["lexical_map"]["surface"]: r.openui for r in typed}
    assert rendered["true"] == "Boolean(true)"
    for record in typed:
        assert record.target_kind == "typed_node"
        validate_output(record.openui, "typed_node")


def test_scope_rows_inherit_root_lineage(corpus) -> None:
    records, _ = corpus
    for record in records:
        assert record.meta["split_group_id"] == "p1"
        assert record.meta["parent_id"] == "p1"
        assert record.meta["program_family_id"] == "fam:p1"
        assert record.meta["determinacy"] == "deterministic"
