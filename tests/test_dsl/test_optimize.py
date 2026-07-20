"""Schema-checked AST optimizer tests (dsl.analysis.optimize)."""

from __future__ import annotations

import pytest

from slm_training.dsl import validate
from slm_training.dsl.canonicalize import canonical_equal, canonicalize
from slm_training.dsl.lang_core import library_schema
from slm_training.dsl.analysis.optimize import (
    SCHEMA_DEFAULTS,
    OptimizeOptions,
    optimize,
    semantic_fingerprint,
)

HERO = 'root = Stack([hero], "column")\nhero = Card([t])\nt = TextContent(":x.t")'
HERO_RENAMED = (
    'root = Stack([box], "column")\nbox = Card([label])\nlabel = TextContent(":x.t")'
)


def test_trailing_default_direction_elided_row_kept() -> None:
    elided = optimize('root = Stack([t], "column")\nt = TextContent(":a.b")')
    assert '"column"' not in elided.source
    assert elided.rewrites["defaults_elided"] == 1

    kept = optimize('root = Stack([t], "row")\nt = TextContent(":a.b")')
    assert '"row"' in kept.source
    assert kept.rewrites["defaults_elided"] == 0


def test_form_empty_fields_default_elided() -> None:
    src = (
        'root = Stack([f])\n'
        'f = Form(":form.name", [b], [])\n'
        'b = Button(":form.submit")'
    )
    result = optimize(src)
    assert ", []" not in result.source
    assert result.rewrites["defaults_elided"] >= 1
    validate(result.source)


def test_dead_binding_removed_and_fingerprint_stable() -> None:
    src = HERO + '\ndead = TextContent(":unused.x")'
    result = optimize(src, options=OptimizeOptions(flatten_single_child=False))
    assert ":unused.x" not in result.source
    assert result.rewrites["dead_bindings_removed"] == 1
    assert semantic_fingerprint(src) == semantic_fingerprint(result.source)
    validate(result.source)


def test_flatten_single_child_stack_and_redirect() -> None:
    src = (
        'root = Stack([card, wrap], "column")\n'
        "card = Card([hdr])\n"
        'hdr = CardHeader(":w.title")\n'
        "wrap = Stack([note])\n"
        'note = TextContent(":w.note")'
    )
    result = optimize(src)
    assert result.rewrites["containers_flattened"] == 1
    assert result.flatten_opportunities == 1
    # Wrapper gone: exactly one Stack (the root) remains.
    assert result.source.count("Stack(") == 1
    validate(result.source)


def test_flatten_respects_schema_child_admissibility() -> None:
    # Card.children admits TextContent but not Modal — the wrapper protecting
    # the Modal must survive, the one around TextContent must not.
    blocked = (
        "root = Card([w])\n"
        "w = Stack([m])\n"
        'm = Modal(":m.title", true, [t])\n'
        't = TextContent(":m.body")'
    )
    result = optimize(blocked)
    assert result.rewrites["containers_flattened"] == 0
    assert result.flatten_opportunities == 1

    allowed = 'root = Card([w])\nw = Stack([t])\nt = TextContent(":a.b")'
    fixed = optimize(allowed)
    assert fixed.rewrites["containers_flattened"] == 1
    assert "Stack(" not in fixed.source
    validate(fixed.source)


def test_flatten_never_touches_root_top_node() -> None:
    src = 'root = Stack([t])\nt = TextContent(":a.b")'
    result = optimize(src)
    assert result.rewrites["containers_flattened"] == 0
    assert result.source.startswith("root = Stack(")


def test_flatten_skips_protected_components() -> None:
    src = (
        'root = Card([w])\nw = Stack([t])\nt = TextContent(":a.b")'
    )
    result = optimize(
        src, options=OptimizeOptions(protected_components=frozenset({"Stack"}))
    )
    assert result.rewrites["containers_flattened"] == 0
    assert result.flatten_opportunities == 1


def test_optimize_is_idempotent_and_canonical() -> None:
    src = (
        'root = Stack([card, wrap], "column")\n'
        "card = Card([hdr])\n"
        'hdr = CardHeader(":w.title")\n'
        "wrap = Stack([note])\n"
        'note = TextContent(":w.note")\n'
        'dead = TextContent(":unused.x")'
    )
    once = optimize(src)
    again = optimize(once.source)
    assert again.source == once.source
    assert not again.changed
    assert once.source == canonicalize(once.source)


def test_pure_rename_is_canonical_equal_to_input() -> None:
    result = optimize(HERO_RENAMED, options=OptimizeOptions(flatten_single_child=False))
    # The D2 half (rename/reorder) is certified by canonical_equal; the
    # default-elision half is not (D2 deliberately excludes it).
    assert canonical_equal(HERO, HERO_RENAMED)
    assert semantic_fingerprint(result.source) == semantic_fingerprint(HERO)


def test_alpha_equivalent_inputs_optimize_to_identical_bytes() -> None:
    assert optimize(HERO).source == optimize(HERO_RENAMED).source


def test_schema_defaults_are_pinned_to_the_schema() -> None:
    defs = library_schema()["$defs"]
    for (component, prop), default in SCHEMA_DEFAULTS.items():
        spec = defs[component]["properties"][prop]
        enum = spec.get("enum")
        if enum is not None:
            assert default in enum, (component, prop, default)
        if "default" in spec:
            assert spec["default"] == default, (component, prop)
        else:
            # Prose-documented defaults must still be stated in the schema
            # description so curation stays evidence-backed.
            assert f'default "{default}"' in defs[component].get("description", ""), (
                component,
                prop,
            )


def test_validate_false_reads_policy_failing_sources() -> None:
    lit = 'root = Stack([t], "column")\nt = TextContent("Welcome back")'
    result = optimize(lit, validate=False)
    assert result.rewrites["defaults_elided"] == 1
    assert '"Welcome back"' in result.source


def test_unparseable_source_raises() -> None:
    with pytest.raises(Exception):
        optimize('root = Card(["unclosed')
