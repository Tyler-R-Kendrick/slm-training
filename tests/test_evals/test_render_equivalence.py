"""Tests for slm_training.evals.render_equivalence."""

from __future__ import annotations

import pytest

from slm_training.evals.render_equivalence import render_equivalence


_BASE = (
    'root = Stack([title, cta])\n'
    'title = TextContent(":card.title")\n'
    'cta = Button(":card.action")\n'
)


@pytest.mark.parametrize("pred,gold", [
    (_BASE, _BASE),
    (
        'root = Stack([t, c])\nt = TextContent(":card.title")\nc = Button(":card.action")\n',
        _BASE,
    ),
    (
        'root = Stack([title, cta])\n'
        'title = TextContent(":card.title")\n'
        'cta = Button(":card.action", "primary")\n',
        _BASE,
    ),
])
def test_golden_equivalent_pairs(pred: str, gold: str) -> None:
    report = render_equivalence(pred, gold)
    assert report.equivalent is True
    assert report.tier0.canonical_exact is True
    assert report.tier0.binding_graph_equal is True
    assert report.tier1.component_type_match == 1.0
    assert report.tier1.normalized_render_tree_distance == 1.0
    assert report.tier2.status == "not_available"


@pytest.mark.parametrize("pred,gold", [
    (
        'root = Stack([title, cta])\n'
        'title = TextContent(":card.title")\n'
        'cta = TextContent(":card.action")\n',
        _BASE,
    ),
    (
        'root = Stack([cta, title])\n'
        'title = TextContent(":card.title")\n'
        'cta = Button(":card.action")\n',
        _BASE,
    ),
    (
        'root = Stack([Card([title]), cta])\n'
        'title = TextContent(":card.title")\n'
        'cta = Button(":card.action")\n',
        _BASE,
    ),
])
def test_non_equivalent_pairs(pred: str, gold: str) -> None:
    report = render_equivalence(pred, gold)
    assert report.equivalent is False


def test_binding_corruption_prevents_equivalence() -> None:
    gold = _BASE
    pred = (
        'root = Stack([title, cta])\n'
        'title = TextContent(":card.title")\n'
        'cta = Button(":card.action")\n'
        'dead = TextContent(":dead.text")\n'
    )
    report = render_equivalence(pred, gold)
    assert report.equivalent is False
    assert "binding_graph_mismatch" in report.reason_codes or "canonical_mismatch" in report.reason_codes


def test_missing_renderer_returns_not_available() -> None:
    report = render_equivalence(_BASE, _BASE)
    assert report.tier2.status == "not_available"
    assert report.tier2.visual_similarity is None


def test_compiler_failure_yields_false() -> None:
    report = render_equivalence("root = not_a_valid_component()", _BASE)
    assert report.equivalent is False
    assert "parser_failure" in report.reason_codes


def test_report_round_trip() -> None:
    report = render_equivalence(_BASE, _BASE)
    data = report.to_dict()
    assert data["equivalent"] is True
    assert data["tier2"]["status"] == "not_available"
    assert data["version_stamp"]["stamp_schema"] == "version_stamp/v1"
