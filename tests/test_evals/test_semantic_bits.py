"""E1 bits-per-semantic-decision tests."""

from __future__ import annotations

import math

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.semantic_bits import (
    SemanticBitsConfig,
    categorize,
    compare_representations,
    semantic_bits,
)

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(id="h", prompt="Hero", openui=HERO, placeholders=[":hero.title", ":hero.body"]),
        ExampleRecord(id="c", prompt="CTA", openui=CTA, placeholders=[":cta.label"]),
    ]


def test_production_stream_reports_bits_and_categories() -> None:
    report = semantic_bits(_records(), stream="production")
    assert report["n_programs"] == 2
    assert report["n_scored_programs"] == 2
    assert report["n_decisions"] > 0
    assert report["bits_per_decision"] is not None
    assert report["bits_per_decision"] >= 0.0
    # Category breakdown covers the decisions and includes real productions.
    assert sum(report["by_category"].values()) == report["n_decisions"]
    assert report["by_category"].get("production", 0) > 0


def test_bits_per_decision_is_entropy_bounded_by_alphabet() -> None:
    report = semantic_bits(_records(), stream="production")
    # Shannon entropy is bounded above by log2(alphabet size).
    assert report["bits_per_decision"] <= math.log2(report["alphabet_size"]) + 1e-9


def test_params_per_bit_scales_with_params() -> None:
    a = semantic_bits(_records(), stream="production", params=100_000)
    b = semantic_bits(_records(), stream="production", params=200_000)
    assert a["params_per_bit"] > 0
    assert math.isclose(b["params_per_bit"], 2 * a["params_per_bit"], rel_tol=1e-9)


def test_excluding_slots_reduces_decisions() -> None:
    withslots = semantic_bits(_records(), stream="production")
    without = semantic_bits(
        _records(), stream="production", config=SemanticBitsConfig(include_slots=False)
    )
    assert without["n_decisions"] < withslots["n_decisions"]


def test_compare_shows_externalization_ratio() -> None:
    cmp = compare_representations(_records())
    assert cmp["production"]["total_bits"] >= 0.0
    assert cmp["surface"]["total_bits"] >= 0.0
    # Surface stream carries at least as many decisions as the production stream
    # (it still includes structural symbols the codec externalizes).
    assert cmp["decision_reduction_ratio"] is None or cmp["decision_reduction_ratio"] >= 1.0


def test_empty_and_deterministic() -> None:
    assert semantic_bits([], stream="production")["n_decisions"] == 0
    a = semantic_bits(_records(), stream="production")
    b = semantic_bits(_records(), stream="production")
    assert a["total_bits"] == b["total_bits"]


def test_categorize_known_prefixes() -> None:
    assert categorize("+Stack") == "production"
    assert categorize("@0") == "slot"
    assert categorize("&2") == "reference"
    assert categorize("$@state") == "state_ref"
