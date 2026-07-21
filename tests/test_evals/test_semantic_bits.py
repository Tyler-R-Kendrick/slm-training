"""E1 bits-per-semantic-decision tests."""

from __future__ import annotations

import math

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.semantic_bits import (
    SemanticBitsConfig,
    categorize,
    compare_representations,
    compiler_state_conditional_bits,
    semantic_bits,
    semantic_bits_per_success,
    verified_utility_per_neural_evaluation,
    verified_utility_per_non_singleton_decision,
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


# --- B1/B3: choice stream ----------------------------------------------------


def test_choice_stream_has_fewer_decisions_than_production() -> None:
    production = semantic_bits(_records(), stream="production")
    choice = semantic_bits(_records(), stream="choice")
    # The choice stream elides exactly the grammar-forced framing tokens
    # (one '=' per statement here), never a semantic decision.
    assert 0 < choice["n_decisions"] < production["n_decisions"]
    assert choice["total_bits"] < production["total_bits"]
    assert choice["by_category"].get("production", 0) == production["by_category"].get(
        "production", 0
    )
    # No structural framing left in the choice stream for these documents.
    assert choice["by_category"].get("structural", 0) < production["by_category"].get(
        "structural", 1
    )


def test_compare_includes_choice_arm() -> None:
    cmp = compare_representations(_records())
    assert cmp["choice"]["total_bits"] > 0
    assert cmp["surface_to_choice_bit_ratio"] >= cmp["surface_to_production_bit_ratio"]


def test_compiler_state_conditional_bits_groups_by_signature() -> None:
    report = compiler_state_conditional_bits(_records())
    assert report["n_programs"] == 2
    assert report["n_groups"] > 0
    assert report["total_bits"] >= 0.0
    assert report["non_singleton_bits"] >= 0.0
    assert report["singleton_bits"] >= 0.0
    assert report["total_bits"] == pytest.approx(
        report["non_singleton_bits"] + report["singleton_bits"], rel=1e-9
    )
    assert "by_group" in report


def test_compiler_state_conditional_bits_bits_per_non_singleton() -> None:
    report = compiler_state_conditional_bits(_records())
    if report["bits_per_non_singleton_decision"] is not None:
        assert report["bits_per_non_singleton_decision"] >= 0.0


def test_semantic_bits_per_success_safe_ratio() -> None:
    result = semantic_bits_per_success(100.0, 5)
    assert result["ratio"] == pytest.approx(20.0)
    zero = semantic_bits_per_success(100.0, 0)
    assert zero["ratio"] is None


def test_verified_utility_per_neural_evaluation() -> None:
    result = verified_utility_per_neural_evaluation(0.8, 4)
    assert result["ratio"] == pytest.approx(0.2)
    zero = verified_utility_per_neural_evaluation(0.8, 0)
    assert zero["ratio"] is None


def test_verified_utility_per_non_singleton_decision() -> None:
    result = verified_utility_per_non_singleton_decision(0.9, 3)
    assert result["ratio"] == pytest.approx(0.3)
    zero = verified_utility_per_non_singleton_decision(0.9, 0)
    assert zero["ratio"] is None
