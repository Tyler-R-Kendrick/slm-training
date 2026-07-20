"""Tests for the schema-derived action-description catalog."""

from __future__ import annotations

import pytest

from slm_training.dsl.action_descriptions import (
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
    centroid_distance,
    compute_nearest_neighbor_metrics,
    coverage_report,
    masked_mean_pool,
)


@pytest.fixture
def catalog() -> ActionDescriptionCatalog:
    return ActionDescriptionCatalog.build()


def test_catalog_has_component_entries(catalog: ActionDescriptionCatalog) -> None:
    assert "+Card" in catalog.by_key
    assert "+Stack" in catalog.by_key
    assert "+Button" in catalog.by_key


def test_catalog_has_structural_entries(catalog: ActionDescriptionCatalog) -> None:
    assert "-" in catalog.by_key
    assert "r=" in catalog.by_key
    assert "a=" in catalog.by_key
    assert "q=" in catalog.by_key
    assert "m=" in catalog.by_key
    assert "$=" in catalog.by_key


def test_catalog_has_builtin_entries(catalog: ActionDescriptionCatalog) -> None:
    assert "*Run" in catalog.by_key
    assert "*Set" in catalog.by_key


def test_component_entry_has_fields(catalog: ActionDescriptionCatalog) -> None:
    card = catalog.by_key["+Card"]
    assert card.short_name == "Card"
    assert "children" in card.signature
    assert card.result_type == "element"
    assert card.provenance == "schema"


def test_coverage_report_full(catalog: ActionDescriptionCatalog) -> None:
    desc = catalog.descriptions_for("schema_description")
    report = coverage_report(desc, catalog)
    assert report["coverage_fraction"] == pytest.approx(1.0)
    assert report["missing"] == []


def test_coverage_report_none(catalog: ActionDescriptionCatalog) -> None:
    desc = catalog.descriptions_for("none")
    report = coverage_report(desc, catalog)
    assert report["coverage_fraction"] == pytest.approx(0.0)
    assert report["missing"]


def test_stub_differs_from_schema(catalog: ActionDescriptionCatalog) -> None:
    stub = catalog.descriptions_for("current_stub")
    schema = catalog.descriptions_for("schema_description")
    assert stub["+Card"] != schema["+Card"]
    assert "UI component" in stub["+Card"]


def test_shuffled_differs_from_schema(catalog: ActionDescriptionCatalog) -> None:
    shuffled = catalog.descriptions_for("shuffled")
    schema = catalog.descriptions_for("schema_description")
    assert set(shuffled.keys()) == set(schema.keys())
    assert shuffled != schema


def test_expanded_description_loads(catalog: ActionDescriptionCatalog) -> None:
    expanded = catalog.descriptions_for("expanded_description")
    schema = catalog.descriptions_for("schema_description")
    # Expanded JSON overrides at least the listed components.
    for key in ("+Card", "+Stack", "+Button", "+Input", "+Select"):
        assert key in expanded
        assert expanded[key] != schema[key]
    # Missing actions fall back to schema descriptions.
    assert "*Run" in expanded
    assert expanded["*Run"] == schema["*Run"]


def test_fixture_encoder_deterministic() -> None:
    enc = FixtureDescriptionEncoder(32)
    v1 = enc.encode("hello")
    v2 = enc.encode("hello")
    assert v1.shape == (32,)
    assert (v1 == v2).all()


def test_fixture_encoder_different_inputs_differ() -> None:
    enc = FixtureDescriptionEncoder(32)
    v1 = enc.encode("hello")
    v2 = enc.encode("world")
    assert not (v1 == v2).all()


def test_masked_mean_pool() -> None:
    import torch

    emb = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [0.0, 0.0]],
            [[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = masked_mean_pool(emb, mask)
    assert pooled.shape == (2, 2)
    assert pooled[0].tolist() == pytest.approx([1.5, 0.5])
    assert pooled[1].tolist() == pytest.approx([0.0, 2.0])


def test_nearest_neighbor_metrics() -> None:
    import torch

    vectors = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.99, 0.01]),
        "c": torch.tensor([0.0, 1.0]),
    }
    metrics = compute_nearest_neighbor_metrics(vectors)
    assert 0.0 <= metrics["mean_nearest_cosine"] <= 1.0
    assert "a" in metrics["nearest_neighbor_map"]


def test_centroid_distance() -> None:
    import torch

    vectors = {
        "a1": torch.tensor([0.0, 0.0]),
        "a2": torch.tensor([0.0, 1.0]),
        "b1": torch.tensor([10.0, 0.0]),
        "b2": torch.tensor([10.0, 1.0]),
    }
    dist = centroid_distance(vectors, {"a1", "a2"}, {"b1", "b2"})
    assert dist == pytest.approx(10.0, abs=1e-4)
