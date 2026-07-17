"""Source-family catalog, lineage, and exposure-cap tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.harnesses.train_data.catalog import (
    apply_parent_cap,
    classify_source_family,
    family_stats,
    resolve_lineage,
)
from slm_training.harnesses.train_data.synth import (
    ComponentPromptSynthesizer,
    SemanticSlotSynthesizer,
)

pytestmark_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _record(rid: str, source: str = "fixture", **meta) -> ExampleRecord:
    return ExampleRecord(
        id=rid,
        prompt=f"prompt {rid}",
        openui=CTA,
        placeholders=[":cta.label"],
        split="train",
        source=source,
        meta=meta,
    )


def test_classify_source_family() -> None:
    assert classify_source_family(_record("a", "fixture")) == "human_curated"
    assert classify_source_family(_record("a", "human")) == "human_feedback"
    assert classify_source_family(_record("a", "rico")) == "rico_real"
    assert classify_source_family(_record("a", "awwwards")) == "awwwards_real"
    assert classify_source_family(_record("a", "synth+stress")) == "stress_adversarial"
    assert (
        classify_source_family(
            _record("a_syn_0", "fixture+template", synth="template", parent_id="a")
        )
        == "prompt_paraphrase"
    )
    # Outermost transformation wins for stacked lineages.
    assert (
        classify_source_family(
            _record(
                "a_syn_0_ns",
                "fixture+template+namespace",
                synth="namespace_augment",
                parent_id="a_syn_0",
            )
        )
        == "namespace_augment"
    )
    assert (
        classify_source_family(
            _record(
                "a_component_prompt",
                "fixture+component_prompt",
                synth="component_prompt",
                parent_id="a",
            )
        )
        == "prompt_paraphrase"
    )


def test_component_prompt_synthesizer_describes_inventory_and_content() -> None:
    [derived] = ComponentPromptSynthesizer().expand(
        ExampleRecord(
            id="hero",
            prompt="Update hero content.",
            openui=HERO,
            placeholders=[":shop.hero_title", ":shop.hero_body"],
            split="train",
            meta={"task": "generation"},
        )
    )
    assert derived.prompt == (
        "Build an OpenUI layout with one Stack, 2 Text Content components, "
        "and one Card. Include content slots for hero title and hero body."
    )
    assert derived.openui == HERO
    assert derived.meta["component_inventory"] == {
        "Stack": 1,
        "TextContent": 2,
        "Card": 1,
    }
    assert derived.meta["content_concepts"] == ["hero title", "hero body"]


def test_semantic_slot_synthesizer_renames_generation_slots_by_schema_role() -> None:
    record = ExampleRecord(
        id="form",
        prompt="Build a form for :cov.placeholder and :cov.label.",
        openui=(
            'root = Stack([field, submit])\n'
            'field = Input("email", ":cov.placeholder")\n'
            'submit = Button(":cov.label")'
        ),
        placeholders=[":cov.placeholder", ":cov.label"],
        meta={"task": "generation"},
    )

    [variant] = SemanticSlotSynthesizer().expand(record)

    assert variant.id == "form_semantic_slots"
    assert all(slot not in variant.openui for slot in record.placeholders)
    assert variant.placeholders[0].startswith(":input.")
    assert variant.placeholders[0].rsplit(".", 1)[-1] in {
        "email",
        "name",
        "search",
        "query",
        "value",
    }
    assert variant.placeholders[1].startswith(":button.")
    assert variant.meta["slot_role_map"][":cov.placeholder"] == variant.placeholders[0]


def test_semantic_slot_synthesizer_skips_non_generation_tasks() -> None:
    record = ExampleRecord(
        id="repair",
        prompt="Repair it.",
        openui='root = TextContent(":copy.text")',
        placeholders=[":copy.text"],
        meta={"task": "repair"},
    )

    assert SemanticSlotSynthesizer().expand(record) == []


def test_resolve_lineage_walks_to_root() -> None:
    index = {
        "a": (None, None),
        "a_syn_0": ("a", "template"),
        "a_syn_0_ns": ("a_syn_0", "namespace_augment"),
    }
    root, lineage = resolve_lineage("a_syn_0_ns", index)
    assert root == "a"
    assert lineage == ["template", "namespace_augment"]
    root, lineage = resolve_lineage("a", index)
    assert root == "a"
    assert lineage == []


def test_apply_parent_cap_prefers_root_and_is_deterministic() -> None:
    records = [
        ExampleRecord(
            id=rid,
            prompt=f"p {rid}",
            openui=CTA,
            split="train",
            meta={"root_parent_id": "a", "source_family": "prompt_paraphrase"},
        )
        for rid in ("a_syn_2", "a_syn_0", "a_syn_1")
    ] + [
        ExampleRecord(
            id="a",
            prompt="p a",
            openui=CTA,
            split="train",
            meta={"root_parent_id": "a", "source_family": "human_curated"},
        )
    ]
    kept, dropped = apply_parent_cap(records, 2)
    kept_ids = sorted(r.id for r in kept)
    assert "a" in kept_ids  # root always kept first
    assert len(kept_ids) == 2
    assert kept_ids == ["a", "a_syn_0"]  # then sorted-id order
    assert {d["id"] for d in dropped} == {"a_syn_1", "a_syn_2"}
    # Uncapped passthrough.
    kept_all, dropped_none = apply_parent_cap(records, None)
    assert len(kept_all) == 4 and dropped_none == []


def test_family_stats_counts_parents() -> None:
    records = [
        ExampleRecord(
            id=rid,
            prompt="p",
            openui=CTA,
            split="train",
            meta={"root_parent_id": root, "source_family": family},
        )
        for rid, root, family in (
            ("a", "a", "human_curated"),
            ("a_syn_0", "a", "prompt_paraphrase"),
            ("a_syn_1", "a", "prompt_paraphrase"),
            ("b", "b", "human_curated"),
        )
    ]
    stats = family_stats(records)
    assert stats["total_records"] == 4
    assert stats["unique_root_parents"] == 2
    fam = stats["families"]
    assert fam["human_curated"]["unique_records"] == 2
    assert fam["prompt_paraphrase"]["unique_records"] == 2
    assert fam["prompt_paraphrase"]["unique_root_parents"] == 1
    assert fam["prompt_paraphrase"]["records_per_root_parent"]["max"] == 2
    assert stats["records_per_root_parent"]["max"] == 3
    assert fam["human_curated"]["target_tokens"] > 0


@pytestmark_bridge
def test_pipeline_manifest_source_families(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="t1",
                prompt="Hero card",
                openui=HERO,
                placeholders=[":hero.title", ":hero.body"],
                split="train",
            ),
            ExampleRecord(
                id="t2",
                prompt="Button only",
                openui=CTA,
                placeholders=[":cta.label"],
                split="train",
            ),
        ],
    )
    result = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="vfam",
            synthesizer="quality",
            namespace_augment=True,
        )
    )
    manifest = result["manifest"]
    families = manifest["source_families"]["families"]
    assert "human_curated" in families
    assert "prompt_paraphrase" in families
    assert "namespace_augment" in families
    # Every record carries lineage metadata pointing back to a seed.
    records = [
        json.loads(line)
        for line in (Path(result["output_dir"]) / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in records:
        assert row["meta"]["source_family"]
        assert row["meta"]["root_parent_id"] in {"t1", "t2"}
        assert isinstance(row["meta"]["transformation_lineage"], list)
    # Namespace of a template variant records the full chain.
    stacked = [
        r
        for r in records
        if r["meta"]["transformation_lineage"] == ["template", "namespace_augment"]
    ]
    assert stacked


@pytestmark_bridge
def test_pipeline_parent_cap(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="t1",
                prompt="Hero card",
                openui=HERO,
                placeholders=[":hero.title", ":hero.body"],
                split="train",
            ),
        ],
    )
    uncapped = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="vuncapped",
            synthesizer="quality",
            namespace_augment=True,
        )
    )
    capped = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="vcapped",
            synthesizer="quality",
            namespace_augment=True,
            max_records_per_parent=3,
        )
    )
    assert uncapped["stats"]["record_count"] > 3
    assert capped["stats"]["record_count"] == 3
    assert capped["stats"]["parent_cap_dropped"] > 0
    exposure = capped["manifest"]["source_families"]["records_per_root_parent"]
    assert exposure["max"] <= 3
    # The original seed record survives the cap.
    ids = capped["manifest"]["ids"]
    assert "t1" in ids


def test_apply_parent_cap_per_family_groups_by_family() -> None:
    records = [
        ExampleRecord(
            id=f"{family}_{i}",
            prompt=f"p {family} {i}",
            openui=CTA,
            split="train",
            meta={"root_parent_id": "a", "source_family": family},
        )
        for family in ("scope_identity_lexical", "scope_repair_lexical")
        for i in range(3)
    ]
    # Cross-family cap keeps 2 rows for the whole parent...
    kept_global, _ = apply_parent_cap(records, 2)
    assert len(kept_global) == 2
    # ...per-family cap keeps 2 per (family, parent) without cross-eviction.
    kept_family, dropped = apply_parent_cap(records, 2, per_family=True)
    assert len(kept_family) == 4
    families = {r.meta["source_family"] for r in kept_family}
    assert families == {"scope_identity_lexical", "scope_repair_lexical"}
    assert all(d["reason"] == "max_records_per_parent" for d in dropped)
