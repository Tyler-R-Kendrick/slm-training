"""Source-family catalog, lineage, and exposure-cap tests."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.harnesses.train_data.catalog import (
    apply_parent_cap,
    classify_source_family,
    family_stats,
    resolve_lineage,
)
from slm_training.harnesses.train_data.synth import NoopSynthesizer, get_synthesizer

pytestmark_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":slot_0")'


def _record(rid: str, source: str = "fixture", **meta) -> ExampleRecord:
    return ExampleRecord(
        id=rid,
        prompt=f"prompt {rid}",
        openui=CTA,
        placeholders=[":slot_0"],
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
    with pytest.raises(ValueError, match="namespace_augment is prohibited"):
        classify_source_family(
            _record(
                "a_syn_0_ns",
                "fixture+template+namespace",
                synth="namespace_augment",
                parent_id="a_syn_0",
            )
        )


def test_namespace_synthesizer_is_not_available() -> None:
    with pytest.raises(ValueError, match="unknown synthesizer"):
        get_synthesizer("namespace_augment")


def test_resolve_lineage_walks_to_root() -> None:
    index = {
        "a": (None, None),
        "a_syn_0": ("a", "template"),
        "a_syn_0_layout": ("a_syn_0", "layout_augment"),
    }
    root, lineage = resolve_lineage("a_syn_0_layout", index)
    assert root == "a"
    assert lineage == ["template", "layout_augment"]
    root, lineage = resolve_lineage("a", index)
    assert root == "a"
    assert lineage == []


def test_apply_parent_cap_prefers_root_and_is_deterministic() -> None:
    records = [
        ExampleRecord(
            id=rid,
            prompt="p z",
            openui=CTA,
            split="train",
            meta={"root_parent_id": "z", "source_family": "prompt_paraphrase"},
        )
        for rid in ("b_syn_2", "a_syn_0", "c_syn_1")
    ] + [
        ExampleRecord(
            id="z",
            prompt="p z",
            openui=CTA,
            split="train",
            meta={
                "root_parent_id": "z",
                "parent_id": "z",
                "source_family": "human_curated",
            },
        )
    ]
    kept, dropped = apply_parent_cap(records, 2)
    kept_ids = sorted(r.id for r in kept)
    assert "z" in kept_ids  # self-parent root beats child whose ID sorts first
    assert len(kept_ids) == 2
    assert kept_ids == ["a_syn_0", "z"]  # then sorted-id order
    assert {d["id"] for d in dropped} == {"b_syn_2", "c_syn_1"}
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
                placeholders=[":slot_0", ":slot_1"],
                split="train",
            ),
            ExampleRecord(
                id="t2",
                prompt="Button only",
                openui=CTA,
                placeholders=[":slot_0"],
                split="train",
            ),
        ],
    )
    config = TrainDataConfig(
        seed_path=seeds,
        rico_path=None,
        source="fixture",
        output_root=tmp_path / "out",
        version="vfam",
        synthesizer="quality",
        namespace_augment=True,
    )
    assert config.namespace_augment is True


@pytestmark_bridge
def test_pipeline_parent_cap(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.jsonl"
    contaminated = ExampleRecord(
        id="heldout_copy",
        prompt="An experimental archive catalog for rare mineral samples",
        openui=CTA,
        placeholders=[":slot_0"],
        split="train",
    )
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="t1",
                prompt="Hero card",
                openui=HERO,
                placeholders=[":slot_0", ":slot_1"],
                split="train",
            ),
            contaminated,
        ],
    )
    eval_root = tmp_path / "eval"
    heldout = ExampleRecord(
        **{**contaminated.__dict__, "id": "heldout", "openui": canonicalize(CTA)}
    )
    assert canonicalize(contaminated.openui) == heldout.openui
    write_jsonl(
        eval_root / "v1/suites/controlled/records.jsonl",
        [heldout],
    )
    quality = get_synthesizer("quality")

    class RootQualityOnly:
        def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
            return quality.expand(record) if record.id == "t1" else []

    uncapped = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="vuncapped",
            synthesizer="quality",
            decontam_eval_root=eval_root,
            test_seed_path=None,
        ),
        synthesizer=RootQualityOnly(),
    )
    capped = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="vcapped",
            synthesizer="quality",
            max_records_per_parent=3,
            decontam_eval_root=eval_root,
            test_seed_path=None,
        ),
        synthesizer=RootQualityOnly(),
    )
    assert uncapped["stats"]["record_count"] > 3
    assert capped["stats"]["record_count"] == 3
    assert capped["stats"]["parent_cap_dropped"] > 0
    exposure = capped["manifest"]["source_families"]["records_per_root_parent"]
    assert exposure["max"] <= 3
    # The original seed record survives the cap.
    ids = capped["manifest"]["ids"]
    assert "t1" in ids
    for result in (uncapped, capped):
        rejected = [
            json.loads(line)
            for line in (Path(result["output_dir"]) / "rejected.jsonl")
            .read_text()
            .splitlines()
        ]
        assert any(
            row["id"] == "heldout_copy"
            and row["stage"] == "decontamination"
            and row["reason"] == "ngram_overlap"
            for row in rejected
        )
        assert "heldout_copy" not in result["manifest"]["ids"]

    duplicate_seeds = tmp_path / "duplicate-seeds.jsonl"
    write_jsonl(
        duplicate_seeds,
        [
            ExampleRecord(
                id="a_syn_0",
                prompt="Same content",
                openui=HERO,
                placeholders=[":slot_0", ":slot_1"],
                split="train",
                source="fixture+template",
                meta={"parent_id": "z", "synth": "template"},
            ),
            ExampleRecord(
                id="z",
                prompt="Same content",
                openui=HERO,
                placeholders=[":slot_0", ":slot_1"],
                split="train",
                meta={"parent_id": "z"},
            ),
        ],
    )
    exact_dedup = build_train_data(
        TrainDataConfig(
            seed_path=duplicate_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="vselfparent",
            synthesizer="none",
            max_records_per_parent=1,
            decontam_eval_root=eval_root,
            test_seed_path=None,
        ),
        synthesizer=NoopSynthesizer(),
    )
    assert exact_dedup["manifest"]["ids"] == ["z"]


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
