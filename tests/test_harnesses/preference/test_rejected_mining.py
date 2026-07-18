"""Rejected-ledger → preference-pair mining tests."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.preference import load_pairs
from slm_training.harnesses.preference.rejected_mining import (
    mine_rejected_pairs,
    pairs_fingerprint,
)


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "train" / "vmine"
    dataset.mkdir(parents=True)
    admitted = ExampleRecord(
        id="hero_best",
        prompt="Hero card with title and body.",
        openui=(
            'root = Stack([hero], "column")\n'
            'hero_title = TextContent(":hero.title")\n'
            "hero = Card([hero_title])"
        ),
        placeholders=[":hero.title"],
        split="train",
        meta={
            "parent_id": "hero_root",
            "curation_score": 0.95,
            "quality": {"score": 0.97},
        },
        design_md="# design",
    )
    write_jsonl(dataset / "records.jsonl", [admitted])
    entries = [
        {
            "id": "hero_wide",
            "stage": "quality",
            "reason": "quality_gate_failed",
            "detail": {"score": 0.3, "reasons": ["too_many_components"]},
            "record": {
                "id": "hero_wide",
                "prompt": "Hero card with far too many tiles.",
                "openui": 'root = Stack([a, b])\na = TextContent(":a")\nb = TextContent(":b")',
                "meta": {"parent_id": "hero_root"},
            },
        },
        # Orphan root: no admitted twin → skipped.
        {
            "id": "orphan",
            "stage": "quality",
            "reason": "quality_gate_failed",
            "detail": {"score": 0.1},
            "record": {
                "id": "orphan",
                "prompt": "Orphan",
                "openui": "root = Stack([z])",
                "meta": {"parent_id": "nowhere"},
            },
        },
        # Dedup drops are id-only → never minable.
        {"id": "dup", "stage": "dedup", "reason": "exact_pair_duplicate"},
    ]
    (dataset / "rejected.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    return dataset


def test_mines_pairs_against_best_admitted_twin(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    pairs = mine_rejected_pairs(dataset)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.prompt == "Hero card with far too many tiles."
    assert ":hero.title" in pair.chosen
    assert pair.rejected.startswith("root = Stack([a, b])")
    assert pair.chosen_score == 0.95
    assert pair.rejected_score == 0.3
    meta = pair.meta or {}
    assert meta["pair_corpus"] == "rejected_ledger"
    assert meta["chosen_id"] == "hero_best"
    assert meta["rejected_id"] == "hero_wide"
    assert len(pairs_fingerprint(pairs)) == 64


def test_cli_writes_versioned_preference_dataset(tmp_path: Path) -> None:
    from scripts.mine_rejected_preferences import main

    dataset = _dataset(tmp_path)
    out_root = tmp_path / "preference"
    lineage_root = tmp_path / "lineage"
    assert (
        main(
            [
                "--dataset",
                str(dataset),
                "--version",
                "vmine_pairs",
                "--out-root",
                str(out_root),
                "--lineage-root",
                str(lineage_root),
            ]
        )
        == 0
    )
    out_dir = out_root / "vmine_pairs"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "preference"
    assert manifest["record_count"] == 1
    assert manifest["content_fingerprint"]
    assert manifest["artifacts"]  # common-store envelope applied
    assert len(load_pairs(out_dir / "pairs.jsonl")) == 1
    assert list((lineage_root / "data_snapshots").glob("preference-vmine_pairs-*.json"))


def test_missing_ledger_yields_no_pairs(tmp_path: Path) -> None:
    empty = tmp_path / "train" / "vempty"
    empty.mkdir(parents=True)
    assert mine_rejected_pairs(empty) == []
