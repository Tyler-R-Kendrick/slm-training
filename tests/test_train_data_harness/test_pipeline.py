"""Training-data harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data


def _seed_file(tmp_path: Path) -> Path:
    path = tmp_path / "seeds.jsonl"
    write_jsonl(
        path,
        [
            ExampleRecord(
                id="t1",
                prompt="Hero card",
                openui=(
                    'root = Stack([hero], "column")\n'
                    'hero_title = TextContent(":hero.title")\n'
                    'hero_body = TextContent(":hero.body")\n'
                    'hero = Card([hero_title, hero_body])'
                ),
                placeholders=[":hero.title", ":hero.body"],
                split="train",
            ),
            ExampleRecord(
                id="t2",
                prompt="Button only",
                openui='root = Stack([cta])\ncta = Button(":cta.label")',
                placeholders=[":cta.label"],
                split="train",
            ),
        ],
    )
    return path


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)
def test_build_train_data_writes_artifacts(tmp_path: Path) -> None:
    seeds = _seed_file(tmp_path)
    out_root = tmp_path / "train_data"
    result = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=out_root,
            version="vtest",
            synthesizer="template",
        )
    )
    out_dir = Path(result["output_dir"])
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "records.jsonl").exists()
    assert (out_dir / "stats.json").exists()
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["record_count"] >= 2
    assert stats["error_count"] == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "train_data"
    assert "prompt_fingerprints" in manifest
    assert "openui_fingerprints" in manifest
    assert len(manifest["ids"]) == stats["record_count"]


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)
def test_build_train_data_rejects_invalid_openui(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    write_jsonl(
        path,
        [
            ExampleRecord(
                id="bad1",
                prompt="Bad",
                openui='root = Stack([cta])\ncta = Button("nope")',
                split="train",
            )
        ],
    )
    result = build_train_data(
        TrainDataConfig(
            seed_path=path,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="vbad",
            synthesizer="none",
        )
    )
    assert result["stats"]["record_count"] == 0
    assert result["stats"]["error_count"] >= 1


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)
def test_build_train_data_from_rico_fixtures(tmp_path: Path) -> None:
    rico = Path("fixtures/rico/semantic_train.jsonl")
    if not rico.exists():
        pytest.skip("RICO fixtures missing")
    result = build_train_data(
        TrainDataConfig(
            seed_path=None,
            rico_path=rico,
            source="rico",
            output_root=tmp_path / "train_data",
            version="vrico",
            synthesizer="none",
            rico_limit=10,
        )
    )
    assert result["stats"]["record_count"] >= 5
    assert result["stats"]["error_count"] == 0
    assert result["manifest"]["source"] == "rico"
