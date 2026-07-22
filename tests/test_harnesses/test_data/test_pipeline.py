"""Testing-data harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.language_contract import output_contract_violations
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.data.leakage import load_train_fingerprints
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)


def test_build_test_data_suites(tmp_path: Path) -> None:
    seeds = tmp_path / "test_seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="smoke_1",
                prompt="Smoke",
                openui='root = Stack([cta])\ncta = Button(":cta.label")',
                placeholders=[":cta.label"],
                split="smoke",
                meta={"suite": "smoke"},
            ),
            ExampleRecord(
                id="held_1",
                prompt="Held",
                openui='root = Stack([blurb])\nblurb = TextContent(":page.blurb")',
                placeholders=[":page.blurb"],
                split="held_out",
                meta={"suite": "held_out"},
            ),
            ExampleRecord(
                id="adv_1",
                prompt="x",
                openui='root = Stack([fallback])\nfallback = TextContent(":fallback.text")',
                placeholders=[":fallback.text"],
                split="adversarial",
                meta={"suite": "adversarial"},
            ),
        ],
    )
    result = build_test_data(
        TestDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "test_data",
            version="vtest",
            suites=("smoke", "held_out", "adversarial"),
            train_manifest=None,
            require_train_manifest=False,
        )
    )
    out_dir = Path(result["output_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite_counts"]["smoke"] == 1
    assert manifest["version_stamp"]["components"] == {"data.test_build": "v9"}
    assert result["stats"]["version_stamp"] == manifest["version_stamp"]
    assert (out_dir / "suites" / "smoke" / "records.jsonl").exists()


def test_fixture_normalization_failure_is_fatal(tmp_path: Path) -> None:
    seeds = tmp_path / "bad_test_seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="bad_fixture",
                prompt="Broken fixture",
                openui="root = Broken(",
                split="smoke",
                source="fixture",
                meta={"suite": "smoke"},
            )
        ],
    )
    with pytest.raises(ValueError, match="fixture test record 'bad_fixture'"):
        build_test_data(
            TestDataConfig(
                seed_path=seeds,
                rico_path=None,
                source="fixture",
                output_root=tmp_path / "test_data",
                version="bad",
                suites=("smoke",),
                train_manifest=None,
                require_train_manifest=False,
            )
        )


def test_leakage_detection_by_id(tmp_path: Path) -> None:
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="shared_id",
                prompt="Train",
                openui='root = Stack([cta])\ncta = Button(":cta.label")',
                split="train",
            )
        ],
    )
    train_result = build_train_data(
        TrainDataConfig(
            seed_path=train_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="v0",
            synthesizer="none",
        )
    )
    train_manifest = Path(train_result["output_dir"]) / "manifest.json"

    test_seeds = tmp_path / "test.jsonl"
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="shared_id",
                prompt="Test different prompt",
                openui='root = Stack([blurb])\nblurb = TextContent(":page.blurb")',
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        build_test_data(
            TestDataConfig(
                seed_path=test_seeds,
                rico_path=None,
                source="fixture",
                output_root=tmp_path / "test_data",
                version="v0",
                suites=("smoke",),
                train_manifest=train_manifest,
            )
        )


def test_leakage_detection_by_openui(tmp_path: Path) -> None:
    openui = 'root = Stack([cta])\ncta = Button(":cta.label")'
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="train_1",
                prompt="Train prompt",
                openui=openui,
                split="train",
            )
        ],
    )
    train_result = build_train_data(
        TrainDataConfig(
            seed_path=train_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="v0",
            synthesizer="none",
        )
    )
    train_manifest = Path(train_result["output_dir"]) / "manifest.json"

    test_seeds = tmp_path / "test.jsonl"
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="test_1",
                prompt="Completely different prompt text",
                openui=openui,
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )
    with pytest.raises(ValueError, match="openui"):
        build_test_data(
            TestDataConfig(
                seed_path=test_seeds,
                rico_path=None,
                source="fixture",
                output_root=tmp_path / "test_data",
                version="v0",
                suites=("smoke",),
                train_manifest=train_manifest,
            )
        )


def test_rico_train_and_test_are_disjoint(tmp_path: Path) -> None:
    train_rico = Path("src/slm_training/resources/rico/semantic_train.jsonl")
    test_rico = Path("src/slm_training/resources/rico/semantic_test.jsonl")
    if not train_rico.exists() or not test_rico.exists():
        pytest.skip("RICO fixtures missing")

    train_result = build_train_data(
        TrainDataConfig(
            seed_path=None,
            rico_path=train_rico,
            source="rico",
            output_root=tmp_path / "train_data",
            version="v0",
            synthesizer="none",
            rico_limit=20,
        )
    )
    test_result = build_test_data(
        TestDataConfig(
            seed_path=None,
            rico_path=test_rico,
            source="rico",
            output_root=tmp_path / "test_data",
            version="v0",
            suites=("smoke", "held_out"),
            train_manifest=Path(train_result["output_dir"]) / "manifest.json",
            rico_limit=15,
        )
    )
    assert test_result["stats"]["total_records"] >= 5
    train_ids = set(train_result["manifest"]["ids"])
    test_ids = set(test_result["manifest"]["ids"])
    assert train_ids.isdisjoint(test_ids)


def test_test_builder_sanitizes_gold_with_shared_transform(tmp_path: Path) -> None:
    from slm_training.harnesses.train_data.sanitize import (
        SanitizeOptions,
        sanitize_openui,
    )

    seeds = tmp_path / "sanitize_seeds.jsonl"
    gold = (
        'root = Stack([hero], "column")\n'
        "hero = Card([hdr])\n"
        'hdr = CardHeader(":hero.title")'
    )
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="sanitize_smoke",
                prompt="Hero card.",
                openui=gold,
                placeholders=[":hero.title"],
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )
    result = build_test_data(
        TestDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "eval",
            version="vsan",
            suites=("smoke",),
            train_manifest=None,
            require_train_manifest=False,
        )
    )
    assert result["stats"]["sanitize_mode"] == "enforce"
    assert result["stats"]["sanitize"]["sanitized"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "eval" / "vsan" / "suites" / "smoke" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = sanitize_openui(
        gold, options=SanitizeOptions(mode="enforce")
    ).openui.replace(":hero.title", ":slot_0")
    assert records[0]["openui"] == expected
    assert '"column"' not in records[0]["openui"]

    off = build_test_data(
        TestDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "eval_off",
            version="voff",
            suites=("smoke",),
            train_manifest=None,
            require_train_manifest=False,
            sanitize_mode="off",
        )
    )
    off_records = [
        json.loads(line)
        for line in (
            tmp_path / "eval_off" / "voff" / "suites" / "smoke" / "records.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert '"column"' in off_records[0]["openui"]
    assert off["stats"]["sanitize"] == {"mode": "off"}


def test_test_builder_templatizes_or_rejects_free_form_targets(tmp_path: Path) -> None:
    seeds = tmp_path / "free_form_seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="free_form_form",
                prompt="Contact form.",
                openui=(
                    'root = Form("contact", actions, [])\n'
                    'actions = Buttons([Button(":form.submit")])'
                ),
                split="held_out",
                meta={"suite": "held_out"},
            )
        ],
    )
    result = build_test_data(
        TestDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "eval",
            version="enforced",
            suites=("held_out",),
            train_manifest=None,
            require_train_manifest=False,
        )
    )
    record = json.loads(
        (
            Path(result["output_dir"])
            / "suites"
            / "held_out"
            / "records.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert output_contract_violations(record["openui"]) == ()
    assert record["meta"]["sanitize"]["template_fills"] == {":slot_0": "contact"}

    rejected_root = tmp_path / "eval_off"
    with pytest.raises(ValueError, match="symbol-only output contract"):
        build_test_data(
            TestDataConfig(
                seed_path=seeds,
                rico_path=None,
                source="fixture",
                output_root=rejected_root,
                version="off",
                suites=("held_out",),
                train_manifest=None,
                require_train_manifest=False,
                sanitize_mode="off",
            )
        )
    assert not rejected_root.exists()


def test_train_manifest_records_resolve_from_owning_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "owner"
    dataset = checkout / "outputs" / "data" / "train" / "fixture"
    dataset.mkdir(parents=True)
    write_jsonl(
        dataset / "records.jsonl",
        [
            ExampleRecord(
                id="train_record",
                prompt="Button.",
                openui='root = Button(":button.label")',
                split="train",
            )
        ],
    )
    manifest = dataset / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": "outputs/data/train/fixture/records.jsonl",
                "ids": [],
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_train_fingerprints(manifest)["ids"] == {"train_record"}
