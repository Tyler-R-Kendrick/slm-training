"""Quality-report + rejected.jsonl emission tests for train-data builds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)


def _seed_file(tmp_path: Path) -> Path:
    """Seeds that exercise the rejection stages deterministically."""
    good = ExampleRecord(
        id="good_hero",
        prompt="Hero card with a title and body.",
        openui=(
            'root = Stack([hero], "column")\n'
            'hero_title = TextContent(":hero.title")\n'
            'hero_body = TextContent(":hero.body")\n'
            "hero = Card([hero_title, hero_body])"
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
    )
    exact_twin = ExampleRecord(
        id="good_hero_twin",
        prompt=good.prompt,
        openui=good.openui,
        placeholders=list(good.placeholders),
        split="train",
    )
    garbage = ExampleRecord(
        id="garbage_literal",
        prompt="Show a welcome banner.",
        openui='root = Stack([msg])\nmsg = TextContent("Welcome!")',
        placeholders=[],
        split="train",
    )
    broken = ExampleRecord(
        id="broken_dsl",
        prompt="Completely broken program.",
        openui="root = Stack([",
        placeholders=[],
        split="train",
    )
    wide_items = [f"w{i}" for i in range(9)]
    wide = ExampleRecord(
        id="wide_grid",
        prompt="A wall of nine text tiles.",
        openui=(
            f'root = Stack([{", ".join(wide_items)}], "column")\n'
            + "\n".join(
                f'{name} = TextContent(":wide.item{i}")'
                for i, name in enumerate(wide_items)
            )
        ),
        placeholders=[f":wide.item{i}" for i in range(9)],
        split="train",
    )
    path = tmp_path / "seeds.jsonl"
    write_jsonl(path, [good, exact_twin, garbage, broken, wide])
    return path


def _build(tmp_path: Path, **overrides) -> dict:
    return build_train_data(
        TrainDataConfig(
            seed_path=_seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="vreport",
            synthesizer="template",
            # 10-component wide_grid seed survives normalization but trips the
            # quality gate's hard too_many_components reason.
            max_components=8,
            **overrides,
        )
    )


def test_build_emits_quality_report_and_rejected_ledger(tmp_path: Path) -> None:
    result = _build(tmp_path)
    out_dir = Path(result["output_dir"])
    report_path = out_dir / "quality_report.json"
    rejected_path = out_dir / "rejected.jsonl"
    assert report_path.is_file()
    assert rejected_path.is_file()

    stats = result["stats"]
    assert stats["profile"] == "strict"
    assert stats["fuzzy_dedup"] is True  # strict profile applied
    assert stats["semantic_dedup"] is True
    assert stats["ngram_decontam"] is True
    assert stats["semantic_dedup_engine"] in {"lexical-tfidf", "embeddings"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == result["quality_report"]
    assert report["schema_version"] == 1
    assert report["profile"] == "strict"

    # The ledger and the report agree, and nothing was dropped silently.
    rejected_rows = [
        json.loads(line)
        for line in rejected_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rejected_rows) == report["counts"]["rejected_total"] > 0
    by_stage = report["counts"]["by_stage"]
    assert sum(by_stage.values()) == len(rejected_rows)

    # The broken seed and the literal-content seed both fail normalization
    # (the contract layer enforces the placeholder rules before scoring) and
    # keep their payloads for repair mining.
    normalize_rows = [r for r in rejected_rows if r["stage"] == "normalize"]
    assert any(r["id"] == "broken_dsl" for r in normalize_rows)
    assert any(r["id"] == "garbage_literal" for r in normalize_rows)
    assert all("record" in r for r in normalize_rows)
    assert by_stage["normalize"] >= 2

    # The wide seed parses but fails the quality gate with a hard reason.
    quality_rows = [r for r in rejected_rows if r["stage"] == "quality"]
    assert any(r["id"] == "wide_grid" for r in quality_rows)
    wide_row = next(r for r in quality_rows if r["id"] == "wide_grid")
    assert "too_many_components" in wide_row["detail"]["reasons"]
    assert "record" in wide_row

    # The exact twin is recorded as a dedup drop, id-only.
    dedup_rows = [r for r in rejected_rows if r["stage"] == "dedup"]
    assert any(r["reason"] == "exact_pair_duplicate" for r in dedup_rows)
    assert all("record" not in r for r in dedup_rows)

    # Constraint-fitness metrics reflect the parse failure.
    fitness = report["constraint_fitness"]
    assert fitness["parse_failures"] >= 1
    assert fitness["parse_rate"] is not None and fitness["parse_rate"] < 1.0
    assert fitness["tier_histogram"]
    assert report["garbage"]["reason_histogram"]
    assert report["redundancy"]["dropped"]["exact_pair"] >= 1
    assert report["redundancy"]["top_clusters"]

    # Decontamination ran against the committed eval suites with no flags on
    # the fixture corpus, and the engines are recorded for reproducibility.
    assert report["decontamination"]["ngram_size"] == 8
    assert report["decontamination"]["suites_indexed"]
    assert report["engines"]["semantic_dedup"] in {"lexical-tfidf", "embeddings"}
    assert report["engines"]["decontam"] == "ngram-8"

    # Admitted corpus stayed healthy.
    assert report["counts"]["admitted"] == stats["record_count"] > 0

    # The manifest points at both artifacts.
    manifest = result["manifest"]
    assert manifest["profile"] == "strict"
    assert manifest["quality_report"].endswith("quality_report.json")
    assert manifest["rejected"].endswith("rejected.jsonl")


def test_dedup_against_excludes_pairs_already_in_other_corpora(tmp_path: Path) -> None:
    first = _build(tmp_path)
    second = build_train_data(
        TrainDataConfig(
            seed_path=tmp_path / "seeds.jsonl",
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="vderived",
            synthesizer="template",
            max_components=8,
            dedup_against=(first["output_dir"],),
        )
    )
    stats = second["stats"]
    assert stats["dedup_against"] == [first["output_dir"]]
    assert stats["cross_corpus_dropped"] > 0
    assert second["quality_report"]["redundancy"]["dropped"]["cross_corpus"] > 0
    # Everything admitted by the identical first build is excluded now.
    assert stats["record_count"] < first["stats"]["record_count"] or (
        stats["record_count"] == 0
    )


def test_permissive_profile_still_emits_report(tmp_path: Path) -> None:
    result = _build(tmp_path, profile="permissive")
    stats = result["stats"]
    assert stats["profile"] == "permissive"
    assert stats["fuzzy_dedup"] is False
    assert stats["semantic_dedup"] is False
    assert stats["ngram_decontam"] is False
    report = result["quality_report"]
    assert report["profile"] == "permissive"
    # Exact dedup + quality gate still run under permissive.
    assert report["counts"]["by_stage"].get("dedup", 0) >= 1
    assert report["counts"]["by_stage"].get("quality", 0) >= 1
