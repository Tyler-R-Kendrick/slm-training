"""Training-data harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.language_contract import contract_id
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
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
                    "hero = Card([hero_title, hero_body])"
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
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
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
    assert result["manifest"]["synthesis_telemetry_sha256"]
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["record_count"] >= 2
    assert stats["error_count"] == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "train_data"
    assert "prompt_fingerprints" in manifest
    assert "openui_fingerprints" in manifest
    assert len(manifest["synthesis_telemetry_sha256"]) == 64
    assert len(manifest["ids"]) == stats["record_count"]
    assert {
        row.meta.get("contract_id") for row in load_jsonl(out_dir / "records.jsonl")
    } == {contract_id()}


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_prompt_contracts_expose_component_counts_and_slots(tmp_path: Path) -> None:
    baseline = build_train_data(
        TrainDataConfig(
            seed_path=_seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="baseline",
            synthesizer="none",
        )
    )
    result = build_train_data(
        TrainDataConfig(
            seed_path=_seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="contracts",
            synthesizer="none",
            prompt_component_contract=True,
            prompt_slot_contract=True,
        )
    )
    rows = {row.id: row for row in load_jsonl(Path(result["output_dir"]) / "records.jsonl")}
    assert "Components: Card x1, Stack x1, TextContent x2" in rows["t1"].prompt
    assert "Placeholders: :slot_0, :slot_1" in rows["t1"].prompt
    assert result["stats"]["prompt_component_contract"] is True
    assert result["stats"]["prompt_slot_contract"] is True
    assert result["manifest"]["ids"] == baseline["manifest"]["ids"]


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_component_contract_can_expose_types_without_counts(tmp_path: Path) -> None:
    result = build_train_data(
        TrainDataConfig(
            seed_path=_seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="types",
            synthesizer="none",
            prompt_component_contract=True,
            prompt_component_contract_mode="types",
        )
    )
    rows = {row.id: row for row in load_jsonl(Path(result["output_dir"]) / "records.jsonl")}
    assert "Components: Card, Stack, TextContent" in rows["t1"].prompt
    assert " x" not in rows["t1"].prompt
    assert result["stats"]["prompt_component_contract_mode"] == "types"


def test_semantic_role_contract_is_prohibited(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="template markers are opaque"):
        TrainDataConfig(
            seed_path=_seed_file(tmp_path),
            prompt_semantic_role_contract=True,
        )


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_build_train_data_derives_from_existing_records(tmp_path: Path) -> None:
    roots = _seed_file(tmp_path)
    result = build_train_data(
        TrainDataConfig(
            source="existing",
            derive_from=roots,
            output_root=tmp_path / "train_data",
            version="derived",
            synthesizer="template",
            include_frontier_artifacts=False,
            include_edit_derivatives=False,
            repairs_per_program=0,
        )
    )
    rows = load_jsonl(Path(result["output_dir"]) / "records.jsonl")
    assert result["stats"]["source"] == "existing"
    assert result["stats"]["derive_from"] == str(roots)
    derived = [row for row in rows if not row.source.startswith("language_contract")]
    assert {row.meta.get("derivation_source") for row in derived} == {str(roots)}
    assert any(row.target_kind == "lexical" for row in rows)
    assert any(row.meta.get("synth") == "template" for row in rows)
    assert len(rows) > 2


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_documentizes_expressions_and_records_target_selection(tmp_path: Path) -> None:
    result = build_train_data(
        TrainDataConfig(
            source="language_contract",
            output_root=tmp_path / "train_data",
            version="documents",
            synthesizer="none",
            include_frontier_artifacts=False,
            include_scope_corpus=False,
            documentize_expressions=True,
            target_kinds=("document",),
            require_design_md=False,
            test_seed_path=None,
        )
    )
    out_dir = Path(result["output_dir"])
    rows = load_jsonl(out_dir / "records.jsonl")
    assert rows
    assert {row.target_kind for row in rows} == {"document"}
    projected = [row for row in rows if row.meta.get("target_projection")]
    assert projected
    assert all(row.openui.startswith("root = ") for row in projected)
    rejected = [
        json.loads(line)
        for line in (out_dir / "rejected.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = [row for row in rejected if row["stage"] == "selection"]
    assert selected
    assert {row["detail"]["target_kind"] for row in selected} <= {
        "lexical",
        "statement",
        "typed_node",
    }
    feedback = json.loads(
        (out_dir / "synthesis_feedback.json").read_text(encoding="utf-8")
    )
    assert all(
        item["evidence"].get("top_reason") != "target_kind_excluded"
        for item in feedback["recommendations"]
    )


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_immutable_build_refuses_to_overwrite_snapshot(tmp_path: Path) -> None:
    config = TrainDataConfig(
        seed_path=_seed_file(tmp_path),
        rico_path=None,
        source="fixture",
        output_root=tmp_path / "train_data",
        version="immutable",
        synthesizer="none",
        immutable=True,
    )
    build_train_data(config)
    with pytest.raises(FileExistsError, match="immutable training-data snapshot"):
        build_train_data(config)


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_build_train_data_rejects_invalid_openui(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    write_jsonl(
        path,
        [
            # Structurally broken source — content-policy literals are no
            # longer "invalid" (sanitization templatizes them; see
            # test_sanitize.py), but a syntax error can never be rescued.
            ExampleRecord(
                id="bad1",
                prompt="Bad",
                openui="root = Stack([cta])\ncta = Button(",
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
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_build_train_data_from_rico_fixtures(tmp_path: Path) -> None:
    rico = Path("src/slm_training/resources/rico/semantic_train.jsonl")
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
    # Opaque-slot canonicalization collapses semantic-name-only duplicates.
    assert result["stats"]["record_count"] >= 3
    assert result["stats"]["error_count"] == 0
    assert result["manifest"]["source"] == "rico"


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_all_sources_are_tiered_and_rebuild_stably(tmp_path: Path) -> None:
    config = TrainDataConfig(
        seed_path=Path("src/slm_training/resources/train_seeds.jsonl"),
        rico_path=Path("src/slm_training/resources/rico/semantic_train.jsonl"),
        rico_limit=1,
        source="all",
        output_root=tmp_path / "integrated",
        version="v12",
        synthesizer="none",
        programspec_count=1,
        require_design_md=False,
        test_seed_path=None,
    )
    first = build_train_data(config)
    second = build_train_data(config)

    families = set(first["manifest"]["source_families"]["families"])
    assert {
        "programspec_generated",
        "language_contract",
        "corruption_repair",
        "edit_trajectory",
        "frontier_described",
        "abstraction_ladder",
        "renderer_visual",
        "web_distilled",
    } <= families
    rows = load_jsonl(Path(first["output_dir"]) / "records.jsonl")
    assert rows
    assert all(row.meta.get("verification_tier") for row in rows)
    assert all(row.meta["verification_tier"] != "Quarantine" for row in rows)
    assert first["manifest"]["diffusion_online"]["policies"]
    assert Path(first["manifest"]["mixture"]).is_file()
    assert all(
        Path(path).is_file() for path in first["manifest"]["governance"].values()
    )
    assert (
        first["manifest"]["content_fingerprint"]
        == second["manifest"]["content_fingerprint"]
    )
