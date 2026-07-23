from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.harnesses.train_data.operator_corpus import (
    OperatorCorpusConfig,
    build_symbolic_operator_corpus,
)

SOURCE = (
    'root = Card([TextContent(":hero.title"), '
    'TextContent(":hero.body")], "clear")'
)
STAMP = {
    "stamp_schema": "version_stamp/v1",
    "code_commit": "fixture",
    "dirty": False,
    "components": {"harness.train_data": "v12"},
    "timestamp": "2026-07-23T00:00:00+00:00",
}


def _record() -> ExampleRecord:
    return ExampleRecord(
        id="operator-root",
        prompt="fixture",
        openui=SOURCE,
        placeholders=[":hero.title", ":hero.body"],
        source="fixture",
        meta={
            "source_family": "operator_fixture",
            "program_family_id": "operator_fixture:card",
        },
    )


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing",
)
def test_symbolic_operator_corpus_is_closed_replayable_and_deterministic(
    tmp_path: Path,
) -> None:
    config = OperatorCorpusConfig(
        max_roots=1,
        actions_per_state=2,
        max_combinations_per_operator=32,
        sibling_forks=True,
    )
    first = build_symbolic_operator_corpus(
        records=[_record()],
        output_dir=tmp_path / "first",
        version="fixture-v1",
        version_stamp=STAMP,
        config=config,
    )
    second = build_symbolic_operator_corpus(
        records=[_record()],
        output_dir=tmp_path / "second",
        version="fixture-v1",
        version_stamp=STAMP,
        config=config,
    )
    assert first["content_fingerprint"] == second["content_fingerprint"]
    assert first["record_count"] == 10
    assert first["report"]["invalid_family_count"] == 0
    assert first["report"]["coverage_gaps"]
    assert first["report"]["version_stamp"] == STAMP
    assert first["report"]["application_coverage"]["legal_successes"] > 0
    assert first["report"]["application_coverage"]["rejected_combinations"] > 0
    assert first["report"]["application_coverage"]["emitted_illegal_targets"] == 0
    assert {item["phase"] for item in first["report"]["legal_sets"]} == {
        "single_turn",
        "next_turn",
    }
    assert any(
        gap["rejection_samples"] for gap in first["report"]["coverage_gaps"]
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "first" / "operator_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {
        row["target_view"]
        for row in rows
        if row["kind"] == "single_turn"
    } == {"operator_only", "result_ast_only", "dual"}
    assert {"single_turn", "next_turn", "sibling_fork"} == {
        row["kind"] for row in rows
    }
    assert all(
        set(row["question"])
        == {
            "opcode",
            "view",
            "state_ast",
            "legal_set_fingerprint",
            "trace_fingerprint",
        }
        for row in rows
    )
    successful = [row for row in rows if row["outcome"] == "success"]
    assert successful
    assert all(row["application"]["proof"] for row in successful)
    assert all(row["application"]["effect"] for row in successful)
    assert all(row["canonical_preference"]["steps"] for row in successful)
    assert all(row["conversation_trace"]["turns"] for row in rows)


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing",
)
def test_symbolic_operator_corpus_fails_closed_without_document_roots(
    tmp_path: Path,
) -> None:
    record = _record()
    record.target_kind = "expression"
    with pytest.raises(
        ValueError, match="requires an admitted document root"
    ):
        build_symbolic_operator_corpus(
            records=[record],
            output_dir=tmp_path,
            version="fixture-v1",
            version_stamp=STAMP,
            config=OperatorCorpusConfig(max_roots=1),
        )


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing",
)
def test_train_build_registers_operator_sibling_artifacts(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.jsonl"
    write_jsonl(seeds, [_record()])
    result = build_train_data(
        TrainDataConfig(
            profile="permissive",
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train",
            version="operator-v1",
            synthesizer="none",
            require_design_md=False,
            test_seed_path=None,
            include_language_contract=False,
            include_frontier_artifacts=False,
            include_design_md_contrastive=False,
            include_scope_corpus=False,
            governance_artifacts=False,
            mixture_manifest=False,
            emit_preference_pairs=False,
            diffusion_online=False,
            include_operator_corpus=True,
            operator_corpus_max_roots=1,
            operator_corpus_actions_per_state=1,
            operator_corpus_max_combinations=16,
        )
    )
    out_dir = Path(result["output_dir"])
    assert (out_dir / "operator_records.jsonl").is_file()
    assert (out_dir / "operator_coverage.json").is_file()
    assert result["manifest"]["operator_corpus"]["record_count"] >= 4
    assert result["stats"]["operator_corpus"]["content_fingerprint"]
    assert result["operator_corpus"]["report"]["invalid_family_count"] == 0
