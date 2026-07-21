from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_judge_resolution_audit import main


def test_describe_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--mode", "describe"]) == 0
    captured = capsys.readouterr()
    assert "SLM-185 judge resolution audit schema" in captured.out
    assert "SemanticResolutionManifestV1" in captured.out


def test_build_corpus_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "corpus"
    assert main(["--mode", "build-corpus", "--output-dir", str(output_dir), "--repeats", "3"]) == 0

    corpus_path = output_dir / "judge_resolution_corpus.jsonl"
    summary_path = output_dir / "judge_resolution_corpus_summary.json"
    assert corpus_path.is_file()
    assert summary_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema"] == "JudgeResolutionCorpusSummaryV1"
    assert summary["item_n"] == 15
    assert summary["repeats"] == 3
    assert "version_stamp" in summary

    lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    first = json.loads(lines[0])
    assert "item_id" in first
    assert "expected_class" in first


def test_run_mode_with_corpus(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    corpus_dir = tmp_path / "corpus"
    main(["--mode", "build-corpus", "--output-dir", str(corpus_dir), "--repeats", "3"])

    assert (
        main(
            [
                "--mode",
                "run",
                "--output-dir",
                str(output_dir),
                "--corpus",
                str(corpus_dir / "judge_resolution_corpus.jsonl"),
            ]
        )
        == 0
    )

    report_path = output_dir / "judge_resolution_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "SemanticResolutionManifestV1"
    assert report["status"] == "fixture"
    assert report["claim_class"] == "wiring"
    assert len(report["endpoints"]) == 3
    assert "version_stamp" in report
    assert report["provenance"]["item_n"] == 15


def test_run_mode_writes_design_docs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    design_json = tmp_path / "design.json"
    design_md = tmp_path / "design.md"
    assert (
        main(
            [
                "--mode",
                "run",
                "--output-dir",
                str(output_dir),
                "--write-design-docs",
                "--design-json",
                str(design_json),
                "--design-md",
                str(design_md),
            ]
        )
        == 0
    )
    assert design_json.is_file()
    assert design_md.is_file()
    assert "SLM-185" in design_md.read_text(encoding="utf-8")


def test_analyze_history_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    main(["--mode", "run", "--output-dir", str(output_dir), "--repeats", "3"])
    report_path = output_dir / "judge_resolution_report.json"

    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "suites": {
                    "smoke": {"fixture_binding_aware_v2": 0.8},
                    "held_out": {"fixture_binding_aware_v2": 0.02},
                }
            }
        ),
        encoding="utf-8",
    )

    analysis_dir = tmp_path / "analysis"
    assert (
        main(
            [
                "--mode",
                "analyze-history",
                "--output-dir",
                str(analysis_dir),
                "--history",
                str(history_path),
                "--manifest",
                str(report_path),
            ]
        )
        == 0
    )

    reclass_path = analysis_dir / "judge_resolution_history_reclassification.json"
    assert reclass_path.is_file()
    payload = json.loads(reclass_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "JudgeResolutionHistoryReclassificationV1"
    assert "semantic_resolution" in payload
    assert (
        payload["semantic_resolution"]["fixture_binding_aware_v2"]["deltas"]["smoke"]
        == "directional"
    )


def test_analyze_history_without_manifest(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "suites": {
                    "smoke": {"fixture_binding_aware_v2": 0.001},
                }
            }
        ),
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "analysis"
    assert (
        main(
            [
                "--mode",
                "analyze-history",
                "--output-dir",
                str(analysis_dir),
                "--history",
                str(history_path),
            ]
        )
        == 0
    )
    reclass_path = analysis_dir / "judge_resolution_history_reclassification.json"
    assert reclass_path.is_file()
    payload = json.loads(reclass_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "JudgeResolutionHistoryReclassificationV1"
    assert payload["manifest_source"] == "default_fixture"


def test_analyze_history_requires_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--mode", "analyze-history", "--output-dir", str(tmp_path)]) != 0
    assert "--history" in capsys.readouterr().err
