from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_legal_edit_bridges import build, main
from slm_training.data.flow.bridge_corpus import load_corpus

FIXTURE = Path("tests/fixtures/slm196_legal_edit_bridge")


def test_describe_build_validate_and_stats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--describe"]) == 0
    assert "UNKNOWN" in capsys.readouterr().out
    output = tmp_path / "fixture"
    assert (
        main(
            [
                "--fixture",
                "--records",
                str(FIXTURE / "records.jsonl"),
                "--planner-manifest",
                str(FIXTURE / "planner_manifest.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert main(["--validate", "--output", str(output)]) == 0
    assert main(["--stats", "--output", str(output)]) == 0
    rows, _, manifest = load_corpus(output)
    assert len(rows) == 4
    assert manifest["publishable"] is False


def test_production_rejects_dev_fixture_planner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen selected"):
        build(
            FIXTURE / "records.jsonl",
            FIXTURE / "planner_manifest.json",
            tmp_path / "production",
            fixture=False,
        )


def test_planner_floor_and_claim_binding_are_hard_gates(tmp_path: Path) -> None:
    planner = json.loads(
        (FIXTURE / "planner_manifest.json").read_text(encoding="utf-8")
    )
    planner["reachability_rate"] = 0.94
    bad = tmp_path / "planner.json"
    bad.write_text(json.dumps(planner), encoding="utf-8")
    with pytest.raises(ValueError, match="reachability"):
        build(
            FIXTURE / "records.jsonl",
            bad,
            tmp_path / "rejected",
            fixture=True,
        )
