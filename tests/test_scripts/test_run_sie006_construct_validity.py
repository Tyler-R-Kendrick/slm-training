"""CLI smoke for scripts.run_sie006_construct_validity."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_sie006_construct_validity import main


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["catalogue_id"] == "exp-sr-3"
    assert payload["promotion"] is False


def test_fixture_writes_docs(tmp_path: Path) -> None:
    docs_out = tmp_path / "iter-slm483-sie-006.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(out_dir),
                "--docs-out",
                str(docs_out),
            ]
        )
        == 0
    )
    payload = json.loads(docs_out.read_text(encoding="utf-8"))
    assert payload["status"] == "fixture"
    assert payload["external_blocked"] is False
    assert payload["training_excluded"] is True
    assert payload["evaluator_human_agreement_kappa"] is not None
    assert docs_out.with_suffix(".md").is_file()


def test_external_blocked_writes_docs(tmp_path: Path) -> None:
    docs_out = tmp_path / "iter-slm483-sie-006-blocked.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "external-blocked",
                "--output-dir",
                str(out_dir),
                "--docs-out",
                str(docs_out),
            ]
        )
        == 0
    )
    payload = json.loads(docs_out.read_text(encoding="utf-8"))
    assert payload["status"] == "external_blocked"
    assert payload["external_blocked"] is True
    assert payload["evaluator_human_agreement_kappa"] is None
    assert payload["blocked_claims"]
