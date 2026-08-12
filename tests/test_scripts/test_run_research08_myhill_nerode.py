"""CLI smoke for RESEARCH-08 runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_research08_myhill_nerode import main


def test_plan_only_exits_zero(capsys) -> None:  # noqa: ANN001
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_key"] == "RESEARCH-08"


def test_fixture_writes_docs(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    docs = tmp_path / "iter-revmath-research-08-preregistered.json"
    out = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--out-dir",
                str(out),
                "--docs-out",
                str(docs),
                "--cold-trials",
                "5",
            ]
        )
        == 0
    )
    assert docs.is_file()
    assert docs.with_suffix(".md").is_file()
    payload = json.loads(docs.read_text(encoding="utf-8"))
    assert payload["pilot"]["zero_semantic_disagreement"] is True
    summary = json.loads(capsys.readouterr().out)
    assert summary["decision"] in {"accept_research_only", "reject"}
