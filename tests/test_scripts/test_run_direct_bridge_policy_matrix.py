from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_direct_bridge_policy_matrix as runner


def test_describe_lists_every_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main(["--describe"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["modes"] == [
        "describe",
        "train",
        "eval",
        "matrix",
        "resume",
        "confirm",
    ]


def test_confirm_fails_closed_without_publishable_corpus() -> None:
    with pytest.raises(SystemExit) as exc:
        runner.main(["--confirm"])
    assert exc.value.code == 2


def test_matrix_writes_evidence_with_agentv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "DESIGN_JSON", tmp_path / "design.json")
    monkeypatch.setattr(runner, "DESIGN_MD", tmp_path / "design.md")
    monkeypatch.setattr(runner, "DESIGN_AGENTV", tmp_path / "design-agentv")
    monkeypatch.setattr(
        runner,
        "publish_agentv_evaluation",
        lambda *args, **kwargs: {
            "format": "AgentEvals JSONL",
            "sdk": "@agentv/core",
            "spec": str(tmp_path / "spec.jsonl"),
            "summary": {"passed": 5, "total": 5, "failed": 0, "errors": 0},
        },
    )
    output = tmp_path / "run"
    assert (
        runner.main(
            [
                "--matrix",
                "--seeds",
                "0",
                "--steps",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["agentv"]["sdk"] == "@agentv/core"
    assert report["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert runner.DESIGN_JSON.is_file()
    assert runner.DESIGN_MD.is_file()
