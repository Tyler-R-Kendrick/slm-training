"""Tests for scripts/run_twotower_adapter_subspace_geometry.py (SLM-125 CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_twotower_adapter_subspace_geometry import main


def test_fixture_run_writes_report_with_authorization(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = main(
        [
            "--fixture",
            "--ranks",
            "2,4",
            "--target-modules",
            "attn_q,attn_v",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kind"] == "adapter_subspace_geometry"
    assert report["status"] == "completed"
    assert report["authorization"]["decision"] == "no_safe_direction"
    assert "result_content_sha" in report
    assert set(report["result"]) == {"rank2:attn_q+attn_v", "rank4:attn_q+attn_v"}


def test_fixture_stdout_when_no_out(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "--fixture",
            "--ranks",
            "2",
            "--target-modules",
            "attn_q",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    report = json.loads(captured.out)
    assert report["status"] == "completed"


def test_missing_checkpoint_or_events_without_fixture_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["--ranks", "2"])
    assert rc == 2
    assert "--checkpoint and --events are required" in capsys.readouterr().err
