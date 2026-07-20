"""Tests for scripts/run_slm148_x22_conflict_campaign.py (SLM-148)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_slm148_x22_conflict_campaign


def _repo_root() -> Path:
    return Path(run_slm148_x22_conflict_campaign.__file__).resolve().parents[1]


def test_plan_only_cli_writes_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm148_x22_conflict_campaign.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    assert (out_dir / "slm148_x22_conflict_campaign_report.json").exists()
    report = json.loads(
        (out_dir / "slm148_x22_conflict_campaign_report.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "Slm148X22ConflictCampaignReportV1"
    assert report["status"] == "plan_only"
    assert report["claim_class"] == "wiring"
    assert "manifest" in report
    assert "version_stamp" in report


def test_fixture_cli_writes_json_and_design_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm148_x22_conflict_campaign.main(
        ["--mode", "fixture", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    report_path = out_dir / "slm148_x22_conflict_campaign_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "Slm148X22ConflictCampaignReportV1"
    assert report["status"] == "fixture"
    assert "rows" in report
    assert "manifest" in report
    ids = {row["arm_id"] for row in report["rows"]}
    assert "S0_minimal" in ids
    assert "S5_gold_plan_oracle" in ids

    design_json = _repo_root() / "docs/design/iter-slm148-x22-conflict-campaign-20260720.json"
    design_md = _repo_root() / "docs/design/iter-slm148-x22-conflict-campaign-20260720.md"
    assert design_json.exists()
    assert design_md.exists()


def test_fixture_cli_prints_path(capsys, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm148_x22_conflict_campaign.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir)]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert str(out_dir / "slm148_x22_conflict_campaign_report.json") in captured.out


def test_plan_only_default_out_path_uses_date() -> None:
    rc = run_slm148_x22_conflict_campaign.main(["--mode", "plan-only"])
    assert rc == 0
