"""End-to-end tests for scripts/sync_abstract_planning_linear.py (AP-038 / SLM-339)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.sync_abstract_planning_linear import main
from slm_training.harnesses.autoresearch.linear_sync import (
    COMMENT_CREATE_MUTATION,
    COMMENTS_QUERY,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "planning_result"
    / "ap037_fixture.json"
)


def test_dry_run_prints_plan_and_makes_no_network(tmp_path: Path, capsys) -> None:
    assert main(["--result", str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "Issue key: AP-037" in out
    assert "ap-sync:sha256=" in out
    assert "Planned comment:" in out


def test_offline_out_writes_exact_graphql(tmp_path: Path, capsys) -> None:
    out_file = tmp_path / "payloads.json"
    assert main(["--result", str(FIXTURE), "--offline-out", str(out_file)]) == 0
    text = out_file.read_text(encoding="utf-8")
    assert COMMENTS_QUERY.split("{")[0].strip() in text
    assert COMMENT_CREATE_MUTATION.split("{")[0].strip() in text
    assert "ap-sync:sha256=" in text
    report_line = capsys.readouterr().out
    assert '"action": "create"' in report_line


def test_live_without_api_key_is_clean_error(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    assert main(["--result", str(FIXTURE), "--live"]) == 2
    err = capsys.readouterr().err
    assert "LINEAR_API_KEY" in err
    assert "lin_api_" not in err


def test_result_without_ap_key_errors(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["campaign_id"] = "no-key-here"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    try:
        main(["--result", str(bad)])
    except Exception as exc:
        assert "AP-###" in str(exc) or "no AP" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected an error for a keyless campaign_id")
