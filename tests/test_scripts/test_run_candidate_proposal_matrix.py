from __future__ import annotations

import json

import pytest

from scripts import run_candidate_proposal_matrix as cli


def test_describe(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--describe"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "CandidateProposalManifestV1"
    assert payload["k_grid"] == [1, 2, 4, 8, 16, "all"]


def test_eval_writes_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    monkeypatch.setattr(
        cli,
        "publish_agentv_evaluation",
        lambda *args, **kwargs: {"summary": {"passed": 5, "failed": 0}},
    )
    monkeypatch.setattr(cli, "_rewrite_agentv_paths", lambda: None)
    assert (
        cli.main(
            [
                "--eval",
                "--steps",
                "2",
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )
    report = json.loads(output_json.read_text())
    assert report["schema"] == "CandidateProposalManifestV1"
    assert report["requested_mode"] == "eval"
    assert "retain exact cached enumeration" in output_md.read_text().lower()


def test_confirm_refuses_without_development_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_candidate_proposal_matrix",
        lambda **kwargs: {"positive_claim_eligible": []},
    )
    with pytest.raises(SystemExit, match="confirmation refused"):
        cli.main(["--confirm"])
