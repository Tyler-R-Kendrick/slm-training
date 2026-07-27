"""Tests for slm_training.harnesses.experiments.slm269_dca0_activation (SLM-269)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm269_dca0_activation import (
    build_activation_report,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_real_artifacts_block_activation() -> None:
    """Against the currently-committed artifacts: SLM-260's replay is
    fixture-only, SLM-261's loss ledger is missing, and SLM-268 closed
    blocked with no selected regime and no negative result — so DCA0-01
    cannot launch honestly.
    """
    report = build_activation_report(REPO_ROOT)

    assert report["activation_status"] == "blocked"
    assert report["activation_verdict"] == "prerequisite_evidence_unqualified"
    assert report["disposition"] == "inconclusive"
    assert all(g["qualified"] is False for g in report["gates"])
    reasons = {g["issue"]: g["reason"] for g in report["gates"]}
    assert reasons["SLM-260"] == "fixture_only_not_full_scorer"
    assert reasons["SLM-268"] == "blocked_no_selected_regime_no_negative_result"
    assert all(not arm["available"] for arm in report["corpus_arms"].values())


def test_missing_artifact_is_recorded_not_hidden(tmp_path: Path) -> None:
    report = build_activation_report(tmp_path)

    assert all(g["sha256"] == "missing" for g in report["gates"])
    assert report["activation_status"] == "blocked"


def test_report_is_self_hashed_and_no_spend() -> None:
    report = build_activation_report(REPO_ROOT)

    assert report["manifest_sha256"]
    assert report["claim_class"] == "diagnostic"
    assert "No train" in report["honesty"]
    assert report["version_stamp"]["stamp_schema"] == "version_stamp/v1"


def test_render_markdown_lists_gates_and_blockers() -> None:
    markdown = render_markdown(build_activation_report(REPO_ROOT))

    assert "SLM-269" in markdown
    assert "blocked" in markdown
    assert "SLM-268" in markdown
    assert "remains open" in markdown
