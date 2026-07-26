"""Tests for slm_training.harnesses.experiments.dca_activation_gate (SLM-270+)."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.dca_activation_gate import (
    DCA_ISSUES,
    PENDING_CAMPAIGN,
    evaluate_dca_gate,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_qualified_artifacts(root: Path) -> None:
    design = root / "docs" / "design"
    design.mkdir(parents=True, exist_ok=True)
    (design / "training-loss-ledger.md").write_text("# LossLedgerV1\n")
    (design / "iter-slm268-verified-data-scaling-activation-20260725.json").write_text(
        json.dumps({"activation_status": "complete", "activation_verdict": "hybrid_wins"})
    )
    (design / "iter-slm269-dca0-01-activation-20260725.json").write_text(
        json.dumps({"activation_status": "complete", "disposition": "no_target_policy_effect"})
    )


def test_slm270_real_artifacts_are_blocked() -> None:
    """Against the currently-committed artifacts: no LossLedgerV1, SLM-268
    blocked, SLM-269 inconclusive — the objective-surgery campaign cannot
    launch honestly.
    """
    contract = evaluate_dca_gate(DCA_ISSUES["SLM-270"], REPO_ROOT)

    assert contract.verdict == "blocked"
    assert contract.disposition == "inconclusive"
    assert contract.prerequisite_verdicts["SLM-261_loss_ledger"] == "missing_or_unqualified"
    assert contract.prerequisite_verdicts["SLM-268_data_regime"] == "blocked"
    assert contract.prerequisite_verdicts["SLM-269_target_policy"] == "blocked_or_inconclusive"
    assert contract.successor_conditions


def test_qualified_artifacts_flip_to_pending_campaign(tmp_path: Path) -> None:
    """The gate is not hardcoded to fail: synthetic qualified artifacts
    flip it — but only to prerequisites_met_pending_campaign, never to a
    result or default-change authorization.
    """
    _write_qualified_artifacts(tmp_path)
    contract = evaluate_dca_gate(DCA_ISSUES["SLM-270"], tmp_path)

    assert contract.verdict == PENDING_CAMPAIGN
    assert contract.disposition == "pending"


def test_missing_artifacts_are_recorded(tmp_path: Path) -> None:
    contract = evaluate_dca_gate(DCA_ISSUES["SLM-270"], tmp_path)

    assert contract.verdict == "blocked"
    assert contract.prerequisite_verdicts["SLM-268_data_regime"] == "missing"


def test_registry_entries_are_well_formed() -> None:
    for issue_id, spec in DCA_ISSUES.items():
        assert spec.linear_issue == issue_id
        assert spec.gate_contract_id
        assert spec.slug
        assert spec.activation_requirement


def test_render_markdown_covers_legs() -> None:
    markdown = render_markdown(evaluate_dca_gate(DCA_ISSUES["SLM-270"], REPO_ROOT))

    assert "SLM-270" in markdown
    assert "blocked" in markdown
    assert "Successor conditions" in markdown


def test_slm271_real_artifacts_are_blocked() -> None:
    contract = evaluate_dca_gate(DCA_ISSUES["SLM-271"], REPO_ROOT)

    assert contract.verdict == "blocked"
    assert contract.contract_id == "mask-curriculum-activation-gate-v1"
