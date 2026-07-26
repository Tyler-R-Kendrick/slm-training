"""Tests for slm_training.harnesses.experiments.lot_downstream_gate (SLM-252+)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.lot1_01_activation_gate import (
    AUTHORIZED_WIRING_ONLY,
    NOT_AUTHORIZED,
    REQUIRED_FIDELITY_VERDICT,
    REQUIRED_TRACE_VERDICT,
    load_upstream_contract,
)
from slm_training.harnesses.experiments.lot_downstream_gate import (
    DOWNSTREAM_ISSUES,
    PENDING_CAMPAIGN_GATE,
    evaluate_downstream_gate,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIDELITY_CONTRACT_PATH = REPO_ROOT / "docs/design/lotus-openui-fidelity-contract-v1.json"
TRACE_GATE_CONTRACT_PATH = REPO_ROOT / "docs/design/compiler-reasoning-trace-v1.json"


def _real_contracts() -> tuple[dict, dict]:
    return (
        load_upstream_contract(FIDELITY_CONTRACT_PATH),
        load_upstream_contract(TRACE_GATE_CONTRACT_PATH),
    )


def _positive_contracts() -> tuple[dict, dict]:
    fidelity = {
        "contract_id": "synthetic-fidelity",
        "contract_hash": "deadbeef",
        "authorization": {
            "linear_issue": "SLM-248",
            "verdict": REQUIRED_FIDELITY_VERDICT,
            "verdict_rationale": "synthetic positive",
        },
    }
    trace = {
        "contract_id": "synthetic-trace",
        "contract_hash": "cafef00d",
        "gate": {
            "linear_issue": "SLM-249",
            "verdict": REQUIRED_TRACE_VERDICT,
            "verdict_rationale": "synthetic positive",
            "allowed_lot1_implementation": [
                "kxc_bounded_implementation: K=6, c=479, stages per contract"
            ],
        },
    }
    return fidelity, trace


def test_slm252_real_contracts_are_not_authorized() -> None:
    """The currently-committed SLM-248/SLM-249 artifacts must yield
    not_authorized for LOT2-01: LOT1-02's launch gate is unmet, so no
    parent checkpoint, treatment recipe, control, or curriculum exists.
    """
    fidelity, trace = _real_contracts()
    contract = evaluate_downstream_gate(DOWNSTREAM_ISSUES["SLM-252"], fidelity, trace)

    assert contract.verdict == NOT_AUTHORIZED
    assert contract.lot1_02_disposition == NOT_AUTHORIZED
    assert contract.lot1_01_disposition == NOT_AUTHORIZED
    assert contract.contract_id == "causal-supervision-routing-gate-v1"


def test_positive_upstream_is_pending_not_authorized() -> None:
    """Even with a synthetic positive LOT1 chain, a LOT2+ issue is only
    upstream_authorized_pending_campaign_gate — its own campaign gate must
    still be published by a real run. The evaluator never jumps straight
    to a launch/quality authorization.
    """
    fidelity, trace = _positive_contracts()
    contract = evaluate_downstream_gate(DOWNSTREAM_ISSUES["SLM-252"], fidelity, trace)

    assert contract.verdict == PENDING_CAMPAIGN_GATE
    assert contract.lot1_02_disposition == AUTHORIZED_WIRING_ONLY


def test_registry_entries_are_well_formed() -> None:
    for issue_id, spec in DOWNSTREAM_ISSUES.items():
        assert spec.linear_issue == issue_id
        assert spec.gate_contract_id
        assert spec.slug
        assert spec.activation_requirement


def test_render_markdown_covers_chain() -> None:
    fidelity, trace = _real_contracts()
    contract = evaluate_downstream_gate(DOWNSTREAM_ISSUES["SLM-252"], fidelity, trace)
    markdown = render_markdown(contract)
    assert "SLM-252" in markdown
    assert "not_authorized" in markdown
    assert "LOT1-02" in markdown
