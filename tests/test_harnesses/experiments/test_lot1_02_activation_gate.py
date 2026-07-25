"""Tests for slm_training.harnesses.experiments.lot1_02_activation_gate (SLM-251)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.lot1_01_activation_gate import (
    AUTHORIZED_WIRING_ONLY,
    NOT_AUTHORIZED,
    REQUIRED_FIDELITY_VERDICT,
    REQUIRED_TRACE_VERDICT,
    load_upstream_contract,
)
from slm_training.harnesses.experiments.lot1_02_activation_gate import (
    evaluate_lot1_02_launch_gate,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIDELITY_CONTRACT_PATH = REPO_ROOT / "docs/design/lotus-openui-fidelity-contract-v1.json"
TRACE_GATE_CONTRACT_PATH = REPO_ROOT / "docs/design/compiler-reasoning-trace-v1.json"


def test_real_upstream_contracts_are_not_authorized() -> None:
    """The currently-committed SLM-248/SLM-249 artifacts must yield
    not_authorized: SLM-248 reports needs_target_trace_contract, SLM-249
    reports inconclusive, and its allowed_lot1_implementation authorizes no
    LOT1 implementation. This is the load-bearing proof that LOT1-02
    correctly closes without any training or curriculum code.
    """
    contract = evaluate_lot1_02_launch_gate(
        load_upstream_contract(FIDELITY_CONTRACT_PATH),
        load_upstream_contract(TRACE_GATE_CONTRACT_PATH),
    )

    assert contract.verdict == NOT_AUTHORIZED
    assert contract.lot1_01_disposition == NOT_AUTHORIZED
    assert contract.gate_1_transfer_authorization.met is False
    assert contract.gate_2_trace_oracle_ceiling.met is False
    assert contract.gate_3_explicit_kc_stage_authorization.met is False
    assert "none" in contract.gate_3_explicit_kc_stage_authorization.actual_verdict


def test_all_prerequisites_met_authorizes_wiring_only() -> None:
    """The evaluator is not hardcoded to always fail: synthetic contracts
    with both required verdicts and an explicit LOT1 implementation
    allowance flip the disposition, but only to wiring-only authorization.
    """
    fidelity_contract = {
        "contract_id": "synthetic-fidelity",
        "contract_hash": "deadbeef",
        "authorization": {
            "linear_issue": "SLM-248",
            "verdict": REQUIRED_FIDELITY_VERDICT,
            "verdict_rationale": "synthetic positive",
        },
    }
    trace_gate_contract = {
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

    contract = evaluate_lot1_02_launch_gate(fidelity_contract, trace_gate_contract)

    assert contract.verdict == AUTHORIZED_WIRING_ONLY
    assert contract.lot1_01_disposition == AUTHORIZED_WIRING_ONLY
    assert contract.gate_3_explicit_kc_stage_authorization.met is True


def test_explicit_allowance_alone_does_not_authorize() -> None:
    """An explicit K/c allowance cannot substitute for the oracle-ceiling
    verdict: gate 2 unmet keeps the campaign not_authorized even when
    SLM-249 names an allowed implementation.
    """
    fidelity_contract = {
        "contract_id": "synthetic-fidelity",
        "contract_hash": "deadbeef",
        "authorization": {
            "linear_issue": "SLM-248",
            "verdict": REQUIRED_FIDELITY_VERDICT,
            "verdict_rationale": "synthetic positive",
        },
    }
    trace_gate_contract = {
        "contract_id": "synthetic-trace",
        "contract_hash": "cafef00d",
        "gate": {
            "linear_issue": "SLM-249",
            "verdict": "inconclusive",
            "verdict_rationale": "ceiling not measured",
            "allowed_lot1_implementation": [
                "kxc_bounded_implementation: K=6, c=479, stages per contract"
            ],
        },
    }

    contract = evaluate_lot1_02_launch_gate(fidelity_contract, trace_gate_contract)

    assert contract.verdict == NOT_AUTHORIZED
    assert contract.gate_3_explicit_kc_stage_authorization.met is True


def test_render_markdown_covers_all_gates() -> None:
    contract = evaluate_lot1_02_launch_gate(
        load_upstream_contract(FIDELITY_CONTRACT_PATH),
        load_upstream_contract(TRACE_GATE_CONTRACT_PATH),
    )
    markdown = render_markdown(contract)
    assert "SLM-251" in markdown
    assert "not_authorized" in markdown
    assert "Gate 3" in markdown
