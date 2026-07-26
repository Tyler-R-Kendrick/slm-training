"""Tests for slm_training.harnesses.experiments.lot1_01_activation_gate (SLM-250)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.lot1_01_activation_gate import (
    AUTHORIZED_WIRING_ONLY,
    NOT_AUTHORIZED,
    REQUIRED_FIDELITY_VERDICT,
    REQUIRED_TRACE_VERDICT,
    evaluate_activation_gates,
    load_upstream_contract,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIDELITY_CONTRACT_PATH = REPO_ROOT / "docs/design/lotus-openui-fidelity-contract-v1.json"
TRACE_GATE_CONTRACT_PATH = REPO_ROOT / "docs/design/compiler-reasoning-trace-v1.json"


def test_real_upstream_contracts_are_not_authorized() -> None:
    """The actual, currently-committed SLM-248/SLM-249 verdicts must yield
    not_authorized: SLM-248 reports needs_target_trace_contract (not
    authorize_bounded_implementation) and SLM-249 reports inconclusive (not
    oracle_ceiling_positive). This is the load-bearing proof that LOT1-01
    correctly closes without any K x c model/training code.
    """
    fidelity_contract = load_upstream_contract(FIDELITY_CONTRACT_PATH)
    trace_gate_contract = load_upstream_contract(TRACE_GATE_CONTRACT_PATH)

    contract = evaluate_activation_gates(fidelity_contract, trace_gate_contract)

    assert contract.verdict == NOT_AUTHORIZED
    assert contract.gate_1_fidelity_authorization.met is False
    assert contract.gate_1_fidelity_authorization.actual_verdict == "needs_target_trace_contract"
    assert contract.gate_2_trace_oracle_ceiling.met is False
    assert contract.gate_2_trace_oracle_ceiling.actual_verdict == "inconclusive"


def test_both_gates_met_authorizes_wiring_only() -> None:
    """The evaluator is not hardcoded to always fail: synthetic contracts

    reporting both required verdicts must flip the disposition, but only to
    a wiring-only authorization, never a semantic-quality claim.
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
        },
    }

    contract = evaluate_activation_gates(fidelity_contract, trace_gate_contract)

    assert contract.verdict == AUTHORIZED_WIRING_ONLY
    assert contract.gate_1_fidelity_authorization.met is True
    assert contract.gate_2_trace_oracle_ceiling.met is True


def test_only_one_gate_met_stays_not_authorized() -> None:
    fidelity_contract = {
        "contract_id": "synthetic-fidelity",
        "contract_hash": "deadbeef",
        "authorization": {"linear_issue": "SLM-248", "verdict": REQUIRED_FIDELITY_VERDICT},
    }
    trace_gate_contract = {
        "contract_id": "synthetic-trace",
        "contract_hash": "cafef00d",
        "gate": {"linear_issue": "SLM-249", "verdict": "inconclusive"},
    }

    contract = evaluate_activation_gates(fidelity_contract, trace_gate_contract)

    assert contract.verdict == NOT_AUTHORIZED
    assert contract.gate_1_fidelity_authorization.met is True
    assert contract.gate_2_trace_oracle_ceiling.met is False


def test_missing_fields_fail_closed_not_authorized() -> None:
    contract = evaluate_activation_gates({}, {})
    assert contract.verdict == NOT_AUTHORIZED
    assert contract.gate_1_fidelity_authorization.actual_verdict == ""
    assert contract.gate_2_trace_oracle_ceiling.actual_verdict == ""


def test_contract_hash_is_stable_and_changes_with_verdict() -> None:
    fidelity_contract = load_upstream_contract(FIDELITY_CONTRACT_PATH)
    trace_gate_contract = load_upstream_contract(TRACE_GATE_CONTRACT_PATH)

    a = evaluate_activation_gates(fidelity_contract, trace_gate_contract)
    b = evaluate_activation_gates(fidelity_contract, trace_gate_contract)
    assert a.contract_hash() == b.contract_hash()

    synthetic_fidelity = dict(fidelity_contract)
    synthetic_fidelity["authorization"] = dict(fidelity_contract["authorization"])
    synthetic_fidelity["authorization"]["verdict"] = REQUIRED_FIDELITY_VERDICT
    c = evaluate_activation_gates(synthetic_fidelity, trace_gate_contract)
    assert c.contract_hash() != a.contract_hash()


def test_to_dict_and_to_json_round_trip(tmp_path: Path) -> None:
    fidelity_contract = load_upstream_contract(FIDELITY_CONTRACT_PATH)
    trace_gate_contract = load_upstream_contract(TRACE_GATE_CONTRACT_PATH)
    contract = evaluate_activation_gates(fidelity_contract, trace_gate_contract)

    out_path = tmp_path / "contract.json"
    contract.to_json(out_path)
    assert out_path.exists()

    data = contract.to_dict()
    assert data["verdict"] == NOT_AUTHORIZED
    assert data["gate_1_fidelity_authorization"]["source_contract_id"] == "lotus-openui-fidelity-contract-v1"
    assert data["gate_2_trace_oracle_ceiling"]["source_contract_id"] == "compiler-reasoning-trace-v1"


def test_render_markdown_includes_verdict_and_gates() -> None:
    fidelity_contract = load_upstream_contract(FIDELITY_CONTRACT_PATH)
    trace_gate_contract = load_upstream_contract(TRACE_GATE_CONTRACT_PATH)
    contract = evaluate_activation_gates(fidelity_contract, trace_gate_contract)

    md = render_markdown(contract)
    assert "not_authorized" in md
    assert "needs_target_trace_contract" in md
    assert "inconclusive" in md
    assert "No K x c latent-workspace model code" in md
