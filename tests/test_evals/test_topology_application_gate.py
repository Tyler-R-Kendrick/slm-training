from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.cap2_disposition import Cap2CapabilityVerdict
from slm_training.evals.topology_application_gate import (
    TopologyApplicationGateVerdict,
    evaluate_topology_application_gate,
)

ROOT = Path(__file__).resolve().parents[2]
DISPOSITION_REPORT = (
    ROOT / "docs/design/dsh3-17-cap2-disposition-20260723/report.json"
)


def _real_capabilities() -> list[dict]:
    report = json.loads(DISPOSITION_REPORT.read_text())
    return report["disposition"]["capabilities"]


def test_gate_is_not_authorized_against_the_real_rejected_disposition() -> None:
    gate = evaluate_topology_application_gate(_real_capabilities())
    assert gate.verdict is TopologyApplicationGateVerdict.NOT_AUTHORIZED
    assert gate.hierarchical_head_verdict is Cap2CapabilityVerdict.REJECTED
    assert gate.hierarchical_head_evidence_ids
    assert "not an accepted" in gate.reason


def test_gate_flips_to_authorized_given_a_supported_hierarchical_head() -> None:
    """Proves fail-closed logic is conditional, not a hardcoded rejection."""
    capabilities = [
        {
            "capability": "hierarchical_head",
            "verdict": "supported",
            "reason": "fixture: causally improved beyond token baseline",
            "evidence_ids": ["fixture-evidence"],
            "implemented_benefit": True,
        }
    ]
    gate = evaluate_topology_application_gate(capabilities)
    assert gate.verdict is TopologyApplicationGateVerdict.AUTHORIZED
    assert gate.hierarchical_head_verdict is Cap2CapabilityVerdict.SUPPORTED
    assert gate.hierarchical_head_evidence_ids == ("fixture-evidence",)


def test_gate_stays_closed_for_supported_without_implemented_benefit() -> None:
    capabilities = [
        {
            "capability": "hierarchical_head",
            "verdict": "supported",
            "reason": "fixture",
            "evidence_ids": ["fixture-evidence"],
            "implemented_benefit": False,
        }
    ]
    gate = evaluate_topology_application_gate(capabilities)
    assert gate.verdict is TopologyApplicationGateVerdict.NOT_AUTHORIZED


def test_missing_hierarchical_head_row_fails_closed() -> None:
    with pytest.raises(ValueError, match="hierarchical_head row"):
        evaluate_topology_application_gate([])
