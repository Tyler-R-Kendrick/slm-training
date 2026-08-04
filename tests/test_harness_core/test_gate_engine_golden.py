"""Characterization pin for the ship-gate payload."""

from __future__ import annotations

import json

from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates
from tests.casefiles import snapshot_cases

CASES = snapshot_cases(__file__, "gate_payloads")


def _legacy_payload(payload: dict) -> dict:
    """Project additive diagnostics away from the pre-extraction payload pin."""
    additive = {
        "evidence_volume_failures",
        "measurement_integrity_failures",
        "quality_threshold_failures",
        "runtime_failures",
    }
    return {key: value for key, value in payload.items() if key not in additive}


def test_payloads_match_golden() -> None:
    for case in CASES:
        payload = evaluate_ship_gates(
            case.input["suites"], thresholds=case.input["thresholds"]
        )
        case.expect(_legacy_payload(payload))


def test_payload_json_bytes_stable() -> None:
    """Key order and value formatting must survive, not just dict equality."""
    for case in CASES:
        payload = evaluate_ship_gates(
            case.input["suites"], thresholds=case.input["thresholds"]
        )
        got = json.dumps(_legacy_payload(payload), indent=2)
        want = json.dumps(case.expected, indent=2)
        assert got == want, f"serialized payload drifted for case {case.id!r}"
