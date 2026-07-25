"""Integration checks for the committed SLM-232 evidence."""

from __future__ import annotations

import json

from scripts.run_slm232_latent_state_use import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    _markdown,
    _scientific_hash,
)
from slm_training.harnesses.experiments.slm232_latent_state_use import (
    LatentAblationResultV1,
    LatentStateUseGateV1,
)


def test_committed_report_markdown_hash_and_gate_are_consistent() -> None:
    report = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    assert report["report_hash"] == _scientific_hash(report)
    assert DEFAULT_MARKDOWN.read_text(encoding="utf-8") == _markdown(report)
    gate_payload = dict(report["gate"])
    gate_payload["evaluated_depths"] = tuple(gate_payload["evaluated_depths"])
    gate_payload["ablations"] = tuple(
        LatentAblationResultV1(**row) for row in gate_payload["ablations"]
    )
    gate_payload["allowed_downstream_work"] = tuple(
        gate_payload["allowed_downstream_work"]
    )
    gate_payload["blocking_evidence"] = tuple(gate_payload["blocking_evidence"])
    gate = LatentStateUseGateV1(**gate_payload)
    gate.validate()
    assert report["agentv"]["summary"]["passed"] == 5
    assert report["agentv"]["summary"]["failed"] == 0
    assert report["verdict"] == "unstable"
    assert report["representation"]["by_depth"][0][
        "z_after_context_and_position_removal"
    ]["effective_rank"] == 0.0
    assert report["training_default_changed"] is False
    assert report["generation_default_changed"] is False
    assert report["checkpoint_created"] is False
    assert report["ship_gate_claim"] is False
