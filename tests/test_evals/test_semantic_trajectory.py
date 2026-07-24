from __future__ import annotations

import json

import pytest

from scripts.collect_semantic_trajectory import collect
from slm_training.evals.semantic_trajectory import (
    EarlyWarningAction,
    FeatureAvailability,
    SemanticEarlyWarningPolicyV1,
    SemanticTrajectoryTelemetryV1,
    bounded_empty_mass,
    entropy_and_margin,
)


def _row(availability: FeatureAvailability = FeatureAvailability.EXACT) -> SemanticTrajectoryTelemetryV1:
    return SemanticTrajectoryTelemetryV1(
        "r", "a" * 64, 10, "family", "e", "s", availability,
        0.5, 0.5, 0.0, 2, 1.0, 1.0, 0.2,
    )


def test_telemetry_roundtrip_and_unknown_no_action() -> None:
    row = _row(FeatureAvailability.UNKNOWN)
    assert SemanticTrajectoryTelemetryV1.from_dict(row.to_dict()) == row
    assert SemanticEarlyWarningPolicyV1(default_off=False).action_for(row) is EarlyWarningAction.UNKNOWN_NO_ACTION
    assert SemanticEarlyWarningPolicyV1().action_for(_row()) is EarlyWarningAction.UNKNOWN_NO_ACTION


def test_entropy_margin_and_exactness() -> None:
    entropy, margin = entropy_and_margin((0.75, 0.25))
    assert entropy > 0 and margin == pytest.approx(0.5)
    assert bounded_empty_mass((0.1, 0.2), coverage=FeatureAvailability.EXACT) == (0.30000000000000004, FeatureAvailability.EXACT)
    assert bounded_empty_mass((0.1,), coverage=FeatureAvailability.PARTIAL) == (None, FeatureAvailability.PARTIAL)
    with pytest.raises(ValueError, match="probabilities"):
        entropy_and_margin((float("nan"), 1.0))


def test_fixture_inventory_is_explicitly_insufficient(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row(FeatureAvailability.PARTIAL).to_dict()) + "\n")
    assert collect(path)["status"] == "insufficient_trajectory_evidence"


def test_inventory_requires_complete_trajectories_and_two_families(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = []
    for trajectory in range(20):
        for checkpoint in range(4):
            row = SemanticTrajectoryTelemetryV1(
                run_id=f"trajectory-{trajectory}",
                checkpoint_sha256=f"{checkpoint:064x}",
                exposure_updates=checkpoint,
                data_family=f"family-{trajectory % 2}",
                example_id="e",
                state_fingerprint="s",
                feature_availability=FeatureAvailability.EXACT,
                full_vocab_entropy=0.5,
                legal_entropy=0.5,
                top1_top2_margin=0.5,
                legal_set_size=2,
                legal_mass=1.0,
                empty_vs_populated_log_score_gap=0.0,
                constrained_empty_mass=0.1,
            )
            rows.append(json.dumps(row.to_dict()))
    path.write_text("\n".join(rows) + "\n")
    assert collect(path)["status"] == "ready_for_grouped_fit"
