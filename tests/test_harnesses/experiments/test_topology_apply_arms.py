from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.topology_apply_arms import (
    TopologyApplyArm,
    build_fixture_requests,
    run_topology_apply_arms,
)


@pytest.fixture(scope="module")
def result():
    return run_topology_apply_arms(max_steps=6)


def test_fixture_requests_are_deterministic():
    left = [spec.to_dict() for spec in build_fixture_requests()]
    right = [spec.to_dict() for spec in build_fixture_requests()]
    assert left == right
    assert len(left) >= 2


def test_arm_comparison_is_deterministic(result):
    singleton = next(
        spec for spec in build_fixture_requests() if "singleton" in spec.request_id
    )
    again = run_topology_apply_arms(requests=[singleton], max_steps=6)
    for arm, data in again["arms"].items():
        assert data["requests"][0] == result["arms"][arm]["requests"][0]


def test_matched_quality_and_authority(result):
    assert result["acceptance"]["legal_terminal_set_unchanged"] is True
    assert result["acceptance"]["matched_workload_identical_action_sequences"] is True
    assert result["acceptance"]["matched_quality_every_step"] is True
    assert result["acceptance"]["exact_singleton_authority_preserved"] is True
    assert result["acceptance"]["head_always_defers_off_mode"] is True
    assert result["acceptance"]["zero_model_forwards_without_head"] is True


def test_singleton_request_exercises_forced_bypass(result):
    for arm, data in result["arms"].items():
        run = data["requests"][0]
        assert "singleton" in run["request_id"]
        assert run["singleton_bypasses"] > 0, arm
        assert run["model_forwards"] == (
            2 * (run["steps"] - run["singleton_bypasses"])
            if arm == TopologyApplyArm.HIERARCHICAL_HEAD.value
            else 0
        )


def test_topology_apply_reduces_node_passes_at_matched_quality(result):
    totals = {arm: data["totals"] for arm, data in result["arms"].items()}
    topology = totals[TopologyApplyArm.TOPOLOGY_APPLY.value]
    recompute = totals[TopologyApplyArm.RECOMPUTE_ONLY.value]
    head = totals[TopologyApplyArm.HIERARCHICAL_HEAD.value]
    assert topology["steps"] > 0
    assert topology["direct_applies"] > 0
    assert topology["node_passes"] < recompute["node_passes"]
    # The head arm does the recompute arm's work plus scorer forwards.
    assert head["node_passes"] == recompute["node_passes"]
    assert head["model_forwards"] > 0
    assert topology["model_forwards"] == 0
    # Compiler and verifier work is never skipped by direct topology apply.
    assert topology["compiler_expansions"] == recompute["compiler_expansions"]
    assert topology["verifier_calls"] == recompute["verifier_calls"]
    assert result["disposition"] == "execution_sparsity_supported"
    assert result["systems_claim"] is False


def test_edit_kinds_cover_mapped_topology_actions(result):
    kinds: set[str] = set()
    for run in result["arms"][TopologyApplyArm.TOPOLOGY_APPLY.value]["requests"]:
        kinds.update(run["edit_kinds"])
    assert kinds  # at least one mapped edit kind exercised end to end
    assert kinds <= {"expand", "contract", "move", "rewrite"}
