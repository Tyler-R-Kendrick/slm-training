"""Tests for the SLM-233 fairness and RecursiveCoreGateV2 contracts."""

from slm_training.harnesses.experiments.slm233_recursive_campaign import (
    RecursiveCoreVerdict,
    RecursiveFairnessManifestV1,
    classify_recursive_core_gate,
)


def _manifest() -> RecursiveFairnessManifestV1:
    digest = "a" * 64
    return RecursiveFairnessManifestV1(
        common_tensor_hashes={"tok.weight": digest},
        common_tensor_hashes_match=True,
        architecture_seed_namespaces={"stacked": 1, "shared_recursive": 2},
        stacked_to_shared_layer_mapping={"A.0": "B.0"},
        accounting_by_arm={"A": {"parameters": 10}, "B": {"parameters": 10}},
        optimizer_contract={"name": "AdamW", "state": "empty"},
        corpus_hash=digest,
        data_order_hash=digest,
        corruption_schedule_hash=digest,
        exposure_hash=digest,
        checkpoint_eval_schedule_hash=digest,
        decode_evaluator_gate_hashes={"floor": digest},
        hardware_runtime_budget={"device": "cpu", "max_wall_minutes": 3},
    )


def _gate(**overrides):
    values = {
        "floor_verdict": "inconclusive",
        "observability_verdict": "stagnant",
        "dynamics_verdict": "expansive_unstable",
        "z_verdict": "unstable",
        "matrix_complete": True,
        "controls_matched": True,
        "all_finite": True,
        "semantic_outcomes_available": False,
        "fairness_manifest_hash": _manifest().to_dict()["manifest_hash"],
        "gate_refs": {"floor": {"hash": "a" * 64}},
        "matched_matrix_ref": "agentv-dir://raw_campaign.json",
        "primary_effect_sizes": {"B": {"nll_delta": -0.2}},
        "equivalence_margins": {"nll": 0.05},
        "cost_frontier": [{"arm": "A", "parameters": 10}],
    }
    values.update(overrides)
    return classify_recursive_core_gate(**values)


def test_manifest_hashes_complete_fairness_contract() -> None:
    manifest = _manifest().to_dict()
    assert manifest["schema"] == "RecursiveFairnessManifestV1"
    assert len(manifest["manifest_hash"]) == 64


def test_proxy_floor_dominates_favorable_proxy_loss() -> None:
    gate = _gate()
    assert (
        gate.verdict
        == RecursiveCoreVerdict.ARCHITECTURE_NOT_IDENTIFIABLE.value
    )
    assert "rsc3" in gate.blocked_claims
    assert not gate.checkpoint_refs


def test_nonfinite_matrix_is_unstable_before_floor_classification() -> None:
    gate = _gate(all_finite=False)
    assert gate.verdict == RecursiveCoreVerdict.UNSTABLE.value


def test_incomplete_fairness_is_inconclusive() -> None:
    gate = _gate(controls_matched=False)
    assert gate.verdict == RecursiveCoreVerdict.INCONCLUSIVE.value


def test_floor_escaped_positive_fixture_requires_identifiable_semantics() -> None:
    gate = _gate(
        floor_verdict="floor_escaped",
        dynamics_verdict="stable",
        z_verdict="causal_use",
        semantic_outcomes_available=True,
        positive_classification=RecursiveCoreVerdict.EXPLICIT_Z_POSITIVE.value,
    )
    assert gate.verdict == RecursiveCoreVerdict.EXPLICIT_Z_POSITIVE.value


def test_explicit_z_fixture_fails_closed_without_causal_z_gate() -> None:
    gate = _gate(
        floor_verdict="floor_escaped",
        dynamics_verdict="stable",
        semantic_outcomes_available=True,
        positive_classification=RecursiveCoreVerdict.EXPLICIT_Z_POSITIVE.value,
    )
    assert gate.verdict == RecursiveCoreVerdict.INCONCLUSIVE.value
