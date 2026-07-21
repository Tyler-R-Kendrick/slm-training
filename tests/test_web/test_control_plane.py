"""Tests for the observability + control-plane web layer.

Covers the read-only observability API, cold-start fallback + provenance, the
pure-compute gate endpoint, the execution capability gate, the job allowlist
(the security boundary), and an end-to-end job run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates
from slm_training.lineage.store import LineageStore
from slm_training.web import jobs as jobs_mod
from slm_training.web.app import create_app
from slm_training.web.observability import Readers

SMOKE_SUITE = {
    "smoke": {
        # Above the DEFAULT_MIN_SUITE_N evidence floor.
        "n": 32,
        "parse_rate": 0.9,
        "structural_similarity": 0.5,
        "placeholder_fidelity": 0.4,
        "reward_score": 0.5,
        # Measured (zero) fallback telemetry — certified_fallback fails closed
        # when unmeasured.
        "fallback_count": 0,
    }
}


@pytest.fixture
def ro_client() -> TestClient:
    with TestClient(create_app(execution=False)) as client:
        yield client


# --- observability reads ---------------------------------------------------
def test_overview_aggregates_committed_evidence(ro_client: TestClient) -> None:
    overview = ro_client.get("/api/overview").json()
    assert {"scoreboards", "experiment_totals", "checkpoints", "system"} <= set(overview)
    assert overview["experiment_totals"]["count"] >= 1


def test_scoreboard_unknown_kind_is_404(ro_client: TestClient) -> None:
    assert ro_client.get("/api/scoreboards/bogus").status_code == 404
    assert ro_client.get("/api/scoreboards/quality").json()["kind"] == "quality"


def test_checkpoints_roster_includes_fixture(ro_client: TestClient) -> None:
    roster = ro_client.get("/api/checkpoints").json()["checkpoints"]
    assert any("playground_demo" in (c.get("run_id") or "") for c in roster)


def test_e499_cold_start_evidence_is_persisted() -> None:
    root = Path(__file__).parents[2]
    snapshot = (
        root / "src" / "slm_training" / "web" / "static" / "dashboard_snapshot.json"
    )
    assert json.loads(snapshot.read_text())["schema_version"] == 1

    readers = Readers(root)
    run_id = "e499-choice-compatible-strict-hf-choice-candidate-r6"
    assert any(row.get("run_id") == run_id for row in readers.runs()["runs"])
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert {
        "e499-remediated-roots-hf-choice-control-r4",
        "e499-strict-r4-hf-choice-candidate-r4",
        run_id,
    } <= checkpoint_ids


def test_e500_cold_start_evidence_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_ids = {
        row.get("run_id") for row in readers.runs()["runs"]
    }
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    expected = {
        "e500-document-control-hf-choice-r1",
        "e500-documentized-expression-hf-choice-r2",
        "e500-document-control-hf-choice-r3-5k",
        "e500-documentized-expression-hf-choice-r4-5k",
    }
    assert expected <= run_ids
    assert expected <= checkpoint_ids

    train = readers.train_data()
    version = "e500_documentized_expression_candidate_r2_20260718"
    assert version in train["versions"]
    records = readers.train_records(version, limit=300)
    assert records["count"] == 260
    assert len(records["records"]) == 260


def test_e521_visible_slot_contract_data_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    version = "e521_visible_slot_contract_r2_20260719"
    assert version in readers.train_data()["versions"]

    records = readers.train_records(version, limit=300)
    assert records["count"] == 244
    assert len(records["records"]) == 244
    assert all(
        all(slot in row["prompt"] for slot in row["placeholders"])
        for row in records["records"]
    )


def test_e524_visible_component_contract_data_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    version = "e524_visible_component_slot_contract_r4_20260719"
    assert version in readers.train_data()["versions"]

    records = readers.train_records(version, limit=300)
    assert records["count"] == 244
    assert len(records["records"]) == 244
    assert all(
        any(line.startswith("Components: ") for line in row["prompt"].splitlines())
        for row in records["records"]
    )
    assert all(
        all(slot in row["prompt"] for slot in row["placeholders"])
        for row in records["records"]
    )


def test_e527_visible_component_types_data_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    version = "e527_visible_component_types_slot_contract_r1_20260719"
    records = readers.train_records(version, limit=300)
    assert records["count"] == 244
    assert all("Components: " in row["prompt"] for row in records["records"])
    assert all(
        " x" not in next(
            line
            for line in row["prompt"].splitlines()
            if line.startswith("Components: ")
        )
        for row in records["records"]
    )


def test_e530_visible_semantic_roles_data_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    version = "e530_visible_semantic_roles_r2_20260719"
    records = readers.train_records(version, limit=300)
    assert records["count"] == 244
    assert all("Semantic roles: " in row["prompt"] for row in records["records"])
    assert any(" -> " in row["prompt"] for row in records["records"])


def test_e501_matched_runs_and_checkpoints_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_ids = {
        row.get("run_id") for row in readers.runs()["runs"]
    }
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    expected_runs = {
        "e501-e396-control-r1",
        "e501-e396-e500-init-r1",
        "e501-e396-e500-uniform-init-r2",
        "e501-e396-e500-uniform-init-r3-1k",
    }
    expected_checkpoints = expected_runs - {"e501-e396-control-r1"}
    assert expected_runs <= run_ids
    assert expected_checkpoints <= checkpoint_ids
    assert all(readers.run(run_id)["scoreboard"] for run_id in expected_runs)


def test_e502_matched_runs_and_checkpoints_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    expected = {
        "e502-e396-e500-uniform-lr1e4-r1",
        "e502-e396-e500-uniform-lr3e5-r2",
        "e502-e396-e500-prior-retained-lr3e4-r3",
        "e502-e396-e500-prior-retained-lr3e4-r4-5k",
    }
    run_ids = {row.get("run_id") for row in readers.runs()["runs"]}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert expected <= run_ids
    assert expected <= checkpoint_ids
    assert all(readers.run(run_id)["scoreboard"] for run_id in expected)


def test_e503_matched_runs_and_checkpoints_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    expected = {
        "e503-e396-e500-retention0-r1-5k",
        "e503-e396-e500-retention001-r2-5k",
        "e503-e396-e500-retention005-r3-5k",
        "e503-e396-e500-retention003-r4-5k",
    }
    run_ids = {row.get("run_id") for row in readers.runs()["runs"]}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert expected <= run_ids
    assert expected <= checkpoint_ids
    assert all(readers.run(run_id)["scoreboard"] for run_id in expected)


def test_e504_matched_runs_and_checkpoints_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    expected = {
        "e504-e396-e500-replay000-r1-5k",
        "e504-e396-e500-replay0125-r2-5k",
        "e504-e396-e500-replay025-r3-5k",
        "e504-e396-e500-replay050-r4-5k",
        "e504-e396-e500-replay050-retention001-r5-5k",
    }
    run_ids = {row.get("run_id") for row in readers.runs()["runs"]}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert expected <= run_ids
    assert expected <= checkpoint_ids
    assert all(readers.run(run_id)["scoreboard"] for run_id in expected)


def test_e505_run_and_checkpoint_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e505-e396-e500-replay050-loss-attribution-r1-5k"
    assert run_id in {row.get("run_id") for row in readers.runs()["runs"]}
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert readers.run(run_id)["scoreboard"]


def test_e506_multi_suite_eval_runs_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    expected = {
        "e506-e505-contract-off-r1",
        "e506-e505-contract-on-r2",
    }
    assert expected <= {row.get("run_id") for row in readers.runs()["runs"]}
    for run_id in expected:
        scoreboard = readers.run(run_id)["scoreboard"]
        assert set(scoreboard["suites"]) == {"held_out", "ood", "adversarial"}


def test_e507_length_safe_ood_runs_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    expected = {
        "e507-e505-ood160-contract-off-r1",
        "e507-e505-ood160-contract-on-r2",
    }
    assert expected <= {row.get("run_id") for row in readers.runs()["runs"]}
    for run_id in expected:
        scoreboard = readers.run(run_id)["scoreboard"]
        assert set(scoreboard["suites"]) == {"ood"}


def test_e508_default_generation_ood_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e508-e505-ood160-contract-on-fullgen-r1"
    assert run_id in {row.get("run_id") for row in readers.runs()["runs"]}
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}


def test_e509_slot_contract_context_ood_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e509-e505-ood160-contract-context-r1"
    assert run_id in {row.get("run_id") for row in readers.runs()["runs"]}
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}


def test_e510_component_plan_ood_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e510-e505-ood160-component-plan4-r1"
    assert run_id in {row.get("run_id") for row in readers.runs()["runs"]}
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}


def test_e511_component_plan_three_suite_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e511-e505-three-suite192-component-plan4-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"held_out", "ood", "adversarial"}
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {
        "held_out",
        "ood",
        "adversarial",
    }


def test_e512_slot_component_weight_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e512-e505-ood160-component-plan4-slot8-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}


def test_e513_durable_checkpoint_and_run_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e513-e396-e500-replay050-slotrole4-focal2-r3-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e515_focal_control_checkpoint_and_run_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e515-e396-e500-replay050-slotrole4-focal0-r1-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e517_context_control_checkpoint_and_run_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e517-e396-e500-replay050-slotrole1-context-r1-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e519_honest_context_checkpoint_and_run_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e519-e396-e500-replay050-slotrole1-honest-context-r1-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e522_visible_inventory_checkpoint_and_run_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e522-e396-e521-replay050-slotrole1-honest-context-r2-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e525_visible_component_checkpoint_and_run_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e525-e396-e524-replay050-slotrole1-honest-context-r2-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    assert set(readers.run(run_id)["scoreboard"]["suites"]) == {"ood"}
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e528_visible_component_types_checkpoint_and_run_are_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e528-e396-e527-replay050-slotrole1-honest-context-r1-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    monkeypatch.setattr(readers, "_run_dir", lambda *_: tmp_path / "missing")
    detail = readers.run(run_id)
    assert set(detail["scoreboard"]["suites"]) == {"ood"}
    assert detail["train_summary"]["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 99
    assert detail["training_data"]["provenance"] == "committed"
    assert detail["training_data"]["dataset"]["version"] == (
        "e527_visible_component_types_slot_contract_r1_20260719"
    )
    assert detail["training_data"]["dataset"]["fingerprint_matches_run"] is True
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids
    assert run_id in readers.train_data(
        version="e527_visible_component_types_slot_contract_r1_20260719"
    )["used_by_runs"]


def test_e531_visible_semantic_roles_checkpoint_and_run_are_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e531-e396-e530-replay050-slotrole1-honest-context-r1-5k"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    monkeypatch.setattr(readers, "_run_dir", lambda *_: tmp_path / "missing")
    detail = readers.run(run_id)
    assert set(detail["scoreboard"]["suites"]) == {"ood"}
    assert detail["train_summary"]["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 99
    assert detail["training_data"]["provenance"] == "committed"
    assert detail["training_data"]["dataset"]["version"] == (
        "e530_visible_semantic_roles_r2_20260719"
    )
    assert detail["training_data"]["dataset"]["fingerprint_matches_run"] is True
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids
    assert run_id in readers.train_data(
        version="e530_visible_semantic_roles_r2_20260719"
    )["used_by_runs"]


def test_e533_visible_role_inference_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e533-e531-ood160-visible-role-inference-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert set(listed["suites"]) == {"ood"}
    assert listed["pass"] is False
    detail = readers.run(run_id)
    assert set(detail["scoreboard"]["suites"]) == {"ood"}
    assert detail["scoreboard"]["suites"]["ood"]["n"] == 4


def test_e534_visible_role_decode_bias_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e534-e531-ood160-visible-role-bias4-r2"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert listed["pass"] is False
    detail = readers.run(run_id)
    assert set(detail["scoreboard"]["suites"]) == {"ood"}
    assert detail["scoreboard"]["suites"]["ood"]["n"] == 4
    assert detail["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25


def test_e535_visible_reference_completeness_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e535-e531-ood160-visible-role4-reference4-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert listed["pass"] is False
    detail = readers.run(run_id)
    ood = detail["scoreboard"]["suites"]["ood"]
    assert ood["n"] == 4
    assert ood["visible_reference_applications"] == 0
    assert ood["ref_graph_exact"] == 0.0


def test_e536_choice_decision_evidence_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e536-e531-ood160-choice-evidence-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert listed["pass"] is False
    detail = readers.run(run_id)
    ood = detail["scoreboard"]["suites"]["ood"]
    assert ood["n"] == 4
    assert ood["generation_evidence_schemas"] == ["choice_decision_trace/v1"]
    assert detail["scoreboard"]["agentv"]["passed"] == 0


def test_e538_role_plan_composition_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e538-e531-ood160-role4-plan4-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert listed["pass"] is False
    detail = readers.run(run_id)
    ood = detail["scoreboard"]["suites"]["ood"]
    assert ood["n"] == 4
    assert ood["meaningful_program_rate"] == 0.0
    assert ood["generation_evidence_schemas"] == ["choice_decision_trace/v1"]
    assert detail["scoreboard"]["agentv"]["passed"] == 0


def test_e539_structural_reference_runs_are_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    runs = {row.get("run_id"): row for row in readers.runs()["runs"]}
    control_id = "e539-control-e531-ood160-role4-reference0-r1"
    intervention_id = "e539-e531-ood160-role4-structural-reference4-r1"
    assert runs[control_id]["pass"] is False
    assert runs[intervention_id]["pass"] is False

    control = readers.run(control_id)["scoreboard"]["suites"]["ood"]
    intervention = readers.run(intervention_id)
    ood = intervention["scoreboard"]["suites"]["ood"]
    assert control["n"] == ood["n"] == 4
    assert control["placeholder_fidelity"] == 0.3833333333333333
    assert ood["placeholder_fidelity"] == 0.4666666666666667
    assert intervention["scoreboard"]["agentv"]["passed"] == 0


def test_e540_reference_phase_telemetry_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e540-e531-ood160-role4-reference4-phase-trace-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert listed["pass"] is False
    detail = readers.run(run_id)
    ood = detail["scoreboard"]["suites"]["ood"]
    assert ood["n"] == 4
    assert ood["placeholder_fidelity"] == 0.4666666666666667
    assert ood["generation_evidence_schemas"] == ["choice_decision_trace/v2"]
    assert detail["scoreboard"]["agentv"]["passed"] == 0


def test_e541_root_reference_run_is_persisted() -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    run_id = "e541-e531-ood160-role4-root-reference4-r1"
    listed = next(
        row for row in readers.runs()["runs"] if row.get("run_id") == run_id
    )
    assert listed["pass"] is False
    detail = readers.run(run_id)
    ood = detail["scoreboard"]["suites"]["ood"]
    assert ood["n"] == 4
    assert ood["placeholder_fidelity"] == 0.3833333333333333
    assert ood["generation_evidence_schemas"] == ["choice_decision_trace/v2"]
    assert detail["scoreboard"]["agentv"]["passed"] == 0


def test_e544_root_identity_run_and_checkpoint_are_persisted(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e544-e543-root-identity1-r2-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25
    assert detail["scoreboard"]["agentv"]["passed"] == 0
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e545_matched_train_summaries_and_checkpoints_are_persisted(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e545-e544-root-identity-neg1-control-r1-24s",
        "e545-e544-root-identity-neg4-r2-24s",
    }

    for run_id in run_ids:
        detail = readers.run(run_id)
        assert detail["provenance"] == "committed"
        assert detail["train_summary"]["steps"] == 24
        assert detail["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.0
        assert detail["scoreboard"]["agentv"]["passed"] == 0

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_ids <= checkpoint_ids


def test_e546_matched_train_summaries_and_checkpoints_are_persisted(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e546-e544-strict-subset1-control-r1-24s",
        "e546-e544-strict-subset5-r2-24s",
    }

    for run_id in run_ids:
        detail = readers.run(run_id)
        assert detail["provenance"] == "committed"
        assert detail["train_summary"]["steps"] == 24
        assert detail["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.0
        assert detail["scoreboard"]["agentv"]["passed"] == 0

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_ids <= checkpoint_ids


def test_e547_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e547-e544-strict-subset2-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.2248
    assert detail["scoreboard"]["agentv"]["passed"] == 0
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e548_eval_run_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    detail = readers.run("e548-e547-semantic-role8-eval-r2")
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.2248
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == (
        0.2583333333333333
    )
    assert detail["scoreboard"]["agentv"]["passed"] == 0


def test_e549_eval_run_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    detail = readers.run("e549-e547-slot-component0-eval-r1")
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.27125
    assert detail["scoreboard"]["suites"]["ood"]["component_type_recall"] == 0.0
    assert detail["scoreboard"]["agentv"]["passed"] == 0


def test_e550_eval_run_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    detail = readers.run("e550-e547-slot-component2-eval-r1")
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.2248
    assert detail["scoreboard"]["agentv"]["passed"] == 0


def test_e551_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e551-e544-strict-subset2-no-lexeme-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.3
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e552_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e552-e544-strict-subset2-lexeme05-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == (
        0.13333333333333333
    )
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e553_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e553-e544-prior-proportional-r3-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.3
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == (
        0.12437500000000001
    )
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e554_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e554-e544-slot-next-context-r2-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == (
        0.2583333333333333
    )
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id in checkpoint_ids


def test_e555_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e555-e544-slot-pair-interaction-r2-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.3
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e556_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e556-e544-slot-context-combined-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == (
        0.21666666666666667
    )
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e557_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e557-e544-slot-pair-balance1-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.3
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e558_runs_and_checkpoints_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    trial_id = "e558-e544-owner-coverage-r1-24s"
    run_id = "e558-e544-owner-coverage-r2-24s"

    trial = readers.run(trial_id)
    detail = readers.run(run_id)
    assert trial["provenance"] == "committed"
    assert trial["train_summary"]["steps"] == 24
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.425
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert {trial_id, run_id} <= checkpoint_ids


def test_e559_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e559-e544-owner-coverage2-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert (
        detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"]
        == 0.44166666666666665
    )
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e560_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e560-e544-owner-threshold4-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.218125
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e561_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e561-e544-owner-threshold7-r1-24s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 24
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.575
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e562_eval_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e562-e561-component-plan-decode1-eval-r1"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert (
        detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"]
        == 0.7416666666666667
    )
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.273175


def test_e563_eval_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e563-e561-component-plan-decode05-eval-r1"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert (
        detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"]
        == 0.4083333333333333
    )
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.201925


def test_e564_eval_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e564-e561-semantic-role-decode2-eval-r1"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.575
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.241925


def test_e565_eval_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e565-e561-semantic-role-decode0-eval-r1"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.575
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.241925


def test_e566_eval_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e566-e561-slot-component-decode2-eval-r1"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.575
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.241925


def test_e567_eval_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e567-e561-slot-component-decode0-eval-r1"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert (
        detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"]
        == 0.5333333333333333
    )
    assert detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.219425


def test_e568_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e568-e561-cont48-r1-48s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 48
    assert detail["scoreboard"]["suites"]["ood"]["reward_score"] == 0.692
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e569_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e569-e561-matched-cont48-r1-48s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 48
    assert detail["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e570_eval_is_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e570-e569-component-plan1-eval-r1"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert (
        detail["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.335
    )
    assert detail["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7695


def test_e571_eval_and_control_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    detail = readers.run("e571-e569-component-plan05-eval-r2")
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25
    assert detail["scoreboard"]["suites"]["ood"]["component_type_recall"] == (
        0.3333333333333333
    )

    control = readers.run("e571-e569-component-plan05-eval-r1")
    assert control["provenance"] == "committed"


def test_e572_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e572-e569-fidelity2-r1-48s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 48
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.65
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e573_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e573-e569-fidelity1-r1-48s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 48
    assert detail["scoreboard"]["suites"]["ood"]["placeholder_fidelity"] == 0.475
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e574_run_and_checkpoint_are_persisted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e574-e569-slotloss2-r1-48s"

    detail = readers.run(run_id)
    assert detail["provenance"] == "committed"
    assert detail["train_summary"]["steps"] == 48
    assert detail["scoreboard"]["suites"]["ood"]["reward_score"] == 0.757
    assert run_id in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e575_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e575-e569-prompt-plan1-r3"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7345

    for run_id in (
        "e575-e569-prompt-plan-control-r3",
        "e575-e569-prompt-plan2-r3",
    ):
        assert readers.run(run_id)["provenance"] == "committed"

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e576_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e576-e569-plan-binding1-r2"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7345

    for run_id in (
        "e576-e569-plan-binding-control-r2",
        "e576-e569-plan-binding2-r2",
    ):
        assert readers.run(run_id)["provenance"] == "committed"

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e577_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e577-e569-plan-binding-order1-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7345

    control = readers.run("e577-e569-plan-binding-order-control-r1")
    assert control["provenance"] == "committed"

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e578_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e578-e569-plan-root1-r2"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.25
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7345

    for run_id in (
        "e578-e569-plan-root-control-r2",
        "e578-e569-plan-root2-r2",
    ):
        assert readers.run(run_id)["provenance"] == "committed"

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e579_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e579-e569-verified-root4-r2"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["ast_edge_f1"] == 0.2
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.748

    for weight in (0, 1, 2):
        run_id = f"e579-e569-verified-root{weight}-r2"
        assert readers.run(run_id)["provenance"] == "committed"

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e580_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e580-e569-cardinality-root4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["ast_edge_f1"] == 0.0
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7345
    assert readers.run(
        "e580-e569-cardinality-root0-r1"
    )["provenance"] == "committed"

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e581_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e581-e569-count-components4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["ast_edge_f1"] == 0.2
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.748

    for weight in (0, 1, 2):
        run_id = f"e581-e569-count-components{weight}-r1"
        assert readers.run(run_id)["provenance"] == "committed"

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e582_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e582-e569-distinct-slots4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert (
        primary["scoreboard"]["suites"]["ood"]["ast_edge_f1"]
        == 0.16666666666666666
    )
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.751
    assert (
        readers.run("e582-e569-distinct-slots0-r1")["provenance"]
        == "committed"
    )

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e583_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e583-e569-slot-family4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.751
    control = readers.run("e583-e569-slot-family0-r1")
    assert control["provenance"] == "committed"
    assert control["scoreboard"]["suites"]["ood"]["reward_score"] == 0.776

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e584_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e584-e569-role-gate4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.57425
    control = readers.run("e584-e569-role-gate0-r1")
    assert control["provenance"] == "committed"
    assert control["scoreboard"]["suites"]["ood"]["reward_score"] == 0.776

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e585_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e585-e569-role-coverage4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.57425
    control = readers.run("e585-e569-role-coverage0-r1")
    assert control["provenance"] == "committed"
    assert control["scoreboard"]["suites"]["ood"]["reward_score"] == 0.776

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e586_matched_eval_arms_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e586-e569-original-coverage4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.751
    control = readers.run("e586-e569-original-coverage0-r1")
    assert control["provenance"] == "committed"
    assert control["scoreboard"]["suites"]["ood"]["reward_score"] == 0.776

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e587_matched_eval_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e587-e586-schema-value1-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7635
    control = readers.run("e587-e586-schema-value0-r1")
    aggressive = readers.run("e587-e586-schema-value4-r1")
    assert control["provenance"] == "committed"
    assert control["scoreboard"]["suites"]["ood"]["reward_score"] == 0.751
    assert aggressive["provenance"] == "committed"
    assert aggressive["scoreboard"]["suites"]["ood"]["reward_score"] == 0.692

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e588_root_closure_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e588-e587-root8-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7585
    control = readers.run("e588-e587-root4-control-r1")
    plateau = readers.run("e588-e587-root12-r1")
    assert control["provenance"] == "committed"
    assert control["scoreboard"]["suites"]["ood"]["reward_score"] == 0.692
    assert plateau["provenance"] == "committed"
    assert plateau["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7585

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e589_opaque_slot_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e589-e588-opaque4-r1"
    primary = readers.run(primary_id)
    assert primary["provenance"] == "committed"
    assert primary["scoreboard"]["suites"]["ood"]["structural_similarity"] == (
        0.331875
    )
    control = readers.run("e589-e588-opaque0-control-r1")
    plateau = readers.run("e589-e588-opaque8-r1")
    assert control["provenance"] == "committed"
    assert control["scoreboard"]["suites"]["ood"]["structural_similarity"] == (
        0.406875
    )
    assert plateau["provenance"] == "committed"
    assert plateau["scoreboard"]["suites"]["ood"]["structural_similarity"] == (
        0.331875
    )

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e590_opaque_close_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    readers = Readers(root)
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")

    primary_id = "e590-e589-close4-r1"
    for run_id in (
        "e590-e589-close0-control-r1",
        "e590-e589-close2-r1",
        primary_id,
        "e590-e589-close8-r1",
    ):
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["reward_score"] == 0.7585

    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert primary_id not in checkpoint_ids


def test_e591_role_slot_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    for run_id in (
        "e591-e590-role0-control-r1",
        "e591-e590-role2-r1",
        "e591-e590-role4-r1",
    ):
        assert readers.run(run_id)["provenance"] == "committed"
    assert "e591-e590-role2-r1" not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e592_array_item_result_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e592-e591-array-items-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.5
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e593_enum_close_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e593-e592-enum-close2-r1": 0.6447499999999999,
        "e593-e592-enum-close4-r1": 0.65725,
    }
    for run_id, reward in run_ids.items():
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["reward_score"] == reward
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e594_inline_plan_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e594-e592-inline-plan2-r1",
        "e594-e592-inline-plan4-r1",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["reward_score"] == 0.8115
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e595_action_plan_result_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e595-e592-action-plan-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["ood"]["component_type_recall"] == 0.625
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e596_role_alias_result_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e596-e595-role-alias-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["ood"]["reward_score"] == 0.8115
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e597_schema_role_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e597-e596-role8-r1",
        "e597-e596-role12-r1",
        "e597-e596-schema-roles-r1",
        "e597-e596-schema-roles8-r1",
        "e597-e596-schema-roles8-r2",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["reward_score"] == 0.8115
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e598_owner_slot_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e598-e597-slotowner4-r1",
        "e598-e597-slotowner6-r1",
        "e598-e597-slotowner8-r1",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["reward_score"] == 0.8115
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e599_slot_coverage_close_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e599-e598-slotclose2-r1",
        "e599-e598-slotclose2-r2",
        "e599-e598-slotclose4-r1",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert (
            run["scoreboard"]["suites"]["ood"]["structural_similarity"]
            == 0.516875
        )
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e600_modified_count_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e600-e599-modified-count-r1",
        "e600-e599-modified-count-plan8-r1",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["component_type_recall"] == 0.625
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e601_first_component_seed_ladder_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e601-e600-planseed8-r1",
        "e601-e600-planseed16-r1",
        "e601-e600-planseed32-r1",
        "e601-e600-rootseed8-r1",
        "e601-e600-rootseed32-r1",
        "e601-e600-rootseed8-r2",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert (
            run["scoreboard"]["suites"]["ood"]["structural_similarity"]
            == 0.516875
        )
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e602_plan_seed_trace_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e602-e601-rootseed32-trace-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.516875
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e603_reachability_trace_runs_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    expected_structure = {
        "e603-e602-final-choice-trace-r1": 0.2175,
        "e603-e602-final-choice-trace-r2": 0.516875,
    }
    for run_id, structure in expected_structure.items():
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["structural_similarity"] == structure
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(expected_structure)


def test_e604_plan_pressure_runs_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e604-e603-plan16-r1",
        "e604-e603-plan32-r1",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.575625
        assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e605_missing_family_trace_runs_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e605-e604-plan32-trace-r1",
        "e605-e604-plan32-trace-r2",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        assert run["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.575625
        assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert not checkpoints.intersection(run_ids)


def test_e606_plan_margin_run_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e606-e605-planmargin2-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.575625
    assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e607_root_trace_run_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e607-e606-roottrace-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["ood"]["structural_similarity"] == 0.575625
    assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e608_root_margin_run_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e608-e607-rootmargin2-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["ood"]["meaningful_program_rate"] == 0.75
    assert run["scoreboard"]["suites"]["ood"]["reward_score"] == 0.67875
    assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e621_coverage_closure_runs_are_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    expected = {
        "e621-coverage-aware-closure-r1": (0.5, 0.66675),
        "e621-coverage-aware-closure-r2": (0.75, 0.8175),
    }
    for run_id, (meaningful, reward) in expected.items():
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["meaningful_program_rate"] == meaningful
        assert suite["reward_score"] == reward
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert checkpoints.isdisjoint(expected)


def test_e622_coverage_closure_trace_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e622-coverage-closure-trace-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.75
    assert suite["reward_score"] == 0.8175
    assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e630_prompt_owned_closure_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e630-prompt-owned-closure-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.5
    assert suite["reward_score"] == 0.785
    assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e631_owner_escape_is_persisted_without_new_checkpoint(
    tmp_path: Path,
) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e631-frame-aware-owner-escape-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.75
    assert suite["structural_similarity"] == 0.572925
    assert suite["reward_score"] == 0.8515
    assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e633_input_role_runs_persist_without_new_checkpoints(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    expected = {
        "e633-input-active-role-routing-r1": (0.75, 0.8515),
        "e633-input-active-role-routing-r2": (0.5, 0.785),
        "e633-input-active-role-routing-r3": (0.75, 0.8515),
    }
    for run_id, (meaningful, reward) in expected.items():
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["meaningful_program_rate"] == meaningful
        assert suite["reward_score"] == reward
        assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert checkpoints.isdisjoint(expected)


def test_e634_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e634-final-precontent-routing-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.5
    assert suite["reward_score"] == 0.785
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e635_confirmed_runs_persist_without_new_checkpoints(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e635-property-compatible-coverage-r1",
        "e635-property-compatible-coverage-r2",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.25
        assert suite["reward_score"] == 0.8515
        assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert checkpoints.isdisjoint(run_ids)


def test_e636_runs_persist_without_new_checkpoints(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    expected = {
        "e636-modal-schema-reach-r1": 0.25,
        "e636-modal-schema-reach-r2": 0.5,
    }
    for run_id, strict_rate in expected.items():
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["binding_aware_meaningful_v2_rate_strict"] == strict_rate
        assert suite["reward_score"] == 0.8575
        assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert checkpoints.isdisjoint(expected)


def test_e637_confirmed_runs_persist_without_new_checkpoints(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e637-nested-family-accounting-r1",
        "e637-nested-family-accounting-r2",
    }
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.5
        assert suite["structural_similarity"] == 0.581675
        assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert checkpoints.isdisjoint(run_ids)


def test_e638_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e638-root-slot-coverage-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.5
    assert suite["reward_score"] == 0.819
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e639_neutral_runs_persist_without_new_checkpoints(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    expected = {
        "e639-root-sibling-coverage-r1": 0.0,
        "e639-root-sibling-coverage-r2": 0.5,
    }
    for run_id, strict in expected.items():
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["binding_aware_meaningful_v2_rate_strict"] == strict
        assert run["scoreboard"]["agentv"]["passed"] == 0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert checkpoints.isdisjoint(expected)


def test_e640_neutral_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e640-root-slot-references-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.5
    assert suite["structural_similarity"] == 0.581675
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e641_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e641-role-plan-completion-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.25
    assert suite["reward_score"] == 0.884
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e642_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e642-root-only-role-plans-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.5
    assert suite["structural_similarity"] == 0.48835
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e643_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e643-bound-role-plans-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.25
    assert suite["placeholder_fidelity"] == 0.7583333333333333
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e644_retained_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e644-role-obligation-margin-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.605625
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e645_neutral_runs_persist_without_new_checkpoints(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {"e645-root-binding-w4-r1", "e645-root-binding-w8-r1"}
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
        assert suite["structural_similarity"] == 0.605625
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_ids.isdisjoint(checkpoints)


def test_e646_neutral_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e646-complete-root-reachability-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.605625
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e647_diagnostic_runs_persist_without_new_checkpoints(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {"e647-root-abstention-trace-r1", "e647-root-abstention-trace-r2"}
    for run_id in run_ids:
        run = readers.run(run_id)
        assert run["provenance"] == "committed"
        suite = run["scoreboard"]["suites"]["ood"]
        assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
        assert suite["structural_similarity"] == 0.605625
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_ids.isdisjoint(checkpoints)


def test_e648_positive_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e648-dynamic-literal-root-probe-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 1.0
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.73545
    assert suite["component_type_recall"] == 0.875
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e649_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e649-refresh-action-role-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.75
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["placeholder_fidelity"] == 1.0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e650_positive_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e650-planned-family-role-binding-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 1.0
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["placeholder_fidelity"] == 1.0
    assert suite["placeholder_validity"] == 1.0
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e651_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e651-schema-enum-literal-margin-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.75
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.7629250000000001
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e652_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e652-value-text-role-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.75
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.78235
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e653_positive_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e653-nested-role-ownership-r2"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 1.0
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.769175
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e654_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e654-nested-role-enum-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["placeholder_fidelity"] == 0.95
    assert suite["structural_similarity"] == 0.7629250000000001
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e655_neutral_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e655-direct-role-slot-ownership-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["placeholder_fidelity"] == 1.0
    assert suite["structural_similarity"] == 0.769175
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e656_neutral_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e656-repeated-slot-role-ownership-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["reward_score"] == 0.973
    assert suite["structural_similarity"] == 0.769175
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e657_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e657-combined-role-ownership-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["structural_similarity"] == 0.75125
    assert suite["ast_edge_f1"] == 0.6964285714285714
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e658_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e658-property-role-ownership-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["placeholder_fidelity"] == 0.95
    assert suite["reward_score"] == 0.958
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e659_negative_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e659-property-role-guard-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["meaningful_program_rate"] == 0.75
    assert suite["placeholder_fidelity"] == 0.65
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e666_positive_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e666-schema-enum-finalize-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["placeholder_fidelity"] == 1.0
    assert suite["structural_similarity"] == 0.769175
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e667_neutral_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e667-nested-typed-array-owner-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.769175
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_spa_routes_and_retired_classic_redirect(ro_client: TestClient) -> None:
    """The SPA owns /playground and old classic bookmarks redirect to it."""
    root = ro_client.get("/")
    assert root.status_code == 200 and 'id="root"' in root.text
    # /playground is now a SPA route (the React playground).
    assert 'id="root"' in ro_client.get("/playground").text
    classic = ro_client.get("/playground/classic", follow_redirects=False)
    assert classic.status_code == 308
    assert classic.headers["location"] == "/playground"
    # Client-side deep routes (incl. /runs/<id>) fall through to the SPA shell.
    assert 'id="root"' in ro_client.get("/checkpoints").text
    assert 'id="root"' in ro_client.get("/runs/qx_e70_stability").text


def test_readers_cold_start_fallback(tmp_path) -> None:
    """An empty repo root must never raise; it reports committed provenance."""
    from slm_training.dsl.schema import ExampleRecord, write_jsonl

    write_jsonl(
        tmp_path / "src" / "slm_training" / "resources" / "train_seeds.jsonl",
        [
            ExampleRecord(
                id="example-1",
                prompt="Show the training data",
                openui='root = TextContent(":copy.value")',
                source="fixture",
            )
        ],
    )
    readers = Readers(tmp_path)
    assert readers.scoreboard("quality")["results"] == []
    train = readers.train_data()
    assert train["provenance"] == "committed"
    assert train["version"] == "examples"
    assert train["versions"] == ["examples"]
    assert train["record_count"] == 1
    assert readers.train_records("examples")["records"][0]["id"] == "example-1"
    assert readers.test_data()["provenance"] == "committed"
    assert readers.runs()["provenance"] == "committed"


def test_train_records_supports_browsing_filters_and_pagination(tmp_path) -> None:
    from slm_training.dsl.schema import ExampleRecord, write_jsonl

    path = tmp_path / "outputs" / "train_data" / "v1" / "records.jsonl"
    write_jsonl(
        path,
        [
            ExampleRecord(
                id=f"row-{i}",
                prompt=f"Prompt {i}",
                openui='root = TextContent(":copy.value")',
                source="template" if i % 2 else "layout",
            )
            for i in range(6)
        ],
    )
    readers = Readers(tmp_path)
    assert readers.train_data()["versions"] == ["examples", "v1"]
    page = readers.train_records("v1", offset=2, limit=2)
    assert page["count"] == 6
    assert [row["id"] for row in page["records"]] == ["row-2", "row-3"]
    assert page["sources"] == ["layout", "template"]
    filtered = readers.train_records("v1", source="template", query="Prompt 3")
    assert filtered["count"] == 1
    assert filtered["records"][0]["id"] == "row-3"


def test_preference_data_lists_committed_event_corpora(tmp_path) -> None:
    directory = (
        tmp_path
        / "src/slm_training/resources/data/preference/events-v1"
    )
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "decision_event_corpus",
                "dataset_id": "events-v1",
                "record_count": 12,
                "splits": {"train": 9, "held_out": 3},
                "evidence_kinds": {"constraint_shadow": 12},
                "set_valued_events": 0,
                "content_fingerprint": "abcdef1234567890",
            }
        )
    )
    data = Readers(tmp_path).preference_data()
    assert data["provenance"] == "committed"
    assert data["rows"] == [
        {
            "dataset_id": "events-v1",
            "kind": "exact-state decisions",
            "records": 12,
            "train": 9,
            "held_out": 3,
            "evidence": "constraint_shadow:12",
            "usage": "decoder evidence only",
            "fingerprint": "abcdef123456",
        }
    ]


def test_preference_data_describes_counterfactual_corpora_by_capability(
    tmp_path,
) -> None:
    directory = tmp_path / "src/slm_training/resources/data/preference/events-v1"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "decision_event_corpus",
                "dataset_id": "events-v1",
                "record_count": 12,
                "splits": {"train": 9, "held_out": 3},
                "evidence_kinds": {"counterfactual": 12},
                "content_fingerprint": "abcdef1234567890",
            }
        )
    )

    assert Readers(tmp_path).preference_data()["rows"][0]["usage"] == (
        "semantic preference training"
    )


def test_committed_train_version_is_default_and_browsable(tmp_path) -> None:
    from slm_training.dsl.schema import ExampleRecord, write_jsonl

    vdir = (
        tmp_path
        / "src"
        / "slm_training"
        / "resources"
        / "data"
        / "train"
        / "remediated_roots_judged"
    )
    write_jsonl(
        vdir / "records.jsonl",
        [
            ExampleRecord(
                id="judged-1",
                prompt="A judged prompt/output pair",
                openui='root = TextContent(":copy.value")',
                source="judged",
            )
        ],
    )
    (vdir / "stats.json").write_text('{"record_count": 1}\n', encoding="utf-8")
    (vdir / "manifest.json").write_text("{}\n", encoding="utf-8")

    readers = Readers(tmp_path)
    data = readers.train_data()
    assert data["provenance"] == "committed"
    assert data["version"] == "remediated_roots_judged"
    assert data["record_count"] == 1
    assert readers.train_records(data["version"])["records"][0]["id"] == "judged-1"


def test_run_detail_merges_scoreboard(ro_client: TestClient) -> None:
    board = ro_client.get("/api/scoreboards/quality").json()["results"]
    # The committed matrix can contain metadata-only rows before the actual
    # suite scoreboards; choose a row whose suites can produce gate output.
    row = next(row for row in board if "suites" in row)
    run_id = row.get("run_id") or row.get("id")
    detail = ro_client.get(f"/api/runs/{run_id}").json()
    assert detail["scoreboard"] is not None
    assert detail["scoreboard"]["matrix"] == "quality"
    # gates are derived from the scoreboard suites even with an empty outputs/.
    assert detail["gates"] is not None and "pass" in detail["gates"]


def test_run_detail_missing_is_graceful(ro_client: TestClient) -> None:
    detail = ro_client.get("/api/runs/nope_xyz").json()
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"] is None
    assert detail["gates"] is None


def test_research_evidence_and_autoresearch_run_are_current(tmp_path) -> None:
    design = tmp_path / "docs" / "design"
    design.mkdir(parents=True)
    run_dir = (
        tmp_path / "outputs" / "autoresearch" / "e9-current" / "runs" / "e9-run"
    )
    run_dir.mkdir(parents=True)
    suites = {
        "smoke": {
            "n": 3,
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.25,
            "structural_similarity": 0.4,
            "placeholder_fidelity": 0.5,
            "reward_score": 0.6,
        }
    }
    (design / "iter-e9-current-20260716.json").write_text(
        json.dumps(
            {
                "campaign": "E9 current experiment",
                "date_utc": "2026-07-16",
                "run_id": "e9-run",
                "train_result": {"trace_id": "a" * 32},
                "suites": suites,
                "ship_gates": {"pass": False},
                "agentv": {"total": 5, "passed": 1},
                "scoreboard": "outputs/autoresearch/e9-current/runs/e9-run/scoreboard.json",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train_summary.json").write_text(
        json.dumps({"run_id": "e9-run", "steps": 12, "last_loss": 1.25}),
        encoding="utf-8",
    )
    (run_dir / "train_telemetry.json").write_text(
        json.dumps({"spans": {"forward": {"pct": 75.0}}}), encoding="utf-8"
    )
    (run_dir / "trace.json").write_text(
        json.dumps({"trace_id": "a" * 32}), encoding="utf-8"
    )
    (run_dir / "gates.json").write_text(
        json.dumps({"pass": False, "failures": ["smoke:meaningful_program_rate"]}),
        encoding="utf-8",
    )

    readers = Readers(tmp_path)
    research = readers.scoreboard("research")
    assert research["results"][0]["run_id"] == "e9-run"
    assert research["results"][0]["agentv"] == {"total": 5, "passed": 1}
    assert any(row["run_id"] == "e9-run" for row in readers.runs()["runs"])
    detail = readers.run("e9-run")
    assert detail["provenance"] == "live"
    assert detail["train_summary"]["steps"] == 12
    assert detail["telemetry"]["spans"]["forward"]["pct"] == 75.0
    assert detail["trace"]["trace_id"] == "a" * 32
    assert detail["scoreboard"]["suites"] == suites
    # Headline comparisons use meaningful output, not syntax-only parse success,
    # keyed by the ship-gate policy's lever names.
    comparisons, _ = readers._performance_rows([])
    current = next(row for row in comparisons if row["run_id"] == "e9-run")
    assert current["metrics"]["meaningful_program_rate"] == 0.25


def test_research_evidence_accepts_nested_train_and_evaluation(tmp_path) -> None:
    design = tmp_path / "docs" / "design"
    design.mkdir(parents=True)
    suites = {"smoke": {"n": 3, "meaningful_program_rate": 1 / 3}}
    (design / "iter-e230-diverse-roots-20260716.json").write_text(
        json.dumps(
            {
                "campaign": "E230 diverse judged generation roots",
                "date": "2026-07-16",
                "train": {
                    "run_id": "e230-diverse-roots-32step",
                    "path": "outputs/autoresearch/e230/runs/e230-diverse-roots-32step",
                    "trace_id": "b" * 32,
                },
                "evaluation": {
                    "suites": suites,
                    "failed_gates": 4,
                    "agentv": {"total": 5, "passed": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    result = Readers(tmp_path).scoreboard("research")["results"][0]
    assert result["run_id"] == "e230-diverse-roots-32step"
    assert result["pass"] is False
    assert result["suites"] == suites
    assert result["agentv"] == {"total": 5, "passed": 1}
    assert result["trace_id"] == "b" * 32
    assert result["run_dir"].endswith("e230-diverse-roots-32step")


def test_research_evidence_does_not_green_branch_only_result(tmp_path) -> None:
    design = tmp_path / "docs" / "design"
    design.mkdir(parents=True)
    (design / "iter-e9.json").write_text(
        json.dumps(
            {
                "run_id": "e9-run",
                "date": "2026-07-18",
                "suites": {"smoke": {"n": 1, "parse_rate": 1.0}},
                "ship_gates": {"pass": True, "failures": []},
                "reproducibility": {
                    "classification": "branch_only_diagnostic",
                    "current_main_reproduced": False,
                },
            }
        ),
        encoding="utf-8",
    )

    result = Readers(tmp_path).scoreboard("research")["results"][0]
    assert result["raw_gate_pass"] is True
    assert result["pass"] is False
    assert result["claim_class"] == "branch_only_diagnostic"


def test_committed_sde0_evidence_is_visible_on_research_scoreboard() -> None:
    root = Path(__file__).resolve().parents[2]

    result = next(
        row
        for row in Readers(root).scoreboard("research")["results"]
        if row["run_id"] == "sde0-01-e396-baseline"
    )

    assert result["pass"] is False
    assert result["agentv"] == {"total": 5, "passed": 0, "execution_errors": 0}
    assert set(result["suites"]) == {
        "smoke",
        "held_out",
        "adversarial",
        "ood",
        "rico_held",
    }
    assert result["suites"]["rico_held"]["diagnostic_subset"] is True


def test_committed_e498_evidence_is_visible_on_research_scoreboard() -> None:
    root = Path(__file__).resolve().parents[2]

    result = next(
        row
        for row in Readers(root).scoreboard("research")["results"]
        if row["run_id"] == "e498-current-main-slot-component-restore"
    )

    assert result["pass"] is False
    assert result["claim_class"] == "diagnostic"
    assert result["agentv"]["passed"] == 0
    assert result["suites"]["smoke"]["slot_component_applications"] == 20


def test_rl_traces_are_paginated_and_malformed_rows_are_skipped(tmp_path) -> None:
    path = tmp_path / "outputs" / "runs" / "molt-smoke" / "rl_traces.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"run_id": "molt-smoke", "engine": "molt", "rollout_id": f"r-{i}"}
        for i in range(3)
    ]
    path.write_text(
        json.dumps(rows[0])
        + "\nnot-json\n"
        + json.dumps({"run_id": "other"})
        + "\n"
        + json.dumps(rows[1])
        + "\n"
        + json.dumps(rows[2])
        + "\n"
    )
    with TestClient(create_app(execution=False, root=tmp_path)) as client:
        response = client.get("/api/runs/molt-smoke/rl-traces?offset=1&limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert payload["count"] == 1
    assert payload["invalid_rows"] == 2
    assert payload["traces"][0]["rollout_id"] == "r-1"

    missing = Readers(tmp_path).rl_traces("../escape")
    assert missing["provenance"] == "missing"
    assert missing["traces"] == []


# --- capability gate -------------------------------------------------------
def test_read_only_reports_capabilities(ro_client: TestClient) -> None:
    caps = ro_client.get("/api/capabilities").json()
    assert caps["execution"] is False
    assert caps["read_only"] is True
    assert len(caps["jobs"]) >= 5


def test_read_only_blocks_execution(ro_client: TestClient) -> None:
    resp = ro_client.post(
        "/api/jobs",
        json={"job": "build_train_data", "params": {"source": "fixture", "version": "v0"}},
    )
    assert resp.status_code == 403


# --- pure-compute gate endpoint (works even read-only) ---------------------
def test_gates_evaluate_matches_pure_function(ro_client: TestClient) -> None:
    thresholds = {"smoke": {"parse_rate": 0.66}}
    resp = ro_client.post(
        "/api/gates/evaluate", json={"suites": SMOKE_SUITE, "thresholds": thresholds}
    ).json()
    assert resp == evaluate_ship_gates(SMOKE_SUITE, thresholds=thresholds)
    assert resp["pass"] is True


# --- remote dispatch monitoring --------------------------------------------
def test_dispatches_endpoint_shape(ro_client: TestClient) -> None:
    payload = ro_client.get("/api/dispatches").json()
    assert set(payload) >= {"jobs", "remotes", "bucket_url"}
    assert payload["bucket_url"].startswith("https://huggingface.co/")


def test_remote_url_extraction() -> None:
    from slm_training.web.observability import _first_remote_url

    assert _first_remote_url("submitted https://huggingface.co/jobs/x9 ok") == (
        "https://huggingface.co/jobs/x9"
    )
    assert _first_remote_url("trackio https://tk-openui.hf.space/.") == (
        "https://tk-openui.hf.space/"
    )
    assert _first_remote_url("no url here") is None


def test_remote_train_allowlisted_and_safe(tmp_path) -> None:
    assert jobs_mod.JOB_SPECS["remote_train"].kind == "dispatch"
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        assert (
            client.post(
                "/api/jobs",
                json={"job": "remote_train", "params": {"host": "a;rm -rf /", "run_id": "r"}},
            ).status_code
            == 422
        )


# --- execution mode + allowlist (the security boundary) --------------------
def test_allowlist_rejects_unknown_and_malicious(tmp_path) -> None:
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        assert client.get("/api/capabilities").json()["execution"] is True
        assert client.post("/api/jobs", json={"job": "nope", "params": {}}).status_code == 400
        # shell-injection / path-escape attempts are rejected at validation.
        assert (
            client.post(
                "/api/jobs",
                json={"job": "build_train_data", "params": {"source": "x;rm -rf /", "version": "v0"}},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/jobs",
                json={"job": "build_test_data", "params": {"source": "both", "version": "../etc"}},
            ).status_code
            == 422
        )


def test_train_data_job_renders_existing_derivative_controls() -> None:
    argv = jobs_mod.JOB_SPECS["build_train_data"].render_argv(
        {
            "source": "existing",
            "base_version": "v1",
            "version": "v1-derived",
            "synthesizer": "layout",
            "namespace_augment": True,
            "edit_derivatives": False,
            "repairs_per_program": 0,
        }
    )
    assert ["--derive-from", "outputs/data/train/v1/records.jsonl"] == argv[
        argv.index("--derive-from") : argv.index("--derive-from") + 2
    ]
    assert "--namespace-augment" in argv
    assert "--no-edit-derivatives" in argv


def test_job_runs_to_completion(tmp_path, monkeypatch) -> None:
    # Inject a trivial stdlib-module job so the runner is exercised without the
    # full training toolchain. `python -m this` prints and exits 0.
    monkeypatch.setitem(jobs_mod.JOB_SPECS, "_zen", jobs_mod.JobSpec("this"))
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        job = client.post("/api/jobs", json={"job": "_zen", "params": {}}).json()
        job_id = job["id"]
        assert job["status"] in {"queued", "running"}
        status = _await_terminal(client, job_id)
        assert status == "succeeded"
        detail = client.get(f"/api/jobs/{job_id}").json()
        assert detail["returncode"] == 0
        assert len(detail["tail"]) > 0


def test_job_cancel(tmp_path, monkeypatch) -> None:
    # A long-lived stdlib server we can cancel (bound to an ephemeral port).
    monkeypatch.setitem(
        jobs_mod.JOB_SPECS,
        "_serve",
        jobs_mod.JobSpec(
            "http.server", positional=("port",), params={"port": jobs_mod.IntRange(20000, 65000)}
        ),
    )
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        job = client.post("/api/jobs", json={"job": "_serve", "params": {"port": 48231}}).json()
        job_id = job["id"]
        # wait until it is actually running
        for _ in range(50):
            if client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
                break
            time.sleep(0.1)
        client.post(f"/api/jobs/{job_id}/cancel")
        assert _await_terminal(client, job_id) == "cancelled"


def _await_terminal(client: TestClient, job_id: str, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if status in {"succeeded", "failed", "cancelled"}:
            return status
        time.sleep(0.1)
    return "timeout"
