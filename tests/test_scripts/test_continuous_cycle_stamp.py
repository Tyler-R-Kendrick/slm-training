"""The continuous closeout must stamp cycle records with the eval-comparability
version components, so their deltas land in a real cross-version partition of
the evidence ledger instead of the catch-all ``unstamped`` bucket.

Torch-free: exercises only the closeout payload builder and the ledger miner.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values
from tests.test_scripts.test_run_autotrain_continuous import document_closeout as document_closeout

import scripts.run_autotrain_continuous as driver
from slm_training.autoresearch.evidence_ledger import (
    EVAL_KEY_COMPONENTS,
    eval_key_from_stamp,
    extract_observations,
)


def _stamp():
    return {"stamp_schema": "version_stamp/v1", "code_commit": "a" * 40,
            "code_dirty": False, "stamped_at": "2026-08-01T00:00:00Z",
            "components": {key: "retained-v1" for key in EVAL_KEY_COMPONENTS}}


def _handoff() -> SimpleNamespace:
    # `_render_continuous_cycle_docs` only reads these attributes off the
    # handoff; a lightweight stand-in keeps the test independent of the full
    # AutotrainCycleHandoffV1 construction (and of torch).
    return SimpleNamespace(
        cycle_index=7,
        cycle_role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        evidence_class="fixture",
        created_at="2026-08-10T00:00:00Z",
        reasons=[],
    )


def _delivery() -> dict:
    return {
        "positive": False,
        "stack_layer": False,
        "measurement_complete": True,
        "primary_metric": "smoke.structural_similarity",
        "control_metrics": {"smoke.structural_similarity": 0.0575},
        "candidate_metrics": {"smoke.structural_similarity": 0.1742},
        # candidate id token carries the arm slug the miner recovers.
        "reasons": [
            "quality_metric_win: c20260810-demo-c7-container-close "
            "structural_similarity control=0.0575 candidate=0.1742"
        ],
    }


def test_cycle_record_carries_eval_stamp() -> None:
    _md, payload = driver._render_continuous_cycle_docs(
        campaign_id="c20260810-demo-c7",
        loop_id="loop-demo",
        handoff=_handoff(),
        delivery=_delivery(),
        measurement_version_stamp=_stamp(),
    )
    assert payload["schema"] == "continuous_cycle_results/v1"
    hill = payload.get("hillclimb")
    assert isinstance(hill, dict)
    assert "went_well" in hill and "went_wrong" in hill and "speculate" in hill
    stamp = payload.get("version_stamp")
    assert isinstance(stamp, dict), "cycle record must carry a version_stamp"
    assert stamp.get("stamp_schema")
    assert stamp == _stamp()
    assert stamp["stamped_at"] != _handoff().created_at
    for cid in EVAL_KEY_COMPONENTS:
        assert cid in stamp["components"], f"missing eval component {cid}"

    key = eval_key_from_stamp(payload)
    assert key is not None, "stamped record must resolve a non-null eval_key"
    for cid in EVAL_KEY_COMPONENTS:
        assert f"{cid}=" in key


def test_stamped_record_mines_into_a_real_partition() -> None:
    """End-to-end: the closeout payload, fed to the ledger miner, produces an
    observation whose eval_key is the stamped partition — never ``unstamped``."""
    _md, payload = driver._render_continuous_cycle_docs(
        campaign_id="c20260810-demo-c7",
        loop_id="loop-demo",
        handoff=_handoff(),
        delivery=_delivery(),
        measurement_version_stamp=_stamp(),
    )
    observations = extract_observations(payload, source="test")
    assert observations, "miner should recover an observation from the record"
    obs = observations[0]
    assert obs.slug == "container-close"
    assert obs.eval_key is not None and obs.eval_key != "unstamped"
    assert obs.eval_key == eval_key_from_stamp(payload)
    # the delta is the real screening delta, carried into the right partition.
    assert obs.delta is not None and abs(obs.delta - (0.1742 - 0.0575)) < 1e-9


@pytest.mark.parametrize("stamp", [None, {}, {**_stamp(), "components": {}}])
def test_cycle_closeout_rejects_unavailable_version_provenance(stamp):
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable

    with pytest.raises(MeasurementProvenanceUnavailable):
        driver._render_continuous_cycle_docs(
            campaign_id="c20260810-demo-c7", loop_id="loop-demo",
            handoff=_handoff(), delivery=_delivery(), measurement_version_stamp=stamp)


def test_measurement_unchanged_across_controller_registries(monkeypatch):
    from slm_training.harness_core import versioning

    payloads = []
    for version in ("controller-v1", "controller-v999"):
        monkeypatch.setattr(versioning, "load_registry", lambda: {"components": {
            key: {"version": version} for key in EVAL_KEY_COMPONENTS}})
        delivery = {**_delivery(), "schema": "autotrain_sdlc_delivery/v1",
                    "version_stamp": {**_stamp(), "components": dict.fromkeys(EVAL_KEY_COMPONENTS, version)}}
        _, payload = driver._render_continuous_cycle_docs(
            campaign_id="c20260810-demo-c7", loop_id="loop-demo", handoff=_handoff(),
            delivery=delivery, measurement_version_stamp=_stamp())
        assert payload["controller_provenance"]["version_stamp"] == delivery["version_stamp"]
        payloads.append(payload)
    assert payloads[0]["version_stamp"] == payloads[1]["version_stamp"] == _stamp()
    assert [o.eval_key for o in extract_observations(payloads[0], source="test")] == [
        o.eval_key for o in extract_observations(payloads[1], source="test")]


def _measurement_workload(store, tmp_path):
    from slm_training.harness_core.execution_release import _prepare_release
    from tests.test_autoresearch.test_harness import campaign

    source = tmp_path / "source"
    registry = source / "src/slm_training/resources/versions.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"schema": "version_registry/v1", "components": {
        key: {"version": "retained-v1"} for key in EVAL_KEY_COMPONENTS}}))
    execution = tmp_path / "execution"
    manifest = _prepare_release(source, (tmp_path / "release", execution, tmp_path / "outputs"),
        lambda: {"integration_commit": "a" * 40, "upstream_commit": "a" * 40, "code_dirty": False})
    spec = campaign().model_copy(update={"campaign_id": store.campaign_id})
    store.initialize(spec)
    locked = {"schema_version": "driver_cycle/v1", "campaign_id": store.campaign_id,
        "loop_id": "measurement-loop", "cwd": str(execution), "order": ["control", "candidate"],
        "arms": {arm: {} for arm in ("control", "candidate")},
        "total_seconds": spec.budget.logical_seconds, "initial_spent_seconds": 0,
        "publication_source": {"source_digest": manifest["source_digest"], "commit": "a" * 40}}
    artifact = store.write_artifact("driver_cycle_inputs", locked)
    store.append_event("driver_cycle_locked", artifact_sha256=artifact.stem)


@pytest.fixture
def measured_pair(tmp_path):
    from slm_training.autoresearch.storage import CampaignStore
    from scripts.autotrain_ledgers import publish_cycle_delivery
    from tests.test_autoresearch.test_experiment_campaign import _manifest

    store = CampaignStore("measured", tmp_path)
    _measurement_workload(store, tmp_path)
    locks = {arm: store.lock_experiment_campaign(_manifest(
        campaign_id=store.campaign_id, experiment_id=arm, source_commit="a" * 40))
        for arm in ("control", "candidate")}
    def publish(arm, stamp, *, exit_code=0, score=None):
        scoreboard = {**(score or {}), "measurement_complete": True, "version_stamp": stamp,
                      "suites": (score or {}).get("suites", {"smoke": {"score": 1}}),
                      "checkpoint_sha256": arm, "eval_data_manifest_sha": "data"}
        run = store.root / "runs" / arm
        run.mkdir(parents=True, exist_ok=True)
        (run / "scoreboard.json").write_text(json.dumps(scoreboard))
        outcome = {"campaign_id": store.campaign_id, "experiment_id": arm,
                   "campaign_manifest_sha256": locks[arm].manifest_sha256, "stage_telemetry": [{
                       "command": ["python", "-m", "scripts.evaluate_model", "--ship-gates"],
                       "exit_code": exit_code, "parsed_output": scoreboard}]}
        artifact = store.write_artifact("outcomes", outcome)
        store.append_event("experiment_finished", experiment_id=arm, artifact_sha256=artifact.stem,
                           detail={"campaign_manifest_sha256": locks[arm].manifest_sha256})
    publish("control", _stamp())
    publish("candidate", {**_stamp(), "stamped_at": "2026-08-02T00:00:00Z"})
    delivery = publish_cycle_delivery(tmp_path, {"schema": "autotrain_sdlc_delivery/v1",
        "campaign_id": store.campaign_id, "loop_id": "measurement-loop", "positive": False,
        "measurement_complete": True, "arm_order": ["control", "candidate"],
        "control_metrics": {"smoke.score": 1}, "candidate_metrics": {"smoke.score": 1}})
    return store, publish, delivery


def test_resolve_retains_actual_stamp(measured_pair):
    from scripts.autotrain_docs import resolve_measurement_provenance
    store, _, delivery = measured_pair
    stamp, proof = resolve_measurement_provenance(store, delivery)
    assert stamp == _stamp()
    assert proof["arm_stamps"]["candidate"]["stamped_at"] == "2026-08-02T00:00:00Z"


def test_contradictory_arms_refused(measured_pair):
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable, resolve_measurement_provenance
    store, publish, delivery = measured_pair
    changed = copy.deepcopy(_stamp())
    changed["components"]["evals.scoring"] = "contradiction"
    publish("candidate", changed)
    with pytest.raises(MeasurementProvenanceUnavailable, match="arm_stamps_conflict"):
        resolve_measurement_provenance(store, delivery)


def test_uncommitted_scoreboard_change_refused(measured_pair):
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable, resolve_measurement_provenance
    store, _, delivery = measured_pair
    path = store.root / "runs/control/scoreboard.json"
    value = json.loads(path.read_text())
    value["version_stamp"]["components"]["gates.ship"] = "forged"
    path.write_text(json.dumps(value))
    with pytest.raises(MeasurementProvenanceUnavailable, match="stage_scoreboard_mismatch"):
        resolve_measurement_provenance(store, delivery)


def test_missing_gate_requires_workload_proof(measured_pair, monkeypatch):
    from scripts import autotrain_cycle_lock
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable, resolve_measurement_provenance
    store, publish, delivery = measured_pair
    stamp = _stamp()
    del stamp["components"]["gates.ship"]
    for arm in delivery["arm_order"]:
        publish(arm, stamp)
    monkeypatch.setattr(autotrain_cycle_lock, "load_context", lambda store: None)
    with pytest.raises(MeasurementProvenanceUnavailable, match="workload_lock_missing"):
        resolve_measurement_provenance(store, delivery)


@pytest.mark.parametrize("drift", [False, True])
def test_gate_version_from_verified_workload_only(measured_pair, monkeypatch, drift):
    from scripts.autotrain_cycle_lock import load_context
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable, resolve_measurement_provenance
    from slm_training.harness_core import versioning
    from pathlib import Path

    store, publish, delivery = measured_pair
    monkeypatch.setattr(versioning, "load_registry", lambda: pytest.fail("controller registry consulted"))
    stamp = _stamp()
    del stamp["components"]["gates.ship"]
    for arm in delivery["arm_order"]:
        publish(arm, stamp)
    if drift:
        (Path(load_context(store)["cwd"]) / "src/slm_training/resources/versions.json").write_text("{}")
        with pytest.raises(MeasurementProvenanceUnavailable):
            resolve_measurement_provenance(store, delivery)
    else:
        resolved, proof = resolve_measurement_provenance(store, delivery)
        assert resolved == _stamp()
        assert proof["arm_stamps"]["control"] == stamp


def test_newer_unfinished_attempt_refused(measured_pair):
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable, resolve_measurement_provenance
    store, _, delivery = measured_pair
    store.append_event("experiment_started", experiment_id="control")
    with pytest.raises(MeasurementProvenanceUnavailable, match="newer_attempt_unfinished"):
        resolve_measurement_provenance(store, delivery)


def test_tampered_committed_outcome_refused(measured_pair):
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable, resolve_measurement_provenance
    store, _, delivery = measured_pair
    event = next(row for row in store.verify_event_chain() if row["event_type"] == "experiment_finished")
    path = store.root / "artifacts/outcomes" / (event["artifact_sha256"] + ".json")
    path.write_text("{}")
    with pytest.raises(MeasurementProvenanceUnavailable):
        resolve_measurement_provenance(store, delivery)

def test_document_provenance_wait_preserves_measurement(document_closeout, monkeypatch):
    from scripts import autotrain_docs

    repo, root, loop_id, campaign_id = document_closeout
    def unavailable(*args):
        raise autotrain_docs.MeasurementProvenanceUnavailable("measurement_arm_stamps_conflict")
    monkeypatch.setattr(autotrain_docs, "resolve_measurement_provenance", unavailable)
    camp = root / campaign_id
    before = {name: (camp / name).read_bytes() for name in ("sdlc_delivery.json", "cycle_handoff.json")}
    assert driver._self_heal_document_actions(cwd=repo, root=root, loop_id=loop_id, campaign_id=campaign_id) is None
    events = driver.CampaignStore(campaign_id, root).verify_event_chain()
    assert events[-1]["event_type"] == "documentation_waiting_provenance"
    assert events[-1]["detail"]["reason"] == "measurement_arm_stamps_conflict"
    assert not (camp / "delivery_workspace").exists()
    assert all((camp / name).read_bytes() == content for name, content in before.items())


@pytest.mark.parametrize("field,value", case_values(__file__, "test_delivery_projection_mutation_is_pending"))
def test_delivery_projection_mutation_is_pending(measured_pair, field, value):
    from scripts.autotrain_docs import document_measurement_provenance

    store, _, delivery = measured_pair
    (store.root / "cycle_handoff.json").write_text("{}")
    delivery[field] = value
    (store.root / "sdlc_delivery.json").write_text(json.dumps(delivery))
    assert document_measurement_provenance(store, delivery) is None
    assert store.verify_event_chain()[-1]["detail"]["reason"] == "measurement_delivery_projection_mismatch"


@pytest.mark.parametrize("field", ["code_commit", "evals.scoring", "gates.ship"])
def test_complete_stamps_cannot_claim_other_workload(measured_pair, field):
    from scripts.autotrain_docs import MeasurementProvenanceUnavailable, resolve_measurement_provenance

    store, publish, delivery = measured_pair
    stamp = _stamp()
    if field == "code_commit":
        stamp[field] = "f" * 40
    else:
        stamp["components"][field] = "forged-version"
    for arm in delivery["arm_order"]:
        publish(arm, stamp)
    with pytest.raises(MeasurementProvenanceUnavailable, match="source_mismatch|registry_conflict"):
        resolve_measurement_provenance(store, delivery)


@pytest.mark.parametrize("exit_code", [1, -9, 8, None, False])
def test_failed_or_unproved_stage_is_pending(measured_pair, exit_code):
    from scripts.autotrain_docs import document_measurement_provenance

    store, publish, delivery = measured_pair
    (store.root / "cycle_handoff.json").write_text("{}")
    publish("candidate", _stamp(), exit_code=exit_code)
    assert document_measurement_provenance(store, delivery) is None
    assert store.verify_event_chain()[-1]["detail"]["reason"] == "measurement_stage_scoreboard_mismatch"


def test_canonical_expected_gate_rejection_remains_measurement(measured_pair, tmp_path):
    from scripts.autotrain_docs import resolve_measurement_provenance
    from tests.test_autoresearch.test_harness import _complete_gate_scoreboard

    store, publish, delivery = measured_pair
    scoreboard = _complete_gate_scoreboard(tmp_path / "agentv")
    publish("candidate", _stamp(), exit_code=8, score=scoreboard)
    assert resolve_measurement_provenance(store, delivery)[0] == _stamp()
