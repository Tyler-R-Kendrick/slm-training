"""Operational locked-pair repair and completion regressions; no train/eval."""

import threading
import json
from types import SimpleNamespace

import pytest

from scripts import autotrain_supervision, autotrain_locked_diagnostic as diagnostic
from slm_training.autoresearch.storage import CampaignStore
from tests.test_autoresearch.test_locked_prereg_supervisor import _fixture


def test_locked_supervisor_rejects_nonzero_driver_with_stale_completion(tmp_path, monkeypatch):
    plan, path, root, _store, _commands = _fixture(tmp_path, monkeypatch)
    args = SimpleNamespace(loop_id=plan["campaign_id"], root=root, max_cycles=1,
                           stop_after_pass=None, locked_preregistration=path)
    runtime = SimpleNamespace(
        store=CampaignStore("runtime", root / "loops" / plan["campaign_id"]),
        cancel_event=threading.Event())
    monkeypatch.setattr(autotrain_supervision, "pre_cycle", lambda *_: {
        "campaign_id": plan["campaign_id"]})
    monkeypatch.setattr(autotrain_supervision, "handle_pending", lambda *_: "run")
    monkeypatch.setattr(autotrain_supervision, "_driver_argv", lambda *_: ["fixture-driver"])
    def operation(_runtime, request, **_kwargs):
        assert request["operation"] == "driver"
        return {"returncode": 1, "campaign_id": plan["campaign_id"],
                "completion": {"campaign_id": plan["campaign_id"]}}
    with pytest.raises(ValueError, match="did not exit successfully"):
        autotrain_supervision.supervise(args, runtime, {"root": str(root)},
            run_operation=operation, watchdog=lambda **_: None)


def test_locked_pre_cycle_propagates_source_blocker(tmp_path, monkeypatch):
    plan, _path, root, _store, _commands = _fixture(tmp_path, monkeypatch)
    runtime = SimpleNamespace()
    from scripts import autotrain_verification
    monkeypatch.setattr(autotrain_verification, "drain_source_verification",
                        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("typed source blocker")))
    with pytest.raises(ValueError, match="typed source blocker"):
        autotrain_supervision.pre_cycle(runtime, {"locked_diagnostic": True}, 1,
                                        lambda *_: None, lambda *_a, **_k: None)


def test_document_handoff_is_not_dispatched_as_repair():
    assert diagnostic.locked_repair_rows([{"kind": "document"}]) == []


@pytest.mark.parametrize("kind", ["next_experiment", "retry_measurement", "rebuild_data"])
def test_locked_scientific_actions_do_not_false_clear(kind):
    with pytest.raises(ValueError, match="scientific plan"):
        diagnostic.locked_repair_rows([{"kind": kind, "action_sha256": "a" * 64}])


def test_locked_driver_pending_dispatches_environment_repair(tmp_path, monkeypatch):
    from scripts import autotrain_pending
    job = {"activity_id": "fixture-activity", "pending_digest": "a" * 64,
           "payload": {"reason": "AgentV SDK unavailable", "campaign_id": "fixture-campaign",
                       "wake": {"predicate": "AgentV dependency available"},
                       "blocker": {"kind": "heal_postcondition_failed",
                                   "blocker_code": "npm_bridge_unavailable",
                                   "reason": "AgentV SDK unavailable"}}}
    monkeypatch.setattr(autotrain_pending, "next_driver_pending", lambda *_: [job])
    seen = []
    autotrain_pending.drain_driver_pending(
        SimpleNamespace(), {"loop_id": "fixture-campaign"}, 1, lambda *_: None,
        lambda _rt, request, **_kw: seen.append(request), locked_diagnostic=True,
    )
    assert [request["operation"] for request in seen] == ["repair"]
    assert seen[0]["hard_pending"][0]["blocker_code"] == "npm_bridge_unavailable"


def test_locked_driver_continuation_rechecks_without_repair(monkeypatch):
    from scripts import autotrain_pending, autotrain_readiness_wake
    job = {"activity_id": "fixture-activity", "pending_digest": "a" * 64,
           "payload": {"reason": "cursor continuation", "campaign_id": "fixture-campaign",
                       "wake": {"predicate": "cursor reconciled"},
                       "blocker": {"kind": "driver_continuation",
                                   "required_capability": "driver_continuation_reconciliation"}}}
    monkeypatch.setattr(autotrain_pending, "next_driver_pending", lambda *_: [job])
    seen = []
    def bind(_common, bound_job, blocker):
        bound_job["probe"] = {"input_kind": "driver_pending_inputs"}
        seen.append("bind")
        return blocker
    monkeypatch.setattr(autotrain_pending, "bind_continuation_probe", bind)
    monkeypatch.setattr(autotrain_readiness_wake, "recheck_driver_pending",
                        lambda *_a, **_k: seen.append("recheck"))
    autotrain_pending.drain_driver_pending(
        SimpleNamespace(), {"loop_id": "fixture-campaign"}, 1, lambda *_: None,
        lambda *_a, **_k: seen.append("repair"), locked_diagnostic=True)
    assert seen == ["bind", "recheck"]


@pytest.mark.parametrize("candidate_nll, expected_positive", [(1.8, True), (2.1, False)])
def test_diagnostic_requires_complete_pair_and_reports_canonical_verdict(
    tmp_path, monkeypatch, candidate_nll, expected_positive,
):
    plan, path, root, store, _ = _fixture(tmp_path, monkeypatch)
    ids = [plan["arms"][role]["run_id"] for role in ("control", "candidate")]
    value = {"order": ids, "loop_id": plan["campaign_id"], "cycle": 1,
             "upstream": plan["source_commit"], "integration": plan["source_commit"],
             "effective_primary": "smoke.eval_nll", "preregistration_path": str(path)}
    state = {"arm_exits": {ids[0]: 0, ids[1]: 1}}
    journal = SimpleNamespace(value=value, state=state, store=store)
    from scripts import run_autotrain_continuous as continuous
    owner = SimpleNamespace(_classify_positive=continuous._classify_positive,
                            _created_checkpoint_paths=lambda *_: ())
    with pytest.raises(ValueError, match="successful arm exits"):
        diagnostic.finalize_diagnostic(journal, owner)
    assert not (store.root / "sdlc_delivery.json").exists()
    state["arm_exits"][ids[1]] = 0
    with pytest.raises(ValueError, match="comparison incomplete"):
        diagnostic.finalize_diagnostic(journal, owner)
    for eid, nll in zip(ids, (2.0, candidate_nll), strict=True):
        run = store.root / "runs" / eid
        run.mkdir(parents=True)
        (run / "scoreboard.json").write_text(json.dumps({"suites": {"smoke": {
            "document_n": 6, "completed_document_n": 6,
            "incomplete_document_n": 0, "decode_timeout_count": 0,
            "eval_nll": nll, "parse_rate": 1.0}}}))
        (run / "eval_smoke.json").write_text(json.dumps({
            "eval_nll": nll, "parse_rate": 1.0}))
        (run / "eval_nll_records.json").write_text(json.dumps({
            "schema": "eval_nll_records/v1", "definition_hash": "b" * 64,
            "records": {f"record-{i}": nll for i in range(6)},
            "selection": {"selected_record_ids": [f"record-{i}" for i in range(6)],
                          "selected_root_ids": plan["inputs"]["selected_root_ids"],
                          "input_sha256s": plan["inputs"]["input_sha256s"],
                          "selection_sha256": plan["inputs"]["selection_sha256"]},
        }))
    for eid in ids:
        record_path = store.root / "runs" / eid / "eval_nll_records.json"
        record = json.loads(record_path.read_text())
        record["selection"]["selected_record_ids"] = [f"wrong-{i}" for i in range(6)]
        record["records"] = {f"wrong-{i}": candidate_nll for i in range(6)}
        record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="selection differs"):
        diagnostic.finalize_diagnostic(journal, owner)
    for eid, nll in zip(ids, (2.0, candidate_nll), strict=True):
        record_path = store.root / "runs" / eid / "eval_nll_records.json"
        record = json.loads(record_path.read_text())
        record["selection"]["selected_record_ids"] = plan["inputs"]["selected_record_ids"]
        record["records"] = {f"record-{i}": nll for i in range(6)}
        record_path.write_text(json.dumps(record))
    diagnostic.finalize_diagnostic(journal, owner)
    record = json.loads((store.root / "sdlc_delivery.json").read_text())
    assert record["paired_test"]["n_pairs"] == 6
    assert record["measured_effect"] == pytest.approx(2.0 - candidate_nll)
    assert record["minimum_effect"] == 0.02
    assert record["effect_gate_pass"] is expected_positive
    assert record["positive"] is expected_positive
    if expected_positive:
        assert record["paired_test"]["verdict"] == "win"
    assert record["promotion_allowed"] is False and record["stack_layer"] is False
    assert record["claim_class"] == "diagnostic"
    from slm_training.versioning import component_version
    assert record["version_stamp"]["components"]["harness.autoresearch.experiment_campaign"] == component_version("harness.autoresearch.experiment_campaign")
    assert record["version_stamp"]["code_commit"] == plan["source_commit"]
    assert json.loads((store.root / "cycle_handoff.json").read_text())["ship_state"] == "blocked"
