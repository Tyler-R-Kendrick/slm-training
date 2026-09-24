"""Real producer/journal/runtime/CLI boundaries; no live coding-agent claim."""

import hashlib
import json
from pathlib import Path

import pytest

from tests.casefiles import case_values

from scripts.autotrain_pending import data_pending, drain_driver_pending, publish_pending, unresolved_driver_pending
from scripts.autotrain_readiness import resolve_matrix_readiness
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ActivitySpec, contract_digest
from tests.test_scripts.test_autotrain_readiness import compiled, _matrix_fixture  # noqa: F401


def waiting_driver(runtime, common, pending, *, name="driver"):
    from scripts.autotrain_pending import validate_pending

    request = {**common, "operation": "driver", "driver_argv": []}
    runtime.register(ActivitySpec(activity_id=name, family=common["loop_id"], kind="control",
        source_digest=common["source_digest"], environment_digest=common["environment_digest"],
        input_digest=contract_digest(request), output_namespace="attempts/" + name))
    lease = runtime.claim_next(activity_id=name, capabilities={"local_process"})
    directory = runtime.attempt_dir(lease)
    directory.mkdir(parents=True)
    request.update(lease=lease.model_dump(mode="json"), parent_event="parent")
    (directory / "request.json").write_text(json.dumps(request))
    result = {"schema_version": "supervisor_operation/v1", "operation": "driver",
        "request_digest": contract_digest(request), "payload": {"returncode": 10, "pending": pending}}
    path = directory / "result.json"
    path.write_text(json.dumps(result))
    publish_pending(Path(common["root"]), common["loop_id"], pending)
    outcome, wake = validate_pending(pending)
    runtime.finish(lease, outcome=outcome, wake=wake, spent_seconds=.01,
                   outputs={"result.json": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def common_for(compiled, monkeypatch):
    source = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", f"{source}:{source / 'src'}:{source / 'outputs/runs/autonomy-validation/dependencies'}")
    from scripts.merge_verification_evidence import digest, environment_identity

    return {"cwd": str(compiled["cwd"]), "root": str(compiled["cwd"] / "campaigns"),
            "loop_id": "pair", "source_digest": "a" * 64,
            "environment_digest": digest(environment_identity())}


def test_pending_actual_code_producer_dispatches_then_current_probe_wakes_only_exact_driver(compiled, monkeypatch):
    from scripts import autotrain_readiness as owner

    matrix = _matrix_fixture(compiled)
    common = common_for(compiled, monkeypatch)
    original = owner._matrix_contexts
    def broken(*args, **kwargs):
        raise KeyError("injected current code defect")
    monkeypatch.setattr(owner, "_matrix_contexts", broken)
    pending = data_pending(resolve_matrix_readiness(matrix, cwd=compiled["cwd"],
        root=Path(common["root"]), loop_id="pair", minimum=1))
    argv = pending["blocker"]["original_reproducer"]["argv"]
    assert argv[1:3] == ["-m", "scripts.autotrain_readiness_probe"]
    monkeypatch.setattr(owner, "_matrix_contexts", original)
    calls = []
    def repair(_runtime, request, **kwargs):
        calls.append(request)
        assert request["operation"] == "repair"
        assert request["hard_pending"][0]["blocker_code"] == pending["blocker"]["blocker_code"]
        assert request["hard_pending"][0]["affected_activity_id"] == "driver"
        return {"any_healed": True}  # Plumbing only; actual probe owns acceptance.
    store = CampaignStore("runtime", Path(common["root"]) / "loops/pair")
    with ActivityRuntime(store) as runtime:
        waiting_driver(runtime, common, pending)
        unrelated = {**pending, "wake": {**pending["wake"], "identity_digest": "f" * 64}}
        waiting_driver(runtime, common, unrelated, name="unrelated")
        drain_driver_pending(runtime, common, 1, lambda _: None, repair)
        assert runtime.snapshot()["driver"].status == "runnable"
        assert runtime.snapshot()["unrelated"].status == "waiting_dependency"
        verification = runtime.snapshot()["driver-readiness-" + contract_digest(pending)]
        assert verification.attempts == 1 and verification.charged_seconds > 0
        assert len(calls) == 1
        assert not any(e["event_type"] == "experiment_finished" for e in store.verify_event_chain())


def test_fake_heal_original_still_bad_is_not_a_wake_and_replay_does_not_mint_probe(compiled, monkeypatch):
    from scripts import autotrain_controller_repair as dispatcher

    matrix = _matrix_fixture(compiled)
    common = common_for(compiled, monkeypatch)
    monkeypatch.setattr(dispatcher, "dispatch_screening_rebuild", lambda **_: None)
    pending = data_pending(resolve_matrix_readiness(matrix, cwd=compiled["cwd"],
        root=Path(common["root"]), loop_id="pair", minimum=2))
    store = CampaignStore("runtime", Path(common["root"]) / "loops/pair")
    with ActivityRuntime(store) as runtime:
        path = waiting_driver(runtime, common, pending)
        for cycle in range(3):
            drain_driver_pending(runtime, common, cycle, lambda _: None, lambda *a, **k: {"any_healed": True})
        assert runtime.snapshot()["driver"].status == "waiting_dependency"
        verification = runtime.snapshot()["driver-readiness-" + contract_digest(pending)]
        assert verification.attempts == 1
        assert not any(e["event_type"] == "driver_pending_resolved" for e in store.verify_event_chain())
        path.write_text("{}")
        with pytest.raises(ValueError, match="result changed"):
            unresolved_driver_pending(runtime)


@pytest.mark.parametrize("n,report,actual,binding", case_values(__file__, "test_unknown_zero_and_wall_selection_never_runnable"))
def test_unknown_zero_and_wall_selection_never_runnable(n, report, actual, binding):
    from scripts.autotrain_pending import screening_deficit_report
    assert binding in screening_deficit_report(n, report, actual)["binding_constraints"]


@pytest.mark.parametrize("report", [None, {"n_min": 6, "chosen_n": None}])
def test_automatic_fallback_is_unknown_even_with_positive_configured_count(report):
    from scripts.autotrain_pending import screening_deficit_report
    assert screening_deficit_report(3, report, 96, automatic=True)["binding_constraints"] == ["unknown"]
    assert screening_deficit_report(3, None, 96, automatic=False) is None


def test_selector_includes_all_and_only_actual_scheduled_arms(compiled):
    from copy import deepcopy
    from scripts.autotrain_pending import selected_readiness_matrix
    from scripts.autotrain_readiness import _matrix_contexts
    from scripts.autotrain_screening import screening_multi_arm_ids

    matrix = _matrix_fixture(compiled)
    extra = deepcopy(matrix["hypotheses"][1])
    extra["experiment"]["experiment_id"] = "extra"
    extra["experiment"]["knobs"]["design_md_dropout"] = .3
    matrix["hypotheses"].append(extra)
    selected = selected_readiness_matrix(matrix, fitted_candidates=2)
    later, _ = screening_multi_arm_ids(matrix=matrix, control_id="control", recommended_id="candidate",
        fitted_candidates=2, by_id={row["experiment"]["experiment_id"]: Path("unused") for row in matrix["hypotheses"]})
    assert selected["selected_experiment_ids"] == ["control", *later] == ["control", "candidate", "extra"]
    context, _, _ = _matrix_contexts(selected, CampaignStore(matrix["campaign_id"], compiled["cwd"] / "campaigns"),
                                     compiled["cwd"], compiled["cwd"] / "campaigns")
    assert len(context["additional_trainer_configs"]) == 2
    assert "selected_experiment_ids" not in matrix


@pytest.fixture
def local_compiled(monkeypatch):
    """Tiny disposable data under the checkout; canonical read-only Git works.

    Python owners are imported from the candidate, not a copied source release.
    This proves operation/data plumbing, not immutable-release isolation.
    """
    from tempfile import TemporaryDirectory
    directory = Path(__file__).resolve().parents[2] / "outputs/runs"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="data-pending-test-", dir=directory) as scratch:
        yield compiled.__wrapped__(Path(scratch), monkeypatch)


def test_actual_pre_cycle_repairs_data_and_wakes_without_starving_behind_capability(local_compiled, monkeypatch):
    from scripts import autotrain_controller_repair as dispatcher
    from scripts.autotrain_supervisor_operations import run_operation
    from scripts.autotrain_supervision import pre_cycle
    from slm_training.data.readiness_receipt import current_successors
    from scripts.run_autotrain_supervisor import _source_identity

    compiled = local_compiled
    matrix = _matrix_fixture(compiled)
    common = common_for(compiled, monkeypatch)
    common["source_digest"] = _source_identity(compiled["cwd"])
    original = dispatcher.dispatch_screening_rebuild
    monkeypatch.setattr(dispatcher, "dispatch_screening_rebuild", lambda **_: None)
    pending = data_pending(resolve_matrix_readiness(matrix, cwd=compiled["cwd"],
        root=Path(common["root"]), loop_id="pair", minimum=2))
    monkeypatch.setattr(dispatcher, "dispatch_screening_rebuild", original)
    store = CampaignStore("runtime", Path(common["root"]) / "loops/pair")
    logs = []
    with ActivityRuntime(store) as runtime:
        run = runtime.run
        def observed_run(*args, **kwargs):
            result = run(*args, **kwargs)
            logs.append({"stderr": result.stderr, "stdout": result.stdout, "launch_error": result.launch_error})
            return result
        monkeypatch.setattr(runtime, "run", observed_run)
        parked = {"schema_version": "driver_pending/v1", "measurement_complete": False,
            "outcome": "capability", "reason": "unconfigured fixture repair provider",
            "wake": {"predicate": "configured source repair verified", "source": "fixture_provider_grant",
                     "identity_digest": "e" * 64},
            "blocker": {"kind": "repair_harness", "blocker_code": "harness_code_failure",
                        "required_capability": "configured_source_repair"}}
        waiting_driver(runtime, common, parked, name="parked")
        waiting_driver(runtime, common, pending)
        pre_cycle(runtime, common, 1, logs.append, run_operation)
        assert runtime.snapshot()["parked"].status == "waiting_capability"
        assert runtime.snapshot()["driver"].status == "waiting_dependency"
        pre_cycle(runtime, common, 2, logs.append, run_operation)
        assert runtime.snapshot()["driver"].status == "runnable", "\n".join(str(row) for row in logs)
        assert runtime.snapshot()["parked"].status == "waiting_capability"
        assert any(s.status == "succeeded" and "-inspect-" in key for key, s in runtime.snapshot().items())
        assert any(row.get("operation") == "repair" for row in logs)
        assert any(row.get("event") == "driver_readiness_rechecked" and row["ready"] for row in logs)
        activity = CampaignStore(pending["readiness"]["readiness_campaign_id"], Path(common["root"]))
        accepted = current_successors(activity, cwd=compiled["cwd"], loop_id="pair", kind="eval")
        assert len(accepted) == 1
        assert not any(e["event_type"] == "experiment_campaign_locked" for e in activity.verify_event_chain())


def test_continuation_capability_producer_dispatches_repair_without_fabricated_wake(tmp_path, monkeypatch):
    from scripts.autotrain_cycle_context import CycleJournal

    root = tmp_path / "campaigns"
    campaign = CampaignStore("unfinished", root)
    journal = CycleJournal(campaign, {"initial_spent_seconds": 0, "order": []})
    pending = journal.pending("driver_pending_no_progress", capability=True)
    common = common_for({"cwd": tmp_path}, monkeypatch)
    calls = []
    with ActivityRuntime(CampaignStore("runtime", root / "loops/pair")) as runtime:
        waiting_driver(runtime, common, pending)
        def dispatch(_runtime, request, **kwargs):
            calls.append(request)
            return {"any_healed": True}
        drain_driver_pending(runtime, common, 1, lambda _: None, dispatch)
        blocker = calls[0]["hard_pending"][0]
        assert blocker["blocker_code"] == "driver_pending_no_progress"
        assert blocker["campaign_id"] == "unfinished"
        assert blocker["affected_activity_id"] == "driver"
        assert blocker["unmet_predicate"] == pending["wake"]["predicate"]
        assert blocker["original_reproducer"]["argv"][2] == "scripts.autotrain_readiness_probe"
        assert runtime.snapshot()["driver"].status == "waiting_capability"


def test_driver_rebuild_rechecks_actual_final_config_and_records_selection(compiled, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    from scripts import run_autotrain_continuous as driver
    from scripts.autotrain_pending import resolve_screening_matrix

    matrix = _matrix_fixture(compiled)
    original_version = matrix["hypotheses"][0]["experiment"]["knobs"]["eval_version"]
    def rebuild(**kwargs):
        result = deepcopy(matrix)
        for row in result["hypotheses"]:
            row["experiment"]["knobs"]["eval_version"] = kwargs["eval_version"]
        result["hypotheses"][1]["experiment"]["knobs"]["design_md_dropout"] = .4
        return result
    monkeypatch.setattr(driver, "_matrix", rebuild)
    monkeypatch.setattr(driver, "_screening_n_report", lambda *a, **k: (2, {"n_min": 2}))
    monkeypatch.setattr(driver, "_screening_suite_records", lambda *_: 2)
    resolved = {"eval_version": original_version, "successions": []}
    context = {"cwd": compiled["cwd"], "root": compiled["cwd"] / "campaigns", "loop_id": "pair",
        "policy": SimpleNamespace(identity_dict=lambda: {"fixture": "controlled-selection"}),
        "fitted_candidates": 1, "resolved_data": resolved}
    actual, pending = resolve_screening_matrix(matrix, {"eval_version": original_version},
        {"n_min": 2, "binding_constraints": ["suite_volume"]}, context=context)
    assert pending is None
    assert actual["hypotheses"][1]["experiment"]["knobs"]["design_md_dropout"] == .4
    selected = resolved["successions"][0]
    assert selected["selected_experiment_ids"] == ["control", "candidate"]
    assert selected["readiness"]["request_digest"] != selected["resolved_readiness"]["request_digest"]


@pytest.mark.parametrize("binding", ["unknown", "wall_budget"])
def test_driver_wall_or_unknown_dispatches_source_repair_not_data(compiled, monkeypatch, binding):
    from types import SimpleNamespace
    from scripts import autotrain_readiness as readiness
    from scripts.autotrain_pending import resolve_screening_matrix

    matrix = _matrix_fixture(compiled)
    def forbidden(*args, **kwargs):
        pytest.fail("unknown/wall constraint generated evaluation data")
    monkeypatch.setattr(readiness, "resolve_matrix_readiness", forbidden)
    _, pending = resolve_screening_matrix(matrix, {"eval_version": "original"},
        {"n_min": 6, "binding_constraints": [binding]}, context={"cwd": compiled["cwd"],
            "root": compiled["cwd"] / "campaigns", "loop_id": "pair", "fitted_candidates": 1,
            "policy": SimpleNamespace(identity_dict=lambda: {"fixture": "controlled-selection"})})
    assert pending["blocker"]["kind"] == "repair_harness"
    assert pending["blocker"]["original_reproducer"]["argv"][2] == "scripts.autotrain_readiness_probe"


def test_continuation_wake_requires_current_canonical_cursor_reconciliation(local_compiled, monkeypatch):
    from scripts.autotrain_cycle_context import CycleJournal, register
    from scripts.autoresearch_command_cursor import resolved_continuation_grant
    from slm_training.autoresearch.climb_policy import load_climb_policy
    from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
    from tests.test_autoresearch.test_experiment_campaign import _manifest_payload
    from tests.test_autoresearch.test_harness import campaign

    common = common_for(local_compiled, monkeypatch)
    root = Path(common["root"])
    store, journal_store = CampaignStore("cursor", root), CampaignStore("runtime", root / "loops/pair")
    store.initialize(campaign().model_copy(update={"campaign_id": "cursor"}))
    manifest = ExperimentCampaignV1.model_validate(_manifest_payload(campaign_id="cursor", claim_class="fixture"))
    store.lock_experiment_campaign(manifest)
    total_seconds = store.load_campaign().budget.logical_seconds
    value = {"schema_version": "driver_cycle/v1", "campaign_id": "cursor", "total_seconds": total_seconds,
        "initial_spent_seconds": 0, "order": [manifest.experiment_id],
        "arms": {manifest.experiment_id: {"manifest_digest": store.load_experiment_campaign(manifest.experiment_id).manifest_sha256}}, "files": {},
        "policy_sha256": load_climb_policy().sha256,
        "execution_identity": resolved_continuation_grant(local_compiled["cwd"], total_seconds).execution_identity}
    register(store, journal_store, value)
    journal = CycleJournal(store, value)
    journal.state["repair_required"] = "driver_pending_no_progress"
    pending = journal.pending("driver_pending_no_progress", capability=True)
    with ActivityRuntime(journal_store) as runtime:
        waiting_driver(runtime, common, pending)
        def repair(*args, **kwargs):
            return {"any_healed": True}
        drain_driver_pending(runtime, common, 1, lambda _: None, repair)
        assert runtime.snapshot()["driver"].status == "waiting_capability"
        # Controller fixture reconciliation, NOT a worker asserting its own repair.
        # The dispatcher never performs this mutation on the producer's behalf.
        journal.state.pop("repair_required")
        journal.save()
        drain_driver_pending(runtime, common, 2, lambda _: None, repair)
        assert runtime.snapshot()["driver"].status == "waiting_capability"  # Flag clearing alone is not progress.
        from scripts.autoresearch_command_cursor import record_execution_outcome
        from slm_training.autoresearch.schemas import ExperimentOutcome
        digest = value["arms"][manifest.experiment_id]["manifest_digest"]
        record_execution_outcome(store, ExperimentOutcome(campaign_id="cursor", experiment_id=manifest.experiment_id,
            status="failed", exit_code=2, error="operational fixture, not a model score",
            campaign_manifest_sha256=digest), digest, pending=False)
        journal.state.update(index=1, seen=[manifest.experiment_id], arm_exits={manifest.experiment_id: 2}, phase="finalizing")
        journal.save()
        drain_driver_pending(runtime, common, 3, lambda _: None, repair)
        assert runtime.snapshot()["driver"].status == "runnable", [
            json.loads(path.read_text()) for path in journal_store.root.rglob("readiness.json")]
        checked = runtime.snapshot()["driver-readiness-" + contract_digest(pending)]
        assert checked.attempts == 3 and checked.charged_seconds > 0


@pytest.mark.parametrize("remaining,restored", [(3, True), (4, False), (5, False)])
def test_pending_row_wake_uses_canonical_monotone_progress(tmp_path, remaining, restored):
    from scripts.autotrain_readiness_probe import _verified_pending_advance
    from scripts.autoresearch_command_cursor import record_execution_outcome
    from tests.test_autoresearch.test_evaluation_continuation import outcome, pending, MANIFEST

    previous, current = outcome("stopped", pending(n=4)), outcome("stopped", pending(n=remaining))
    store = CampaignStore(current.campaign_id, tmp_path)
    record_execution_outcome(store, current, MANIFEST, pending=True)
    value = {"order": [current.experiment_id], "arms": {current.experiment_id: {"manifest_digest": MANIFEST}}}
    assert _verified_pending_advance(store, value, set(),
        {"last_yield": previous.model_dump(mode="json")},
        {"index": 0, "last_yield": current.model_dump(mode="json")}) is restored


def test_pending_dispatch_round_robin_is_bounded_and_survives_controller_restart(tmp_path, monkeypatch):
    common = common_for({"cwd": tmp_path}, monkeypatch)
    store = CampaignStore("runtime", Path(common["root"]) / "loops/pair")
    pending = {"schema_version": "driver_pending/v1", "measurement_complete": False,
        "outcome": "capability", "reason": "fixture capability absent",
        "wake": {"predicate": "provider restored", "source": "provider_grant", "identity_digest": "e" * 64},
        "blocker": {"kind": "repair_harness", "blocker_code": "harness_code_failure"}}
    with ActivityRuntime(store) as runtime:
        for name in ("a", "b", "c"):
            waiting_driver(runtime, common, pending, name=name)
    calls = []
    def repair(_runtime, request, **kwargs):
        calls.append(request["affected_activity_id"])
        return None  # Unavailable capability; not a successful repair.
    for cycle in range(6):
        with ActivityRuntime(store) as runtime:
            drain_driver_pending(runtime, common, cycle, lambda _: None, repair)
            assert len(calls) == cycle + 1  # At most one remedy per pass.
            assert all(s.status == "waiting_capability" for s in runtime.snapshot().values())
    assert calls == ["a", "b", "c", "a", "b", "c"]


def test_pre_cycle_missing_source_verification_grant_does_not_starve_actual_inspection(local_compiled, monkeypatch):
    from scripts.autotrain_supervision import pre_cycle
    from scripts.autotrain_supervisor_operations import run_operation
    from scripts.run_autotrain_supervisor import _source_identity
    from tests.test_scripts.test_autotrain_verification import request

    common = common_for(local_compiled, monkeypatch)
    common["source_digest"] = _source_identity(local_compiled["cwd"])
    store = CampaignStore("runtime", Path(common["root"]) / "loops/pair")
    dependency = {"schema_version": "repair_verification_dependency/v1", "grant": None,
        "wake": {"predicate": "complete_current_source_verification", "source": "source_verification_completed",
                 "identity_digest": "e" * 64}}
    observations = []
    with ActivityRuntime(store) as runtime:
        request(runtime, dependency, "source-repair")
        result = pre_cycle(runtime, common, 1, observations.append, run_operation)
        assert result is not None and "report" in result, observations
        assert runtime.snapshot()["source-repair"].status == "waiting_dependency"
        assert any(row.get("event") == "source_verification_wait"
                   and row["status"] == "waiting_capability" for row in observations)
        assert any(s.status == "succeeded" and "-inspect-" in key for key, s in runtime.snapshot().items())
        assert not any(row["event_type"] == "source_verification_completed" for row in store.verify_event_chain())
