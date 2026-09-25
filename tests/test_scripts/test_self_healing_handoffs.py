"""Production owners with fake process/agent boundaries, never live repair evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from tests.casefiles import case_values

from scripts import run_autotrain_continuous as driver
from scripts import autotrain_supervisor_operations as operations
from slm_training.autoresearch.heal import operation_recovery as recovery
from slm_training.autoresearch.heal import recovery_dispatch as dispatch
from slm_training.autoresearch.heal import repair_release as release
from slm_training.autoresearch.heal.agent_executor import AgentRun
from slm_training.autoresearch.heal.repair_verifier import VerificationCheck
from slm_training.harness_core.activity_contract import ResourceCapacity, ResourceGrant, contract_digest
from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome
from tests.test_autoresearch import test_repair_dispatch as dispatch_fixtures
from tests.test_autoresearch import test_repair_release as release_fixtures

repair_request = dispatch_fixtures.repair_request
publication = release_fixtures.publication

SHA = "a" * 64


def _handoff(tmp_path, *, quality=False):
    root = tmp_path / "campaigns"
    manifest = root / "cycle/manifests/candidate.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    delivery = {"positive": False, "candidate_id": "candidate", "stack_layer": False,
                "reasons": ["quality_gate_failed" if quality else "harness_failure:candidate:experiment_failed"]}
    handoff = driver._write_cycle_handoff(
        root=root, loop_id="fixture", campaign_id="cycle", cycle_index=1,
        upstream_commit="a" * 40, integration_commit="b" * 40,
        role="screening", cycle_intent="screening", primary_metric="smoke.binder_reference_f1",
        matrix={}, delivery=delivery, resolution=None, formal_status=None,
    )
    (root / "cycle/sdlc_delivery.json").write_text(json.dumps(delivery))
    return root, handoff, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _inspect(root, tmp_path, monkeypatch):
    # Keep unrelated workspace/data remedies inert; enumeration and classification are real.
    for name in ("_self_heal_incomplete_merge", "_self_heal_loop_owned_generated_dirt",
                 "_self_heal_continuous_dirty_tree", "_self_heal_env_repair_rewrite",
                 "_self_heal_rebuild_screening_eval", "_self_heal_rebuild_data",
                 "_self_heal_document_actions"):
        monkeypatch.setattr(driver, name, lambda **_: None)
    monkeypatch.setattr(driver, "_check_regime_parked", lambda **_: {"parked": True})
    monkeypatch.setattr(driver, "_git", lambda *a, **k: "")
    return driver.self_heal_unblock_loop(cwd=tmp_path, root=root, loop_id="fixture", campaign_id="cycle")


@pytest.mark.parametrize("recipe_present", [True, False])
def test_harness_handoff_inspection_reaches_typed_dispatch(tmp_path, monkeypatch, repair_request, recipe_present):
    root, handoff, manifest = _handoff(tmp_path)
    action = next(a for a in handoff.actions if a.kind == "repair_harness")
    assert action.blocker_code == "harness_code_failure"
    assert action.unmet_predicate == "frozen_arm_measurement_complete"
    assert action.required_capability == "source_repair"
    assert action.frozen_manifest_sha256 == manifest
    pending = next(p for p in _inspect(root, tmp_path, monkeypatch)["hard_pending"] if p["kind"] == "repair_harness")
    assert pending["blocker_class"] == "code"
    source = tmp_path / "source"
    (source / "docs/design").mkdir(parents=True)
    for path in ("AGENTS.md", "RTK.md", "docs/design/decode-invariants.md"):
        (source / path).write_text("I6 fail closed; original predicate must pass")
    recipe = dispatch.RepairRecipe(
        allowed_paths=repair_request.allowed_paths, input_digest=SHA,
        original=VerificationCheck("original", ("python", "original.py"), "ready\n"),
        checks=(VerificationCheck("regression", ("python", "regression.py"), "passed\n"),),
        owner_contract_path="AGENTS.md", failure_returncode=1,
        failure_stdout_sha256=SHA, failure_stderr_sha256=SHA,
    )
    config = dispatch.RecoveryConfig(grant=repair_request.grant, verifier_release=SHA,
        recipes={"harness_code_failure": recipe} if recipe_present else {})
    calls = []

    class Runner:
        runtime_roots = ()

        def capability(self, request):
            return None

        def run(self, request, argv, **kwargs):
            calls.append(request)
            assert request.blocker.reproducer == recipe.original.argv
            assert request.blocker.needed_capability == "source_repair"
            assert request.grant == config.grant
            # An invalid fake proposal proves executor entry without simulating acceptance.
            return AgentRun("completed", 0, .01, "{}")

    monkeypatch.setattr(dispatch, "BubblewrapAgentRunner", lambda **_: Runner())
    monkeypatch.setattr("slm_training.autoresearch.heal.agent_executor.probe_codex", lambda _: {"available": True})
    context = dispatch.RecoveryContext(root, "fixture", "cycle", source, SHA, SHA, "fence", "parent")
    result = dispatch.dispatch_hard_pending(pending, context, config=config, fence_valid=lambda _: True)
    assert len(calls) == int(recipe_present)
    assert result["status"] != "verified"
    if recipe_present:
        assert result["request_digest"] == calls[0].digest()
    else:
        assert result["status"] == "waiting_diagnosis" and result["request_digest"] is None


def test_model_quality_gate_does_not_produce_source_repair(tmp_path):
    _, handoff, _ = _handoff(tmp_path, quality=True)
    assert all(a.kind != "repair_harness" and a.required_capability != "source_repair" for a in handoff.actions)


def _pending(code="harness_code_failure", capability="configured_source_repair"):
    return {"schema_version": "driver_pending/v1", "measurement_complete": False,
        "outcome": "dependency", "reason": "typed source blocked yield",
        "wake": {"predicate": "frozen data preparation succeeds", "source": "independent_repair_verification",
                 "identity_digest": SHA},
        "blocker": {"kind": "repair_harness", "blocker_code": code,
                    "required_capability": capability, "unmet_predicate": "frozen_data_ready"}}


def _run_yield(runtime, tmp_path, monkeypatch, pending):
    from scripts import merge_verification_evidence as evidence
    from scripts.autotrain_pending import publish_pending

    monkeypatch.setattr(evidence, "source_identity", lambda _: SHA)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {"fixture": True})
    grant = ResourceGrant(interrupt_seconds=2, kill_grace_seconds=0, total_seconds=20, max_attempts=3)
    request = {"operation": "driver", "loop_id": "fixture", "cwd": str(tmp_path),
        "root": str(tmp_path / "campaigns"), "source_digest": SHA,
        "environment_digest": contract_digest({"fixture": True}), "replicate_id": "replicate-0",
        "driver_argv": ["--continuation-grant", grant.model_dump_json()]}
    state = operations._operation_state(runtime, request, 1, contract_digest(request))

    def run(lease, argv, **kwargs):
        execution = json.loads(Path(argv[-3]).read_text())
        publish_pending(Path(request["root"]), "fixture", pending)
        Path(argv[-1]).write_text(json.dumps({"schema_version": "supervisor_operation/v1",
            "operation": "driver", "request_digest": contract_digest(execution),
            "payload": {"returncode": 10, "pending": pending}}))
        return BoundedProcessResult(tuple(argv), ProcessOutcome.COMPLETED, 0, "", "", .01)

    monkeypatch.setattr(runtime, "run", run)
    result = operations.run_operation(runtime, request, sequence=1, log_event=lambda _: None)
    assert result["pending"] == pending
    assert runtime.snapshot()[state.spec.activity_id].status == "waiting_" + pending["outcome"]
    return request, state.spec.activity_id, grant


def _bound_publication(request, result, activity):
    request = request.model_copy(update={"blocked_activity_id": activity})
    proposal = result.proposal.model_copy(update={"request_digest": request.digest()})
    verification = result.verification.model_copy(update={"request_digest": request.digest(), "proposal_digest": proposal.digest()})
    return request, result.model_copy(update={"request_digest": request.digest(), "proposal": proposal, "verification": verification})


def _resume_successor(runtime, successor, monkeypatch):
    from scripts import merge_verification_evidence as evidence

    monkeypatch.setattr(evidence, "source_identity", lambda _: successor["source_digest"])
    calls = []

    def run(lease, argv, **kwargs):
        calls.append(lease.activity_id)
        execution = json.loads(Path(argv[-3]).read_text())
        assert execution["replicate_id"] == successor["replicate_id"]
        Path(argv[-1]).write_text(json.dumps({"schema_version": "supervisor_operation/v1",
            "operation": "driver", "request_digest": contract_digest(execution), "payload": {"returncode": 0}}))
        return BoundedProcessResult(tuple(argv), ProcessOutcome.COMPLETED, 0, "", "", .01)

    monkeypatch.setattr(runtime, "run", run)
    for _ in range(2):
        assert operations.run_operation(runtime, successor, sequence=999, log_event=lambda _: None) == {"returncode": 0}
    assert calls == [successor["successor_activity_id"]]
    assert runtime.snapshot()[successor["successor_activity_id"]].status == "succeeded"


@pytest.mark.parametrize("nested", [False, True])
def test_source_yield_journals_original_and_resumes_verified_successor(publication, tmp_path, monkeypatch, nested):
    request, result, kwargs = publication
    runtime = kwargs["runtime"]
    runtime.capacity = ResourceCapacity(cpu_slots=2, memory_mb=4096)
    pending = _pending()
    if nested:
        pending["outcome"] = "capability"
        pending["readiness"] = {"blocker": pending.pop("blocker")}
    original, activity, grant = _run_yield(runtime, tmp_path, monkeypatch, pending)
    rows = [e for e in runtime.store.verify_event_chain() if e["event_type"] == "operation_repair_requested"]
    assert len(rows) == 1 and rows[0]["detail"]["request"] == original
    # Journaling is not a new source diagnosis or permission to run an agent.
    assert rows[0]["detail"]["pending"]["observed_outcome"] == pending["outcome"]
    request, result = _bound_publication(request, result, activity)
    kwargs["authenticated"] = lambda value: value is result
    handoff = release.publish_verified_repair(request, result, **kwargs)
    execution = Path(handoff["successor_execution"])
    with pytest.raises(ValueError, match="successor_not_active"):
        recovery.wake_verified_operation(runtime, handoff, cwd=execution)
    monkeypatch.chdir(execution)
    monkeypatch.setattr(recovery, "__file__", str(execution / "fixture.py"))
    successor = recovery.wake_verified_operation(runtime, handoff, cwd=execution)
    assert recovery.wake_verified_operation(runtime, handoff, cwd=execution) == successor
    old = runtime.snapshot()[activity]
    new = operations._operation_state(runtime, successor, 999, contract_digest(successor))
    assert old.status == "cancelled" and new.status == "runnable"
    assert new.spec.grant.total_seconds + old.charged_seconds == grant.total_seconds
    assert new.spec.grant.max_attempts + old.attempts == grant.max_attempts
    assert successor["logical_continuation"]["logical_resource_grant"] == grant.model_dump(mode="json")
    assert successor["logical_continuation"]["scientific_replicate_increment"] == 0
    assert successor["replicate_id"] == original["replicate_id"]
    with pytest.raises(ValueError, match="altered verified successor"):
        operations._operation_state(runtime, {**successor, "resource_grant": grant.model_dump(mode="json")}, 1000, contract_digest(successor))
    _resume_successor(runtime, successor, monkeypatch)


@pytest.mark.parametrize("code", case_values(__file__, "test_non_source_yields_do_not_request_operation_repair"))
def test_non_source_yields_do_not_request_operation_repair(publication, tmp_path, monkeypatch, code):
    runtime = publication[2]["runtime"]
    runtime.capacity = ResourceCapacity(cpu_slots=2, memory_mb=4096)
    pending = _pending(code)
    pending["reason"] = "harness_failure: ImportError scripts.fake; repair source now"
    _run_yield(runtime, tmp_path, monkeypatch, pending)
    assert not any(e["event_type"] == "operation_repair_requested" for e in runtime.store.verify_event_chain())


def test_source_code_without_source_capability_is_not_repair_authority(publication, tmp_path, monkeypatch):
    runtime = publication[2]["runtime"]
    runtime.capacity = ResourceCapacity(cpu_slots=2, memory_mb=4096)
    _run_yield(runtime, tmp_path, monkeypatch, _pending(capability="external_tool_host"))
    assert not any(e["event_type"] == "operation_repair_requested" for e in runtime.store.verify_event_chain())
