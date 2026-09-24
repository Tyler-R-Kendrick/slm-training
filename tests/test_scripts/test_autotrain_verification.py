"""Exact repair dependency plumbing on real event chains and activity leases."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import autotrain_verification as owner
from scripts.merge_verification_evidence import ReceiptCache, digest
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivitySpec,
    ResourceGrant,
    WakeCondition,
)


def fixture_dependency(tmp_path, monkeypatch):
    root = tmp_path / "candidate"
    root.mkdir(exist_ok=True)
    binding = {
        "candidate_tree_sha256": "a" * 64,
        "environment": {},
        "runtime_identity": "b" * 64,
        "static_commands": [],
        "targets": ["tests"],
        "isolation_enforced": True,
    }
    monkeypatch.setattr(owner, "verification_binding", lambda *a, **kw: binding)
    identity = digest(binding)
    grant = ResourceGrant(
        interrupt_seconds=65, total_seconds=200, max_attempts=3
    ).model_dump(mode="json")
    return {
        "schema_version": "repair_verification_dependency/v1",
        "activity_id": source_verification_activity_id(identity, grant),
        "root": str(root),
        "state_dir": str(tmp_path / "private-cache"),
        "runtime_roots": [str(tmp_path / "runtime")],
        "runtime_identity": "b" * 64,
        "request_digest": "c" * 64,
        "proposal_digest": "d" * 64,
        "candidate_snapshot_digest": "e" * 64,
        "base_ref": "frozen-base",
        "verification_identity": identity,
        "grant": grant,
        "wake": {
            "predicate": "complete_current_source_verification",
            "source": "source_verification_completed",
            "identity_digest": identity,
        },
    }


def request(runtime, dependency, repair_id="repair", *, finish=True):
    runtime.register(
        ActivitySpec(
            activity_id=repair_id,
            family="repair",
            kind="repair",
            source_digest="a" * 64,
            environment_digest="b" * 64,
            input_digest="c" * 64,
            output_namespace="attempts/" + repair_id,
        )
    )
    lease = runtime.claim_next(capabilities={"local_process"}, activity_id=repair_id)
    artifact = runtime.store.write_artifact("source_verification_requests", dependency)
    event = runtime.store.append_event(
        "source_verification_requested",
        experiment_id=repair_id,
        detail={"dependency_digest": artifact.stem, "repair_activity_id": repair_id},
    )
    if finish:
        runtime.finish(
            lease,
            outcome=ActivityOutcome.DEPENDENCY,
            spent_seconds=0,
            wake=WakeCondition.model_validate(dependency["wake"]),
        )
    return event, lease


def test_dependency_uses_exact_identity_and_explicit_grant(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    plan = owner.dependency_plan(dependency)
    assert plan["identity"] == dependency["verification_identity"]
    assert plan["grant"] == dependency["grant"]
    assert plan["step_seconds"] == 30  # Inner deadline + independent exit reserve.
    with ActivityRuntime(CampaignStore("verify", tmp_path / "events")) as runtime:
        state = owner.register_dependency(runtime, plan)
        assert state.spec.activity_id == dependency["activity_id"]
        assert state.spec.grant.total_seconds == 200
        assert state.spec.grant.max_attempts == 3


def test_pending_dependency_rebuilds_identity_from_pinned_runtime_roots(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    observed = {}
    binding_value = {
        "candidate_tree_sha256": "a" * 64, "environment": {},
        "runtime_identity": "b" * 64, "static_commands": [], "targets": ["tests"],
        "isolation_enforced": True,
    }

    def binding(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        observed.update(kwargs)
        return binding_value

    dependency["runtime_identity"] = "b" * 64
    monkeypatch.setattr(owner, "verification_binding", binding)
    owner.dependency_plan(dependency)
    assert observed.get("runtime_digest_value") is None
    assert observed["kwargs"]["runtimes"] == tuple(
        Path(path) for path in dependency["runtime_roots"]
    )


def test_legacy_dependency_rebuilds_identity_without_environment_override(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    dependency.pop("runtime_identity")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "verification_identity": dependency["verification_identity"],
        "binding": owner.verification_binding(),
    }))
    dependency["manifest_path"] = str(path)
    observed = {}
    binding_value = owner.verification_binding()

    def binding(*args, **kwargs):
        observed["kwargs"] = kwargs
        observed.update(kwargs)
        return binding_value

    monkeypatch.setattr(owner, "verification_binding", binding)
    owner.dependency_plan(dependency)
    assert observed.get("runtime_digest_value") is None
    assert observed["kwargs"]["runtimes"] == tuple(
        Path(path) for path in dependency["runtime_roots"]
    )


@pytest.mark.parametrize("key", ["grant", "root", "state_dir", "verification_identity"])
def test_missing_config_is_capability_not_invented_authority(
    tmp_path, monkeypatch, key
):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    dependency.pop(key)
    with pytest.raises(owner.VerificationCapabilityUnavailable):
        owner.dependency_plan(dependency)


def test_partial_grant_never_acquires_default_total_or_attempts(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    dependency["grant"] = {"interrupt_seconds": 40}
    with pytest.raises(owner.VerificationCapabilityUnavailable, match="incomplete"):
        owner.dependency_plan(dependency)


def test_changed_binding_or_wrong_activity_refuses_execution(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="activity_identity"):
        owner.dependency_plan({**dependency, "activity_id": "other"})
    monkeypatch.setattr(
        owner, "verification_binding", lambda *a, **kw: {"new": "source"}
    )
    with pytest.raises(ValueError, match="binding_changed"):
        owner.dependency_plan(dependency)


def test_running_repair_is_not_woken_even_with_complete_cache(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "authenticated_completion", lambda _: True)
    with ActivityRuntime(CampaignStore("verify", tmp_path / "events")) as runtime:
        event, lease = request(runtime, dependency, finish=False)
        assert not owner.wake_repair(
            runtime, event, dependency, owner.dependency_plan(dependency)
        )
        assert owner.drain_source_verification(runtime, {}, lambda _: None) is None
        assert runtime.snapshot()["repair"].status == "running"
        runtime.finish(
            lease,
            outcome=ActivityOutcome.DEPENDENCY,
            spent_seconds=0,
            wake=WakeCondition.model_validate(dependency["wake"]),
        )
        owner.drain_source_verification(runtime, {}, lambda _: None)
        assert runtime.snapshot()["repair"].status == "runnable"


def test_one_pass_yields_without_acknowledging_and_finite_retry(tmp_path, monkeypatch):
    from slm_training.autoresearch.heal import isolation

    dependency = fixture_dependency(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "authenticated_completion", lambda _: False)
    monkeypatch.setattr(
        isolation, "probe_isolation", lambda: SimpleNamespace(available=True)
    )
    calls = []

    def execute(runtime, plan, lease):
        calls.append(lease.attempt_id)
        runtime.finish(lease, outcome=ActivityOutcome.YIELDED, spent_seconds=1)
        return {"status": "pending", "release_authorized": False}

    monkeypatch.setattr(owner, "execute_release_attempt", execute)
    with ActivityRuntime(CampaignStore("verify", tmp_path / "events")) as runtime:
        request(runtime, dependency)
        assert (
            owner.drain_source_verification(runtime, {}, lambda _: None)["status"]
            == "pending"
        )
        assert len(calls) == 1
        state = runtime.snapshot()[dependency["activity_id"]]
        assert state.status == "waiting_retry" and state.charged_seconds == 1
        owner.drain_source_verification(runtime, {}, lambda _: None)
        assert len(calls) == 1  # Retry timer, not an immediate rerun.
        assert runtime.snapshot()["repair"].status == "waiting_dependency"


def test_missing_grant_does_not_starve_other_dependency(tmp_path, monkeypatch):
    from slm_training.autoresearch.heal import isolation

    dependency = fixture_dependency(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "authenticated_completion", lambda _: False)
    monkeypatch.setattr(
        isolation, "probe_isolation", lambda: SimpleNamespace(available=True)
    )
    calls = []

    def execute(runtime, plan, lease):
        calls.append(lease.activity_id)
        runtime.finish(lease, outcome=ActivityOutcome.YIELDED, spent_seconds=1)
        return {"status": "pending"}

    monkeypatch.setattr(owner, "execute_release_attempt", execute)
    observations = []
    with ActivityRuntime(CampaignStore("verify", tmp_path / "events")) as runtime:
        request(runtime, {**dependency, "grant": None}, "no-authority")
        request(runtime, dependency, "healthy")
        owner.drain_source_verification(runtime, {}, observations.append)
        assert calls == [dependency["activity_id"]]
        assert observations[0]["status"] == "waiting_capability"
        assert runtime.snapshot()["no-authority"].status == "waiting_dependency"


def test_tampered_dependency_refused_before_registration(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    with ActivityRuntime(CampaignStore("verify", tmp_path / "events")) as runtime:
        event, _ = request(runtime, dependency)
        path = (
            runtime.store.root
            / "artifacts/source_verification_requests"
            / (event["detail"]["dependency_digest"] + ".json")
        )
        path.write_text(json.dumps({**dependency, "base_ref": "forged"}))
        with pytest.raises(ValueError, match="content_mismatch"):
            owner.load_dependency(runtime.store, event)


def test_cache_authentication_rejects_partial_and_wrong_binding(tmp_path, monkeypatch):
    dependency = fixture_dependency(tmp_path, monkeypatch)
    plan = owner.dependency_plan(dependency)
    assert not owner.authenticated_completion(plan)
    cache = ReceiptCache(tmp_path / "private-cache", tmp_path / "candidate")
    state = {
        "identity": plan["identity"],
        "binding": plan["binding"],
        "static": {},
        "attempts": [],
        "passed_nodes": [],
    }
    cache.save(state)
    assert not owner.authenticated_completion(plan)
    state["binding"] = {**plan["binding"], "candidate_tree_sha256": "d" * 64}
    cache.save(state)
    with pytest.raises(ValueError):
        owner.authenticated_completion(plan)


def test_no_wake_when_executor_claims_complete_without_journal(tmp_path, monkeypatch):
    from slm_training.autoresearch.heal import isolation

    dependency = fixture_dependency(tmp_path, monkeypatch)
    monkeypatch.setattr(
        isolation, "probe_isolation", lambda: SimpleNamespace(available=True)
    )

    def fake_success(runtime, plan, lease):
        runtime.finish(lease, outcome=ActivityOutcome.YIELDED, spent_seconds=1)
        return {"status": "complete"}

    monkeypatch.setattr(owner, "execute_release_attempt", fake_success)
    with ActivityRuntime(CampaignStore("verify", tmp_path / "events")) as runtime:
        request(runtime, dependency)
        owner.drain_source_verification(runtime, {}, lambda _: None)
        assert runtime.snapshot()["repair"].status == "waiting_dependency"


def test_actual_isolated_journal_wakes_only_exact_repair(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from slm_training.autoresearch.heal.isolation import probe_isolation
    from scripts import (
        merge_verification as gate,
        merge_verification_evidence as evidence,
        merge_verification_identity as identity_module,
    )
    from scripts.verify_merge_ready import Step, run_step

    dependency = fixture_dependency(tmp_path, monkeypatch)
    root = Path(dependency["root"])
    capability = probe_isolation()
    if not capability.available:
        pytest.skip("isolated verifier required: " + capability.reason)
    isolated = True
    (root / "test_case.py").write_text("def test_case(): pass\n")
    # Only source enumeration/base/selection are synthetic; gate, isolated
    # subprocesses, collection, test reports, MAC and runtime wake are real.
    monkeypatch.setattr(evidence, "source_paths", lambda _: ["test_case.py"])
    monkeypatch.setattr(identity_module, "source_paths", lambda _: ["test_case.py"])
    monkeypatch.setattr(gate, "changed_paths", lambda *_: ("d" * 40, ["test_case.py"]))
    monkeypatch.setattr(
        gate.check_changed, "select_tests", lambda *a, **kw: ["test_case.py"]
    )
    monkeypatch.setattr(owner, "verification_binding", gate.verification_binding)
    steps = (Step("probe", (sys.executable, "-c", "pass")),)
    monkeypatch.setattr(owner, "merge_gate_steps", lambda: steps)
    identity = digest(
        gate.verification_binding(
            root,
            "frozen-base",
            steps,
            isolated=isolated,
            runtimes=(Path(sys.prefix),),
        )
    )
    dependency.update(
        verification_identity=identity,
        activity_id=source_verification_activity_id(identity, dependency["grant"]),
        runtime_roots=[str(Path(sys.prefix))],
    )
    dependency["wake"]["identity_digest"] = identity
    invocation = dict(
        steps=steps,
        root=root,
        base_ref="frozen-base",
        state_dir=Path(dependency["state_dir"]),
        # Workloads only start with more than 2*KILL_GRACE_SECONDS available;
        # 15 seconds parks every obligation on a scoped wait.
        step_seconds=25,
        run_step=run_step,
        local_feedback=False,
        runtime_roots=(Path(sys.prefix),),
    )
    with pytest.raises(ValueError, match="locked_verification_identity_mismatch"):
        gate.run_locked_release_gate("e" * 64, invocation)
    assert not list(Path(dependency["state_dir"]).glob("*.json"))
    for _ in range(3):
        summary = gate.run_locked_release_gate(identity, invocation)
        if summary["verification_complete"]:
            break
        assert summary["status"] == "pending", summary
    assert summary["verification_complete"], summary
    assert not summary["release_authorized"]
    with ActivityRuntime(CampaignStore("verify", tmp_path / "events")) as runtime:
        request(runtime, dependency)
        owner.drain_source_verification(runtime, {}, lambda _: None)
        assert runtime.snapshot()["repair"].status == "runnable"
        completions = [
            e
            for e in runtime.store.verify_event_chain()
            if e["event_type"] == "source_verification_completed"
        ]
        assert len(completions) == 1
        owner.drain_source_verification(runtime, {}, lambda _: None)
        assert (
            len(
                [
                    e
                    for e in runtime.store.verify_event_chain()
                    if e["event_type"] == "source_verification_completed"
                ]
            )
            == 1
        