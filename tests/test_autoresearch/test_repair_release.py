"""Real filesystem/CampaignStore publication; receipts below are trusted fixtures."""

import json
import sys
import time
from pathlib import Path

import pytest

from slm_training.autoresearch.heal import repair_release as release
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_contracts import (
    RepairBlocker,
    RepairDispatchResult,
    RepairGrant,
    RepairProposal,
    RepairRequest,
    RepairVerification,
)
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivitySpec,
    ResourceCapacity,
    ResourceGrant,
    WakeCondition,
    contract_digest,
)
from slm_training.autoresearch.storage import CampaignStore

SHA = "a" * 64


@pytest.fixture
def publication(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "fixture.py").write_text("answer = 42\n")
    store = CampaignStore("runtime", tmp_path / "outputs")
    with ActivityRuntime(store) as runtime:
        runtime.register(
            ActivitySpec(
                activity_id="repair",
                family="fixture",
                kind="control",
                source_digest=SHA,
                environment_digest=SHA,
                input_digest=SHA,
                output_namespace="attempt",
            )
        )
        lease = runtime.claim_next(capabilities={"local_process"})
        request = RepairRequest(
            activity_id="repair",
            blocked_activity_id="original-eval",
            attempt_id=lease.attempt_id,
            campaign_id="campaign",
            fence=lease.token,
            parent_event="fixture",
            blocker=RepairBlocker(
                code="harness_code_failure",
                blocker_class="code",
                owner="fixture",
                source_digest=SHA,
                environment_digest=SHA,
                input_digest=SHA,
                reproducer=("python", "fixture.py"),
                predicate="answer_42",
                needed_capability="source_repair",
                evidence=("fixture failure",),
            ),
            grant=RepairGrant(
                grant_id="fixture",
                provider="fixture",
                executable=sys.executable,
                executable_sha256=SHA,
                expires_at=time.time() + 600,
                max_attempts=1,
                total_seconds=30.0,
                interrupt_seconds=10,
            ),
            allowed_paths=("fixture.py",),
            verification_manifest_digest=SHA,
            project_instructions="fixture",
            owner_contract="fixture",
            existing_tests=("original",),
            failure_returncode=1,
            failure_stdout_sha256=SHA,
            failure_stderr_sha256=SHA,
        )
        proposal = RepairProposal(
            request_digest=request.digest(),
            tree_digest=manifest_digest(tree_manifest(candidate)),
            patch_digest=SHA,
            root_cause="fixture",
            regression_test="tests/test_fixture.py",
            reproduction_artifacts=(SHA,),
            classification="implementation",
        )
        receipt = RepairVerification(
            request_digest=request.digest(),
            proposal_digest=proposal.digest(),
            verifier_release=SHA,
            manifest_digest=SHA,
            source_digest=SHA,
            environment_digest=SHA,
            input_digest=SHA,
            grant_id="fixture",
            fence=lease.token,
            original_failure_reproduced=True,
            original_predicate_restored=True,
            required_checks_passed=True,
            protected_surfaces_unchanged=True,
            release_digest=proposal.tree_digest,
            evidence_digest=SHA,
        )
        result = RepairDispatchResult(
            status="verified",
            request_digest=request.digest(),
            reason="fixture_verified",
            proposal=proposal,
            verification=receipt,
        )
        kwargs = dict(
            runtime=runtime,
            lease=lease,
            candidate=candidate,
            destinations=(tmp_path / "releases", tmp_path / "outputs"),
            expected_previous=None,
            authenticated=lambda value: value is result,
        )
        yield request, result, kwargs


def test_publish_and_reconcile_without_activation(publication):
    request, result, kwargs = publication
    first = release.publish_verified_repair(request, result, **kwargs)
    again = release.publish_verified_repair(request, result, **kwargs)
    assert first == again
    assert first["resume_activity_id"] == "original-eval"
    assert first["requires_fresh_process"] and not first["service_activated"]
    assert not first["promotion_authority"]
    events = kwargs["runtime"].store.verify_event_chain()
    assert sum(e["event_type"] == "repair_release_intent" for e in events) == 1
    assert sum(e["event_type"] == "repair_release_accepted" for e in events) == 1


@pytest.mark.parametrize("boundary", ["pointer", "terminal"])
def test_publication_crash_reconciles_same_intent(publication, monkeypatch, boundary):
    request, result, kwargs = publication
    store = kwargs["runtime"].store
    writer, append = release._atomic_write, store.append_event

    def fail_pointer(*args, **kw):
        raise OSError("injected disk full")

    def fail_terminal(kind, **kw):
        if kind == "repair_release_accepted":
            raise OSError("injected terminal write failure")
        return append(kind, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(
            release, "_atomic_write", fail_pointer if boundary == "pointer" else writer
        )
        patch.setattr(
            store, "append_event", fail_terminal if boundary == "terminal" else append
        )
        with pytest.raises(OSError):
            release.publish_verified_repair(request, result, **kwargs)
    release.publish_verified_repair(request, result, **kwargs)
    events = store.verify_event_chain()
    assert sum(e["event_type"] == "repair_release_intent" for e in events) == 1
    assert sum(e["event_type"] == "repair_release_accepted" for e in events) == 1


@pytest.mark.parametrize("fault", ["forged", "stale", "changed", "expired", "outputs"])
def test_rejected_publication_cannot_write_pointer(publication, fault):
    request, result, kwargs = publication
    if fault == "forged":
        kwargs["authenticated"] = lambda _: False
    elif fault == "stale":
        kwargs["runtime"].cancel(kwargs["lease"].activity_id, reason="revoked")
    elif fault == "changed":
        (kwargs["candidate"] / "fixture.py").write_text("answer = 0\n")
    elif fault == "expired":
        kwargs["lease"] = kwargs["lease"].model_copy(update={"expires_at": 1.0})
    else:
        outputs = kwargs["destinations"][1]
        kwargs["destinations"] = (outputs / "releases", outputs)
    with pytest.raises((ValueError, RuntimeError)):
        release.publish_verified_repair(request, result, **kwargs)
    assert not (kwargs["runtime"].store.root / "source_release_pointer.json").exists()


def test_candidate_drift_during_materialization_is_rejected(publication, monkeypatch):
    request, result, kwargs = publication
    prepare = release.prepare_release

    def drifting(source, *args, **kw):
        (source / "fixture.py").write_text("answer = 0\n")
        return prepare(source, *args, **kw)

    monkeypatch.setattr(release, "prepare_release", drifting)
    with pytest.raises(ValueError, match="changed_during"):
        release.publish_verified_repair(request, result, **kwargs)
    assert not (kwargs["runtime"].store.root / "source_release_pointer.json").exists()


def test_marker_source_excludes_live_git_and_linked_outputs(tmp_path):
    source = tmp_path / "authoring"
    source.mkdir()
    (source / "fixture.py").write_text("answer = 42\n")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("private metadata")
    execution, original, outputs = (
        tmp_path / name for name in ("execution", "release", "outputs")
    )
    marker = release.prepare_release(source, original, execution, outputs)
    (outputs / "irrelevant").write_text("not part of repair identity")
    with pytest.raises(ValueError, match="external symlink"):
        tree_manifest(execution)
    resolved, digest = release.pinned_repair_source(execution, marker["source_digest"])
    assert resolved == original
    assert digest == manifest_digest(tree_manifest(original))
    assert not (resolved / ".git").exists() and not (resolved / "outputs").exists()
    (execution / "fixture.py").write_text("drift\n")
    with pytest.raises(ValueError, match="execution_source_drift"):
        release.pinned_repair_source(execution, marker["source_digest"])


def test_unpinned_source_refused(tmp_path):
    with pytest.raises(ValueError, match="requires_pinned"):
        release.pinned_repair_source(tmp_path, SHA)


def test_trusted_callback_and_activation_require_recorded_acceptance(publication):
    request, result, kwargs = publication
    callback = release.verified_release_callback(
        kwargs["runtime"], kwargs["lease"], destinations=kwargs["destinations"]
    )
    handoff = callback(request, result, kwargs["candidate"])
    validated = release.verified_activation_handoff(kwargs["runtime"].store, handoff)
    assert validated == handoff
    assert not validated["service_activated"] and "any_healed" not in validated
    with pytest.raises(ValueError, match="fresh_repair_operation"):
        callback(request, result, kwargs["candidate"])
    with pytest.raises(ValueError, match="identity_mismatch"):
        release.verified_activation_handoff(
            kwargs["runtime"].store, {**handoff, "resume_activity_id": "other"}
        )


def test_pointer_hash_without_acceptance_cannot_activate(publication, monkeypatch):
    request, result, kwargs = publication
    store = kwargs["runtime"].store
    append = store.append_event

    def refuse_acceptance(kind, **fields):
        if kind == "repair_release_accepted":
            raise OSError("controller died before acceptance")
        return append(kind, **fields)

    monkeypatch.setattr(store, "append_event", refuse_acceptance)
    with pytest.raises(OSError):
        release.publish_verified_repair(request, result, **kwargs)
    pointer = json.loads((store.root / "source_release_pointer.json").read_text())
    handoff = release._handoff(request, pointer)
    with pytest.raises(ValueError, match="requires_controller_acceptance"):
        release.verified_activation_handoff(store, handoff)


def test_missing_original_activity_cannot_activate(publication):
    request, result, kwargs = publication
    # An accepted historical handoff without a resume target stays non-activating.
    handoff = release.publish_verified_repair(request, result, **kwargs)
    assert handoff["resume_activity_id"]
    with pytest.raises(ValueError, match="identity_mismatch"):
        release.verified_activation_handoff(
            kwargs["runtime"].store, {**handoff, "resume_activity_id": None}
        )


@pytest.mark.parametrize("fault", ["none", "register_crash", "exhausted"])
def test_same_request_wake_requires_successor_import_identity(
    publication, monkeypatch, fault
):
    """Real ledger/process/publication; successful import identity is simulated."""
    from slm_training.autoresearch.heal import operation_recovery as bridge

    request, result, kwargs = publication
    runtime = kwargs["runtime"]
    runtime.capacity = ResourceCapacity(cpu_slots=2, memory_mb=4096)
    original = {
        "operation": "inspect",
        "loop_id": "fixture",
        "cwd": str(kwargs["candidate"]),
        "source_digest": SHA,
        "replicate_id": "replicate-0",
    }
    runtime.register(
        ActivitySpec(
            activity_id="original-eval",
            family="fixture",
            kind="control",
            source_digest=SHA,
            environment_digest=SHA,
            input_digest=contract_digest(original),
            output_namespace="original-attempt",
            grant=ResourceGrant(max_attempts=1 if fault == "exhausted" else 3),
        )
    )
    failed_lease = runtime.claim_next(capabilities={"local_process"})
    failed = runtime.run(
        failed_lease,
        (sys.executable, "-c", "raise SystemExit(1)"),
        cwd=kwargs["candidate"],
    )
    bridge.record_operation_failure(
        runtime, failed_lease, original, failed, outcome=ActivityOutcome.UNKNOWN_FAILURE
    )
    runtime.finish(
        failed_lease,
        outcome=ActivityOutcome.UNKNOWN_FAILURE,
        spent_seconds=failed.duration_seconds,
        wake=WakeCondition(
            predicate="original operation produces valid output",
            source="independent_repair_verification",
            identity_digest=contract_digest(original),
        ),
    )
    handoff = release.publish_verified_repair(request, result, **kwargs)
    execution = Path(handoff["successor_execution"])
    monkeypatch.chdir(execution)
    with pytest.raises(ValueError, match="successor_not_active"):
        bridge.wake_verified_operation(runtime, handoff, cwd=execution)
    monkeypatch.setattr(bridge, "__file__", str(execution / "fixture.py"))
    if fault == "exhausted":
        with pytest.raises(ValueError, match="max_attempts"):
            bridge.wake_verified_operation(runtime, handoff, cwd=execution)
        assert runtime.snapshot()["original-eval"].status == "waiting_repair"
        return
    if fault == "register_crash":

        def crash_register(*args):
            raise OSError("injected crash after old cancellation")

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "register", crash_register)
            with pytest.raises(OSError):
                bridge.wake_verified_operation(runtime, handoff, cwd=execution)
        assert runtime.snapshot()["original-eval"].status == "cancelled"
    successor = bridge.wake_verified_operation(runtime, handoff, cwd=execution)
    assert bridge.wake_verified_operation(runtime, handoff, cwd=execution) == successor
    state = runtime.snapshot()["original-eval"]
    assert state.status == "cancelled" and state.spec.input_digest == contract_digest(
        original
    )
    next_state = runtime.snapshot()[successor["successor_activity_id"]]
    from scripts.autotrain_supervisor_operations import _operation_state

    assert (
        _operation_state(runtime, successor, 999, contract_digest(successor)).spec
        == next_state.spec
    )
    changed = {**successor, "resource_grant": state.spec.grant.model_dump(mode="json")}
    with pytest.raises(ValueError, match="altered verified successor"):
        _operation_state(runtime, changed, 1000, contract_digest(changed))
    assert (
        next_state.status == "runnable"
        and next_state.spec.source_digest == handoff["source_digest"]
    )
    assert (
        successor["cwd"] == str(execution)
        and successor["replicate_id"] == "replicate-0"
    )
    assert (
        next_state.spec.grant.total_seconds + state.charged_seconds
        == state.spec.grant.total_seconds
    )
    assert (
        next_state.spec.grant.max_attempts + state.attempts
        == state.spec.grant.max_attempts
    )
    assert successor["resource_grant"] == next_state.spec.grant.model_dump(mode="json")
    assert successor["logical_continuation"]["scientific_replicate_increment"] == 0
    assert len(bridge.pending_operation_repairs(runtime)) == 0
    events = runtime.store.verify_event_chain()
    assert sum(e["event_type"] == "operation_successor_activated" for e in events) == 1
