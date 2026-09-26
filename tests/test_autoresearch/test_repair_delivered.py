"""Real local bindings and Git hashes; remote reconciliation observations are fixtures."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import merge_verification_evidence as evidence
from slm_training.autoresearch.heal import repair_delivered as owner
from slm_training.autoresearch.heal import repair_release as release
from slm_training.autoresearch.heal.operation_recovery import _successor_plan
from slm_training.autoresearch.runtime.operations_verification import verification_environment
from slm_training.harness_core import execution_release as core
from slm_training.harness_core.activity_contract import ActivityOutcome, ActivitySpec, ResourceGrant, WakeCondition, contract_digest
from slm_training.harness_core.github_delivery_tree import git_object, source_entries, tree_sha
from tests.test_autoresearch import test_repair_delivery as producer_fixtures
from tests.test_autoresearch import test_repair_release as release_fixtures

publication = release_fixtures.publication
accepted_source = producer_fixtures.accepted_source


def _commit(source, *, wrong_tree=False):
    tree = "e" * 40 if wrong_tree else tree_sha(source_entries(source, core._files(source)))
    raw = f"tree {tree}\nparent {'f' * 40}\nauthor Fixture <fixture@invalid> 0 +0000\ncommitter Fixture <fixture@invalid> 0 +0000\n\naccepted repair\n"
    return raw, git_object("commit", raw.encode())


@pytest.fixture
def accepted_execution(accepted_source):
    request, result, kwargs, _ = accepted_source
    handoff = release.publish_verified_repair(request, result, **kwargs)
    return Path(handoff["successor_execution"]), handoff, kwargs


def test_new_clean_release_requires_exact_remote_commit_and_tree(accepted_execution, tmp_path):
    source, accepted, _ = accepted_execution
    before = (source / core.MARKER).read_bytes()
    raw, sha = _commit(source)
    destinations = (tmp_path / "delivered-release", tmp_path / "delivered-execution", tmp_path / "delivered-outputs")
    manifest = core.prepare_delivered_release((source, accepted["source_digest"]), destinations, (raw, sha))
    assert core.runtime_source_identity(destinations[1]) == accepted["source_digest"]
    assert core.runtime_git_provenance(destinations[1]) == {"integration_commit": sha, "upstream_commit": sha, "code_dirty": False}
    assert manifest["files"] == core._files(source)
    assert (source / core.MARKER).read_bytes() == before
    assert core.runtime_git_provenance(source)["code_dirty"] is True
    assert core.validate_source_refs(destinations[1], sha, sha, git=lambda *a, **kw: pytest.fail("Git queried"))["code_dirty"] is False


def test_clean_head_materialization_includes_committed_templates_and_serena(tmp_path):
    source = tmp_path / "source"
    (source / ".serena").mkdir(parents=True)
    (source / ".env.example").write_text("EXAMPLE=value\n")
    (source / ".env.local").write_text("SECRET=private\n")
    (source / ".serena" / ".gitignore").write_text("cache/\n")
    (source / ".serena" / "project.yml").write_text("name: fixture\n")
    (source / ".serena" / "cache").mkdir()
    (source / ".serena" / "cache" / "private").write_text("local only\n")
    initial = tmp_path / "initial"
    core.prepare_release(source, tmp_path / "initial-release", initial, tmp_path / "initial-outputs")
    with pytest.raises(ValueError, match="release_git_provenance_unavailable"):
        core.runtime_git_provenance(initial)
    assert {".env.example", ".serena/.gitignore", ".serena/project.yml"} <= core._files(initial).keys()
    assert ".env.local" not in core._files(initial)
    assert ".serena/cache/private" not in core._files(initial)

    raw, sha = _commit(initial)
    clean = tmp_path / "clean"
    core.prepare_delivered_release((initial, core.runtime_source_identity(initial)),
        (tmp_path / "clean-release", clean, tmp_path / "clean-outputs"), (raw, sha))
    assert core.runtime_git_provenance(clean) == {
        "integration_commit": sha, "upstream_commit": sha, "code_dirty": False}

    other = tmp_path / "other"
    (other / ".serena").mkdir(parents=True)
    (other / ".env.example").write_text("EXAMPLE=changed\n")
    (other / ".serena" / ".gitignore").write_text("cache/\n")
    (other / ".serena" / "project.yml").write_text("name: fixture\n")
    altered = tmp_path / "altered"
    core.prepare_release(other, tmp_path / "altered-release", altered, tmp_path / "altered-outputs")
    with pytest.raises(ValueError, match="delivered_commit_tree_mismatch"):
        core.prepare_delivered_release((altered, core.runtime_source_identity(altered)),
            (tmp_path / "rejected-release", tmp_path / "rejected", tmp_path / "rejected-outputs"), (raw, sha))


@pytest.mark.parametrize("fault", ["sha", "tree", "identity", "copy_drift", "mode", "link"])
def test_delivered_materializer_rejects_unproved_or_changed_source(accepted_execution, tmp_path, monkeypatch, fault):
    source, accepted, _ = accepted_execution
    raw, sha = _commit(source, wrong_tree=fault == "tree")
    expected = accepted["source_digest"]
    if fault == "sha":
        sha = "a" * 40
    elif fault == "identity":
        expected = "a" * 64
    elif fault == "copy_drift":
        copy = core.shutil.copyfile
        def drift(*args):
            result = copy(*args)
            (source / "fixture.py").write_text("changed during copy\n")
            return result
        monkeypatch.setattr(core.shutil, "copyfile", drift)
    elif fault == "mode":
        (source / "fixture.py").chmod(0o755)
    elif fault == "link":
        (source / "fixture.py").unlink()
        (source / "fixture.py").symlink_to("tests/test_regression.py")
    target = tmp_path / "new-execution"
    with pytest.raises(ValueError):
        core.prepare_delivered_release((source, expected), (tmp_path / "new-release", target, tmp_path / "new-outputs"), (raw, sha))
    assert not (target / core.MARKER).exists()


@pytest.fixture
def reconciled(accepted_execution, monkeypatch):
    source, accepted, kwargs = accepted_execution
    runtime = kwargs["runtime"]
    attempt = runtime.attempt_dir(kwargs["lease"])
    attempt.mkdir(parents=True, exist_ok=True)
    output = attempt / "handoff.json"
    output.write_text(json.dumps(accepted))
    runtime.finish(kwargs["lease"], outcome=ActivityOutcome.SUCCEEDED, spent_seconds=0,
                   outputs={output.name: hashlib.sha256(output.read_bytes()).hexdigest()})
    monkeypatch.setattr(evidence, "environment_identity", lambda: {
        "fixture_runtime": "fixed", "execution_environment_sha256": evidence.digest(verification_environment())})
    environment = evidence.digest(evidence.environment_identity())
    original = {"operation": "inspect", "cwd": str(kwargs["destinations"].verified_predecessor[0]),
                "source_digest": kwargs["destinations"].verified_predecessor[1],
                "environment_digest": environment, "replicate_id": "original-replicate"}
    runtime.register(ActivitySpec(activity_id=accepted["resume_activity_id"], family="fixture", kind="control",
        source_digest=original["source_digest"], environment_digest=environment, input_digest=contract_digest(original),
        output_namespace="original", grant=ResourceGrant(interrupt_seconds=10, kill_grace_seconds=10,
                                                       total_seconds=100, max_attempts=3)))
    lease = runtime.claim_next(activity_id=accepted["resume_activity_id"], capabilities={"local_process"})
    runtime.finish(lease, outcome=ActivityOutcome.CODE_FAILURE, spent_seconds=20,
                   wake=WakeCondition(predicate="verified repair", source="accepted_release", identity_digest=contract_digest(original)))
    runtime.store.append_event("operation_repair_requested", experiment_id=accepted["resume_activity_id"], detail={"request": original})
    wait = accepted["source_delivery"]
    runtime.register(ActivitySpec(activity_id="delivery", family="fixture", kind="delivery",
        source_digest=accepted["source_digest"], environment_digest=environment, input_digest=contract_digest(wait),
        output_namespace="delivery", capabilities=("authorized_github_connector_delivery",)))
    lease = runtime.claim_next(activity_id="delivery", capabilities={"authorized_github_connector_delivery"})
    raw, sha = _commit(source)
    remote = {"repository": "owner/repo", "publication_id": wait["publication_id"], "artifact_sha256": wait["artifact_sha256"],
              "source_digest": accepted["source_digest"], "base_ref": "f" * 40, "base_git_tree": "a" * 40,
              "head_sha": "b" * 40, "merge_sha": sha, "candidate_git_tree": tree_sha(source_entries(source, core._files(source))),
              "verification_identity": "c" * 64, "verification_plan_sha256": "d" * 64, "merge_commit_object": raw}
    proof = {"schema_version": "verified_repair_source_delivery_receipt/v1", "wait": wait, "remote": remote}
    artifact = runtime.store.write_artifact("verified_repair_source_delivery_receipts", proof)
    runtime.store.append_event("verified_repair_source_delivered", artifact_sha256=artifact.stem,
        experiment_id=lease.activity_id, detail={"publication_id": wait["publication_id"], "request_artifact_sha256": wait["artifact_sha256"]})
    reconciliation = {"verification_artifact": str(artifact), "receipt": proof}
    request = {"schema_version": "connector_delivery_request/v1", "wait": wait,
               "lease": lease.model_dump(mode="json"),
               "operation": "deliver_and_reconcile_squash_merge", "base_ref": remote["base_ref"],
               "repository": remote["repository"], "source_digest": accepted["source_digest"]}
    runtime.store._replace_durable(runtime.attempt_dir(lease) / "request.json", json.dumps(request))
    return runtime, lease, wait, reconciliation, accepted


def _finish(runtime, lease, reconciliation, reference):
    attempt = runtime.attempt_dir(lease)
    attempt.mkdir(parents=True, exist_ok=True)
    outputs = {}
    request = json.loads((attempt / "request.json").read_text())
    for name, content in {"receipt.json": {"provider": "github_connector", "request_digest": contract_digest(request)},
                          "domain-reconciliation.json": {**reconciliation, "delivered_activation": reference}}.items():
        path = attempt / name
        path.write_text(json.dumps(content))
        outputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    outputs.update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in attempt.glob("*.json")})
    runtime.finish(lease, outcome=ActivityOutcome.SUCCEEDED, spent_seconds=0, outputs=outputs)
    return attempt


def test_record_is_fenced_idempotent_and_inert_until_committed_delivery(reconciled):
    runtime, lease, wait, reconciliation, accepted = reconciled
    pointer = (runtime.store.root / "source_release_pointer.json").read_bytes()
    with runtime.publication(lease):
        ref = owner.record_delivered_activation(runtime, lease, wait, reconciliation)
    assert owner.record_delivered_activation(runtime, lease, wait, reconciliation) == ref
    with pytest.raises(ValueError, match="requires_committed_delivery"):
        owner.resolve_delivered_activation(runtime.store, ref)
    _finish(runtime, lease, reconciliation, ref)
    handoff = owner.resolve_delivered_activation(runtime.store, ref)
    assert release.verified_activation_handoff(runtime.store, handoff) == handoff
    assert handoff["activation_id"] != handoff["publication_id"] == accepted["publication_id"]
    assert handoff["source_digest"] == accepted["source_digest"]
    assert handoff["successor_execution"] != accepted["successor_execution"]
    assert (runtime.store.root / "source_release_pointer.json").read_bytes() == pointer
    assert release.verified_activation_handoff(runtime.store, accepted) == accepted
    with pytest.raises(Exception, match="stale|expired|foreign"):
        owner.record_delivered_activation(runtime, lease, wait, reconciliation)


@pytest.mark.parametrize("fault", ["proof", "materialized", "reference", "output"])
def test_activation_rejects_tampering(reconciled, fault):
    runtime, lease, wait, reconciliation, _ = reconciled
    ref = owner.record_delivered_activation(runtime, lease, wait, reconciliation)
    attempt = _finish(runtime, lease, reconciliation, ref)
    checked = owner.resolve_delivered_activation(runtime.store, ref)
    if fault == "proof":
        Path(reconciliation["verification_artifact"]).write_text("{}")
    elif fault == "materialized":
        (Path(checked["successor_execution"]) / "fixture.py").write_text("wrong\n")
    elif fault == "reference":
        ref = {**ref, "activation_id": "wrong"}
    else:
        (attempt / "domain-reconciliation.json").write_text("{}")
    with pytest.raises(ValueError):
        owner.resolve_delivered_activation(runtime.store, ref)


def test_successor_binds_relocated_environment_and_keeps_remaining_grant(reconciled, monkeypatch):
    runtime, lease, wait, reconciliation, accepted = reconciled
    ref = owner.record_delivered_activation(runtime, lease, wait, reconciliation)
    _finish(runtime, lease, reconciliation, ref)
    handoff = owner.resolve_delivered_activation(runtime.store, ref)
    monkeypatch.setenv("PYTHONPATH", str(Path(handoff["successor_execution"]) / "src"))
    monkeypatch.delenv("PYTHONHOME", raising=False)
    # Historical repair proof remains readable after the controlled path change.
    assert owner.resolve_delivered_activation(runtime.store, ref) == handoff
    plan = _successor_plan(runtime, handoff, runtime.store.verify_event_chain())
    assert plan["request"]["environment_digest"] == evidence.digest(evidence.environment_identity())
    assert plan["spec"]["environment_digest"] == plan["request"]["environment_digest"]
    assert plan["spec"]["grant"]["total_seconds"] == 80
    assert plan["spec"]["grant"]["max_attempts"] == 2
    logical = plan["request"]["logical_continuation"]
    assert logical["scientific_replicate_increment"] == 0 and logical["prior_attempts"] == 1
    assert plan["request"]["replicate_id"] == "original-replicate"
    monkeypatch.setenv("OMP_NUM_THREADS", "987")
    with pytest.raises(ValueError, match="environment_transition_mismatch"):
        _successor_plan(runtime, handoff, runtime.store.verify_event_chain())


def test_environment_drift_before_recording_cannot_be_relabelled(reconciled, monkeypatch):
    runtime, lease, wait, reconciliation, _ = reconciled
    monkeypatch.setenv("OMP_NUM_THREADS", "987")
    with pytest.raises(ValueError, match="predecessor_environment_changed"):
        owner.record_delivered_activation(runtime, lease, wait, reconciliation)
    assert not any(event["event_type"] == "delivered_repair_activation_recorded" for event in runtime.store.verify_event_chain())


@pytest.mark.parametrize("fault", [None, "source", "cache"])
def test_signed_historical_gate_survives_relocation_but_not_tampering(tmp_path, monkeypatch, fault):
    """Real signed cache/pytest workload; fixture source selection and isolation."""
    from scripts import merge_verification as merge
    from slm_training.autoresearch.heal.repair_acceptance import VerificationWorkspace
    from slm_training.autoresearch.heal.repair_delivery import _historical_gate
    from tests.test_autoresearch.test_repair_acceptance import source_gate_fixture

    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    test = source / "tests/test_tiny.py"
    test.write_text("def test_tiny():\n    assert 6 * 7 == 42\n")
    gate = source_gate_fixture(tmp_path, monkeypatch, VerificationWorkspace(source, source))
    state = evidence.ReceiptCache(gate.state_dir, gate.root).load(gate.identity)
    inputs = {"binding": state["binding"]}
    expected = _historical_gate(gate, inputs)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "delivered/src"))
    monkeypatch.setattr(merge, "environment_identity", lambda: pytest.fail("historical environment relabelled"))
    if fault == "source":
        (gate.root / "tests/test_tiny.py").write_text("def test_tiny():\n    assert False\n")
    elif fault == "cache":
        path = gate.state_dir / (gate.identity + ".json")
        saved = json.loads(path.read_text())
        saved["payload"]["passed_nodes"] = []
        path.write_text(json.dumps(saved))
    if fault:
        with pytest.raises(ValueError):
            _historical_gate(gate, inputs)
    else:
        assert _historical_gate(gate, inputs) == expected


@pytest.mark.parametrize("branch", ["main", "acceptance-only/fault"])
def test_real_activation_preserves_branch_authority(reconciled, branch):
    from slm_training.harness_core.source_authority import resolve_source_authority, require_main_source_authority

    runtime, lease, wait, reconciliation, accepted = reconciled
    if branch != "main":
        # Fixture observations represent a genuine separate branch-specific readback.
        remote = {**reconciliation["receipt"]["remote"], "base_branch": branch}
        proof = {**reconciliation["receipt"], "remote": remote}
        artifact = runtime.store.write_artifact("verified_repair_source_delivery_receipts", proof)
        runtime.store.append_event("verified_repair_source_delivered", artifact_sha256=artifact.stem,
            experiment_id=lease.activity_id, detail={"publication_id": wait["publication_id"],
                                                   "request_artifact_sha256": wait["artifact_sha256"]})
        reconciliation = {"verification_artifact": str(artifact), "receipt": proof}
        path = runtime.attempt_dir(lease) / "request.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "base_branch": branch}))
    ref = owner.record_delivered_activation(runtime, lease, wait, reconciliation)
    _finish(runtime, lease, reconciliation, ref)
    handoff = owner.resolve_delivered_activation(runtime.store, ref)
    execution = Path(handoff["successor_execution"])
    assert core.runtime_git_provenance(execution)["code_dirty"] is False
    args = dict(execution=execution, expected_repository="owner/repo",
                expected_commit=reconciliation["receipt"]["remote"]["merge_sha"],
                expected_source_digest=accepted["source_digest"])
    pointer = core.source_authority_reference(execution)
    assert resolve_source_authority(runtime.store, pointer, **args)["ref"] == "refs/heads/" + branch
    if branch == "main":
        assert require_main_source_authority(runtime.store, pointer, **args)["commit"] == args["expected_commit"]
    else:
        with pytest.raises(ValueError, match="main_membership_required"):
            require_main_source_authority(runtime.store, pointer, **args)


def test_interrupted_activation_rebinds_new_delivery_attempt(reconciled, monkeypatch):
    from slm_training.harness_core.source_authority import require_main_source_authority

    runtime, lease, wait, reconciliation, accepted = reconciled
    first_attempt = runtime.attempt_dir(lease)
    request = json.loads((first_attempt / "request.json").read_text())
    append = runtime.store.append_event
    def interrupted(event_type, **kwargs):
        result = append(event_type, **kwargs)
        if event_type == "delivered_repair_activation_recorded":
            raise InterruptedError("after durable activation")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(runtime.store, "append_event", interrupted)
        with pytest.raises(InterruptedError, match="after durable activation"):
            owner.record_delivered_activation(runtime, lease, wait, reconciliation)
    first = next(e for e in runtime.store.verify_event_chain()
                 if e["event_type"] == "delivered_repair_activation_recorded")
    prior = owner._validated_payload(runtime.store, {
        "artifact_sha256": first["artifact_sha256"], "activation_id": first["detail"]["activation_id"]})
    grant = runtime.snapshot()[lease.activity_id].spec.grant
    state = runtime.finish(lease, outcome=ActivityOutcome.RETRY, spent_seconds=1)
    monkeypatch.setattr(runtime, "clock", lambda: state.retry_at + 1)
    retry = runtime.claim_next(activity_id=lease.activity_id, capabilities={"authorized_github_connector_delivery"})
    assert retry.attempt_id != lease.attempt_id
    attempt = runtime.attempt_dir(retry)
    runtime.store._replace_durable(attempt / "request.json", json.dumps(request))
    with pytest.raises(ValueError, match="request_lease_mismatch"):
        owner.record_delivered_activation(runtime, retry, wait, reconciliation)
    request = {**request, "lease": retry.model_dump(mode="json")}
    runtime.store._replace_durable(attempt / "request.json", json.dumps(request))
    ref = owner.record_delivered_activation(runtime, retry, wait, reconciliation)
    assert ref["artifact_sha256"] != first["artifact_sha256"]
    assert (attempt / "source-authority.json").read_bytes() != (first_attempt / "source-authority.json").read_bytes()
    _finish(runtime, retry, reconciliation, ref)
    handoff = owner.resolve_delivered_activation(runtime.store, ref)
    execution = Path(handoff["successor_execution"])
    assert execution != Path(prior["execution"])
    pointer = core.source_authority_reference(execution)
    args = dict(execution=execution, expected_repository="owner/repo",
                expected_commit=reconciliation["receipt"]["remote"]["merge_sha"],
                expected_source_digest=accepted["source_digest"])
    assert require_main_source_authority(runtime.store, pointer, **args)["ref"] == "refs/heads/main"
    assert json.loads((attempt / "receipt.json").read_text())["request_digest"] == contract_digest(request)
    final = runtime.snapshot()[lease.activity_id]
    assert final.attempts == 2 and final.charged_seconds == 1 and final.spec.grant == grant
    with pytest.raises(ValueError, match="terminal_lease_mismatch|committed_output_changed"):
        require_main_source_authority(runtime.store, core.source_authority_reference(Path(prior["execution"])),
            **{**args, "execution": prior["execution"]})
