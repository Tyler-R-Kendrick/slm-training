"""Real subprocess verification plumbing; OS isolation is simulated explicitly."""

import hashlib
import shutil
import sys
import time
from dataclasses import replace

import pytest

from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.activity_contract import ActivitySpec, ResourceGrant
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    tree_manifest,
)
from slm_training.autoresearch.heal.recovery_dispatch import (
    RecoveryConfig,
    RecoveryContext,
    RepairRecipe,
    build_repair_request,
)
from slm_training.autoresearch.heal.repair_acceptance import (
    SourceVerificationGate,
    VerificationWorkspace,
    proposal_patch_digest,
    source_verification_activity_id,
    verify_repair,
)
from slm_training.autoresearch.heal.repair_contracts import (
    RepairDispatchResult,
    RepairGrant,
    RepairProposal,
)
from slm_training.autoresearch.heal.repair_release import (
    verified_activation_handoff,
    verified_release_callback,
)
from slm_training.autoresearch.heal.repair_verifier import (
    VerificationCheck,
    VerificationRequest,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import KILL_GRACE_SECONDS


@pytest.mark.parametrize("restored", [True, False])
@pytest.mark.parametrize("through_seam", [False, True])
@pytest.mark.parametrize("source_available", [False, None, True])
def test_real_original_reproducer_controls_acceptance(
    tmp_path, monkeypatch, restored, through_seam, source_available
):
    source = tmp_path / "source"
    (source / "docs/design").mkdir(parents=True)
    for name in ("AGENTS.md", "RTK.md", "docs/design/decode-invariants.md"):
        (source / name).write_text("I6 fail closed; preserve original reproduction")
    (source / "fixture.py").write_text('print("broken")\nraise SystemExit(1)\n')
    candidate = tmp_path / "candidate"
    shutil.copytree(source, candidate)
    if restored:
        (candidate / "fixture.py").write_text('print("ready")\n')
    (candidate / "tests").mkdir()
    (candidate / "tests/test_added.py").write_text(
        'def test_added():\n    assert 1 + 1 == 2\nprint("checked")\n'
    )
    sha = "a" * 64
    grant = RepairGrant(
        grant_id="fixture",
        provider="fixture",
        executable=sys.executable,
        executable_sha256=sha,
        expires_at=time.time() + 60,
        max_attempts=2,
        total_seconds=80.0,
        interrupt_seconds=10,
    )
    recipe = RepairRecipe(
        allowed_paths=("fixture.py", "tests/test_added.py"),
        input_digest=sha,
        original=VerificationCheck(
            "original", (sys.executable, "fixture.py"), "ready\n"
        ),
        checks=(
            VerificationCheck(
                "regression", (sys.executable, "tests/test_added.py"), "checked\n"
            ),
        ),
        owner_contract_path="AGENTS.md",
        failure_returncode=1,
        failure_stdout_sha256=hashlib.sha256(b"broken\n").hexdigest(),
        failure_stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )
    config = RecoveryConfig(
        grant=grant, recipes={"harness_code_failure": recipe}, verifier_release=sha,
        source_verification_grant=ResourceGrant(),
    )
    context = RecoveryContext(
        tmp_path / "store",
        "loop",
        "campaign",
        source,
        manifest_digest(tree_manifest(source)),
        sha,
        "fence",
        "parent",
    )
    request = build_repair_request(
        {
            "kind": "repair_harness",
            "affected_activity_id": "original-evaluation",
            "blocker_code": "harness_code_failure",
            "unmet_predicate": "ready_stdout",
            "required_capability": "source_repair",
        },
        context,
        config,
    )
    proposal = RepairProposal(
        request_digest=request.digest(),
        tree_digest=manifest_digest(tree_manifest(candidate)),
        patch_digest=proposal_patch_digest(source, candidate),
        root_cause="fixture ordinary code defect",
        regression_test="tests/test_added.py",
        reproduction_artifacts=(sha,),
        classification="implementation",
    )
    spec = VerificationRequest(
        request.digest(),
        request.blocker.fingerprint(),
        context.source_digest,
        proposal.tree_digest,
        sha,
        sha,
        grant.digest(),
        context.fence,
        request.allowed_paths,
        recipe.original,
        recipe.checks,
        1,
        recipe.failure_stdout_sha256,
        recipe.failure_stderr_sha256,
        timeout_seconds=10,
    )
    if restored and not through_seam and source_available is True:
        request = request.model_copy(update={
            "grant": request.grant.model_copy(update={"expires_at": 1.0})
        })
        proposal = proposal.model_copy(update={"request_digest": request.digest()})
        spec = replace(spec, request_digest=request.digest(),
                       authority_digest=request.grant.digest())

    def real_process_without_isolation(spec, argv, **kwargs):
        return run_bounded_process(
            argv,
            cwd=spec.workspace,
            interrupt_after_seconds=spec.timeout_seconds,
            kill_grace_seconds=KILL_GRACE_SECONDS,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )

    monkeypatch.setattr(
        "slm_training.autoresearch.heal.repair_verifier.run_isolated",
        real_process_without_isolation,
    )
    if through_seam:
        result = _two_lease_dispatch(context, config, candidate, proposal, monkeypatch, source_available)
    else:
        workspace = VerificationWorkspace(source, candidate)
        result = verify_repair(
            request,
            proposal,
            spec,
            workspace=workspace,
            journal=CampaignStore("campaign", tmp_path / "store"),
            fence_valid=lambda _: True,
            source_verification=source_gate_fixture(tmp_path, monkeypatch, workspace, complete=source_available is True) if source_available is not False else None,
        )
    if not source_available:
        assert result.status == "waiting_verification"
        assert result.reason == ("source_verification_not_configured" if source_available is False else "source_verification_pending")
        assert result.verification is None
        events = CampaignStore("campaign", tmp_path / "store").verify_event_chain()
        assert any(row["event_type"] == "repair_source_verification_wait" for row in events)
        assert not any(row["event_type"] == "repair_verification_started" for row in events)
        return
    assert result.status == ("verified" if restored else "rejected")
    assert result.verification.original_predicate_restored is restored
    journal = CampaignStore("campaign", tmp_path / "store")
    import json

    combined = json.loads((journal.root / "artifacts/repair_verification" / (result.verification.evidence_digest + ".json")).read_text())
    assert combined["source_release_authorized"] is restored
    assert combined["source_verification"]["verification_complete"]
    assert combined["candidate_snapshot_digest"] == proposal.tree_digest


def _two_lease_dispatch(context, config, candidate, proposal, monkeypatch, source_available):
    from slm_training.autoresearch.heal.recovery_dispatch import dispatch_hard_pending

    calls = []
    publications = []

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def capability(self, request):
            return None

        def execute(self, request, **kwargs):
            calls.append(request.digest())
            output = (
                context.root
                / context.campaign_id
                / "repair_workspaces"
                / request.digest()
                / "candidate"
            )
            shutil.copytree(candidate, output)
            assert proposal.request_digest == request.digest()
            return RepairDispatchResult(
                status="waiting_verification",
                request_digest=request.digest(),
                reason="proposal_received",
                proposal=proposal,
            )

    monkeypatch.setattr(
        "slm_training.autoresearch.heal.recovery_dispatch.CodexExecutor", FakeAgent
    )
    pending = {
        "kind": "repair_harness",
        "affected_activity_id": "original-evaluation",
        "blocker_code": "harness_code_failure",
        "unmet_predicate": "ready_stdout",
        "required_capability": "source_repair",
    }
    first = dispatch_hard_pending(
        pending, context, config=config, fence_valid=lambda value: value == "fence"
    )
    assert first["status"] == "waiting_verification"
    with ActivityRuntime(CampaignStore("runtime", context.root)) as runtime:
        runtime.register(
            ActivitySpec(
                activity_id="verify-publish",
                family="fixture",
                kind="control",
                source_digest=context.source_digest,
                environment_digest=context.environment_digest,
                input_digest=config.recipes["harness_code_failure"].input_digest,
                output_namespace="attempt",
            )
        )
        lease = runtime.claim_next(capabilities={"local_process"})
        next_context = replace(
            context,
            fence=lease.token,
            attempt_id=lease.attempt_id,
            parent_event="new-parent",
        )

        controller_publish = verified_release_callback(
            runtime,
            lease,
            destinations=(context.root.parent / "published-releases", context.root),
        )

        def publish(request, verified, path):
            assert verified.status == "verified"
            publications.append(verified.digest())
            return controller_publish(request, verified, path)

        workspace = VerificationWorkspace(context.source, context.root / context.campaign_id / "repair_workspaces" / proposal.request_digest / "candidate")
        gate = source_gate_fixture(context.root.parent, monkeypatch, workspace, complete=source_available is True) if source_available is not False else None
        second = dispatch_hard_pending(
            pending,
            next_context,
            config=config,
            fence_valid=lambda value: value == lease.token,
            publish_verified=publish,
            source_verification=lambda request, proposal, workspace: gate,
        )
        if "release_handoff" in second:
            assert verified_activation_handoff(runtime.store, second["release_handoff"])
    assert len(calls) == 1, "a new verification lease must not rerun the agent"
    assert second["request_digest"] == first["request_digest"]
    if second["verification"]:
        assert second["verification"]["fence"] == next_context.fence
        assert second["verification"]["authority_binding_digest"]
    dependency = second.pop("verification_dependency", None)
    if not source_available:
        assert dependency["request_digest"] == first["request_digest"]
        assert dependency["candidate_snapshot_digest"] == proposal.tree_digest
        if gate:
            assert dependency["activity_id"] == source_verification_activity_id(
                gate.identity, dependency["grant"]
            )
            assert dependency["wake"]["source"] == "source_verification_completed"
        else:
            assert dependency["wake"]["source"] == "controller_policy"
    assert len(publications) == int(second["status"] == "verified")
    handoff = second.pop("release_handoff", None)
    assert (handoff is not None) == bool(publications)
    if handoff:
        assert handoff["requires_fresh_process"] and not handoff["service_activated"]
        assert not handoff["promotion_authority"]
    original_path = (
        context.root
        / context.campaign_id
        / "artifacts/repair_requests"
        / (first["request_digest"] + ".json")
    )
    assert '"fence": "fence"' in original_path.read_text(), (
        "historical authority must stay immutable"
    )
    return RepairDispatchResult.model_validate_json(__import__("json").dumps(second))


def source_gate_fixture(tmp_path, monkeypatch, workspace, *, complete=True):
    """Actual producer/cache/consumer and pytest processes; isolation/discovery fake.

    This deliberately does not claim OS isolation or production release coverage.
    Discovery is a tiny fixture contract, not the repository's mandatory universe.
    """
    from scripts import merge_verification as merge
    from scripts import merge_verification_evidence as evidence
    from scripts import verify_merge_ready
    from slm_training.autoresearch.heal.isolation import IsolationCapability

    def fixture_binding(root, base_ref, steps, **kwargs):
        return {
            "schema": "merge_verification_binding/v2",
            "base_tree": base_ref,
            "candidate_tree_sha256": manifest_digest(tree_manifest(root)),
            "changed_paths": sorted(tree_manifest(root)),
            "targets": ["tests"], "static_commands": [],
            "environment": {}, "runtime_roots": [],
            "runtime_identity": kwargs.get("runtime_digest_value")
            or merge.runtime_identity(kwargs.get("runtimes", ())),
            "isolation_enforced": True, "max_attempts_per_obligation": 3,
        }

    monkeypatch.setattr(merge, "verification_binding", fixture_binding)
    monkeypatch.setattr(merge, "source_identity", lambda root: manifest_digest(tree_manifest(root)))
    monkeypatch.setattr(merge, "environment_identity", lambda: {})
    monkeypatch.setattr(merge, "runtime_identity", lambda _: "fixture")
    monkeypatch.setattr(verify_merge_ready, "merge_gate_steps", lambda: ())
    monkeypatch.setattr(
        "slm_training.autoresearch.heal.isolation.probe_isolation",
        lambda: IsolationCapability(True, "simulated", "fixture", "/bin/true"),
    )
    from scripts.merge_verification_isolation import run_workload

    def local_workload(*args, **kwargs):
        kwargs["isolated"] = False
        return run_workload(*args, **kwargs)

    monkeypatch.setattr(merge, "run_workload", local_workload)
    frozen = tmp_path / "source-gate-input"
    if not frozen.exists():
        shutil.copytree(workspace.base, frozen / "base")
        shutil.copytree(workspace.candidate, frozen / "root")
    root = frozen / "root"
    directory = tmp_path / "source-gate-cache"
    with monkeypatch.context() as pending:
        if not complete:
            pending.setattr(merge, "_run_shards", lambda *args, **kwargs: None)
        summary = merge.run_release_gate(
            (), root=root, base_ref="fixture-base", state_dir=directory,
            step_seconds=120.0, run_step=lambda *args, **kwargs: pytest.fail("no fixture statics"),
        )
    assert summary["verification_complete"] is complete, summary
    # Same authenticated cache load and canonical identity as the real producer.
    state = evidence.ReceiptCache(directory, root).load(summary["identity"])
    assert bool(state["passed_nodes"]) is complete and state["attempts"]
    return SourceVerificationGate(
        root, directory, "fixture-base", summary["identity"]
    )
