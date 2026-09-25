"""Shared cross-lease repair acceptance fixtures."""

import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.activity_contract import ActivitySpec
from slm_training.autoresearch.heal.repair_acceptance import (
    SourceVerificationGate,
    VerificationWorkspace,
    source_verification_activity_id,
)
from slm_training.autoresearch.heal.isolation_workspace import tree_manifest
from slm_training.autoresearch.heal.repair_contracts import RepairDispatchResult
from slm_training.autoresearch.heal.repair_release import (
    verified_activation_handoff,
    verified_release_callback,
)
from slm_training.autoresearch.storage import CampaignStore


def _two_lease_dispatch(context, config, candidate, proposal, monkeypatch, source_available, restored):
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

        workspace = VerificationWorkspace(
            context.source,
            context.root / context.campaign_id / "repair_workspaces" / proposal.request_digest / "candidate",
            runtime_roots=(Path(sys.prefix),),
        )
        gate = source_gate_fixture(
            context.root.parent, monkeypatch, workspace,
            complete=source_available is True and restored,
        ) if source_available is not False else None
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
    from scripts import merge_verification_identity as identity
    from scripts.merge_verification_identity import source_identity
    from slm_training.autoresearch.heal.isolation import IsolationCapability

    def fixture_binding(root, base_ref, steps, **kwargs):
        return {
            "schema": "merge_verification_binding/v2",
            "base_tree": base_ref,
            "candidate_tree_sha256": source_identity(root),
            "changed_paths": sorted(tree_manifest(root)),
            "targets": ["tests"], "static_commands": [],
            "environment": {}, "runtime_roots": [],
            "runtime_identity": kwargs.get("runtime_digest_value")
            or merge.runtime_identity(kwargs.get("runtimes", ())),
            "isolation_enforced": True, "max_attempts_per_obligation": 3,
        }

    monkeypatch.setattr(merge, "verification_binding", fixture_binding)
    monkeypatch.setattr(merge, "environment_identity", lambda **_: {})
    monkeypatch.setattr(identity, "environment_identity", lambda **_: {})
    monkeypatch.setattr(identity, "runtime_identity", lambda _: "f" * 64)
    monkeypatch.setattr(merge, "runtime_identity", lambda _: "f" * 64)
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
        from slm_training.autoresearch.heal.repair_source_workspace import _private_git

        _private_git(frozen, time.monotonic() + 40, ())
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
