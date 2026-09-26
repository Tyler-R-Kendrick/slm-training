"""Current-source successor preparation retains exact repair artifacts and residual grant."""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from scripts import merge_verification
from scripts.autotrain_verification import dependency_plan
from scripts.autotrain_verification_successor import plan_successor
from scripts.verify_merge_ready import merge_gate_steps
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest, private_snapshot, tree_manifest,
)
from slm_training.autoresearch.heal.recovery_dispatch import (
    RecoveryConfig, RecoveryContext, RepairRecipe, build_repair_request,
)
from slm_training.autoresearch.heal.repair_acceptance import VerificationWorkspace
from slm_training.autoresearch.heal.repair_contracts import (
    RepairDispatchResult, RepairGrant, RepairProposal,
)
from slm_training.autoresearch.heal.repair_release import source_verification_callback
from slm_training.autoresearch.heal.repair_source_workspace import _private_git, proposal_patch_digest
from slm_training.autoresearch.heal.repair_verifier import VerificationCheck
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import (
    ActivityOutcome, ActivitySpec, ResourceGrant, WakeCondition,
)


def _make_repair_case(tmp_path):
    source = tmp_path / "source"
    (source / "docs/design").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "AGENTS.md").write_text("Preserve constrained decoding.\n")
    (source / "RTK.md").write_text("Use bounded local verification.\n")
    (source / "docs/design/decode-invariants.md").write_text("I6 remains mandatory.\n")
    (source / "fixture.py").write_text('raise SystemExit("broken")\n')
    (source / "tests/test_existing.py").write_text("def test_existing(): assert True\n")
    source_digest = manifest_digest(tree_manifest(source))
    recipe = RepairRecipe(
        allowed_paths=("fixture.py", "tests/test_added.py"), input_digest="b" * 64,
        original=VerificationCheck("original", (sys.executable, "fixture.py"), "ready\n"),
        checks=(VerificationCheck("regression", (sys.executable, "tests/test_added.py"), "checked\n"),),
        owner_contract_path="AGENTS.md", failure_returncode=1,
        failure_stdout_sha256=hashlib.sha256(b"").hexdigest(),
        failure_stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )
    verifier_grant = ResourceGrant(interrupt_seconds=65, total_seconds=200, max_attempts=3)
    config = RecoveryConfig(
        grant=RepairGrant(grant_id="repair", provider="fixture", executable=sys.executable,
                          executable_sha256="a" * 64, expires_at=time.time() + 120,
                          max_attempts=2, total_seconds=80, interrupt_seconds=10),
        recipes={"harness_code_failure": recipe}, verifier_release="c" * 64,
        source_verification_grant=verifier_grant,
    )
    context = RecoveryContext(tmp_path / "controller", "loop", "campaign", source,
                              source_digest, "d" * 64, "fence", "parent")
    pending = {"kind": "repair_harness", "blocker_code": "harness_code_failure",
               "affected_activity_id": "driver", "unmet_predicate": "ready_stdout",
               "required_capability": "source_repair"}
    request = build_repair_request(pending, context, config)
    candidate = context.root / context.campaign_id / "repair_workspaces" / request.digest() / "candidate"
    candidate.parent.mkdir(parents=True)
    shutil.copytree(source, candidate)
    (candidate / "fixture.py").write_text('print("ready")\n')
    (candidate / "tests/test_added.py").write_text('assert 2 + 2 == 4\nprint("checked")\n')
    proposal = RepairProposal(
        request_digest=request.digest(), tree_digest=manifest_digest(tree_manifest(candidate)),
        patch_digest=proposal_patch_digest(source, candidate), root_cause="fixture failure",
        regression_test="tests/test_added.py", reproduction_artifacts=("e" * 64,),
        classification="implementation",
    )
    journal = CampaignStore(context.campaign_id, context.root)
    journal.write_artifact("repair_requests", request)
    result = RepairDispatchResult(status="waiting_verification", request_digest=request.digest(),
                                 proposal=proposal, reason="proposal_saved")
    result_artifact = journal.write_artifact("repair_dispatch", result)
    journal.append_event("repair_dispatch", experiment_id=request.activity_id,
                         artifact_sha256=result_artifact.stem,
                         detail={"request_digest": request.digest()})
    workspace = VerificationWorkspace(source, candidate)
    return locals()


def _make_predecessor(tmp_path, case, monkeypatch):
    context, config = case["context"], case["config"]
    request, proposal, workspace = case["request"], case["proposal"], case["workspace"]
    source, candidate = case["source"], case["candidate"]
    verifier_grant = case["verifier_grant"]
    real_binding = merge_verification.verification_binding
    roots = (Path(sys.prefix).resolve(),)
    stage = tmp_path / "old-binding-stage"
    stage.mkdir()
    private_snapshot(source, stage / "base")
    private_snapshot(candidate, stage / "root")
    base_ref = _private_git(stage, time.monotonic() + 60, ("fixture.py", "tests/test_added.py"))
    old_binding = real_binding(stage / "root", base_ref, merge_gate_steps(),
                               isolated=True, runtimes=roots)
    old_binding = {**old_binding, "environment": {"predecessor": "saved"}}
    monkeypatch.setattr(merge_verification, "verification_binding", lambda *a, **kw: old_binding)
    old_gate = source_verification_callback(context, config)(request, proposal, workspace)
    old_manifest_path = old_gate.root.parent / "manifest.json"
    predecessor = {
        "schema_version": "repair_verification_dependency/v1",
        "request_digest": request.digest(), "proposal_digest": proposal.digest(),
        "candidate_snapshot_digest": proposal.tree_digest,
        "campaign_id": request.campaign_id, "activity_id": "repair",
        "root": str(old_gate.root), "state_dir": str(old_gate.state_dir),
        "manifest_path": str(old_manifest_path), "base_ref": old_gate.base_ref,
        "runtime_roots": [str(path) for path in roots],
        "verification_identity": old_gate.identity,
        "grant": verifier_grant.model_dump(mode="json"),
        "wake": {"predicate": "complete_current_source_verification",
                 "source": "source_verification_completed", "identity_digest": old_gate.identity},
    }
    from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id

    predecessor["activity_id"] = source_verification_activity_id(old_gate.identity, verifier_grant)
    predecessor["manifest_path"] = str(old_manifest_path)
    dependency_store = CampaignStore("runtime", context.root / "loops" / "loop")
    predecessor_artifact = dependency_store.write_artifact("source_verification_requests", predecessor)
    monkeypatch.setattr(merge_verification, "verification_binding", lambda *a, **kw: old_binding)
    from scripts import autotrain_verification as owner
    monkeypatch.setattr(owner, "verification_binding", lambda *a, **kw: old_binding)
    old_plan = dependency_plan(predecessor)
    return locals()


def test_stale_runtime_binding_materializes_successor_with_only_remaining_budget(tmp_path, monkeypatch):
    case = _make_repair_case(tmp_path)
    old = _make_predecessor(tmp_path, case, monkeypatch)
    config, request = case["config"], case["request"]
    proposal, verifier_grant = case["proposal"], case["verifier_grant"]
    source_digest = case["source_digest"]
    predecessor, predecessor_artifact = old["predecessor"], old["predecessor_artifact"]
    owner, old_plan = old["owner"], old["old_plan"]
    dependency_store = old["dependency_store"]
    with ActivityRuntime(dependency_store) as runtime:
        runtime.register(ActivitySpec(
            activity_id="repair", family="repair", kind="repair",
            source_digest=source_digest, environment_digest="d" * 64,
            input_digest=request.digest(), output_namespace="attempts/repair",
        ))
        repair_lease = runtime.claim_next(capabilities={"local_process"}, activity_id="repair")
        runtime.finish(repair_lease, outcome=ActivityOutcome.DEPENDENCY, spent_seconds=0,
                       wake=WakeCondition.model_validate(predecessor["wake"]))
        owner.register_dependency(runtime, old_plan)
        old_lease = runtime.claim_next(capabilities={"local_process", "isolated_verifier"},
                                       activity_id=predecessor["activity_id"])
        runtime.finish(old_lease, outcome=ActivityOutcome.YIELDED, spent_seconds=4)
        predecessor_event = runtime.store.append_event(
            "source_verification_requested", experiment_id="repair",
            artifact_sha256=predecessor_artifact.stem,
            detail={"dependency_digest": predecessor_artifact.stem,
                    "repair_activity_id": "repair"},
        )
        monkeypatch.setattr(merge_verification, "verification_binding", old["real_binding"])
        monkeypatch.setattr(owner, "verification_binding", old["real_binding"])
        config_path = tmp_path / "recovery.json"
        config_path.write_text(config.model_dump_json())
        common = {"repair_config": str(config_path),
                  "repair_config_digest": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  "loop_id": "loop"}

        plan_successor(runtime, predecessor_event, predecessor, common)

        successor_event = next(row for row in runtime.store.verify_event_chain()
                               if row["event_type"] == "source_verification_requested"
                               and row["detail"]["dependency_digest"] != predecessor_artifact.stem)
        successor = owner.load_dependency(runtime.store, successor_event)
        plan = dependency_plan(successor)
        state = runtime.snapshot()[successor["activity_id"]]
        assert successor["request_digest"] == request.digest()
        assert successor["proposal_digest"] == proposal.digest()
        assert successor["verification_identity"] != predecessor["verification_identity"]
        assert state.spec.grant.total_seconds == verifier_grant.total_seconds - 4
        assert state.spec.grant.max_attempts == verifier_grant.max_attempts - 1
        assert plan["identity"] == successor["verification_identity"]
        assert json.loads(Path(successor["manifest_path"]).read_text())["grant"] == state.spec.grant.model_dump(mode="json")
