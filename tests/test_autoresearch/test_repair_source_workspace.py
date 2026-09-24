"""Actual private Git/source selection and dispatch; agent execution is simulated."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.merge_verification import verification_binding
from scripts.merge_verification_evidence import digest
from scripts.verify_merge_ready import merge_gate_steps
from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, tree_manifest
from slm_training.autoresearch.heal.recovery_dispatch import (
    RecoveryConfig, RecoveryContext, RepairRecipe, build_repair_request, dispatch_hard_pending,
)
from slm_training.autoresearch.heal.repair_governance import reconcile_version_overlay
from slm_training.autoresearch.heal.repair_acceptance import VerificationWorkspace, proposal_patch_digest
from slm_training.autoresearch.heal.repair_contracts import RepairDispatchResult, RepairGrant, RepairProposal
from slm_training.autoresearch.heal.repair_release import source_verification_callback
from slm_training.autoresearch.heal.repair_source_workspace import _private_git
from slm_training.autoresearch.heal.repair_verifier import VerificationCheck
from slm_training.harness_core.activity_contract import ResourceGrant
from slm_training.autoresearch.storage import CampaignStore


@pytest.fixture
def repair_inputs(tmp_path):
    source = tmp_path / "pinned-source"
    (source / "docs/design").mkdir(parents=True)
    for name in ("AGENTS.md", "RTK.md", "docs/design/decode-invariants.md"):
        (source / name).write_text("Preserve I6 and controller-owned verification.\n")
    (source / "fixture.py").write_text('print("broken")\nraise SystemExit(1)\n')
    (source / "tests").mkdir()
    (source / "tests/test_existing.py").write_text("def test_existing():\n    assert 1 + 1 == 2\n")
    (source / ".gitignore").write_text("tests/test_added.py\n")
    grant = RepairGrant(grant_id="fixture", provider="fixture", executable=sys.executable,
                        executable_sha256="a" * 64, expires_at=time.time() + 120,
                        max_attempts=2, total_seconds=80.0, interrupt_seconds=10)
    recipe = RepairRecipe(
        allowed_paths=("fixture.py", "tests/test_added.py"), input_digest="b" * 64,
        original=VerificationCheck("original", (sys.executable, "fixture.py"), "ready\n"),
        checks=(VerificationCheck("regression", (sys.executable, "tests/test_added.py"), "checked\n"),),
        owner_contract_path="AGENTS.md", failure_returncode=1,
        failure_stdout_sha256=hashlib.sha256(b"broken\n").hexdigest(),
        failure_stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )
    config = RecoveryConfig(grant=grant, recipes={"harness_code_failure": recipe},
                            verifier_release="c" * 64, source_verification_grant=ResourceGrant())
    context = RecoveryContext(tmp_path / "controller", "loop", "campaign", source,
                              manifest_digest(tree_manifest(source)), "d" * 64, "fence", "parent")
    pending = dict(kind="repair_harness", blocker_code="harness_code_failure",
                   affected_activity_id="decode", unmet_predicate="ready_stdout", required_capability="source_repair")
    request = build_repair_request(pending, context, config)
    candidate = context.root / context.campaign_id / "repair_workspaces" / request.digest() / "candidate"
    shutil.copytree(source, candidate)
    (candidate / "fixture.py").write_text('print("ready")\n')
    (candidate / "tests/test_added.py").write_text('def test_added():\n    assert 2 + 2 == 4\nprint("checked")\n')
    proposal = RepairProposal(request_digest=request.digest(), tree_digest=manifest_digest(tree_manifest(candidate)),
                              patch_digest=proposal_patch_digest(source, candidate), root_cause="ordinary fixture defect",
                              regression_test="tests/test_added.py", reproduction_artifacts=("e" * 64,), classification="implementation")
    return context, config, pending, request, proposal, VerificationWorkspace(source, candidate)


def test_actual_factory_binds_full_source_coverage_private_git_and_separate_grant(repair_inputs):
    context, config, _, request, proposal, workspace = repair_inputs
    resolve = source_verification_callback(context, config)
    gate = resolve(request, proposal, workspace)
    assert resolve(request, proposal, workspace) == gate
    assert not (context.source / ".git").exists()
    assert (gate.root / ".git").is_dir() and not (gate.root / ".git").is_symlink()
    assert not (gate.root / ".git/objects/info/alternates").exists()
    assert list((gate.root / ".git/hooks").glob("*")) == []
    assert all(path.stat().st_nlink == 1 for path in (gate.root / ".git").rglob("*") if path.is_file())
    manifest = json.loads((gate.root.parent / "manifest.json").read_text())
    binding = verification_binding(gate.root, gate.base_ref, merge_gate_steps(),
                                   isolated=True, runtimes=(Path(sys.prefix),))
    assert digest(binding) == gate.identity
    assert binding == manifest["binding"]
    assert binding["changed_paths"] == ["fixture.py", "tests/test_added.py"]
    assert binding["targets"] == ["tests"], "unknown source plus added test must select conservative source coverage"
    assert len(binding["static_commands"]) == len([step for step in merge_gate_steps() if step.static])
    assert manifest["grant"] == config.source_verification_grant.model_dump(mode="json")
    assert gate.read(workspace) is None
    assert not gate.state_dir.exists(), "preparing a job must not issue a passing cache record"


def test_private_git_tracks_granted_ignored_path_without_tracking_cache(tmp_path):
    stage = tmp_path / "stage"
    base, root = stage / "base", stage / "root"
    for worktree in (base, root):
        (worktree / ".hypothesis/constants").mkdir(parents=True)
        (worktree / ".hypothesis/constants/cache").write_text("runtime cache\n")
        (worktree / ".gitignore").write_text(
            ".hypothesis/\ntests/test_added.py\ntests/literal*.py\n"
        )
        (worktree / "tests").mkdir()
        (worktree / "tracked.py").write_text("VALUE = 1\n")
        (worktree / ".autonomy-release.json").write_text("{}\n")
        (worktree / "tests/literal1.py").write_text("SHOULD_NOT_BE_TRACKED = True\n")
    (root / "tracked.py").write_text("VALUE = 2\n")
    (root / "tests/test_added.py").write_text("def test_added():\n    assert True\n")
    (root / "tests/literal[1].py").write_text("LITERAL = True\n")

    _private_git(stage, time.monotonic() + 40, (
        "tracked.py", "tests/test_added.py", "tests/literal[1].py",
    ))
    tracked = subprocess.check_output(
        ["/usr/bin/git", "--git-dir=" + str(root / ".git"),
         "--work-tree=" + str(root), "ls-files"],
        text=True,
    ).splitlines()

    assert "tests/test_added.py" in tracked
    assert "tests/literal[1].py" in tracked
    assert "tests/literal1.py" not in tracked
    assert not any(path.startswith(".hypothesis/") for path in tracked)
    assert ".autonomy-release.json" not in tracked


def test_controller_bumps_watched_component_after_worker_teardown(repair_inputs):
    context, _, _, request, proposal, workspace = repair_inputs
    registry = workspace.base / "src/slm_training/resources/versions.json"
    registry.parent.mkdir(parents=True)
    payload = {
        "schema": "version_registry/v1",
        "components": {
            "fixture.repair": {
                "kind": "harness",
                "version": "v1",
                "paths": ["fixture.py"],
                "history": [{"date": "2026-09-01", "note": "initial", "version": "v1"}],
            }
        },
    }
    registry.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry.chmod(0o444)
    target = workspace.candidate / registry.relative_to(workspace.base)
    target.parent.mkdir(parents=True)
    shutil.copyfile(registry, target)
    target.chmod(0o444)
    proposal = proposal.model_copy(update={
        "tree_digest": manifest_digest(tree_manifest(workspace.candidate)),
        "patch_digest": proposal_patch_digest(workspace.base, workspace.candidate),
    })
    result = RepairDispatchResult(status="waiting_verification", request_digest=request.digest(),
                                  proposal=proposal, reason="proposal_received")
    journal = CampaignStore(context.campaign_id, context.root)

    governed = reconcile_version_overlay(request, result, workspace, journal)
    component = json.loads(target.read_text())["components"]["fixture.repair"]

    assert component["version"] == component["history"][0]["version"] == "v2"
    assert target.stat().st_mode & 0o777 == 0o444
    assert governed.proposal.tree_digest == manifest_digest(tree_manifest(workspace.candidate))
    assert governed.proposal != proposal
    assert reconcile_version_overlay(request, governed, workspace, journal) == governed
    assert len(list((journal.root / "artifacts/repair_governance_overlays").glob("*.json"))) == 1


def test_interrupted_source_preparation_removes_unpublished_stage(repair_inputs, monkeypatch):
    context, config, _, request, proposal, workspace = repair_inputs

    def interrupt(_root):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "slm_training.autoresearch.heal.repair_source_workspace._durable", interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        source_verification_callback(context, config)(request, proposal, workspace)

    namespace = context.root / context.campaign_id / "source_verification"
    assert not list(namespace.glob("stage-*"))


def test_readonly_grant_preserves_untouched_mode_change_outside_git_diff(repair_inputs):
    context, config, pending, _, proposal, workspace = repair_inputs
    from slm_training.autoresearch.heal.isolated_agent import _create_regression_mounts

    for root in (workspace.base, workspace.candidate):
        (root / "untouched.py").write_text("UNCHANGED = True\n")
        for path in (root / "fixture.py", root / "untouched.py"):
            path.chmod(0o444)
    recipe = config.recipes["harness_code_failure"]
    recipe = recipe.model_copy(update={"allowed_paths": (*recipe.allowed_paths, "untouched.py")})
    config = config.model_copy(update={"recipes": {"harness_code_failure": recipe}})
    context = replace(context, source_digest=manifest_digest(tree_manifest(workspace.base)))
    request = build_repair_request(pending, context, config)
    destination = context.root / context.campaign_id / "repair_workspaces" / request.digest() / "candidate"
    destination.parent.mkdir(parents=True)
    workspace.candidate.rename(destination)
    workspace = replace(workspace, candidate=destination)
    _create_regression_mounts(destination, ("fixture.py", "untouched.py"))
    proposal = proposal.model_copy(update={"request_digest": request.digest(),
        "tree_digest": manifest_digest(tree_manifest(destination)),
        "patch_digest": proposal_patch_digest(workspace.base, destination)})
    gate = source_verification_callback(context, config)(request, proposal, workspace)
    binding = json.loads((gate.root.parent / "manifest.json").read_text())["binding"]
    assert binding["changed_paths"] == ["fixture.py", "tests/test_added.py"]
    assert (workspace.base / "untouched.py").stat().st_mode & 0o777 == 0o444
    assert (gate.root / "untouched.py").stat().st_mode & 0o777 == 0o644
    assert tree_manifest(gate.root)["untouched.py"] == tree_manifest(destination)["untouched.py"]
    assert proposal.tree_digest == manifest_digest(tree_manifest(destination))
    assert proposal.patch_digest == proposal_patch_digest(workspace.base, destination)


@pytest.mark.parametrize("initially_granted", [False, True])
def test_real_dispatch_factory_and_pending_consumer_do_not_rerun_agent(repair_inputs, monkeypatch, initially_granted):
    context, config, pending, _, proposal, _ = repair_inputs
    from scripts.autotrain_verification import dependency_plan

    extra_runtime = context.root.parent / "dedicated-runtime"
    extra_runtime.mkdir()
    config = config.model_copy(update={"runtime_roots": (sys.prefix, str(extra_runtime))})
    calls = []

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def capability(self, request):
            return None

        def execute(self, request, **kwargs):
            calls.append(request.digest())
            return RepairDispatchResult(status="waiting_verification", request_digest=request.digest(),
                                        proposal=proposal, reason="proposal_received")

    monkeypatch.setattr("slm_training.autoresearch.heal.recovery_dispatch.CodexExecutor", FakeAgent)
    initial_config = config if initially_granted else config.model_copy(update={"source_verification_grant": None})
    resolve = source_verification_callback(context, initial_config)
    first = dispatch_hard_pending(pending, context, config=initial_config, fence_valid=lambda _: True,
                                 source_verification=resolve)
    second_context = replace(context, fence="new-fence", attempt_id="attempt-1")
    second = dispatch_hard_pending(pending, second_context, config=config, fence_valid=lambda _: True,
                                   source_verification=source_verification_callback(second_context, config))
    assert len(calls) == 1
    assert first["status"] == second["status"] == "waiting_verification"
    if initially_granted:
        assert first["verification_dependency"] == second["verification_dependency"]
    else:
        assert first["verification_dependency"]["grant"] is None
        assert first["verification_dependency"]["activity_id"] is None
        assert initial_config.proposal_config_digest() == config.proposal_config_digest()
    dependency = second["verification_dependency"]
    plan = dependency_plan(dependency)
    assert plan["runtime_roots"] == [str(Path(path).resolve()) for path in config.runtime_roots]
    assert plan["identity"] == dependency["verification_identity"]
    assert dependency["grant"] == config.source_verification_grant.model_dump(mode="json")
    assert dependency["wake"]["source"] == "source_verification_completed"
    from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id
    assert dependency["activity_id"] == source_verification_activity_id(
        dependency["verification_identity"], dependency["grant"]
    )
    assert Path(dependency["manifest_path"]).is_file()
    assert not Path(dependency["state_dir"]).exists()
    assert second["verification"] is None and "release_handoff" not in second


def test_legacy_missing_grant_materializes_nothing(repair_inputs):
    context, config, _, request, proposal, workspace = repair_inputs
    legacy = config.model_copy(update={"source_verification_grant": None})
    assert source_verification_callback(context, legacy)(request, proposal, workspace) is None
    assert not (context.root / context.campaign_id / "source_verification").exists()
    payload = config.model_dump(mode="json")
    del payload["source_verification_grant"]
    assert RecoveryConfig.model_validate_json(json.dumps(payload)).source_verification_grant is None


@pytest.mark.parametrize("grant", [{"total_seconds": "1000"}, {"max_attempts": True}, {"interrupt_seconds": 171}, {"unexpected": 1}])
def test_source_grant_is_strict_and_capped(grant):
    with pytest.raises(ValidationError):
        RecoveryConfig(verifier_release="a" * 64, source_verification_grant=grant)


def test_changed_candidate_cannot_reuse_prepared_source(repair_inputs):
    context, config, _, request, proposal, workspace = repair_inputs
    resolve = source_verification_callback(context, config)
    resolve(request, proposal, workspace)
    (workspace.candidate / "fixture.py").write_text("raise SystemExit(99)\n")
    with pytest.raises(ValueError, match="input_mismatch"):
        resolve(request, proposal, workspace)


def test_private_materialization_drift_is_not_rebuilt_or_accepted(repair_inputs):
    context, config, _, request, proposal, workspace = repair_inputs
    resolve = source_verification_callback(context, config)
    gate = resolve(request, proposal, workspace)
    (gate.root / "fixture.py").write_text("raise SystemExit(99)\n")
    with pytest.raises(ValueError, match="materialization_changed"):
        resolve(request, proposal, workspace)

def test_source_verification_reuses_materialized_root_without_copying_it_again(repair_inputs, monkeypatch):
    from slm_training.autoresearch.heal import repair_source_workspace as source_workspace

    context, config, _, request, proposal, workspace = repair_inputs
    copied = []
    private_snapshot = source_workspace.private_snapshot

    def record_copy(source, destination):
        copied.append(destination)
        return private_snapshot(source, destination)

    monkeypatch.setattr(source_workspace, "private_snapshot", record_copy)
    resolve = source_verification_callback(context, config)
    gate = resolve(request, proposal, workspace)

    assert gate is not None
    assert len(copied) == 2  # base and candidate only
    resumed = resolve(request, proposal, workspace)
    assert resumed.root == gate.root
    assert len(copied) == 2  # resume verifies and reuses the durable materialization  # base and candidate; no third full-tree verification copy

def test_cached_source_base_drift_is_not_reused(repair_inputs):
    context, config, _, request, proposal, workspace = repair_inputs
    resolve = source_verification_callback(context, config)
    gate = resolve(request, proposal, workspace)
    (gate.root.parent / "base" / "fixture.py").write_text("tampered base\\n")

    with pytest.raises(ValueError, match="source_verification_base_changed"):
        resolve(request, proposal, workspace)


def test_verifier_rule_change_gets_fresh_materialization(repair_inputs, monkeypatch):
    from scripts import merge_verification

    context, config, _, request, proposal, workspace = repair_inputs
    original = merge_verification.verification_binding
    revision = {"value": "one"}

    def binding(*args, **kwargs):
        value = original(*args, **kwargs)
        return {**value, "fixture_rule_revision": revision["value"]}

    monkeypatch.setattr(merge_verification, "verification_binding", binding)
    resolve = source_verification_callback(context, config)
    first = resolve(request, proposal, workspace)
    revision["value"] = "two"
    second = resolve(request, proposal, workspace)
    assert first.identity != second.identity
    assert first.root.parent != second.root.parent
    assert first.root.is_dir() and second.root.is_dir()


def test_real_private_source_gate_executes_bounded_pending_work_without_false_release(repair_inputs):
    from scripts.merge_verification import run_locked_release_gate
    from slm_training.autoresearch.heal.isolation import probe_isolation

    capability = probe_isolation()
    if not capability.available:
        if os.environ.get("SLM_REQUIRE_ISOLATION"):
            pytest.fail(capability.reason)
        pytest.skip(capability.reason)
    context, config, _, request, proposal, workspace = repair_inputs
    gate = source_verification_callback(context, config)(request, proposal, workspace)
    summary = run_locked_release_gate(
        gate.identity, dict(steps=merge_gate_steps(), root=gate.root, base_ref=gate.base_ref,
                            state_dir=gate.state_dir, step_seconds=2, run_step=None,
                            runtime_roots=(Path(sys.prefix),), local_feedback=False),
    )
    # This tiny source intentionally lacks the full repository's static modules.
    # Actual failed/pending work must remain incomplete, not a release canary.
    assert not summary["verification_complete"] and not summary["release_authorized"]
    assert gate.read(workspace) is None
    from scripts.merge_verification_evidence import ReceiptCache

    state = ReceiptCache(gate.state_dir, gate.root).load(gate.identity)
    assert state["static"], "actual isolated static workers must have run"
    assert all(row["evidence_class"] == "isolated_process" and row["seconds"] > 0
               for row in state["static"].values())
    assert state["binding"]["targets"] == ["tests"]
