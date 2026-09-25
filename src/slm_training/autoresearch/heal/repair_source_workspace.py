"""Controller-owned private Git inputs for the existing resumable source verifier.

No host repository metadata, hooks, credentials or worker-authored coverage is
used. Only the pinned base and independently content-checked proposal are copied.
This prepares inputs; the separately granted finite verifier executes all checks.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.harness_core.execution_release import MARKER
from slm_training.harness_core.lineage.store import _atomic_write
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS, INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS,
)

from .isolation_workspace import (
    manifest_digest, owner_write_preparation, private_snapshot,
    private_snapshot_manifest, tree_manifest,
)
from .repair_acceptance import SourceVerificationGate, proposal_patch_digest
from .repair_scope import require_routine_scope


def reuse_completed_source_verification(context, request, proposal, workspace):
    """Reuse only a completed authenticated receipt for this exact proposal."""
    from scripts import autotrain_verification as owner

    store = CampaignStore("runtime", context.root / "loops" / context.loop_id)
    events = store.verify_event_chain()
    completed = [row for row in events if row["event_type"] == "source_verification_completed"]
    requested = [row for row in events if row["event_type"] == "source_verification_requested"]
    requested.extend(_completed_request_events(completed))
    for event in reversed(requested):
        gate = _completed_gate(store, owner, completed, event, request, proposal, workspace)
        if gate is not None:
            return gate
    return None


def _completed_request_events(completed):
    return [
        {"experiment_id": row["experiment_id"],
         "detail": {"dependency_digest": row["detail"]["dependency_digest"],
                    "repair_activity_id": row["experiment_id"]}}
        for row in completed if row.get("detail", {}).get("dependency_digest")
    ]


def _completed_gate(store, owner, completed, event, request, proposal, workspace):
    try:
        dependency = owner.load_dependency(store, event)
        if (dependency["request_digest"] != request.digest()
                or dependency["proposal_digest"] != proposal.digest()):
            return None
        identity = dependency["verification_identity"]
        if not any(
            row["experiment_id"] == event["experiment_id"]
            and row.get("detail", {}).get("verification_identity", identity) == identity
            for row in completed
        ):
            return None
        owner.dependency_plan(dependency)
        gate = SourceVerificationGate(
            Path(dependency["root"]), Path(dependency["state_dir"]),
            dependency["base_ref"], identity, dependency.get("runtime_identity"),
        )
        return gate if gate.read(workspace) is not None else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _validate(context, config, request, proposal, workspace):
    if config.source_verification_grant is None:
        raise ValueError("source_verification_grant_missing")
    pairs = (
        (request.campaign_id, context.campaign_id),
        (request.blocker.source_digest, context.source_digest),
        (request.blocker.environment_digest, context.environment_digest),
        (proposal.request_digest, request.digest()),
        (workspace.base.resolve(), context.source.resolve()),
        (tuple(path.resolve() for path in (workspace.runtime_roots or (Path(sys.prefix),))),
         tuple(Path(path).resolve() for path in (config.runtime_roots or (sys.prefix,)))),
        (manifest_digest(tree_manifest(workspace.base)), context.source_digest),
        (manifest_digest(tree_manifest(workspace.candidate)), proposal.tree_digest),
        (proposal_patch_digest(workspace.base, workspace.candidate), proposal.patch_digest),
    )
    if any(actual != expected for actual, expected in pairs):
        raise ValueError("source_verification_input_mismatch")
    before, after = tree_manifest(workspace.base), tree_manifest(workspace.candidate)
    changed = tuple(sorted(
        path for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
        and not (path not in before and (workspace.candidate / path).is_dir())
    ))
    require_routine_scope(workspace.base, workspace.candidate, changed,
                          request.allowed_paths, request.semantics_preserving_paths,
                          request_digest=request.digest())
    return changed


def _private_git(stage, deadline, changed):
    """Create local objects from frozen bytes, never clone/link shared metadata."""
    root, base = stage / "root", stage / "base"
    template = stage / "empty-template"
    template.mkdir()
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(stage), "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "",
        "GIT_AUTHOR_NAME": "SLM controller", "GIT_COMMITTER_NAME": "SLM controller",
        "GIT_AUTHOR_EMAIL": "controller@invalid", "GIT_COMMITTER_EMAIL": "controller@invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
    }

    def git(worktree, *args):
        remaining = deadline - time.monotonic() - KILL_GRACE_SECONDS
        if remaining <= 0:
            raise TimeoutError("source_verification_preparation_budget_exhausted")
        result = run_bounded_process(
            ["/usr/bin/git", "--literal-pathspecs", "-c", "core.hooksPath=/dev/null",
             "-c", "core.autocrlf=false",
             "--git-dir=" + str(root / ".git"), "--work-tree=" + str(worktree), *args],
            cwd=worktree, env=env, interrupt_after_seconds=remaining,
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
        if result.returncode != 0 or result.timed_out or result.stdout_truncated:
            raise ValueError("private_source_git_preparation_failed")
        return result.stdout.strip()

    git(root, "init", "--template=" + str(template), "--initial-branch=repair-base")
    (root / ".git/info").mkdir()
    (root / ".git/info/exclude").write_text(MARKER + "\n")
    git(base, "add", "--all", "--", ".")
    tree = git(base, "write-tree")
    commit = git(base, "commit-tree", tree, "-m", "Pinned repair input")
    git(base, "update-ref", "refs/heads/repair-base", commit)
    git(root, "add", "--all", "--", ".")
    if changed:
        git(root, "add", "--force", "--all", "--", *changed)
    return commit


def _durable(root):
    for directory, _, files in os.walk(root, followlinks=False):
        for path in [*(Path(directory) / name for name in files), Path(directory)]:
            if path.is_symlink():
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def _reuse_source_verification(
    namespace, identity_base, steps, runtimes, *, runtime_digest=None
):
    from scripts.merge_verification import runtime_identity, verification_binding
    from scripts.merge_verification_evidence import digest

    runtime_digest = runtime_digest or runtime_identity(runtimes)
    for path in sorted(namespace.glob("*/manifest.json")):
        prior = json.loads(path.read_text())
        if any(prior.get(key) != value for key, value in identity_base.items()):
            continue
        if not (path.parent / "root").is_dir():
            continue
        binding = verification_binding(
            path.parent / "root", prior["base_ref"], steps,
            isolated=True, runtimes=runtimes, runtime_digest_value=runtime_digest,
        )
        identity_inputs = {**identity_base, "verification_identity": digest(binding)}
        if (
            prior.get("verification_identity") == identity_inputs["verification_identity"]
            and prior.get("binding") == binding
        ):
            return path.parent, identity_inputs, binding
    return None


def _materialize_source_verification(context, config, request, proposal, workspace):
    from scripts.merge_verification import runtime_identity, verification_binding
    from scripts.merge_verification_evidence import digest
    from scripts.verify_merge_ready import merge_gate_steps

    started = time.monotonic()
    changed = _validate(context, config, request, proposal, workspace)
    journal = CampaignStore(context.campaign_id, context.root)
    candidate = journal.root / "repair_workspaces" / request.digest() / "candidate"
    if workspace.candidate.resolve() != candidate.resolve():
        raise ValueError("source_verification_candidate_namespace_mismatch")
    runtimes = workspace.runtime_roots or (Path(sys.prefix),)
    namespace = journal.root / "source_verification"
    for exposed in (workspace.base, workspace.candidate, *runtimes):
        if namespace.resolve().is_relative_to(exposed.resolve()) or exposed.resolve().is_relative_to(namespace.resolve()):
            raise ValueError("source_verification_controller_namespace_exposed")
    namespace.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity_base = {
        "request_digest": request.digest(),
        "proposal_digest": proposal.digest(),
        "config_digest": config.digest(),
    }
    steps = merge_gate_steps()
    runtime_digest = runtime_identity(runtimes)
    reused = _reuse_source_verification(namespace, identity_base, steps, runtimes,
                                        runtime_digest=runtime_digest)
    if reused is not None:
        destination, identity_inputs, binding = reused
        return journal, destination, changed, identity_inputs, runtimes, binding

    stage = Path(tempfile.mkdtemp(prefix="stage-", dir=namespace))
    published = False
    try:
        private_snapshot(workspace.base, stage / "base")
        private_snapshot(workspace.candidate, stage / "root")
        if manifest_digest(tree_manifest(stage / "base")) != context.source_digest:
            raise ValueError("source_verification_base_changed_during_copy")
        base_ref = _private_git(
            stage,
            started + INTERRUPT_AFTER_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS,
            changed,
        )
        binding = verification_binding(
            stage / "root", base_ref, steps, isolated=True, runtimes=runtimes,
            runtime_digest_value=runtime_digest,
        )
        identity_inputs = {**identity_base, "verification_identity": digest(binding)}
        destination = namespace / digest(identity_inputs)
        if destination.exists():
            raise ValueError("source_verification_materialization_changed")
        manifest = {
            "schema_version": "repair_source_verification_input/v1",
            **identity_inputs,
            "source_snapshot_digest": context.source_digest,
            "candidate_snapshot_digest": proposal.tree_digest,
            "base_ref": base_ref,
            "binding": binding,
            "grant": config.source_verification_grant.model_dump(mode="json"),
            "coverage_state": "locked_selection_pending_collection",
            "preparation_seconds": time.monotonic() - started,
        }
        _atomic_write(stage / "manifest.json", manifest)
        _durable(stage)
        os.rename(stage, destination)
        published = True
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    if published:
        fd = os.open(namespace, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return journal, destination, changed, identity_inputs, runtimes, binding


def prepare_source_verification(context, config, request, proposal, workspace):
    """Idempotent controller materialization; no source-verification workload runs."""
    from scripts.merge_verification_evidence import digest

    journal, destination, changed, identity_inputs, runtimes, binding = (
        _materialize_source_verification(context, config, request, proposal, workspace)
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in identity_inputs.items()):
        raise ValueError("source_verification_manifest_mismatch")
    if (
        manifest.get("source_snapshot_digest") != context.source_digest
        or manifest.get("candidate_snapshot_digest") != proposal.tree_digest
    ):
        raise ValueError("source_verification_manifest_mismatch")
    if binding != manifest["binding"] or digest(binding) != manifest["verification_identity"]:
        raise ValueError("source_verification_materialization_changed")
    before = tree_manifest(destination / "base")
    after = tree_manifest(destination / "root")
    if manifest_digest(before) != context.source_digest:
        raise ValueError("source_verification_base_changed")
    if any(path not in binding["changed_paths"] and not owner_write_preparation(before.get(path), after.get(path))
           for path in changed):
        raise ValueError("source_verification_omits_repair_changes")
    if manifest_digest(private_snapshot_manifest(destination / "root")) != proposal.tree_digest:
        raise ValueError("source_verification_materialization_changed")
    artifact = journal.write_artifact("repair_source_verification_inputs", manifest)
    journal.append_event("repair_source_verification_prepared", artifact_sha256=artifact.stem,
                         idempotency_key="source-verification-input:" + artifact.stem)
    return SourceVerificationGate(destination / "root", destination / "cache",
                                  manifest["base_ref"], manifest["verification_identity"],
                                  manifest["binding"]["runtime_identity"])
