"""Controller-owned private Git inputs for the existing resumable source verifier.

No host repository metadata, hooks, credentials or worker-authored coverage is
used. Only the pinned base and independently content-checked proposal are copied.
This prepares inputs; the separately granted finite verifier executes all checks.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.harness_core.lineage.store import _atomic_write
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS, INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS,
)

from .isolation_workspace import manifest_digest, private_snapshot, tree_manifest
from .repair_acceptance import SourceVerificationGate, proposal_patch_digest
from .repair_scope import require_routine_scope


def _validate(context, config, request, proposal, workspace):
    if config.source_verification_grant is None:
        raise ValueError("source_verification_grant_missing")
    if config.grant is None or config.grant.expires_at <= time.time():
        raise ValueError("source_verification_repair_authority_expired")
    pairs = (
        (request.campaign_id, context.campaign_id),
        (request.blocker.source_digest, context.source_digest),
        (request.blocker.environment_digest, context.environment_digest),
        (request.grant, config.grant),
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
                          request.allowed_paths, request.semantics_preserving_paths)
    return changed


def _private_git(stage, deadline):
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
            ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", "-c", "core.autocrlf=false",
             "--git-dir=" + str(root / ".git"), "--work-tree=" + str(worktree), *args],
            cwd=worktree, env=env, interrupt_after_seconds=remaining,
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
        if result.returncode != 0 or result.timed_out or result.stdout_truncated:
            raise ValueError("private_source_git_preparation_failed")
        return result.stdout.strip()

    git(root, "init", "--template=" + str(template), "--initial-branch=repair-base")
    git(base, "add", "--force", "--all", "--", ".")
    tree = git(base, "write-tree")
    commit = git(base, "commit-tree", tree, "-m", "Pinned repair input")
    git(base, "update-ref", "refs/heads/repair-base", commit)
    git(root, "add", "--force", "--all", "--", ".")
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


def prepare_source_verification(context, config, request, proposal, workspace):
    """Idempotent controller materialization; no source-verification workload runs."""
    from scripts.merge_verification import verification_binding
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
    identity_inputs = {
        "request_digest": request.digest(), "proposal_digest": proposal.digest(),
        "config_digest": config.digest(),
    }
    destination = namespace / digest(identity_inputs)
    namespace.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not destination.exists():
        stage = Path(tempfile.mkdtemp(prefix="stage-", dir=namespace))
        private_snapshot(workspace.base, stage / "base")
        private_snapshot(workspace.candidate, stage / "root")
        if manifest_digest(tree_manifest(stage / "base")) != context.source_digest:
            raise ValueError("source_verification_base_changed_during_copy")
        base_ref = _private_git(stage, started + INTERRUPT_AFTER_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS)
        binding = verification_binding(stage / "root", base_ref, merge_gate_steps(),
                                       isolated=True, runtimes=runtimes)
        manifest = {
            "schema_version": "repair_source_verification_input/v1",
            **identity_inputs, "source_snapshot_digest": context.source_digest,
            "candidate_snapshot_digest": proposal.tree_digest,
            "base_ref": base_ref, "verification_identity": digest(binding),
            "binding": binding, "grant": config.source_verification_grant.model_dump(mode="json"),
            "coverage_state": "locked_selection_pending_collection",
            "preparation_seconds": time.monotonic() - started,
        }
        _atomic_write(stage / "manifest.json", manifest)
        _durable(stage)
        os.rename(stage, destination)
        fd = os.open(namespace, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    manifest = json.loads((destination / "manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in identity_inputs.items()):
        raise ValueError("source_verification_manifest_mismatch")
    binding = verification_binding(destination / "root", manifest["base_ref"],
                                   merge_gate_steps(), isolated=True, runtimes=runtimes)
    if binding != manifest["binding"] or digest(binding) != manifest["verification_identity"]:
        raise ValueError("source_verification_materialization_changed")
    if not set(changed) <= set(binding["changed_paths"]):
        raise ValueError("source_verification_omits_repair_changes")
    with tempfile.TemporaryDirectory(prefix="slm-source-check-") as temporary:
        frozen = private_snapshot(destination / "root", Path(temporary) / "source")
        if manifest_digest(tree_manifest(frozen)) != proposal.tree_digest:
            raise ValueError("source_verification_materialization_changed")
    artifact = journal.write_artifact("repair_source_verification_inputs", manifest)
    journal.append_event("repair_source_verification_prepared", artifact_sha256=artifact.stem,
                         idempotency_key="source-verification-input:" + artifact.stem)
    return SourceVerificationGate(destination / "root", destination / "cache",
                                  manifest["base_ref"], manifest["verification_identity"])
