"""Private true-base verification input for one authenticated accepted repair.

The host pins repository/base/successor and controller-owned source/cache paths.
Only existing verifier metadata preparation writes private Git state; the host
checkout is read-only and no network or GitHub mutation occurs here.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.merge_verification import verification_binding
from scripts.merge_verification_evidence import digest, source_paths
from scripts.merge_verification_git import _private_git
from scripts.verify_merge_ready import merge_gate_steps
from slm_training.autoresearch.heal.isolation_workspace import private_snapshot
from slm_training.autoresearch.runtime.operations_verification import verification_environment
from slm_training.harness_core.execution_release import _files
from slm_training.harness_core.github_delivery_tree import source_tree
from slm_training.levers import INTERRUPT_AFTER_SECONDS, HARNESS_FINALIZATION_RESERVE_SECONDS


def delivery_environment(host):
    """Launch the pinned -I host with the gate's declared Python environment.

    Those variables cannot choose host imports (-I plus absolute trusted roots).
    Preserve authenticated values for the gate's environment comparison rather
    than dropping them and making every supervisor-produced gate unverifiable.
    The values come from the controller's plan, never a connector request.
    """
    captured = (host.verification_plan or {}).get("environment", {})
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX"):
        env.pop(key, None)
        if key in captured:
            env[key] = captured[key]
    return env


def delivery_host(store, wait, host):
    if (host is None or not host.authorized or host.expires_at <= time.time()
            or wait.get("kind") != "verified_repair_source"):
        return host
    from scripts.github_source_authority import successor_host
    host = successor_host(store, wait, host)
    return resolve_source_host(store, wait, host)


def resolve_source_host(store, wait, host):
    """Resolve an explicitly configured preparation; ordinary pinned plans stay strict.

    Optional host verification_plan keys: repository_source (read-only checkout
    whose HEAD is base_ref), source (new private candidate), state_dir (private
    gate cache), execution (accepted successor), runtime_roots, and grant.
    The returned host carries the derived exact identity; no config is rewritten.
    """
    from slm_training.autoresearch.heal.repair_delivery import resolve_source_delivery

    configured = host.verification_plan or {}
    if not {"repository_source", "remote_base_commit_object"} & configured.keys():
        return host
    subject = resolve_source_delivery(store, wait)
    if (subject["successor_source_digest"] != host.source_digest
            or Path(configured["execution"]).resolve() != Path(subject["successor_execution"]).resolve()):
        raise ValueError("source_preparation_accepted_successor_mismatch")
    root = Path(configured["source"]).resolve()
    repository = Path(configured["repository_source"]).resolve(strict=True) if "repository_source" in configured else None
    candidate = Path(subject["successor"]["verified_source"]).resolve(strict=True)
    roots = tuple(Path(path).resolve() for path in configured.get("runtime_roots", ())) or (Path(sys.prefix).resolve(),)
    exposed = (candidate, Path(subject["successor_execution"]).resolve(),
               Path(subject["predecessor"]["execution"]).resolve(),
               *roots, *((repository,) if repository else ()))
    _destinations(root, Path(configured["state_dir"]).resolve(), exposed)
    if not root.exists():
        if repository is not None:
            _materialize(root, repository, candidate, host.base_ref)
        else:
            _materialize_observed_base(root, candidate, subject, host)
    before = Path(subject["predecessor"]["execution"])
    execution = Path(subject["successor_execution"])
    source_tree(root, host.base_ref, predecessor=(before, _files(before)),
                candidate=(execution, _files(execution)),
                allowed_paths=subject["scope"]["changes"], paths=source_paths(root))
    record = root.parent / (root.name + ".delivery-input.json")
    expected = {"wait": wait, "source_digest": host.source_digest, "base_ref": host.base_ref,
                "repository": host.repository, "configured_sha256": digest(configured)}
    if record.exists():
        saved = json.loads(record.read_text())
        if record.is_symlink() or {key: saved.get(key) for key in expected} != expected:
            raise ValueError("source_delivery_preparation_binding_changed")
        plan = saved["plan"]
    else:
        from slm_training.autoresearch.storage import CampaignStore
        binding = verification_binding(root, host.base_ref, merge_gate_steps(), isolated=True, runtimes=roots)
        plan = {**configured, "identity": digest(binding), "environment": verification_environment(),
                "runtime_roots": [str(path) for path in roots]}
        CampaignStore._replace_durable(record, json.dumps({**expected, "plan": plan}, sort_keys=True))
    return host.model_copy(update={"verification_plan": plan})


def _materialize_observed_base(root, candidate, subject, host):
    from slm_training.harness_core.github_git_snapshot import private_git_snapshot

    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix="source-delivery-", dir=root.parent))
    source = private_snapshot(candidate, stage / "candidate")
    predecessor = Path(subject["predecessor"]["execution"])
    private_git_snapshot(source, predecessor, _files(predecessor),
        commit=host.verification_plan["remote_base_commit_object"], expected=host.base_ref)
    os.rename(source, root)


def _destinations(root, cache, exposed):
    for protected in (root, cache):
        if any(protected.is_relative_to(path) or path.is_relative_to(protected) for path in exposed):
            raise ValueError("source_delivery_controller_namespace_exposed")
    if root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError("source_delivery_issuer_exposed")


def _materialize(root, repository, candidate, base):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                          capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if head != base:
        raise ValueError("source_delivery_repository_checkout_must_match_actual_base")
    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix="source-delivery-", dir=root.parent))
    source = private_snapshot(candidate, stage / "candidate")
    _private_git(repository, source / ".git",
                 INTERRUPT_AFTER_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS, time.monotonic())
    # The clone may race a ref update. Its independent HEAD must still be B.
    copied = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if copied != base or _files(source) != _files(candidate):
        raise ValueError("source_delivery_base_or_candidate_changed_during_copy")
    os.rename(source, root)
