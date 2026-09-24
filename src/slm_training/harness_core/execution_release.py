"""Pinned local source and a separate, disposable execution materialization.

Read-only modes prevent accidental writes; this is NOT a hostile-worker sandbox.
No Git metadata, credentials, caches or human outputs are copied. The controller
must invoke runtime_source_identity at its existing before/after boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

from .activity_contract import contract_digest

MARKER = ".autonomy-release.json"
_READONLY_GIT = frozenset({"status", "rev-parse", "merge-base", "show", "log", "diff",
                          "ls-files", "ls-tree", "cat-file"})


def require_readonly_controller_git(command) -> None:
    """Controller command contract, not an OS sandbox for arbitrary worker code."""
    if not command or Path(command[0]).name not in {"git", "git.exe"}:
        return
    if (len(command) < 2 or command[1] not in _READONLY_GIT
            or any(arg in {"--output", "--ext-diff", "--textconv"}
                   or arg.startswith("--output=") for arg in command[2:])):
        raise PermissionError("capability_unavailable:authorized_github_connector_delivery; controller Git is read-only")


_SKIP = {
    ".git",
    ".venv",
    "node_modules",
    "outputs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".serena",
    # Keep the committed hook policy in the immutable execution tree so the
    # parity verifier sees every configured harness. Private credentials and
    # local overrides remain excluded by their explicit names below.
    ".env",
    "settings.local.json",
}


def _files(root: Path) -> dict:
    entries = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _SKIP)
        for name in sorted(
            files + [name for name in dirs if (Path(directory) / name).is_symlink()]
        ):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if name in _SKIP or relative == MARKER or name.startswith(".env."):
                continue
            info = path.lstat()
            if path.is_symlink():
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("external_source_link:" + relative)
                value = ["link", os.readlink(path)]
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                value = [
                    "file",
                    bool(info.st_mode & 0o111),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                ]
            else:
                raise ValueError("unsupported_source_file:" + relative)
            entries[relative] = value
    return entries


def prepare_release(
    source: Path, release: Path, execution: Path, outputs: Path,
    *, verified_predecessor: tuple[Path, str] | None = None,
) -> dict:
    """Explicit local materialization, never starts a process or overwrites a tree."""
    return _prepare_release(source, (release, execution, outputs),
        lambda: _predecessor_provenance(verified_predecessor)
        if verified_predecessor else _checkout_provenance(source.resolve(strict=True)))


def prepare_delivered_release(verified_source, destinations, remote_commit) -> dict:
    """New materialization of exact delivered bytes, never edit an old marker.

    Caller authenticates observed main membership through its delivery receipt.
    This primitive independently proves the Git commit object and complete tree;
    no caller-supplied provenance dictionary or clean flag is accepted.
    """
    return _prepare_release(verified_source[0], destinations,
                            lambda: _delivered_provenance(verified_source, remote_commit))


def _delivered_provenance(verified_source, remote_commit):
    from .github_delivery_tree import git_object, source_entries, tree_sha

    source, expected_digest = verified_source
    raw, revision = remote_commit
    if expected_digest is None or runtime_source_identity(source) != expected_digest:
        raise ValueError("delivered_source_identity_mismatch")
    if git_object("commit", raw.encode()) != revision:
        raise ValueError("delivered_commit_object_mismatch")
    trees = [line for line in raw.split("\n\n", 1)[0].splitlines() if line.startswith("tree ")]
    if trees != ["tree " + tree_sha(source_entries(source, _files(source)))]:
        raise ValueError("delivered_commit_tree_mismatch")
    return {"integration_commit": revision, "upstream_commit": revision, "code_dirty": False}


def _prepare_release(source, destinations, provenance_read):
    source = source.resolve(strict=True)
    release, execution, outputs = (p.resolve() for p in destinations)
    if any(p.exists() for p in (release, execution)):
        raise ValueError("release_or_execution_exists")
    paths = (source, release, execution)
    if any(
        a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(paths)
        for b in paths[i + 1 :]
    ):
        raise ValueError("source_release_execution_outputs_must_be_disjoint")
    if any(
        outputs.is_relative_to(p) or p.is_relative_to(outputs)
        for p in (release, execution)
    ):
        raise ValueError("source_release_execution_outputs_must_be_disjoint")
    if (
        outputs.is_relative_to(source)
        and not outputs.is_relative_to(source / "outputs")
    ) or source.is_relative_to(outputs):
        raise ValueError("output_must_use_excluded_namespace")
    entries = _files(source)
    provenance = provenance_read()
    for destination in (release, execution):
        destination.mkdir(parents=True)
        for name, value in entries.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if value[0] == "link":
                target.symlink_to(value[1])
            else:
                shutil.copyfile(source / name, target)
                target.chmod(
                    0o555 if value[1] else 0o444 if destination == release else 0o644
                )
    # Concurrent authoring during preparation never produces an accepted release.
    if (
        _files(source) != entries
        or provenance_read() != provenance
        or _files(release) != entries
        or _files(execution) != entries
    ):
        raise ValueError("source_changed_during_release_preparation")
    manifest = {
        "schema_version": "local_execution_release/v1",
        "files": entries,
        "source_digest": contract_digest(entries),
        "git_provenance": provenance,
        "release": str(release),
        "outputs": str(outputs),
        "service_started": False,
    }
    for destination in (release, execution):
        (destination / MARKER).write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (release / MARKER).chmod(0o444)
    outputs.mkdir(parents=True, exist_ok=True)
    (execution / "outputs").symlink_to(outputs, target_is_directory=True)
    return manifest


def _predecessor_provenance(predecessor: tuple[Path, str]) -> dict:
    """Controller-pinned ancestry; candidate bytes never supply Git metadata."""
    execution, expected_digest = predecessor
    if runtime_source_identity(execution) != expected_digest or expected_digest is None:
        raise ValueError("repair_predecessor_source_mismatch")
    provenance = runtime_git_provenance(execution)
    if provenance is None:
        raise ValueError("repair_requires_pinned_execution_release")
    return {**provenance, "code_dirty": True}


def runtime_source_identity(root: Path) -> str | None:
    """Validate immutable original + execution source; legacy checkout => None."""
    marker = root / MARKER
    if not marker.exists():
        return None
    if marker.is_symlink():
        raise ValueError("release_marker_symlink")
    manifest = json.loads(marker.read_text())
    if manifest.get("schema_version") != "local_execution_release/v1":
        raise ValueError("unknown_release_schema")
    entries = manifest["files"]
    if contract_digest(entries) != manifest["source_digest"]:
        raise ValueError("release_manifest_integrity_failure")
    release = Path(manifest["release"])
    if (release == root.resolve() or _files(release) != entries
            or json.loads((release / MARKER).read_text()) != manifest):
        raise ValueError("immutable_release_changed")
    actual = _files(root)
    changed = {
        name
        for name in entries.keys() | actual.keys()
        if entries.get(name) != actual.get(name)
    }
    # Reports belong in the explicit output namespace, not source exemptions.
    if changed:
        raise ValueError("execution_source_drift")
    if (root / "outputs").resolve() != Path(manifest["outputs"]).resolve():
        raise ValueError("output_destination_changed")
    return manifest["source_digest"]


def document_successor_manifest(entries: dict, documents: dict) -> dict:
    """Exact document overlay; the successor still needs its own full gate."""
    if not isinstance(documents, dict):
        raise ValueError("invalid_delivery_document_manifest")
    expected = dict(entries)
    for name, digest in documents.items():
        if not isinstance(name, str):
            raise ValueError("invalid_delivery_document_manifest")
        path = PurePosixPath(name)
        if (str(path) != name
                or path.is_absolute() or ".." in path.parts
                or not (name in {"README.md", "docs/MODEL_CARD.md",
                                 "src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json"}
                        or len(path.parts) == 3 and path.parts[:2] == ("docs", "design")
                        and path.suffix in {".md", ".json"})
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or name in entries and entries[name][:2] != ["file", False]):
            raise ValueError("invalid_delivery_document_manifest")
        expected[name] = ["file", False, digest]
    return expected


def _checkout_provenance(source: Path) -> dict | None:
    """Capture this checkout only; never inherit an enclosing worktree's refs."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=source, capture_output=True,
                              text=True, check=True, timeout=10,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
            return None
        integration = git("rev-parse", "--verify", "HEAD^{commit}")
        upstream = git("rev-parse", "--verify", "origin/main^{commit}")
        git("merge-base", "--is-ancestor", upstream, integration)
        return {"integration_commit": integration, "upstream_commit": upstream,
                "code_dirty": bool(git("status", "--porcelain", "--untracked-files=all"))}
    except (OSError, subprocess.SubprocessError):
        return None


def runtime_git_provenance(root: Path) -> dict | None:
    """Verified historical Git metadata; source_digest remains the runtime identity."""
    if runtime_source_identity(root) is None:
        return None
    value = json.loads((root / MARKER).read_text()).get("git_provenance")
    if (not isinstance(value, dict) or type(value.get("code_dirty")) is not bool
            or any(not isinstance(value.get(key), str) or len(value[key]) != 40
                   or any(c not in "0123456789abcdef" for c in value[key])
                   for key in ("integration_commit", "upstream_commit"))):
        raise ValueError("release_git_provenance_unavailable")
    return value


def validate_source_refs(root: Path, upstream: str, integration: str, *, git):
    """Frozen provenance or the unchanged ordinary-checkout ancestry checks."""
    frozen = runtime_git_provenance(root)
    if frozen is not None:
        if (upstream != frozen["upstream_commit"]
                or integration != frozen["integration_commit"]):
            raise ValueError("continuous source refs differ from immutable release provenance")
        return frozen
    resolved_upstream = git(
        "rev-parse", "--verify", f"{upstream}^{{commit}}"
    ).stdout.strip()
    resolved_integration = git(
        "rev-parse", "--verify", f"{integration}^{{commit}}"
    ).stdout.strip()
    current_upstream = git(
        "rev-parse", "--verify", "origin/main^{commit}"
    ).stdout.strip()
    current_head = git("rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if resolved_integration != current_head:
        raise ValueError("integration_commit must be the current checked-out HEAD")
    main_in_head = (
        git(
            "merge-base",
            "--is-ancestor",
            current_upstream,
            current_head,
            check=False,
        ).returncode
        == 0
    )
    if main_in_head:
        if resolved_upstream != current_upstream:
            raise ValueError(
                "upstream_commit is stale; fetch origin/main before the cycle"
            )
        if git(
            "merge-base",
            "--is-ancestor",
            resolved_upstream,
            resolved_integration,
            check=False,
        ).returncode:
            raise ValueError("integration_commit does not contain upstream_commit")
    elif resolved_upstream != current_head:
        raise ValueError("integration_commit does not contain upstream_commit")
