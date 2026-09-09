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
from pathlib import Path

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
    source: Path, release: Path, execution: Path, outputs: Path
) -> dict:
    """Explicit local materialization, never starts a process or overwrites a tree."""
    source = source.resolve(strict=True)
    release, execution, outputs = (p.resolve() for p in (release, execution, outputs))
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
        or _files(release) != entries
        or _files(execution) != entries
    ):
        raise ValueError("source_changed_during_release_preparation")
    manifest = {
        "schema_version": "local_execution_release/v1",
        "files": entries,
        "source_digest": contract_digest(entries),
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
    if release == root.resolve() or _files(release) != entries:
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
