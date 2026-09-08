"""Private repair snapshots and content/mode/link scope checks.

These checks supplement OS isolation. They do not authenticate worker output.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path, PurePosixPath


class IsolationViolation(ValueError):
    """An input or result crosses the controller's declared boundary."""


_OMIT = frozenset({".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
                   ".ruff_cache", "outputs", ".serena"})


def relative_path(value: str) -> str:
    """Accept a single canonical repository-relative spelling, never traversal."""
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or str(path) != value or path == PurePosixPath(".")):
        raise IsolationViolation(f"noncanonical relative path: {value!r}")
    return value


def checked_path(root: Path, relative: str) -> Path:
    path = root / relative_path(relative)
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise IsolationViolation(f"symlink in mount path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise IsolationViolation(f"path escapes snapshot: {relative}")
    if not path.exists():
        raise IsolationViolation(f"mount path missing: {relative}")
    return path


def tree_manifest(root: Path) -> dict[str, tuple[int, str]]:
    """Hash bytes and modes; refuse sockets/devices/FIFOs and external links."""
    result: dict[str, tuple[int, str]] = {}
    root = root.resolve(strict=True)
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                if not path.resolve().is_relative_to(root):
                    raise IsolationViolation(f"external symlink: {relative}")
                payload = os.readlink(path).encode()
            elif stat.S_ISDIR(info.st_mode):
                payload = b""
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                result[relative] = (info.st_mode, digest)
                continue
            else:
                raise IsolationViolation(f"special file or hardlink: {relative}")
            result[relative] = (info.st_mode, hashlib.sha256(payload).hexdigest())
    return result


def manifest_digest(manifest: dict[str, tuple[int, str]]) -> str:
    import json

    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def scope_changes(before: dict, after: dict, allowed: tuple[str, ...]) -> tuple[str, ...]:
    """Detect repeat edits to already-dirty files, deletion and mode changes."""
    permitted = tuple(relative_path(path) for path in allowed)
    return tuple(sorted(
        path for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
        and not any(path == p or path.startswith(p + "/") for p in permitted)
    ))


def private_snapshot(source: Path, destination: Path) -> Path:
    """Copy without shared Git metadata, caches, hardlinks or external symlinks.

    The controller must pin/quiesce the source first. Existing destinations are
    refused. An incomplete copy is never returned as a usable snapshot.
    """
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_relative_to(source):
        raise IsolationViolation("snapshot destination exists or overlaps source")
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(*_OMIT))
    tree_manifest(destination)
    return destination

