"""Private repair snapshots and content/mode/link scope checks.

These checks supplement OS isolation. They do not authenticate worker output.
"""

from __future__ import annotations

import hashlib
import json
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


def checked_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = root / relative_path(relative)
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise IsolationViolation(f"symlink in mount path: {relative}")
        if part != path and part.exists() and not part.is_dir():
            raise IsolationViolation(f"non-directory ancestor in mount path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise IsolationViolation(f"path escapes snapshot: {relative}")
    if must_exist and not path.exists():
        raise IsolationViolation(f"mount path missing: {relative}")
    return path


def tree_manifest(
    root: Path, *, ignore_names: frozenset[str] = frozenset()
) -> dict[str, tuple[int, str]]:
    """Hash bytes and modes; refuse sockets/devices/FIFOs and external links."""
    result: dict[str, tuple[int, str]] = {}
    root = root.resolve(strict=True)
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in ignore_names)
        for name in sorted(dirs + [name for name in files if name not in ignore_names]):
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

def private_snapshot_manifest(root: Path) -> dict[str, tuple[int, str]]:
    """Hash the same paths retained by private_snapshot without copying them."""
    return tree_manifest(root, ignore_names=_OMIT)


def manifest_digest(manifest: dict[str, tuple[int, str]]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def patch_manifest_digest(before: dict, after: dict) -> str:
    """Hash exact changed manifest entries with the canonical repair encoding."""
    before = {path: list(entry) for path, entry in before.items()}
    after = {path: list(entry) for path, entry in after.items()}
    changes = {
        path: {"before": before.get(path), "after": after.get(path)}
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    }
    payload = json.dumps(changes, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def proposal_digests(
    workspace: Path,
    baseline: dict,
    prepared: dict,
    allowed_paths: tuple[str, ...],
) -> dict[str, str]:
    """Project exact granted-file edits onto the trusted prepared host manifest."""
    projected = {path: list(entry) for path, entry in prepared.items()}
    for relative in allowed_paths:
        path = checked_path(workspace, relative, must_exist=False)
        if not path.exists():
            projected.pop(relative, None)
            continue
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise IsolationViolation("proposal path is not a private regular file")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        stable = ("st_mode", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise IsolationViolation("proposal path changed while hashing")
        projected[relative] = [after.st_mode, digest]
    return {
        "tree_digest": manifest_digest(projected),
        "patch_digest": patch_manifest_digest(baseline, projected),
    }


def _proposal_digest_main() -> None:
    """Trusted read-only command copied into the repair input mount."""
    input_root = Path(__file__).resolve().parent
    root = input_root.parent
    instructions = json.loads((input_root / "repair-instructions.json").read_text())
    baseline = json.loads((input_root / "baseline-manifest.json").read_text())
    prepared = json.loads((input_root / "prepared-manifest.json").read_text())
    expected = instructions["request"]["blocker"]["source_digest"]
    if manifest_digest(baseline) != expected:
        raise IsolationViolation("baseline manifest differs from pinned release")
    allowed = tuple(instructions["request"]["allowed_paths"])
    print(json.dumps(proposal_digests(root, baseline, prepared, allowed), sort_keys=True))


def owner_write_preparation(before, after) -> bool:
    """Exact regular-file owner-write addition; Git cannot represent this mode bit.

    Only source gate path coverage uses this predicate. Full manifest identities
    and independent allowed-path scope checks must still include these changes.
    """
    return (before is not None and after is not None and stat.S_ISREG(before[0])
            and before[1] == after[1] and after[0] == before[0] | stat.S_IWUSR)


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
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise IsolationViolation("snapshot destination exists or overlaps source")
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(*_OMIT))
    tree_manifest(destination)
    return destination


def private_snapshot_with_disposable_dirs(
    source: Path,
    destination: Path,
    extra_dirs: tuple[str, ...],
    allowed_paths: tuple[str, ...],
) -> Path:
    """Retain exact bound paths and empty disposable runtime mount points.

    Validate paths before copying or creating anything. Mount points cannot
    alias source files through symlinks, even when the link stays in the tree.
    """
    extras = tuple(relative_path(path) for path in extra_dirs)
    allowed = {relative_path(path) for path in allowed_paths} | set(extras)
    copy = private_snapshot(source, destination)
    for relative in extras:
        path = copy / relative
        if any(parent.is_symlink() for parent in (path, *path.parents)
               if parent.is_relative_to(copy)):
            raise IsolationViolation(f"symlink in disposable path: {relative}")
        path.mkdir(parents=True, exist_ok=True)
    if not allowed_paths:
        return copy
    for relative in tuple(allowed):
        allowed.update(parent.as_posix() for parent in Path(relative).parents)
    for directory, dirs, files in os.walk(copy, topdown=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.relative_to(copy).as_posix() in allowed:
                continue
            if path.is_dir() and not path.is_symlink():
                path.rmdir()
            else:
                path.unlink()
    tree_manifest(copy)
    return copy


if __name__ == "__main__":
    _proposal_digest_main()
