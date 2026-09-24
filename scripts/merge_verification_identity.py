"""Source and runtime identities used by the merge evidence cache."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import site
import stat
import subprocess
import sys
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(root: Path) -> str:
    """Bind tracked, deleted and nonignored new files, including mode/link data."""
    entries = []
    for name in source_paths(root):
        path = root / name
        if path.is_symlink():
            content = ["symlink", os.readlink(path)]
        elif path.is_file():
            content = ["file", stat.S_IMODE(path.stat().st_mode), file_digest(path)]
        else:
            content = ["missing"]
        entries.append([name, content])
    return digest(entries)


def source_paths(root: Path) -> list[str]:
    """One file universe for identity and isolated workload materialization."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return sorted(set(result.stdout.decode().split("\0")) - {""})


def environment_identity() -> dict:
    # Keep package discovery independent of the caller's sys.path[0].
    package_paths = list(site.getsitepackages())
    try:
        package_paths.append(site.getusersitepackages())
    except (AttributeError, PermissionError):
        pass
    installed = list(importlib.metadata.distributions(path=package_paths))
    commands = {name: shutil.which(name) for name in ("node", "npm", "npx")}
    for key in "OPENUI_BRIDGE_CLI DESIGN_MD_BRIDGE_CLI AGENTV_RUNNER AGENTV_NODE_MODULES".split():
        commands[key] = os.environ.get(key)
    distributions = sorted(
        (
            dist.metadata.get("Name", ""),
            dist.metadata.get("Version", "unknown"),
            digest([dist.read_text("METADATA"), dist.read_text("RECORD")]),
        )
        for dist in installed
    )
    return {
        "python": sys.version,
        "executable_sha256": file_digest(Path(sys.executable)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "distributions_sha256": digest(distributions),
        "dependency_files_sha256": digest(_dependency_files(installed)),
        "command_files": {
            key: [path, file_digest(Path(path))] if path else None
            for key, path in commands.items()
            if key != "AGENTV_NODE_MODULES"
        },
        "javascript_runtime_dependencies": _javascript_runtime_dependencies(commands),
        "execution_environment_sha256": digest(
            {
                key: os.pathsep.join(map(os.path.abspath, value.split(os.pathsep)))
                if key == "PYTHONPATH"
                else value
                for key, value in os.environ.items()
                if key in {"PATH", "NODE_OPTIONS", "ORT_DISABLE_TELEMETRY"}
                or key.startswith(
                    ("PYTHON", "SLM_", "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_")
                )
            }
        ),
    }


def _javascript_runtime_dependencies(commands: dict[str, str | None]) -> dict:
    roots = set()
    locks = {}
    if modules := commands.get("AGENTV_NODE_MODULES"):
        modules_root = Path(modules)
        roots.add(modules_root)
        lock = modules_root.parent / "package-lock.json"
        if lock.is_file():
            locks[str(lock)] = file_digest(lock)
    for key in ("OPENUI_BRIDGE_CLI", "DESIGN_MD_BRIDGE_CLI", "AGENTV_RUNNER"):
        entrypoint = commands.get(key)
        if not entrypoint:
            continue
        parent = Path(entrypoint).resolve().parent
        lock = parent / "package-lock.json"
        if lock.is_file():
            locks[str(lock)] = file_digest(lock)
        for directory in (parent, *parent.parents):
            modules = directory / "node_modules"
            if modules.is_dir():
                roots.add(modules)
    return {
        "trees": {
            str(root.resolve()): runtime_identity((root,))
            for root in sorted(roots, key=str)
            if root.is_dir()
        },
        "package_locks": locks,
    }


def _dependency_files(installed) -> list:
    """Detect ordinary edits via ctime; rollback needs isolated immutable runtimes."""
    entries = []
    for distribution in installed:
        for relative in distribution.files or ():
            path = Path(distribution.locate_file(relative))
            try:
                info = path.stat()
                identity = [
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    info.st_mode,
                ]
            except OSError:
                identity = ["missing"]
            entries.append([str(path), identity])
    return sorted(entries)


def runtime_identity(roots: tuple[Path, ...]) -> str:
    entries = []
    for root in roots:
        root = root.resolve(strict=True)
        for directory, dirs, files in os.walk(root):
            for name in dirs + files:
                path = Path(directory) / name
                info = path.lstat()
                entries.append(
                    [
                        str(path),
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                        info.st_mode,
                        os.readlink(path) if path.is_symlink() else None,
                    ]
                )
    return digest(sorted(entries))
