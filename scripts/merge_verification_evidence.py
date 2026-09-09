"""Merge evidence binding; cache MACs are not isolation or release authority."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import platform
import secrets
import site
import shutil
import stat
import subprocess
import sys
import tempfile
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
    for key in "OPENUI_BRIDGE_CLI DESIGN_MD_BRIDGE_CLI AGENTV_RUNNER".split():
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
        },
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


def _dependency_files(installed) -> list:
    """Linux ctime catches ordinary same-size/mtime-restored installed-file edits.

    Privileged rollback still requires isolation and immutable verifier runtimes.
    """
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


def validate_workload(
    payload: dict, *, request_digest: str, expected: list[str] | None = None
) -> list[str]:
    """Require exact collection and setup/call/teardown success for every node."""
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "merge_test_workload/v1"
    ):
        raise ValueError("unknown workload schema")
    if (
        payload.get("request_digest") != request_digest
        or type(payload.get("exit_code")) is not int
        or payload.get("exit_code") != 0
    ):
        raise ValueError("wrong request or failed workload")
    nodes = payload.get("nodes")
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(not isinstance(n, str) or not n.strip() for n in nodes)
    ):
        raise ValueError("empty or malformed test collection")
    if len(nodes) != len(set(nodes)) or payload.get("deselected") != []:
        raise ValueError("duplicate or deselected tests")
    if payload.get("collection_errors") != []:
        raise ValueError("collection errors")
    if expected is None:
        return nodes
    if set(nodes) != set(expected) or len(expected) != len(set(expected)):
        raise ValueError("collected nodes differ from locked selection")
    _validate_test_phases(payload.get("reports", []), expected)
    return nodes


def validate_targets(targets: list[str]) -> None:
    if not isinstance(targets, list) or any(
        not isinstance(target, str) for target in targets
    ):
        raise ValueError("malformed workload targets")
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("empty or repeated workload targets")
    for target in targets:
        path = target.split("::", 1)[0]
        if (
            not path
            or path.startswith("-")
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or "\\" in path
            or "\0" in target
        ):
            raise ValueError("noncanonical workload target")


def _validate_test_phases(reports: list[dict], expected: list[str]) -> None:
    if not isinstance(reports, list) or any(
        not isinstance(row, dict) for row in reports
    ):
        raise ValueError("invalid test reports")
    wanted = {
        (node, phase) for node in expected for phase in ("setup", "call", "teardown")
    }
    seen = set()
    for row in reports:
        key = (row.get("nodeid"), row.get("when"))
        duration = row.get("duration_seconds")
        if (
            key not in wanted
            or key in seen
            or row.get("outcome") != "passed"
            or row.get("wasxfail")
        ):
            raise ValueError(
                "missing, repeated, skipped, xfailed or failing test obligation"
            )
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (float, int))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("invalid test duration")
        seen.add(key)
    if seen != wanted:
        raise ValueError("missing required test phases")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".verification-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ReceiptCache:
    """Private controller cache. Never mount this directory in repair workloads.

    Local CLI use is same-user development evidence only; OS isolation is a
    separate prerequisite for an autonomous release, regardless of these MACs.
    """

    def __init__(self, directory: Path, root: Path) -> None:
        self.directory = directory.resolve()
        if self.directory.is_relative_to(root.resolve()):
            raise ValueError("verification cache must be outside the candidate tree")
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        info = self.directory.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("verification cache must be owned and mode 0700")
        key_path = self.directory / "issuer.key"
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        info = key_path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("unsafe verification issuer key")
        self.key = key_path.read_bytes()
        if len(self.key) != 32:
            raise ValueError("invalid verification issuer key")

    def load(self, identity: str) -> dict | None:
        _require_identity(identity)
        path = self.directory / f"{identity}.json"
        if not path.exists():
            return None
        envelope = json.loads(path.read_text())
        payload = envelope["payload"]
        signature = hmac.digest(self.key, digest(payload).encode(), "sha256").hex()
        if not hmac.compare_digest(signature, envelope["mac"]):
            raise ValueError("unauthenticated verification cache")
        if payload.get("identity") != identity:
            raise ValueError("stale verification cache")
        return payload

    def save(self, payload: dict) -> None:
        _require_identity(payload["identity"])
        signature = hmac.digest(self.key, digest(payload).encode(), "sha256").hex()
        atomic_json(
            self.directory / f"{payload['identity']}.json",
            {"payload": payload, "mac": signature},
        )


def authorize_release(
    evidence: dict, *, expected_identity: str, independent_verification: dict
) -> bool:
    """Called by a trusted controller with its own independent verifier receipt.

    The receipt argument must come from the authenticated verifier channel,
    never a candidate file. This function checks binding, not its authenticity.
    """
    required = ("scope_passed", "original_reproducer_passed", "isolation_enforced")
    return (
        evidence.get("identity") == expected_identity
        and evidence.get("verification_complete") is True
        and evidence.get("evidence_class") == "isolated_process"
        and independent_verification.get("verification_identity") == expected_identity
        and independent_verification.get("evidence_sha256") == digest(evidence)
        and all(independent_verification.get(key) is True for key in required)
    )


def _require_identity(identity: str) -> None:
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
    ):
        raise ValueError("cache identity must be a SHA256")


def validate_cached_state(state: dict, binding: dict) -> None:
    """Authenticate outside the worker, then recheck proof rather than counters."""
    if state.get("invalidated"):
        raise ValueError("quarantined verification cache")
    if state.get("binding") != binding or state.get("identity") != digest(binding):
        raise ValueError("cached binding mismatch")
    wanted = {name for name, _ in binding["static_commands"]}
    if not set(state.get("static", {})) <= wanted:
        raise ValueError("unknown cached static obligation")
    for row in state.get("static", {}).values():
        if row["status"] == "ok" and (
            type(row.get("exit_code")) is not int or row["exit_code"] != 0
        ):
            raise ValueError("cached static exit was not successful")
    records = (
        list(state.get("static", {}).values())
        + state.get("static_history", [])
        + state.get("attempts", [])
    )
    if any(
        type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
        for seconds in (row.get("seconds") for row in records)
    ):
        raise ValueError("invalid cached resource charge")
    nodes = state.get("nodes", [])
    if not nodes:
        if state.get("passed_nodes") or "nodes" in state:
            raise ValueError("cached passes without collection or empty collection")
        return
    _validate_cached_nodes(state, nodes)
    shards = state.get("shards", [])
    if any(not shard for shard in shards) or sorted(
        n for shard in shards for n in shard
    ) != sorted(nodes):
        raise ValueError("cached shard partition mismatch")


def _validate_cached_nodes(state: dict, nodes: list[str]) -> None:
    collected, passed = [], []
    for attempt in state["attempts"]:
        if attempt.get("kind") not in {"collection", "shard"}:
            raise ValueError("unknown cached workload kind")
        if attempt["status"] != "ok":
            continue
        payload = attempt["workload"]
        if digest(payload) != attempt["workload_sha256"]:
            raise ValueError("cached workload digest mismatch")
        if type(attempt.get("exit_code")) is not int or attempt["exit_code"] != 0:
            raise ValueError("cached workload exit was not successful")
        collection = attempt["kind"] == "collection"
        actual = validate_workload(
            payload,
            request_digest=attempt["request_digest"],
            expected=None if collection else attempt["nodes"],
        )
        (collected if collection else passed).extend(actual)
    if collected != nodes or len(nodes) != len(set(nodes)):
        raise ValueError("cached collection mismatch")
    if len(passed) != len(set(passed)) or not set(passed) <= set(nodes):
        raise ValueError("cached duplicate or unknown pass")
    if sorted(passed) != sorted(state["passed_nodes"]):
        raise ValueError("cached passes lack completed workload evidence")
