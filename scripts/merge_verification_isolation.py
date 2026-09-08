"""Use the canonical OS boundary for merge workloads, not the controller store.

Only candidate bytes, a read-only selection/runner and approved read-only runtime
installations enter the sandbox. The result file is explicitly untrusted input;
it cannot issue a receipt or write the controller's cache/signing key.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from scripts import check_changed
from scripts.merge_verification_evidence import (
    atomic_json,
    digest,
    source_paths,
    validate_targets,
    validate_workload,
)
from slm_training.autoresearch.heal.isolation import (
    IsolationSpec,
    IsolationUnavailable,
    run_isolated,
)
from slm_training.autoresearch.heal.isolation_workspace import private_snapshot
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS


def run_workload(
    root: Path,
    targets: list[str],
    *,
    collect_only: bool,
    seconds: float,
    directory: Path,
    isolated: bool = False,
    runtimes: tuple[Path, ...] = (),
) -> dict:
    """Launch a real pytest child with a pinned verifier-owned protocol runner."""
    import json

    validate_targets(targets)
    if isolated:
        return _isolated_record(root, targets, collect_only, seconds, runtimes)
    request = {"root": str(root), "targets": targets, "collect_only": collect_only}
    request_digest = digest(request)
    request["request_digest"] = request_digest
    with tempfile.TemporaryDirectory(prefix="slm-merge-feedback-") as temporary:
        work = Path(temporary)
        atomic_json(work / "request.json", request)
        shutil.copyfile(
            Path(__file__).with_name("merge_test_worker.py"), work / "worker.py"
        )
        env = check_changed._pytest_worker_env()
        for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            env.pop(key, None)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(root / "src"), env.get("PYTHONPATH")))
        )
        env.pop("PYTHONPYCACHEPREFIX", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = run_bounded_process(
            [
                sys.executable,
                str(work / "worker.py"),
                str(work / "request.json"),
                str(work / "result.json"),
            ],
            cwd=root,
            env=env,
            interrupt_after_seconds=min(seconds, INTERRUPT_AFTER_SECONDS),
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
        record = {
            "request_digest": request_digest,
            "seconds": result.duration_seconds,
            "exit_code": result.returncode,
            "status": "failed",
            "nodes": targets,
            "output_tail": (result.stdout + result.stderr)[-4000:],
        }
        if result.timed_out or result.interrupted or result.killed:
            return {**record, "status": "timeout"}
        try:
            if result.returncode != 0:
                raise ValueError(f"workload exit {result.returncode}")
            payload = json.loads((work / "result.json").read_text())
            nodes = validate_workload(
                payload,
                request_digest=request_digest,
                expected=None if collect_only else targets,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return {**record, "reason": str(exc)}
        return {
            **record,
            "status": "ok",
            "nodes": nodes,
            "workload": payload,
            "workload_sha256": digest(payload),
        }


def _isolated_record(root, targets, collect_only, seconds, runtimes) -> dict:
    result, payload, request_digest = isolated_workload(
        root,
        targets,
        collect_only=collect_only,
        seconds=seconds,
        runtimes=runtimes,
    )
    record = {
        "request_digest": request_digest,
        "seconds": result.duration_seconds,
        "exit_code": result.returncode,
        "nodes": targets,
        "evidence_class": "isolated_process",
        "status": "failed",
        "output_tail": (result.stdout + result.stderr)[-16000:],
    }
    if result.timed_out or result.interrupted or result.killed:
        return {**record, "status": "timeout"}
    try:
        if result.returncode != 0:
            raise ValueError(
                f"workload exit {result.returncode}: {result.stderr[-1000:]}"
            )
        nodes = validate_workload(
            payload,
            request_digest=request_digest,
            expected=None if collect_only else targets,
        )
    except (ValueError, TypeError, KeyError) as exc:
        return {**record, "reason": str(exc), "workload_observation": payload}
    return {
        **record,
        "status": "ok",
        "nodes": nodes,
        "workload": payload,
        "workload_sha256": digest(payload),
    }


def runtime_command(argv: tuple[str, ...], runtimes: tuple[Path, ...]) -> list[str]:
    """Map an executable inside an explicitly mounted runtime, never host home."""
    executable = Path(argv[0]).absolute()
    for index, root in enumerate(runtimes):
        if executable.is_relative_to(root.absolute()):
            return [
                f"/runtime/{index}/{executable.relative_to(root.absolute())}",
                *argv[1:],
            ]
    if executable.is_relative_to("/usr") or executable.is_relative_to("/bin"):
        return list(argv)
    raise ValueError("verification executable is outside approved runtime roots")


def isolated_workload(
    root: Path,
    targets: list[str],
    *,
    collect_only: bool,
    seconds: float,
    runtimes: tuple[Path, ...],
) -> tuple[object, dict, str]:
    """The control directory and issuer key are deliberately not parameters."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="slm-merge-sandbox-") as temporary:
        workspace = Path(temporary)
        _candidate_snapshot(root, workspace / "candidate")
        control = workspace / "control"
        control.mkdir()
        worker = Path(__file__).with_name("merge_test_worker.py")
        (control / "worker.py").write_bytes(worker.read_bytes())
        request = {
            "root": "/workspace/candidate",
            "targets": targets,
            "collect_only": collect_only,
        }
        request_digest = digest(request)
        atomic_json(
            control / "request.json", {**request, "request_digest": request_digest}
        )
        # Exact file mount: writable output is never an authoritative channel.
        output = workspace / "workload-result.json"
        output.write_text("{}")
        argv = runtime_command(
            (
                sys.executable,
                "/workspace/control/worker.py",
                "/workspace/control/request.json",
                "/workspace/workload-result.json",
            ),
            runtimes,
        )
        argv = _workload_argv(root, runtimes, argv, workspace=workspace)
        result = run_isolated(
            IsolationSpec(
                workspace,
                writable_paths=("workload-result.json",),
                runtime_roots=runtimes,
                timeout_seconds=_remaining(seconds, started),
            ),
            argv,
        )
        try:
            payload = json.loads(output.read_text())
        except (OSError, ValueError):
            payload = {}  # An interrupted/malformed file is never passing evidence.
        return (
            replace(result, duration_seconds=time.monotonic() - started),
            payload,
            request_digest,
        )


def isolated_static(
    step, *, budget_seconds: float, root: Path, runtimes: tuple[Path, ...]
) -> dict:
    """Static validators execute candidate imports with the same denied mounts."""
    started = time.monotonic()
    runtime_path = ":".join(
        [f"/runtime/{index}/bin" for index, _ in enumerate(runtimes)]
        + ["/usr/bin", "/bin"]
    )
    source_path = (
        "/workspace/candidate/src" if (root / ".git").exists() else "/workspace/src"
    )
    pythonpath = ":".join(
        [source_path, *[f"/runtime/{index}" for index in range(1, len(runtimes))]]
    )
    environment = [("PATH", runtime_path)]
    if step.name == "ruff":
        environment.append(("RUFF_CACHE_DIR", "/tmp/ruff-cache"))
    if step.name == "compileall":
        environment.append(("PYTHONPYCACHEPREFIX", "/tmp/pycache"))
    with tempfile.TemporaryDirectory(prefix="slm-merge-static-") as temporary:
        workspace = Path(temporary)
        _candidate_snapshot(root, workspace / "candidate")
        argv = runtime_command(step.cmd, runtimes)
        if (root / ".git").exists():
            _private_git(root, workspace / "control" / "git", budget_seconds, started)
            argv = [
                "/usr/bin/env",
                "-C",
                "/workspace/candidate",
                "GIT_DIR=/workspace/control/git",
                "GIT_WORK_TREE=/workspace/candidate",
                "GIT_CONFIG_NOSYSTEM=1",
                "GIT_CONFIG_GLOBAL=/dev/null",
                "GIT_OPTIONAL_LOCKS=0",
                *argv,
            ]
        else:
            argv = ["/usr/bin/env", "-C", "/workspace/candidate", *argv]
        argv = _workload_argv(root, runtimes, argv, workspace=workspace)
        result = run_isolated(
            IsolationSpec(
                workspace,
                runtime_roots=runtimes,
                timeout_seconds=_remaining(budget_seconds, started),
                pythonpath=pythonpath,
                environment=tuple(environment),
            ),
            argv,
        )
    interrupted = result.timed_out or result.interrupted or result.killed
    return {
        "name": step.name,
        "cmd": list(step.cmd),
        "seconds": time.monotonic() - started,
        "exit_code": result.returncode,
        "status": "timeout"
        if interrupted
        else "ok"
        if result.returncode == 0
        else "failed",
        "output_tail": (result.stdout + result.stderr)[-4000:],
        "evidence_class": "isolated_process",
    }


def _remaining(seconds: float, started: float) -> float:
    remaining = seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise IsolationUnavailable("verification_preparation_budget_exhausted")
    return min(remaining, INTERRUPT_AFTER_SECONDS)


def _workload_argv(
    root: Path, runtimes: tuple[Path, ...], argv: list[str], *, workspace=None
) -> list[str]:
    """Resolve only explicit runtime grants, never inherit host environment."""
    from scripts.merge_verification_runtime import runtime_argv

    return runtime_argv(root, runtimes, argv, workspace=workspace)


def _private_git(
    source: Path, destination: Path, seconds: float, started: float
) -> None:
    """Only a new disposable metadata copy is written; source refs never change.

    Non-local file transport copies reachable objects, not hooks/config,
    hardlinks or alternates. No credential or network protocol is available.
    Git-dependent static checks must see real base/index history, not an empty
    `ls-files` result from a metadata-free snapshot.
    """
    destination.parent.mkdir(exist_ok=True)
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(destination.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_TERMINAL_PROMPT": "0",
    }
    commands = [
        [
            "git",
            "clone",
            "--bare",
            "--single-branch",
            "--no-local",
            "--no-hardlinks",
            "--",
            source.resolve().as_uri(),
            str(destination),
        ],
        ["git", "--git-dir", str(destination), "read-tree", "HEAD"],
    ]
    for command in commands:
        result = run_bounded_process(
            command,
            env=env,
            interrupt_after_seconds=_remaining(seconds, started),
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
        if (
            result.returncode != 0
            or result.timed_out
            or result.interrupted
            or result.killed
        ):
            raise IsolationUnavailable(
                "private_git_preparation_failed: " + result.stderr[-1000:]
            )
    (destination / "config").write_text(
        "[core]\nrepositoryformatversion = 0\nbare = false\n"
    )
    shutil.rmtree(destination / "hooks", ignore_errors=True)
    if (destination / "objects/info/alternates").exists():
        raise ValueError("private Git metadata retained shared object authority")


def _candidate_snapshot(source: Path, destination: Path) -> Path:
    """Remove ignored/unbound files before any candidate code is executed."""
    copy = private_snapshot(source, destination)
    if not (source / ".git").exists():
        return copy  # Explicit small test fixture; production binding requires Git.
    allowed = set(source_paths(source))
    for path in tuple(allowed):
        allowed.update(parent.as_posix() for parent in Path(path).parents)
    for directory, dirs, files in os.walk(copy, topdown=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.relative_to(copy).as_posix() in allowed:
                continue
            if path.is_dir() and not path.is_symlink():
                path.rmdir()
            else:
                path.unlink()
    return copy
