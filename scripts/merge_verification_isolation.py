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
from scripts.merge_verification_git import _git_wrapper, _private_git, _remaining
from slm_training.autoresearch.heal.isolation import (
    IsolationSpec,
    IsolationUnavailable,
    run_isolated,
)
from slm_training.autoresearch.heal.isolation_workspace import (
    _OMIT,
    private_snapshot,
    private_snapshot_with_disposable_dirs,
)
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
            filter(
                None,
                (
                    str(root / "src"),
                    *(str(runtime) for runtime in runtimes),
                    env.get("PYTHONPATH"),
                ),
            )
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
            payload = json.loads((work / "result.json").read_text())
        except (OSError, ValueError):
            payload = {}
        try:
            if result.returncode != 0:
                raise ValueError(f"workload exit {result.returncode}")
            if not collect_only and "passed" not in result.stdout:
                raise ValueError("workload exit 0 without pytest execution evidence")
            nodes = validate_workload(
                payload,
                request_digest=request_digest,
                expected=None if collect_only else targets,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return {
                **record,
                "reason": str(exc),
                "workload_observation": payload,
            }
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
        if not collect_only and "passed" not in result.stdout:
            raise ValueError("workload exit 0 without pytest execution evidence")
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
        (workspace / "candidate" / "outputs").mkdir(exist_ok=True)
        control = workspace / "control"
        control.mkdir()
        if (root / ".git").exists():
            _private_git(root, control / "git", seconds, started)
            _git_wrapper(control / "bin")
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
        output = workspace / "workload-output" / "result.json"
        output.parent.mkdir()
        output.write_text("{}")
        argv = runtime_command(
            (
                sys.executable,
                "/workspace/control/worker.py",
                "/workspace/control/request.json",
                "/workspace/workload-output/result.json",
            ),
            runtimes,
        )
        argv = _workload_argv(root, runtimes, argv, workspace=workspace)
        bridge_runtime = _bridge_runtime(runtimes)
        from scripts.merge_verification_runtime import javascript_grants

        runtime_grants = javascript_grants(root, runtimes)
        bridge_grants = runtime_grants["bridges"]
        environment = [
            ("OMP_NUM_THREADS", "1"),
            ("MKL_NUM_THREADS", "1"),
            ("TOKENIZERS_PARALLELISM", "false"),
            ("SLM_REQUIRE_ISOLATION", "1"),
        ]
        if (root / ".git").exists():
            environment.extend([
                ("GIT_DIR", "/workspace/control/git"),
                ("GIT_WORK_TREE", "/workspace/candidate"),
                ("GIT_CONFIG_NOSYSTEM", "1"),
                ("GIT_CONFIG_GLOBAL", "/dev/null"),
                ("GIT_OPTIONAL_LOCKS", "0"),
            ])
        if bridge_runtime is not None:
            environment.extend(
                (("OPENUI_BRIDGE_NODE_MODULES", bridge_runtime), ("NODE_PATH", bridge_runtime))
            )
        environment.extend(bridge_grants.items())
        if runtime_grants["sdk_index"] is not None:
            environment.append(
                ("AGENTV_NODE_MODULES", f"/runtime/{runtime_grants['sdk_index']}")
            )
        for index, runtime in enumerate(runtimes):
            if runtime.name == "slm-hf-cache":
                environment.extend(
                    (("HF_HOME", f"/runtime/{index}"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"))
                )
            if (runtime / "chromium-1228").is_dir():
                environment.append(("PLAYWRIGHT_BROWSERS_PATH", f"/runtime/{index}"))
            if (runtime / "leverproof-lean").is_file():
                environment.append(("SLM_LEVERPROOF_CHECKER", f"/runtime/{index}/leverproof-lean"))
        result = run_isolated(
            IsolationSpec(
                workspace,
                writable_paths=("workload-output/result.json",),
                writable_dirs=("candidate/outputs",),
                runtime_roots=runtimes,
                timeout_seconds=_remaining(seconds, started),
                pythonpath=os.pathsep.join(
                    ("/workspace/candidate/src", *(f"/runtime/{i}" for i in range(1, len(runtimes))))
                ),
                environment=tuple(environment),
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
    bridge_runtime = _bridge_runtime(runtimes)
    if bridge_runtime is not None:
        environment.extend(
            (("OPENUI_BRIDGE_NODE_MODULES", bridge_runtime), ("NODE_PATH", bridge_runtime))
        )
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
            _git_wrapper(workspace / "control" / "bin")
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
def _workload_argv(
    root: Path, runtimes: tuple[Path, ...], argv: list[str], *, workspace=None
) -> list[str]:
    """Resolve only explicit runtime grants, never inherit host environment."""
    from scripts.merge_verification_runtime import runtime_argv
    return runtime_argv(root, runtimes, argv, workspace=workspace)
def _bridge_runtime(runtimes: tuple[Path, ...]) -> str | None:
    for index, root in enumerate(runtimes):
        if root.name == "openui_bridge" and (
            root / "node_modules" / "@openuidev" / "lang-core"
        ).is_dir():
            return f"/runtime/{index}/node_modules"
    return None
def _candidate_snapshot(source: Path, destination: Path) -> Path:
    """Remove ignored/unbound files before any candidate code is executed."""
    if not (source / ".git").exists():
        return private_snapshot_with_disposable_dirs(
            source, destination, (), ()
        )  # Explicit small test fixture; production binding requires Git.
    allowed = tuple(source_paths(source))
    top_level = {path.split("/", 1)[0] for path in allowed}
    extras = {"node_modules", "src"}
    if any(
        child.name not in top_level | extras | _OMIT for child in source.iterdir()
    ):
        return private_snapshot_with_disposable_dirs(
            source, destination, ("node_modules", "src/apps/openui_bridge/node_modules"), allowed
        )
    copy = private_snapshot(source, destination)
    for relative in ("node_modules", "src/apps/openui_bridge/node_modules"):
        (copy / relative).mkdir(parents=True, exist_ok=True)
    return copy
def run_isolated_phase(
    state: dict, kind: str, phase, *, fast_persist
):
    """Run a gate phase, parking budget-exhausted sandbox preparation.

    Preparation that runs out of invocation budget is a transient budget
    wait, never a missing capability: register ``insufficient_budget`` for a
    fresh bounded invocation instead of aborting as ``waiting_capability``.
    Genuine isolation failures still propagate.
    """
    try:
        return phase()
    except IsolationUnavailable as exc:
        if "verification_preparation_budget_exhausted" not in str(exc):
            raise
        state["waiting"][digest([kind, "__phase__"])] = {
            "kind": kind,
            "target_digest": digest([kind]),
            "reason": "insufficient_budget",
            "required_seconds": state.get("workload_budget_seconds", 1.0),
            "available_seconds": 0.0,
            "wake_source": "fresh_bounded_invocation",
        }
        fast_persist()
        return None
