"""Read-only capability probes; installed files alone do not establish readiness."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

from slm_training.autoresearch.climb_policy import load_climb_policy
from slm_training.autoresearch.heal.agent_executor import probe_codex
from slm_training.autoresearch.heal.isolation import probe_isolation
from slm_training.autoresearch.heal.recovery_dispatch import load_recovery_config
from slm_training.harness_core.bounded_process import (
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS


def _probe(argv: tuple[str, ...], cwd: Path, *, env: dict | None = None) -> dict:
    result = run_bounded_process(
        argv,
        cwd=cwd,
        env=env,
        interrupt_after_seconds=15,
        kill_grace_seconds=KILL_GRACE_SECONDS,
        max_output_bytes=2000,
    )
    ok = result.outcome == ProcessOutcome.COMPLETED and result.returncode == 0
    # Raw provider stderr is never emitted into diagnostics or generated docs.
    return {
        "ready": ok,
        "exit_code": result.returncode,
        "outcome": result.outcome.value,
        "stderr_digest": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "seconds": result.duration_seconds,
    }


def storage_health(root: Path, *, minimum_free_bytes: int = 256 * 1024 * 1024) -> dict:
    if (
        isinstance(minimum_free_bytes, bool)
        or not isinstance(minimum_free_bytes, int)
        or minimum_free_bytes < 0
    ):
        raise ValueError("minimum_free_bytes_must_be_nonnegative_integer")
    target = root.absolute()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    try:
        with tempfile.TemporaryFile(dir=target) as probe:
            probe.write(b"slm-readiness")
            probe.flush()
            os.fsync(probe.fileno())
        writable = True
    except OSError:
        writable = False
    available = _available_memory_bytes()
    pressure = usage.free < minimum_free_bytes
    memory_pressure = available is not None and available < 256 * 1024 * 1024
    return {
        "ready": writable and not pressure and not memory_pressure,
        "writable": writable,
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum_free_bytes,
        "available_memory_bytes": available,
        "memory_pressure": memory_pressure,
        "action": "storage_maintenance"
        if pressure or not writable
        else "memory_capacity"
        if memory_pressure
        else None,
        "deletion_performed": False,
    }


def _available_memory_bytes() -> int | None:
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def _javascript_probes(source: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        return {
            name: {"ready": False, "reason": "node_missing"}
            for name in ("agentv", "openui_bridge", "design_md_bridge", "node_version")
        }
    runner = Path(
        os.environ.get("AGENTV_RUNNER", source / "scripts/run_agentv_eval.mjs")
    )
    sdk = runner.resolve().parents[1] / "node_modules/@agentv/core/dist/index.js"
    code = "const m=await import(process.argv[1]);if(typeof m.evaluate!=='function')process.exit(2)"
    result = {
        "agentv": _probe(
            (node, "--input-type=module", "-e", code, sdk.as_uri()), source
        )
    }
    for name, variable, directory in (
        ("openui_bridge", "OPENUI_BRIDGE_CLI", "openui_bridge"),
        ("design_md_bridge", "DESIGN_MD_BRIDGE_CLI", "design_md_bridge"),
    ):
        cli = Path(
            os.environ.get(variable, source / "src/apps" / directory / "cli.mjs")
        ).resolve()
        # Execute the real stdio entrypoint: imported transitive dependencies and
        # a successful protocol response are required, not node_modules existence.
        code = (
            "import json,subprocess,sys;"
            "r=subprocess.run([sys.argv[1],sys.argv[2],'--repl'],"
            "input=json.dumps({'op':'ping'})+'\\n',text=True,capture_output=True,timeout=12);"
            "sys.stderr.write(r.stderr[-2000:]);"
            "sys.exit(0 if r.returncode==0 and json.loads(r.stdout).get('ok') is True else 2)"
        )
        result[name] = _probe((sys.executable, "-c", code, node, str(cli)), source)
    result["node_version"] = _probe(
        (
            node,
            "-e",
            "const n=+process.versions.node.split('.')[0];process.exit(n>=20&&n<23?0:1)",
        ),
        source,
    )
    return result


def _agent_probe(recovery_config: Path | None) -> dict:
    try:
        config = load_recovery_config(recovery_config)
    except (OSError, ValueError):
        return {"ready": False, "reason": "invalid_recovery_configuration"}
    if config is None or config.grant is None:
        return {"ready": False, "reason": "agent_grant_missing"}
    import time

    cli = probe_codex(config.grant.executable)
    return {
        "ready": False,
        "configured_launch_capability": cli["available"]
        and config.grant.expires_at > time.time(),
        "cli_available": cli["available"],
        "grant_id": config.grant.grant_id,
        "reason": "grant_declared_provider_not_contacted",
        "provider_allowance_live_verified": False,
    }


def doctor(
    source: Path,
    root: Path,
    *,
    recovery_config: Path | None = None,
    require_lean: bool = False,
) -> dict:
    probes = {}
    probes["python"] = _probe(
        (
            sys.executable,
            "-c",
            "import sys,torch,pydantic; assert (3,12)<=sys.version_info[:2]<(3,13); print('ready')",
        ),
        source,
    )
    probes.update(_javascript_probes(source))
    isolation = probe_isolation()
    probes["isolation"] = {
        "ready": isolation.available,
        "backend": isolation.backend,
        "reason": "available" if isolation.available else "isolation_probe_failed",
        "detail_digest": hashlib.sha256(isolation.reason.encode()).hexdigest(),
    }
    probes["storage"] = storage_health(root)
    probes["agent"] = _agent_probe(recovery_config)
    if require_lean:
        lean = shutil.which("lean")
        probes["lean"] = (
            {"ready": False, "reason": "lean_missing"}
            if lean is None
            else _probe((lean, "--version"), source)
        )
    required = (
        "python",
        "agentv",
        "openui_bridge",
        "design_md_bridge",
        "node_version",
        "storage",
    )
    if require_lean:
        required += ("lean",)
    return {
        "schema_version": "autonomy_doctor/v1",
        "local_ready": all(probes[k]["ready"] for k in required),
        "repair_ready": False,
        "probes": probes,
        "policy_digest": load_climb_policy().sha256,
        "interrupt_seconds": INTERRUPT_AFTER_SECONDS,
        "kill_grace_seconds": KILL_GRACE_SECONDS,
        "external_writes": False,
        "service_activation": False,
    }
