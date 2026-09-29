"""Native user-systemd ownership and recorded three-hour observations."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from slm_training.autoresearch.runtime.operations_status import (
    loop_status,
    runtime_store,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS


def _quote(value: str) -> str:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("unit_value_contains_control_character")
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def _command(argv: list[str]) -> str:
    return " ".join(_quote(word).replace("$", "$$") for word in argv)


def service_units(config_path: Path, root: Path, settings) -> dict:
    """Generate units only; configuration existence is not release acceptance."""
    config = settings
    execution = Path(config.execution).resolve(strict=True)
    name = "slm-autoresearch-" + config.loop_id
    monitor = name + "-monitor"
    common = [
        sys.executable,
        "-m",
        "scripts.autoresearch",
        "--root",
        str(root.resolve()),
    ]
    _quote(str(execution))  # Reject newline/control injection in the scalar path.
    environment = (
        "WorkingDirectory="
        + str(execution).replace("%", "%%")
        + "\n"
        + "Environment="
        + _quote("PYTHONPATH=" + str(execution / "src"))
        + "\n"
        + "Environment=PYTHONDONTWRITEBYTECODE=1\n"
        + "UnsetEnvironment=PYTHONPYCACHEPREFIX\n"
    )
    environment += "".join(
        "Environment=" + _quote(key + "=" + value) + "\n"
        for key, value in sorted(config.runtime_environment.items())
    )
    stop = f"KillMode=mixed\nKillSignal=SIGINT\nTimeoutStopSec={KILL_GRACE_SECONDS}\nSendSIGKILL=yes\n"
    units = {
        name + ".service": (
            "[Unit]\nDescription=SLM experiment supervisor\n"
            "StartLimitIntervalSec=300\nStartLimitBurst=5\n\n[Service]\nType=exec\n"
            + environment
            + "ExecStart="
            + _command(
                common
                + ["start", "--config", str(config_path.resolve()), "--foreground"]
            )
            + "\n"
            + "Restart=on-failure\nRestartSec=30\n"
            + stop
            + "\n[Install]\nWantedBy=default.target\n"
        ),
        monitor + ".service": (
            "[Unit]\nDescription=SLM recorded progress check\n\n[Service]\nType=oneshot\n"
            + environment
            + "ExecStart="
            + _command(common + ["monitor-check", "--loop-id", config.loop_id])
            + "\n"
            + f"TimeoutStartSec={INTERRUPT_AFTER_SECONDS}\n"
            + stop
        ),
        monitor + ".timer": (
            "[Unit]\nDescription=SLM three-hour progress observation\n\n[Timer]\n"
            "OnCalendar=*-*-* 00/3:00:00 UTC\nPersistent=true\nAccuracySec=1s\n"
            + "Unit="
            + monitor
            + ".service\n\n[Install]\nWantedBy=timers.target\n"
        ),
    }
    return {
        "installed": False,
        "activated": False,
        "units": units,
        "service": name + ".service",
        "timer": monitor + ".timer",
        "enable_command": [
            "systemctl",
            "--user",
            "enable",
            "--now",
            name + ".service",
            monitor + ".timer",
        ],
        "stop_command": [
            "systemctl",
            "--user",
            "stop",
            name + ".service",
            monitor + ".timer",
        ],
        "activation_requires": "parent_verified_release",
    }


def install_service(
    config: Path, root: Path, settings, unit_dir: Path | None = None
) -> dict:
    """Validate and install idempotently. Activation is a separate explicit action."""
    result = service_units(config, root, settings)
    unit_dir = (
        unit_dir
        or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "systemd/user"
    )
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name, value in result["units"].items():
        path = unit_dir / name
        if path.is_symlink() or (path.exists() and path.read_text() != value):
            raise ValueError("existing_service_unit_conflict")
    with tempfile.TemporaryDirectory(prefix="slm-units-") as scratch:
        paths = []
        for name, value in result["units"].items():
            path = Path(scratch) / name
            path.write_text(value)
            paths.append(str(path))
        runtime_dir = Path(scratch) / "runtime"
        runtime_dir.mkdir(mode=0o700)
        check = subprocess.run(
            ["systemd-analyze", "--user", "verify", *paths],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env={**os.environ, "XDG_RUNTIME_DIR": str(runtime_dir)},
        )
        if check.returncode:
            raise ValueError("systemd_unit_validation_failed")
    for name, value in result["units"].items():
        path = unit_dir / name
        if not path.exists():
            CampaignStore._atomic_new(path, value)
    return {
        **result,
        "installed": True,
        "unit_dir": str(unit_dir),
        "reload_command": ["systemctl", "--user", "daemon-reload"],
    }


def timer_status(loop_id: str) -> dict:
    unit = "slm-autoresearch-" + loop_id + "-monitor.timer"
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=ActiveState,NextElapseUSecRealtime,LastTriggerUSec",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"ready": False, "reason": "user_systemd_unavailable"}
    if result.returncode:
        return {"ready": False, "reason": "user_systemd_unavailable"}
    properties = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    return {
        "ready": properties.get("ActiveState") == "active",
        "unit": unit,
        "next_trigger": properties.get("NextElapseUSecRealtime") or None,
        "last_trigger": properties.get("LastTriggerUSec") or None,
    }


def monitor_check(root: Path, loop_id: str) -> dict:
    """Record an actual invocation; never mutate experiments or restart the lab."""
    try:
        status = loop_status(root, loop_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {
            "checked_at": time.time(),
            "assessment": "invalid_evidence",
            "progress_event": None,
            "error_type": type(exc).__name__,
            "timer": timer_status(loop_id),
            "pending_action": "inspect_evidence",
        }
        runtime_store(root, loop_id).append_event(
            "operations_monitor_checked",
            idempotency_key="monitor:" + str(time.time_ns()),
            detail=result,
        )
        return result
    previous = status.get("last_monitor_check")
    progress = status["last_operational_progress_event"]
    if not status["active"]:
        assessment = "stopped"
    elif previous is None or previous["progress_event"] != progress:
        assessment = "observing" if previous is None else "advancing"
    elif status["waiting"]:
        assessment = "waiting"
    else:
        assessment = "stalled"
    result = {
        "checked_at": time.time(),
        "assessment": assessment,
        "progress_event": progress,
        "controller": status["controller"],
        "waiting": status["waiting"],
        "timer": timer_status(loop_id),
        "pending_action": "inspect_controller"
        if assessment in {"stopped", "stalled"}
        else "observe_existing_work",
    }
    runtime_store(root, loop_id).append_event(
        "operations_monitor_checked",
        idempotency_key="monitor:" + str(time.time_ns()),
        detail=result,
    )
    return result
