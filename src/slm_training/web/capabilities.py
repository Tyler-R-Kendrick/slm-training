"""Runtime capability detection: local control plane vs read-only deployment.

The dashboard is a full control plane when served locally (uvicorn via
``scripts/serve_playground.py``) — it can launch allowlisted background jobs. On
Vercel (serverless, read-only filesystem) it must degrade to read-only
observability. ``create_app(execution=...)`` opts in; this module is the runtime
backstop so a mis-set flag on a read-only FS still degrades safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Capabilities:
    """What the running server is allowed to do."""

    execution: bool
    jobs_concurrency: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "execution": self.execution,
            "jobs_concurrency": self.jobs_concurrency,
            "read_only": not self.execution,
        }


def is_serverless() -> bool:
    """Known serverless hosts: read-only FS, per-request short-lived processes."""
    return bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


def detect_execution(
    requested: bool, *, outputs_dir: Path | str = Path("outputs")
) -> bool:
    """Execution is permitted only when *explicitly* requested AND the runtime can
    actually write to ``outputs/`` and is not a known serverless host.

    This is intentionally conservative: read-only is the safe default, so any
    ambiguity (serverless env var present, non-writable FS) disables execution.
    """
    if not requested:
        return False
    if is_serverless():
        return False
    outputs = Path(outputs_dir)
    try:
        outputs.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(outputs, os.W_OK)


def resolve_capabilities(
    requested: bool,
    *,
    outputs_dir: Path | str = Path("outputs"),
    jobs_concurrency: int = 1,
) -> Capabilities:
    return Capabilities(
        execution=detect_execution(requested, outputs_dir=outputs_dir),
        jobs_concurrency=max(1, int(jobs_concurrency)),
    )
