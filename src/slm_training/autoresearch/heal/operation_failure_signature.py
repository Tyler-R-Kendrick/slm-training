"""Narrow, controller-captured identity for a Python operation failure."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from .repair_contracts import Digest, RepairModel


_FRAME = re.compile(
    r'^  File "([^"]+)", line ([1-9][0-9]*), in (<module>|[A-Za-z_][\w.]*)$'
)
_EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception)): (.*)$")


class OperationFailureSignature(RepairModel):
    schema_version: Literal["operation_failure_signature/v1"] = (
        "operation_failure_signature/v1"
    )
    kind: Literal["python_exception"] = "python_exception"
    source_path: str = Field(min_length=1)
    line: int = Field(gt=0)
    function: str = Field(min_length=1)
    exception_type: str = Field(min_length=1)
    message_sha256: Digest
    returncode: int = Field(gt=0, lt=256)


def capture_failure_signature(result, source: Path) -> OperationFailureSignature | None:
    """Capture only a complete traceback ending in source-owned Python code."""
    if (
        result.returncode is None
        or result.returncode <= 0
        or result.returncode >= 256
        or result.stdout_truncated
        or result.stderr_truncated
        or result.launch_error
    ):
        return None
    lines = result.stderr.strip().splitlines()
    if "Traceback (most recent call last):" not in lines:
        return None
    start = len(lines) - 1 - lines[::-1].index("Traceback (most recent call last):")
    exception = _EXCEPTION.fullmatch(lines[-1])
    frames = [_FRAME.fullmatch(line) for line in lines[start + 1 : -1]]
    frames = [frame for frame in frames if frame]
    if not exception or not frames or not Path(frames[-1][1]).is_absolute():
        return None
    frame = frames[-1]
    path = Path(frame[1])
    try:
        relative = path.resolve().relative_to(source.resolve())
    except ValueError:
        return None
    if not relative.parts or relative.suffix != ".py":
        return None
    return OperationFailureSignature(
        source_path=relative.as_posix(),
        line=int(frame[2]),
        function=frame[3],
        exception_type=exception[1],
        message_sha256=hashlib.sha256(exception[2].encode()).hexdigest(),
        returncode=result.returncode,
    )
