"""Run proposed regression tests against baseline and candidate source."""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from slm_training.autoresearch.heal.isolation import IsolationSpec, run_isolated
from slm_training.autoresearch.heal.isolation_workspace import (
    IsolationViolation,
    private_snapshot,
)
from slm_training.harness_core.bounded_process import ProcessOutcome
from .repair_verifier import CheckObservation


def observe_regression(
    path: str,
    source: Path,
    candidate: Path,
    phase: str,
    runtimes: tuple[Path, ...],
    timeout: float,
) -> CheckObservation:
    """Run the one new regression on baseline-with-test and on candidate."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="slm-regression-", dir=source.parent
    ) as directory:
        copy = private_snapshot(source, Path(directory) / "workload")
        test = candidate / path
        if test.is_symlink() or not test.is_file():
            raise IsolationViolation("declared regression test is not a regular file")
        target = copy / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(test, target)
        target.chmod(test.stat().st_mode)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            empty = hashlib.sha256(b"").hexdigest()
            return CheckObservation(
                path,
                phase,
                False,
                None,
                ProcessOutcome.TIMED_OUT.value,
                empty,
                empty,
                time.monotonic() - started,
            )
        result = run_isolated(
            IsolationSpec(copy, runtime_roots=runtimes, timeout_seconds=remaining),
            (
                _runtime_python(runtimes),
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--no-header",
                path,
            ),
        )
    output = result.stdout
    failed_path = re.compile(rf"^FAILED {re.escape(path)}::", re.MULTILINE)
    summary_failure = re.search(r"(?m)^\s*\d+ failed(?:[, ]|$)", output)
    setup_or_collection_error = re.search(
        r"(?m)^ERROR(?:S)?(?: |$)|^\s*\d+ error(?:[, ]|$)", output
    )
    if phase == "baseline_regression":
        passed = (
            result.outcome == ProcessOutcome.COMPLETED
            and result.returncode != 0
            and not result.stdout_truncated
            and failed_path.search(output) is not None
            and summary_failure is not None
            and setup_or_collection_error is None
        )
    else:
        passed = (
            result.outcome == ProcessOutcome.COMPLETED
            and result.returncode == 0
            and not result.stdout_truncated
            and re.search(r"(?m)^\s*\d+ passed(?:[, ]|$)", output) is not None
        )
    return CheckObservation(
        path,
        phase,
        passed,
        result.returncode,
        result.outcome.value,
        hashlib.sha256(output.encode()).hexdigest(),
        hashlib.sha256(result.stderr.encode()).hexdigest(),
        time.monotonic() - started,
    )


def _runtime_python(runtimes: tuple[Path, ...]) -> str:
    executable = Path(sys.executable)
    for index, root in enumerate(runtimes):
        root = root.resolve()
        if executable.is_relative_to(root):
            return f"/runtime/{index}/{executable.relative_to(root)}"
    return sys.executable
