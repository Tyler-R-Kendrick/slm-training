"""Trusted repair acceptance runner; workload artifacts cannot issue receipts.

The controller invokes this module from its immutable release. Every check is
executed in a fresh sandbox. Candidate code never mounts the controller's
request, ledger, receipt directory, verifier installation or host credentials.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from slm_training.harness_core.bounded_process import ProcessOutcome
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
)

from .isolation import IsolationSpec, run_isolated
from .isolation_workspace import (
    IsolationViolation,
    manifest_digest,
    private_snapshot,
    tree_manifest,
)
from .repair_scope import require_routine_scope


@dataclass(frozen=True)
class VerificationCheck:
    """Controller-selected argv and exact externally observed golden response.

    Empty output is not evidence: an early os._exit(0) cannot satisfy a check.
    Expected responses come from the frozen original predicate, not the worker.
    This is behavioral regression evidence, not a proof against overfitting.
    """

    check_id: str
    argv: tuple[str, ...]
    expected_stdout: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.check_id, str)
            or not self.check_id.strip()
            or not isinstance(self.argv, tuple)
            or not self.argv
            or any(not isinstance(arg, str) or "\0" in arg for arg in self.argv)
            or not self.argv[0]
            or not isinstance(self.expected_stdout, str)
            or not self.expected_stdout.strip()
        ):
            raise ValueError(
                "verification requires a named nonempty golden observation"
            )


@dataclass(frozen=True)
class VerificationRequest:
    request_digest: str
    blocker_digest: str
    source_digest: str
    candidate_digest: str
    environment_digest: str
    verifier_release: str
    authority_digest: str
    fencing_token: str
    allowed_paths: tuple[str, ...]
    original: VerificationCheck
    checks: tuple[VerificationCheck, ...]
    failure_returncode: int
    failure_stdout_sha256: str
    failure_stderr_sha256: str
    timeout_seconds: float = INTERRUPT_AFTER_SECONDS
    semantics_preserving_paths: tuple[str, ...] = ()
    equivalence_checks: tuple[VerificationCheck, ...] = ()
    regression_test_path: str = ""

    def __post_init__(self) -> None:
        identities = (
            self.request_digest,
            self.blocker_digest,
            self.source_digest,
            self.candidate_digest,
            self.environment_digest,
            self.verifier_release,
            self.authority_digest,
            self.fencing_token,
        )
        if not all(identities) or not self.checks or not self.regression_test_path:
            raise ValueError(
                "verification requires complete identity, checks and regression test"
            )
        for identity in identities[:-1]:
            _require_digest(identity)
        ids = [
            self.original.check_id,
            *(check.check_id for check in self.checks),
            *(check.check_id for check in self.equivalence_checks),
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("verification check identities must be unique")
        if self.semantics_preserving_paths and not self.equivalence_checks:
            raise ValueError("wiring equivalence requires pinned differential checks")
        if (
            type(self.failure_returncode) is not int
            or not 0 < self.failure_returncode < 256
        ):
            raise ValueError("original failure must have an ordinary nonzero exit")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= INTERRUPT_AFTER_SECONDS
        ):
            raise ValueError("verification timeout exceeds canonical interrupt budget")
        for digest in (self.failure_stdout_sha256, self.failure_stderr_sha256):
            _require_digest(digest)


def _require_digest(digest: str) -> None:
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("verification identity requires exact SHA256")


@dataclass(frozen=True)
class CheckObservation:
    check_id: str
    phase: str
    passed: bool
    returncode: int | None
    outcome: str
    stdout_sha256: str
    stderr_sha256: str
    duration_seconds: float


@dataclass(frozen=True)
class VerificationEvidence:
    request: VerificationRequest
    accepted: bool
    reason: str
    manifest_digest: str
    observations: tuple[CheckObservation, ...]
    changed_paths: tuple[str, ...]
    evidence_class: str = "independent_isolated_process"


def _observe(
    check: VerificationCheck,
    source: Path,
    phase: str,
    runtimes: tuple[Path, ...],
    timeout: float,
) -> CheckObservation:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="slm-verify-", dir=source.parent
    ) as directory:
        copy = private_snapshot(source, Path(directory) / "workload")
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            empty = hashlib.sha256(b"").hexdigest()
            return CheckObservation(
                check.check_id,
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
            check.argv,
        )
    passed = (
        result.outcome == ProcessOutcome.COMPLETED
        and result.returncode == 0
        and not result.stdout_truncated
        and result.stdout == check.expected_stdout
    )
    return CheckObservation(
        check.check_id,
        phase,
        passed,
        result.returncode,
        result.outcome.value,
        hashlib.sha256(result.stdout.encode()).hexdigest(),
        hashlib.sha256(result.stderr.encode()).hexdigest(),
        time.monotonic() - started,
    )


def verification_manifest_digest(request: VerificationRequest) -> str:
    import json
    from dataclasses import asdict

    return hashlib.sha256(
        json.dumps(asdict(request), sort_keys=True).encode()
    ).hexdigest()


def verify_candidate(
    request: VerificationRequest,
    base: Path,
    candidate: Path,
    *,
    runtime_roots: tuple[Path, ...] = (),
    deadline: float | None = None,
) -> VerificationEvidence:
    """Reproduce, independently verify, and return evidence only to controller.

    Caller must atomically check the live fencing token/request and publish the
    approved immutable source. A serialized copy of this result is not trusted
    input to this function and cannot itself grant a release.
    """
    cap = (
        time.monotonic()
        + INTERRUPT_AFTER_SECONDS
        - HARNESS_FINALIZATION_RESERVE_SECONDS
    )
    if deadline is not None and (
        not math.isfinite(deadline) or isinstance(deadline, bool)
    ):
        raise ValueError("invalid verification deadline")
    deadline = cap if deadline is None else min(cap, deadline)
    before, after = tree_manifest(base), tree_manifest(candidate)
    if (
        manifest_digest(before) != request.source_digest
        or manifest_digest(after) != request.candidate_digest
    ):
        raise IsolationViolation("source/candidate identity mismatch")
    changed = tuple(
        sorted(
            path
            for path in before.keys() | after.keys()
            if before.get(path) != after.get(path)
            and not ((candidate / path).is_dir() and path not in before)
        )
    )
    require_routine_scope(
        base,
        candidate,
        changed,
        request.allowed_paths,
        request.semantics_preserving_paths,
        request_digest=request.request_digest,
    )
    new_tests = tuple(
        path
        for path in changed
        if path.startswith("tests/")
        and path.endswith(".py")
        and not (base / path).exists()
    )
    if new_tests != (request.regression_test_path,):
        raise IsolationViolation("repair must identify its sole new regression module")
    try:
        return _verify_observations(
            request, base, candidate, changed, runtime_roots, deadline
        )
    finally:
        if tree_manifest(base) != before or tree_manifest(candidate) != after:
            raise IsolationViolation("source changed during independent verification")


def _verify_observations(
    request: VerificationRequest,
    base: Path,
    candidate: Path,
    changed: tuple[str, ...],
    runtime_roots: tuple[Path, ...],
    deadline: float,
) -> VerificationEvidence:
    manifest = verification_manifest_digest(request)
    observations = []
    reason = None
    remaining = deadline - time.monotonic() - KILL_GRACE_SECONDS
    if remaining <= 0:
        reason = "verification_budget_exhausted"
    else:
        observations.append(
            _observe(
                request.original,
                base,
                "original_reproduction",
                runtime_roots,
                min(request.timeout_seconds, remaining),
            )
        )
        first = observations[0]
        reproduced = (
            not first.passed
            and first.outcome == ProcessOutcome.COMPLETED
            and first.returncode == request.failure_returncode
            and first.stdout_sha256 == request.failure_stdout_sha256
            and first.stderr_sha256 == request.failure_stderr_sha256
        )
        if not reproduced:
            reason = "original_failure_not_reproduced"
        else:
            reason = _verify_regression_stages(
                request, base, candidate, runtime_roots, deadline, observations
            )
    accepted = reason is None and all(item.passed for item in observations[1:])
    if reason is None and not accepted:
        reason = "verification_failed"
    return VerificationEvidence(
        request,
        accepted,
        "restored_predicate" if accepted else reason,
        manifest,
        tuple(observations),
        changed,
    )


def _verify_regression_stages(
    request, base, candidate, runtime_roots, deadline, observations
):
    from .repair_regression import observe_regression

    remaining = deadline - time.monotonic() - KILL_GRACE_SECONDS
    if remaining <= 0:
        return "verification_budget_exhausted"
    baseline = observe_regression(
        request.regression_test_path,
        base,
        candidate,
        "baseline_regression",
        runtime_roots,
        min(request.timeout_seconds, remaining),
    )
    observations.append(baseline)
    if not baseline.passed:
        return "regression_not_reproduced"
    tasks = [
        (check, base, "baseline_equivalence") for check in request.equivalence_checks
    ]
    tasks += [
        (check, candidate, "candidate_verification")
        for check in (request.original, *request.checks, *request.equivalence_checks)
    ]
    for check, source, phase in tasks:
        remaining = deadline - time.monotonic() - KILL_GRACE_SECONDS
        if remaining <= 0:
            return "verification_budget_exhausted"
        observations.append(
            _observe(
                check,
                source,
                phase,
                runtime_roots,
                min(request.timeout_seconds, remaining),
            )
        )
    remaining = deadline - time.monotonic() - KILL_GRACE_SECONDS
    if remaining <= 0:
        return "verification_budget_exhausted"
    observations.append(
        observe_regression(
            request.regression_test_path,
            candidate,
            candidate,
            "candidate_regression",
            runtime_roots,
            min(request.timeout_seconds, remaining),
        )
    )
    return None
