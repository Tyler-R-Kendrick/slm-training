"""Independent formal-object checkers (multi-prover backends).

Backends
--------
``python_structural``
    Pure Python re-implementation of Lean / VSS honesty laws. No Lean kernel.
``python_replay``
    Full VSS certificate search replay (``replay_support_certificate``). Needs
    a live expander+verifier context when provided; otherwise structural-only
    honesty is still enforced and replay is reported as skipped.
``lean_kernel``
    Optional Lean 4 package build / theorem audit. Never sole authority.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
from pathlib import Path
import tempfile
import time
from typing import Any

from slm_training.formal.objects import FormalObjectKind, FormalObjectV1
from slm_training.formal.structural import (
    check_lean_or_closure_laws,
    check_reference_laws,
    check_support_certificate_reference,
    check_support_certificate_structure,
)
from slm_training.levers import (
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    MAX_RUN_SECONDS,
)
from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
    run_bounded_process,
)

CHECKER_PYTHON_STRUCTURAL = "python_structural"
CHECKER_PYTHON_REFERENCE = "python_reference"
CHECKER_PYTHON_REPLAY = "python_replay"
CHECKER_LEAN_KERNEL = "lean_kernel"

REPO_ROOT = Path(__file__).resolve().parents[3]
LEAN_ROOT = REPO_ROOT / "src" / "leverproof_lean"


@dataclass(frozen=True)
class CheckerResult:
    checker_id: str
    backend: str
    ok: bool
    detail: str
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "checker_id": self.checker_id,
            "backend": self.backend,
            "ok": self.ok,
            "detail": self.detail,
            "skipped": self.skipped,
        }


class FormalProjectLock:
    """Process-shared lock for a mutable Lake project build directory.

    Lake's incremental artifacts are shared by every formal surface.  The lock
    is keyed by the canonical project path and stored outside the repository so
    an interrupted process cannot leave a tracked lock artifact behind.
    """

    def __init__(self, project_root: Path, *, timeout_seconds: float) -> None:
        self.project_root = project_root.resolve()
        self.timeout_seconds = max(0.001, float(timeout_seconds))
        digest = hashlib.sha256(str(self.project_root).encode("utf-8")).hexdigest()
        self.lock_path = (
            Path(tempfile.gettempdir()) / f"slm-formal-{digest[:24]}.lock"
        )
        self.wait_seconds = 0.0
        self._handle: Any | None = None

    def __enter__(self) -> "FormalProjectLock":
        started = time.monotonic()
        handle = self.lock_path.open("a+")
        deadline = started + self.timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._handle = handle
                    self.wait_seconds = time.monotonic() - started
                    return self
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "formal project lock timed out after "
                            f"{self.timeout_seconds:.3f}s: {self.project_root}"
                        )
                    time.sleep(min(0.05, remaining))
        except BaseException:
            handle.close()
            raise

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def formal_process_budget(timeout_seconds: float | None) -> tuple[float, float, float]:
    """Return total, interrupt, and kill-grace seconds under the repo cap."""

    requested = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else min(120.0, float(MAX_RUN_SECONDS) - 1.0)
    )
    total = min(float(MAX_RUN_SECONDS), max(0.001, requested))
    grace = min(float(KILL_GRACE_SECONDS), total * 0.1)
    interrupt_after = min(
        float(INTERRUPT_AFTER_SECONDS), max(0.001, total - grace)
    )
    return total, interrupt_after, grace


def run_formal_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float | None,
    on_start: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
) -> BoundedProcessResult:
    """Run a formal command in a bounded, process-group-owned subprocess."""

    _total, interrupt_after, grace = formal_process_budget(timeout_seconds)
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "interrupt_after_seconds": interrupt_after,
        "kill_grace_seconds": grace,
    }
    if on_start is not None:
        kwargs["on_start"] = on_start
    if on_heartbeat is not None:
        kwargs["on_heartbeat"] = on_heartbeat
    return run_bounded_process(command, **kwargs)


# Optional injectable replay context: (state, expander, verifier)
ReplayContext = tuple[Any, Any, Any]
ReplayProvider = Callable[[FormalObjectV1], ReplayContext | None]


def check_python_structural(obj: FormalObjectV1) -> CheckerResult:
    """Structural independent prover — always available, no Lean."""

    violations: list[str] = []
    try:
        FormalObjectV1.from_dict(obj.to_dict())
    except ValueError as exc:
        violations.append(str(exc))

    if obj.kind is FormalObjectKind.SUPPORT_CERTIFICATE:
        violations.extend(check_support_certificate_structure(obj.payload))
    elif obj.kind in {FormalObjectKind.LEAN_CLAIM, FormalObjectKind.CLOSURE_LAW}:
        laws = [str(x) for x in obj.statement.get("laws", ())]
        context = obj.payload.get("context")
        if not isinstance(context, dict):
            context = None
        violations.extend(check_lean_or_closure_laws(laws, context))
    else:
        violations.append(f"unsupported formal object kind: {obj.kind}")

    ok = not violations
    return CheckerResult(
        checker_id=CHECKER_PYTHON_STRUCTURAL,
        backend=CHECKER_PYTHON_STRUCTURAL,
        ok=ok,
        detail="ok" if ok else "; ".join(violations),
    )


def check_python_reference(obj: FormalObjectV1) -> CheckerResult:
    """Second pure-Python prover path (set/digest-first encodings)."""

    violations: list[str] = []
    if obj.kind is FormalObjectKind.SUPPORT_CERTIFICATE:
        violations.extend(check_support_certificate_reference(obj.payload))
    elif obj.kind in {FormalObjectKind.LEAN_CLAIM, FormalObjectKind.CLOSURE_LAW}:
        laws = [str(x) for x in obj.statement.get("laws", ())]
        context = obj.payload.get("context")
        if not isinstance(context, dict):
            context = None
        violations.extend(check_reference_laws(laws, context))
    else:
        violations.append(f"unsupported kind for reference checker: {obj.kind}")
    ok = not violations
    return CheckerResult(
        checker_id=CHECKER_PYTHON_REFERENCE,
        backend=CHECKER_PYTHON_REFERENCE,
        ok=ok,
        detail="ok" if ok else "; ".join(violations),
    )


def check_python_replay(
    obj: FormalObjectV1,
    *,
    replay_provider: ReplayProvider | None = None,
) -> CheckerResult:
    """Independent search-replay prover for support certificates."""

    if obj.kind is not FormalObjectKind.SUPPORT_CERTIFICATE:
        return CheckerResult(
            checker_id=CHECKER_PYTHON_REPLAY,
            backend=CHECKER_PYTHON_REPLAY,
            ok=True,
            detail="not a support certificate; replay not required",
            skipped=True,
        )

    from slm_training.dsl.solver.support import (
        SupportCertificate,
        replay_support_certificate,
    )

    try:
        certificate = SupportCertificate.from_dict(obj.payload["certificate"])
    except (KeyError, TypeError, ValueError) as exc:
        return CheckerResult(
            checker_id=CHECKER_PYTHON_REPLAY,
            backend=CHECKER_PYTHON_REPLAY,
            ok=False,
            detail=f"certificate decode failed: {exc}",
        )

    structural = check_support_certificate_structure(obj.payload)
    if structural:
        return CheckerResult(
            checker_id=CHECKER_PYTHON_REPLAY,
            backend=CHECKER_PYTHON_REPLAY,
            ok=False,
            detail="; ".join(structural),
        )

    context = replay_provider(obj) if replay_provider is not None else None
    if context is None:
        return CheckerResult(
            checker_id=CHECKER_PYTHON_REPLAY,
            backend=CHECKER_PYTHON_REPLAY,
            ok=True,
            detail=(
                "structural honesty ok; full search replay skipped "
                "(no expander/verifier context provided)"
            ),
            skipped=True,
        )

    state, expander, verifier = context
    result = replay_support_certificate(
        certificate, state=state, expander=expander, verifier=verifier
    )
    return CheckerResult(
        checker_id=CHECKER_PYTHON_REPLAY,
        backend=CHECKER_PYTHON_REPLAY,
        ok=result.ok,
        detail="ok" if result.ok else "; ".join(result.violations),
    )


def check_lean_kernel(
    obj: FormalObjectV1,
    *,
    timeout_s: float | None = None,
    enabled: bool = True,
) -> CheckerResult:
    """Optional Lean 4 kernel audit — never sole authority for the loop."""

    if not enabled:
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=True,
            detail="lean kernel disabled",
            skipped=True,
        )
    if obj.kind is FormalObjectKind.SUPPORT_CERTIFICATE:
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=True,
            detail="support certificates are not Lean-kernel objects",
            skipped=True,
        )
    if not LEAN_ROOT.is_dir():
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=False,
            detail=f"lean package missing at {LEAN_ROOT}",
        )

    # The formal checker is an eval-adjacent subprocess and must obey the same
    # repository wall cap as every other run.  ``subprocess.run(timeout=...)``
    # does not reap descendants and previously allowed callers to pass a
    # timeout larger than MAX_RUN_SECONDS.  Use the canonical process-group
    # runner so Lake/Lean children are interrupted and reaped within the
    # derived interrupt + kill-grace budget.
    total, _interrupt_after, _grace = formal_process_budget(timeout_s)
    try:
        with FormalProjectLock(LEAN_ROOT, timeout_seconds=total) as lock:
            remaining = max(0.001, total - lock.wait_seconds)
            bounded = run_formal_process(
                ["lake", "build", "LeverProofLean"],
                cwd=LEAN_ROOT,
                timeout_seconds=remaining,
            )
    except TimeoutError as exc:
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=False,
            detail=str(exc),
        )
    if bounded.outcome is ProcessOutcome.LAUNCH_FAILED:
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=False,
            detail=f"lean build launch failed: {bounded.launch_error or 'unknown error'}",
        )
    if bounded.timed_out:
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=False,
            detail=f"lean build timed out after {total:.3f}s",
        )
    if int(bounded.returncode or 0) != 0:
        tail = (bounded.stderr or bounded.stdout or "")[-500:]
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=False,
            detail=f"lake build failed: {tail}",
        )
    return CheckerResult(
        checker_id=CHECKER_LEAN_KERNEL,
        backend=CHECKER_LEAN_KERNEL,
        ok=True,
        detail=(
            "lake build LeverProofLean ok "
            f"(project_lock_wait_s={lock.wait_seconds:.3f})"
        ),
    )


def run_checkers(
    obj: FormalObjectV1,
    *,
    checkers: Sequence[str] | None = None,
    replay_provider: ReplayProvider | None = None,
    enable_lean: bool = False,
    lean_timeout_s: float | None = None,
) -> list[CheckerResult]:
    """Run the requested independent checkers on one formal object."""

    if checkers is None:
        requested = list(obj.required_checkers)
        if enable_lean and CHECKER_LEAN_KERNEL not in requested:
            requested.append(CHECKER_LEAN_KERNEL)
    else:
        requested = list(checkers)

    results: list[CheckerResult] = []
    for name in requested:
        if name == CHECKER_PYTHON_STRUCTURAL:
            results.append(check_python_structural(obj))
        elif name == CHECKER_PYTHON_REFERENCE:
            results.append(check_python_reference(obj))
        elif name == CHECKER_PYTHON_REPLAY:
            results.append(
                check_python_replay(obj, replay_provider=replay_provider)
            )
        elif name == CHECKER_LEAN_KERNEL:
            results.append(
                check_lean_kernel(
                    obj, timeout_s=lean_timeout_s, enabled=enable_lean
                )
            )
        else:
            results.append(
                CheckerResult(
                    checker_id=name,
                    backend=name,
                    ok=False,
                    detail=f"unknown checker: {name}",
                )
            )
    return results


__all__ = [
    "CHECKER_LEAN_KERNEL",
    "CHECKER_PYTHON_REFERENCE",
    "CHECKER_PYTHON_REPLAY",
    "CHECKER_PYTHON_STRUCTURAL",
    "CheckerResult",
    "FormalProjectLock",
    "ReplayProvider",
    "check_lean_kernel",
    "check_python_reference",
    "check_python_replay",
    "check_python_structural",
    "formal_process_budget",
    "run_formal_process",
    "run_checkers",
]
