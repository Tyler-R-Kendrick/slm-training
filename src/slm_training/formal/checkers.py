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

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.formal.objects import FormalObjectKind, FormalObjectV1
from slm_training.formal.structural import (
    check_lean_or_closure_laws,
    check_reference_laws,
    check_support_certificate_reference,
    check_support_certificate_structure,
)
from slm_training.levers import MAX_RUN_SECONDS

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

    remaining = max(
        1.0,
        float(timeout_s)
        if timeout_s is not None
        else min(120.0, MAX_RUN_SECONDS - 1),
    )
    try:
        proc = subprocess.run(
            ["lake", "build", "LeverProofLean"],
            cwd=LEAN_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=remaining,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=False,
            detail=f"lean build error: {type(exc).__name__}: {exc}",
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
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
        detail="lake build LeverProofLean ok",
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
    "ReplayProvider",
    "check_lean_kernel",
    "check_python_reference",
    "check_python_replay",
    "check_python_structural",
    "run_checkers",
]
