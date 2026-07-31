"""Close the formal loop with multi-backend agreement.

A formal object is **accepted** only when enough **independent** checker
backends agree. Relying on a single Lean kernel (or any single backend) is
explicitly rejected: the loop is not closed unless ``min_backends`` distinct
non-skipped backends report ``ok``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.formal.checkers import (
    CHECKER_LEAN_KERNEL,
    CheckerResult,
    ReplayProvider,
    run_checkers,
)
from slm_training.formal.objects import FormalObjectV1
from slm_training.versioning import build_version_stamp

LOOP_SCHEMA = "formal_loop_report/v1"
DEFAULT_MIN_BACKENDS = 2


@dataclass(frozen=True)
class ObjectLoopResult:
    object_id: str
    kind: str
    accepted: bool
    backends_ok: tuple[str, ...]
    backends_failed: tuple[str, ...]
    backends_skipped: tuple[str, ...]
    checker_results: tuple[CheckerResult, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "accepted": self.accepted,
            "backends_ok": list(self.backends_ok),
            "backends_failed": list(self.backends_failed),
            "backends_skipped": list(self.backends_skipped),
            "checker_results": [r.to_dict() for r in self.checker_results],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FormalLoopReport:
    schema: str
    closed: bool
    min_backends: int
    objects: tuple[ObjectLoopResult, ...]
    single_kernel_reliance: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "closed": self.closed,
            "min_backends": self.min_backends,
            "objects": [o.to_dict() for o in self.objects],
            "single_kernel_reliance": self.single_kernel_reliance,
            "summary": self.summary,
            "version_stamp": build_version_stamp(
                "formal.objects",
                "formal.loop",
            ),
        }


def loop_requires_multi_backend(
    results: Sequence[CheckerResult],
    *,
    min_backends: int = DEFAULT_MIN_BACKENDS,
) -> tuple[bool, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Decide acceptance from checker results.

    Returns ``(accepted, reason, ok_backends, failed_backends, skipped_backends)``.
    """

    ok_backends: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for result in results:
        if result.skipped:
            skipped.append(result.backend)
            continue
        if result.ok:
            if result.backend not in ok_backends:
                ok_backends.append(result.backend)
        else:
            if result.backend not in failed:
                failed.append(result.backend)

    if failed:
        return (
            False,
            f"checker failure(s): {', '.join(failed)}",
            tuple(ok_backends),
            tuple(failed),
            tuple(skipped),
        )
    if len(ok_backends) < min_backends:
        # Special-case: only Lean kernel would pass → single-kernel reliance.
        if ok_backends == [CHECKER_LEAN_KERNEL] or (
            not ok_backends
            and CHECKER_LEAN_KERNEL in skipped
            and len(results) == 1
        ):
            return (
                False,
                "single-kernel reliance: need independent non-Lean backends",
                tuple(ok_backends),
                tuple(failed),
                tuple(skipped),
            )
        return (
            False,
            (
                f"need {min_backends} independent backends, got {len(ok_backends)} "
                f"({', '.join(ok_backends) or 'none'})"
            ),
            tuple(ok_backends),
            tuple(failed),
            tuple(skipped),
        )
    # Reject if the only ok backends are a single family (belt and suspenders).
    if len(ok_backends) == 1 and ok_backends[0] == CHECKER_LEAN_KERNEL:
        return (
            False,
            "single-kernel reliance on lean_kernel",
            tuple(ok_backends),
            tuple(failed),
            tuple(skipped),
        )
    return (
        True,
        f"multi-backend agreement: {', '.join(ok_backends)}",
        tuple(ok_backends),
        tuple(failed),
        tuple(skipped),
    )


def evaluate_object(
    obj: FormalObjectV1,
    *,
    min_backends: int = DEFAULT_MIN_BACKENDS,
    enable_lean: bool = False,
    replay_provider: ReplayProvider | None = None,
    lean_timeout_s: float | None = None,
    extra_checkers: Sequence[str] | None = None,
) -> ObjectLoopResult:
    checkers = list(obj.required_checkers)
    if extra_checkers:
        for name in extra_checkers:
            if name not in checkers:
                checkers.append(name)
    if enable_lean and CHECKER_LEAN_KERNEL not in checkers:
        checkers.append(CHECKER_LEAN_KERNEL)

    # Support certificates declare structural + replay. When replay is skipped
    # (no context), we still need two backends: treat structural as mandatory
    # and require that at least one other non-skipped backend exists OR that
    # replay ran. If replay is skipped, structural alone is insufficient for
    # min_backends=2 — callers should either supply replay context or lower
    # min_backends for offline digest-only audits with honesty.
    results = run_checkers(
        obj,
        checkers=checkers,
        replay_provider=replay_provider,
        enable_lean=enable_lean,
        lean_timeout_s=lean_timeout_s,
    )
    accepted, reason, ok_b, fail_b, skip_b = loop_requires_multi_backend(
        results, min_backends=min_backends
    )
    return ObjectLoopResult(
        object_id=obj.object_id,
        kind=obj.kind.value,
        accepted=accepted,
        backends_ok=ok_b,
        backends_failed=fail_b,
        backends_skipped=skip_b,
        checker_results=tuple(results),
        reason=reason,
    )


def close_formal_loop(
    objects: Iterable[FormalObjectV1],
    *,
    min_backends: int = DEFAULT_MIN_BACKENDS,
    enable_lean: bool = False,
    replay_provider: ReplayProvider | None = None,
    lean_timeout_s: float | None = None,
) -> FormalLoopReport:
    """Run multi-prover checks on all objects; loop is closed iff all accepted."""

    results = [
        evaluate_object(
            obj,
            min_backends=min_backends,
            enable_lean=enable_lean,
            replay_provider=replay_provider,
            lean_timeout_s=lean_timeout_s,
        )
        for obj in objects
    ]
    closed = all(item.accepted for item in results) and bool(results)
    single_kernel = any(
        item.backends_ok == (CHECKER_LEAN_KERNEL,) for item in results
    ) or any(
        "single-kernel reliance" in item.reason for item in results
    )
    accepted_n = sum(1 for item in results if item.accepted)
    summary = (
        f"formal loop {'CLOSED' if closed else 'OPEN'}: "
        f"{accepted_n}/{len(results)} objects accepted "
        f"(min_backends={min_backends}, lean={'on' if enable_lean else 'off'})"
    )
    return FormalLoopReport(
        schema=LOOP_SCHEMA,
        closed=closed,
        min_backends=min_backends,
        objects=tuple(results),
        single_kernel_reliance=single_kernel,
        summary=summary,
    )


def write_loop_report(path: Path, report: FormalLoopReport | Mapping[str, Any]) -> None:
    payload = report.to_dict() if isinstance(report, FormalLoopReport) else dict(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_MIN_BACKENDS",
    "LOOP_SCHEMA",
    "FormalLoopReport",
    "ObjectLoopResult",
    "close_formal_loop",
    "evaluate_object",
    "loop_requires_multi_backend",
    "write_loop_report",
]
