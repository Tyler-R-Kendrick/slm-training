"""Close the formal loop with multi-backend agreement.

A formal object is **accepted** under the production conformance policy when
enough checkers that advertise ``structural_consistency`` succeed. Distinct
backend *names* are not automatically independent trust roots (EVID-08):
``python_structural`` and ``python_reference`` share one trust domain and are
reported as redundant conformance, not independent verification.

Semantic authority requires the distinct-trust-domain policy
(``SEMANTIC_AUTHORITY_POLICY``). Relying on a single Lean kernel is rejected.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.formal.checker_capability import (
    CONFORMANCE_POLICY,
    SEMANTIC_AUTHORITY_POLICY,
    AcceptancePolicyV1,
    PolicyReportV1,
    evaluate_acceptance_policy,
    runs_from_checker_results,
    semantic_authority_requires_distinct_trust_domains,
)
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
    verification_kind: str = "redundant_conformance"
    trust_domains: tuple[str, ...] = ()
    independent_backends: tuple[str, ...] = ()
    redundant_backends: tuple[str, ...] = ()
    policy_report: dict[str, Any] | None = None
    semantic_authority_accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = {
            "object_id": self.object_id,
            "kind": self.kind,
            "accepted": self.accepted,
            "backends_ok": list(self.backends_ok),
            "backends_failed": list(self.backends_failed),
            "backends_skipped": list(self.backends_skipped),
            "checker_results": [r.to_dict() for r in self.checker_results],
            "reason": self.reason,
            "verification_kind": self.verification_kind,
            "trust_domains": list(self.trust_domains),
            "independent_backends": list(self.independent_backends),
            "redundant_backends": list(self.redundant_backends),
            "semantic_authority_accepted": self.semantic_authority_accepted,
        }
        if self.policy_report is not None:
            out["policy_report"] = dict(self.policy_report)
        return out


@dataclass(frozen=True)
class FormalLoopReport:
    schema: str
    closed: bool
    min_backends: int
    objects: tuple[ObjectLoopResult, ...]
    single_kernel_reliance: bool
    summary: str
    policy_id: str = CONFORMANCE_POLICY.policy_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "closed": self.closed,
            "min_backends": self.min_backends,
            "policy_id": self.policy_id,
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
    policy: AcceptancePolicyV1 | None = None,
) -> tuple[bool, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], PolicyReportV1]:
    """Decide acceptance from checker results + capability/trust-domain policy.

    Returns
    ``(accepted, reason, ok_backends, failed_backends, skipped_backends, policy_report)``.
    """

    active = policy or CONFORMANCE_POLICY
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

    report = evaluate_acceptance_policy(
        runs_from_checker_results(results),
        active,
    )

    if failed:
        return (
            False,
            f"checker failure(s): {', '.join(failed)}",
            tuple(ok_backends),
            tuple(failed),
            tuple(skipped),
            report,
        )
    if len(ok_backends) < min_backends:
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
                report,
            )
        return (
            False,
            (
                f"need {min_backends} backends, got {len(ok_backends)} "
                f"({', '.join(ok_backends) or 'none'})"
            ),
            tuple(ok_backends),
            tuple(failed),
            tuple(skipped),
            report,
        )
    if len(ok_backends) == 1 and ok_backends[0] == CHECKER_LEAN_KERNEL:
        return (
            False,
            "single-kernel reliance on lean_kernel",
            tuple(ok_backends),
            tuple(failed),
            tuple(skipped),
            report,
        )
    if not report.accepted:
        return (
            False,
            report.reason,
            tuple(ok_backends),
            tuple(failed),
            tuple(skipped),
            report,
        )
    # Preserve honest labeling: redundant same-domain agreement is conformance,
    # not independent multi-prover semantic authority.
    reason = report.reason
    if report.verification_kind == "redundant_conformance":
        reason = (
            f"redundant conformance agreement: {', '.join(ok_backends)} "
            f"(trust domains: {', '.join(report.trust_domains)})"
        )
    elif report.verification_kind == "independent_verification":
        reason = (
            f"independent verification: {', '.join(ok_backends)} "
            f"(trust domains: {', '.join(report.trust_domains)})"
        )
    return (
        True,
        reason,
        tuple(ok_backends),
        tuple(failed),
        tuple(skipped),
        report,
    )


def evaluate_object(
    obj: FormalObjectV1,
    *,
    min_backends: int = DEFAULT_MIN_BACKENDS,
    enable_lean: bool = False,
    replay_provider: ReplayProvider | None = None,
    lean_timeout_s: float | None = None,
    extra_checkers: Sequence[str] | None = None,
    policy: AcceptancePolicyV1 | None = None,
) -> ObjectLoopResult:
    checkers = list(obj.required_checkers)
    if extra_checkers:
        for name in extra_checkers:
            if name not in checkers:
                checkers.append(name)
    if enable_lean and CHECKER_LEAN_KERNEL not in checkers:
        checkers.append(CHECKER_LEAN_KERNEL)

    results = run_checkers(
        obj,
        checkers=checkers,
        replay_provider=replay_provider,
        enable_lean=enable_lean,
        lean_timeout_s=lean_timeout_s,
    )
    active = policy or CONFORMANCE_POLICY
    accepted, reason, ok_b, fail_b, skip_b, report = loop_requires_multi_backend(
        results, min_backends=min_backends, policy=active
    )
    semantic = evaluate_acceptance_policy(
        runs_from_checker_results(results),
        SEMANTIC_AUTHORITY_POLICY,
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
        verification_kind=report.verification_kind,
        trust_domains=report.trust_domains,
        independent_backends=report.independent_backends,
        redundant_backends=report.redundant_backends,
        policy_report=report.to_dict(),
        semantic_authority_accepted=semantic.accepted,
    )


def close_formal_loop(
    objects: Iterable[FormalObjectV1],
    *,
    min_backends: int = DEFAULT_MIN_BACKENDS,
    enable_lean: bool = False,
    replay_provider: ReplayProvider | None = None,
    lean_timeout_s: float | None = None,
    policy: AcceptancePolicyV1 | None = None,
) -> FormalLoopReport:
    """Run multi-prover checks on all objects; loop is closed iff all accepted."""

    active = policy or CONFORMANCE_POLICY
    results = [
        evaluate_object(
            obj,
            min_backends=min_backends,
            enable_lean=enable_lean,
            replay_provider=replay_provider,
            lean_timeout_s=lean_timeout_s,
            policy=active,
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
    independent_n = sum(
        1 for item in results if item.verification_kind == "independent_verification"
    )
    redundant_n = sum(
        1 for item in results if item.verification_kind == "redundant_conformance"
    )
    summary = (
        f"formal loop {'CLOSED' if closed else 'OPEN'}: "
        f"{accepted_n}/{len(results)} objects accepted "
        f"(min_backends={min_backends}, lean={'on' if enable_lean else 'off'}, "
        f"policy={active.policy_id}, independent={independent_n}, "
        f"redundant_conformance={redundant_n})"
    )
    return FormalLoopReport(
        schema=LOOP_SCHEMA,
        closed=closed,
        min_backends=min_backends,
        objects=tuple(results),
        single_kernel_reliance=single_kernel,
        summary=summary,
        policy_id=active.policy_id,
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
    "semantic_authority_requires_distinct_trust_domains",
    "write_loop_report",
]
