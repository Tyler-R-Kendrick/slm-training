"""Preflight plugin: block candidates whose hypothesis family is concluded.

A family closed in ``closed_approaches.v1.json`` (see
``slm_training.autoresearch.conclusions``) is a durable conclusion — an
approach that adequately powered evidence has already rejected. Spending
another run on it is not research; this check blocks such candidates unless a
recorded reopen condition holds (a new lever key outside the closing
evidence, or a changed harness component version). A closed approach never
closes a goal (AGENTS.md I14), so the block message always names the reopen
conditions that re-legalize the family.

Contract: module-level ``CHECK`` with ``check_id`` and
``run(candidate: dict) -> PreflightVerdict``; never raises.
"""

from __future__ import annotations

from typing import Any

try:  # The preflight package owner defines the canonical verdict model.
    from slm_training.autoresearch.preflight import PreflightVerdict
except ImportError:  # pragma: no cover - exercised until __init__ lands
    from typing import Literal

    from pydantic import BaseModel

    class PreflightVerdict(BaseModel):  # type: ignore[no-redef]
        """Structural fallback matching the preflight package contract."""

        check_id: str
        verdict: Literal["pass", "warn", "block"]
        reasons: list[str]
        data: dict = {}


CHECK_ID = "concluded_family"


def _run(candidate: dict[str, Any]) -> PreflightVerdict:
    from slm_training.autoresearch import conclusions

    lever_keys = [
        str(k) for k in (candidate.get("lever_keys") or []) if str(k).strip()
    ]
    if not lever_keys:
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="pass",
            reasons=["no lever_keys on candidate; no family to conclude about"],
            data={},
        )
    family_key = conclusions.family_key_for_levers(lever_keys)
    closure = conclusions.binding_closure(
        family_key,
        current_lever_keys=lever_keys,
        harness_versions=conclusions.current_harness_versions(),
    )
    if closure is None:
        return PreflightVerdict(
            check_id=CHECK_ID,
            verdict="pass",
            reasons=[f"family {family_key!r} has no binding closure record"],
            data={"family_key": family_key},
        )
    reopen = ", ".join(closure.reopen_conditions) or "none recorded"
    return PreflightVerdict(
        check_id=CHECK_ID,
        verdict="block",
        reasons=[
            (
                f"hypothesis family {family_key!r} is concluded by "
                f"ClosedApproachRecord {closure.id} "
                f"(closed {closure.closed_at_run_date} on evidence "
                f"{closure.evidence_record_ids}); the approach is closed, "
                f"the goal ({closure.goal_invariant_served}) stays open"
            ),
            f"reopen conditions (unmet): {reopen}",
        ],
        data={
            "family_key": family_key,
            "closed_approach_id": closure.id,
            "reopen_conditions": list(closure.reopen_conditions),
            "goal_invariant_served": closure.goal_invariant_served,
        },
    )


class _ConcludedFamilyCheck:
    """Preflight check object (module-level ``CHECK`` instance)."""

    check_id = CHECK_ID

    def run(self, candidate: dict[str, Any]) -> PreflightVerdict:
        try:
            return _run(dict(candidate or {}))
        except Exception as exc:  # noqa: BLE001 — plugins never raise
            return PreflightVerdict(
                check_id=CHECK_ID,
                verdict="pass",
                reasons=[
                    "concluded_family check errored and fails open: "
                    f"{type(exc).__name__}: {exc}"
                ],
                data={"error": f"{type(exc).__name__}: {exc}"},
            )


CHECK = _ConcludedFamilyCheck()
