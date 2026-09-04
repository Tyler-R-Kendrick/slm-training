"""What state a single candidate is in.

One responsibility: reading one candidate's standing -- blocked, quality
re-held rather than reproduced, a confirmed win -- and the priority its role
gives it in the queue.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.autotrain_diagnosis import quality_held_reasons
from scripts.autotrain_measurement import (
    EPS,
    candidate_mpr_positive,
    has_primary_metric_win,
)
from scripts.autotrain_metrics import metric_leaf
from slm_training.autoresearch.schemas import (
    NextRunPriorityV1,
)


def delivery_parse_mpr_held(delivery: dict[str, Any]) -> bool:
    """Parse/MPR non-regression from delivery metrics (fixture SS wins omit quality_held)."""
    control = delivery.get("control_metrics") or {}
    candidate = delivery.get("candidate_metrics") or {}
    if not isinstance(control, Mapping) or not isinstance(candidate, Mapping):
        return False

    c_pr, t_pr = (
        metric_leaf(control, "parse_rate"),
        metric_leaf(candidate, "parse_rate"),
    )
    c_mpr, t_mpr = (
        metric_leaf(control, "meaningful_program_rate"),
        metric_leaf(candidate, "meaningful_program_rate"),
    )
    # Missing metrics must not invent a quality hold.
    if None in (c_pr, t_pr, c_mpr, t_mpr):
        return False
    return bool(t_pr + EPS >= c_pr and t_mpr + EPS >= c_mpr)


def confirm_candidate_blocked(reasons: list[str]) -> bool:
    blocked_prefixes = (
        "primary_metric_null_or_worse:",
        "non_regression_fail:",
        "eg_params_block:",
        "measurement_incomplete:",
        "wall_timeout:",
        "empty_metrics:",
        "harness_failure:",
        "primary_quality_win_rejected",
        "invalid_grammar:",
    )
    return any(reason.startswith(blocked_prefixes) for reason in reasons)


def is_confirm_candidate_win(delivery: dict[str, Any]) -> bool:
    """Screening primary quality win worth confirming (fixture-n may keep positive=False).

    Smoke below the Lean floor cannot mint climb ``positive`` / stack layers.
    A held-quality primary win at certified n may enter the champion queue.
    n=3 fixture SS spikes must not enqueue.
    """
    if delivery.get("measurement_complete") is False:
        return False
    reasons = [str(reason) for reason in delivery.get("reasons") or []]
    if any(r.startswith("fixture_insufficient_n_alone") for r in reasons):
        return False
    if any(r.startswith("mechanism_no_effect:") for r in reasons):
        return False
    if confirm_candidate_blocked(reasons):
        return False
    if any(r.startswith("invalid_grammar:") for r in reasons):
        return False
    from slm_training.autoresearch.hillclimb import invalid_grammar_reasons

    if invalid_grammar_reasons(
        delivery.get("candidate_metrics")
        if isinstance(delivery.get("candidate_metrics"), dict)
        else {},
        arm="candidate",
    ) or invalid_grammar_reasons(
        delivery.get("control_metrics")
        if isinstance(delivery.get("control_metrics"), dict)
        else {},
        arm="control",
    ):
        return False
    if not has_primary_metric_win(delivery, reasons):
        return False
    primary_leaf = str(delivery.get("primary_metric") or "").rsplit(".", 1)[-1]
    if primary_leaf == "latency_ms_p50":
        return quality_held_reasons(reasons)
    if not candidate_mpr_positive(delivery):
        return False
    return quality_held_reasons(reasons) or delivery_parse_mpr_held(delivery)


def confirmation_quality_reheld(delivery: dict[str, Any]) -> bool:
    """Require the policy-owned primary quality win on a confirmation run.

    Screening may surface an efficiency tradeoff as a hypothesis, but a fresh
    confirmation is the gate into promotion.  Faster decode cannot confirm a
    champion when the declared primary or a required non-regression metric got
    worse.
    """

    if not delivery.get("positive") or delivery.get("measurement_complete") is False:
        return False
    reasons = [str(reason) for reason in delivery.get("reasons") or []]
    if confirm_candidate_blocked(reasons):
        return False
    primary_metric = str(delivery.get("primary_metric") or "")
    if not primary_metric:
        return False
    return any(
        reason.startswith(f"primary_metric_win:{primary_metric}:") for reason in reasons
    )


def queued_candidate_priorities(
    candidate_id: str, evidence_id: str
) -> tuple[NextRunPriorityV1, ...]:
    """Project the real successor after a screening candidate enters the queue."""

    return (
        NextRunPriorityV1(
            rank=1,
            area="evaluation",
            hypothesis=(
                "Confirm the fixture candidate on a fresh seed with the exact "
                "size-matched treatment and control recipes before promotion."
            ),
            evidence_ids=(evidence_id,),
            confidence=0.95,
            expected_information_gain=(
                "Tests whether the held-out quality gain reproduces while exposing "
                "the observed binder and latency tradeoffs."
            ),
            authority="observed_result",
            disposition="experiment_next",
            proposed_experiment_id=f"{candidate_id}-fresh-confirmation",
        ),
        NextRunPriorityV1(
            rank=2,
            area="lean_model",
            hypothesis=(
                "Keep promotion formal preflight locked until fresh confirmation "
                "establishes a champion."
            ),
            evidence_ids=(evidence_id,),
            confidence=1.0,
            expected_information_gain=(
                "Prevents screening evidence from bypassing theorem-backed "
                "promotion obligations."
            ),
            authority="lean_assumption",
            disposition="monitor",
        ),
    )


def role_with_confirmation_boundary(cadence_role: str, *, confirming: bool) -> str:
    """Keep unconfirmed champions on screening endpoints and suites."""

    return "screening" if confirming else cadence_role
