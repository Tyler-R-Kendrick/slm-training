"""Judging whether a measurement is adequate and what it says.

One responsibility: given metrics already read, decide whether enough was
measured to conclude anything (sample adequacy, multi-arm completeness) and
whether two payloads are materially identical.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_io import (
    read_json,
)
from scripts.autotrain_metrics import (
    finite_metric,
    metric_leaf,
)
from slm_training.autoresearch.experiment_campaign import (
    SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
)


def multi_arm_measurement(policy: Any | None) -> dict[str, Any]:
    """Read C2 ``measurement.multi_arm``; default when METRIC swarm is unmerged."""

    block: dict[str, Any] = {}
    measurement = getattr(policy, "measurement", None) if policy is not None else None
    if isinstance(measurement, Mapping):
        raw = measurement.get("multi_arm")
        if isinstance(raw, Mapping):
            block = dict(raw)
    max_arms = max(1, int(block.get("max_arms_per_cycle") or 6))
    return {
        "max_arms_per_cycle": max_arms,
        "shared_control": bool(block.get("shared_control", True)),
        "selection_rule": str(
            block.get("selection_rule") or SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST
        ),
    }


EPS = 1e-12


def measurement_is_complete(decision: dict[str, Any]) -> bool:
    incomplete_prefixes = (
        "empty_metrics:",
        "measurement_incomplete:",
        "primary_metric_unavailable",
        "wall_timeout:",
    )
    has_metrics = all(
        any(finite_metric(value) is not None for value in metrics.values())
        for metrics in (
            decision.get("control_metrics") or {},
            decision.get("candidate_metrics") or {},
        )
    )
    return bool(
        has_metrics
        and not any(
            str(reason).startswith(incomplete_prefixes)
            for reason in decision.get("reasons") or []
        )
    )


def yaml_mapping_equal(left: str, right: str) -> bool:
    """True when both texts parse to the same YAML value (comments ignored)."""
    try:
        import yaml
    except ImportError:  # pragma: no cover — PyYAML is a repo dep
        return False
    try:
        return yaml.safe_load(left) == yaml.safe_load(right)
    except yaml.YAMLError:
        return False


def quality_metrics_identical(delivery: Mapping[str, Any]) -> bool:
    """True when SS, MPR, and binder match (mechanism-no-effect on this snapshot)."""
    control = delivery.get("control_metrics") or {}
    candidate = delivery.get("candidate_metrics") or {}
    if not isinstance(control, Mapping) or not isinstance(candidate, Mapping):
        return False
    for name in (
        "structural_similarity",
        "meaningful_program_rate",
        "binder_reference_f1",
    ):
        c_val, t_val = metric_leaf(control, name), metric_leaf(candidate, name)
        if c_val is None or t_val is None:
            return False
        if abs(float(c_val) - float(t_val)) > EPS:
            return False
    return True


def candidate_mpr_positive(delivery: Mapping[str, Any]) -> bool:
    candidate = delivery.get("candidate_metrics") or {}
    if not isinstance(candidate, Mapping):
        return False
    mpr = metric_leaf(candidate, "meaningful_program_rate")
    return mpr is not None and float(mpr) > EPS


def has_primary_metric_win(delivery: dict[str, Any], reasons: list[str]) -> bool:
    primary_metric = str(delivery.get("primary_metric") or "")
    if not primary_metric:
        return False
    return any(
        reason.startswith(f"primary_metric_win:{primary_metric}:") for reason in reasons
    )


def candidate_ship_state(camp_dir: Path, candidate_id: str) -> str:
    gates = read_json(camp_dir / "runs" / candidate_id / "gates.json")
    authoritative = gates.get("authority") == "AgentEvals assertions"
    return "ship_promoted" if authoritative and gates.get("pass") is True else "blocked"
