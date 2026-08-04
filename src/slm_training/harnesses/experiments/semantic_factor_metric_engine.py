"""Execute declarative SFF metric aggregates/derived formulas from metrics.v1.json.

Supported formula ``type`` values are the stable harness surface. Adding a new
metric of an existing type only requires editing the external metrics file.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from slm_training.harnesses.experiments.semantic_factor_config import SFFMetricsConfig

__all__ = [
    "OutcomeRow",
    "compute_arm_metrics",
    "apply_control_relative_metrics",
]


# Minimal row protocol used by the engine (dicts or objects with attributes).
OutcomeRow = Mapping[str, Any]


def _get(row: OutcomeRow, field: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _population(
    rows: Sequence[OutcomeRow],
    population: str,
    *,
    populations: Mapping[str, Any] | None = None,
) -> list[OutcomeRow]:
    specs = populations or {}
    if population in specs:
        spec = dict(specs[population] or {})
        fams = spec.get("include_families")
        require_correct = bool(spec.get("require_correct_not_null"))
        exclude_flags = tuple(str(x) for x in spec.get("exclude_if_true") or ())
        out: list[OutcomeRow] = []
        for r in rows:
            if fams is not None and _get(r, "family") not in set(fams):
                continue
            if require_correct and _get(r, "correct") is None:
                continue
            if any(bool(_get(r, flag)) for flag in exclude_flags):
                continue
            out.append(r)
        return out
    # Legacy fallbacks when metrics resource omits populations.
    if population == "all":
        return list(rows)
    if population == "ranked":
        return [
            r
            for r in rows
            if _get(r, "family") in {"ranked", "adversarial"}
            and _get(r, "correct") is not None
            and not _get(r, "incomplete_skip")
            and not _get(r, "singleton_skip")
        ]
    raise ValueError(f"unknown population {population!r}")


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return [lo, hi]


def _run_aggregate(
    spec: Mapping[str, Any],
    rows: Sequence[OutcomeRow],
    *,
    wilson_z: float,
    populations: Mapping[str, Any] | None = None,
) -> Any:
    kind = str(spec["type"])
    pop = _population(
        rows,
        str(spec.get("population") or "all"),
        populations=populations,
    )
    field = str(spec.get("field") or "")

    if kind == "count":
        return len(pop)
    if kind == "sum":
        return float(sum(float(_get(r, field) or 0.0) for r in pop))
    if kind == "sum_bool":
        return int(sum(1 for r in pop if bool(_get(r, field))))
    if kind == "mean":
        if not pop:
            return None
        return float(sum(float(_get(r, field) or 0.0) for r in pop) / len(pop))
    if kind == "mean_bool":
        if not pop:
            return None
        return float(sum(1 for r in pop if bool(_get(r, field))) / len(pop))
    if kind == "min":
        vals = [float(_get(r, field) or 0.0) for r in pop]
        return min(vals) if vals else 0.0
    if kind == "max":
        vals = [float(_get(r, field) or 0.0) for r in pop]
        return max(vals) if vals else 0.0
    if kind == "percentile":
        vals = [float(_get(r, field) or 0.0) for r in pop]
        return _percentile(vals, float(spec.get("percentile") or 50))
    if kind == "wilson_ci":
        if not pop:
            return [0.0, 0.0]
        success_field = str(spec.get("success_field") or field or "correct")
        successes = sum(1 for r in pop if bool(_get(r, success_field)))
        return _wilson_interval(successes, len(pop), z=wilson_z)
    raise ValueError(f"unsupported aggregate type {kind!r}")


def compute_arm_metrics(
    metrics_cfg: SFFMetricsConfig,
    *,
    arm_id: str,
    role: str,
    rows: Sequence[OutcomeRow],
    kills: Sequence[str],
    trainable_parameters: int = 0,
    claim_class: str = "fixture",
) -> dict[str, Any]:
    """Compute one arm row from external aggregate specs (no control-relative yet)."""

    out: dict[str, Any] = {
        "arm_id": arm_id,
        "role": role,
        "kill_criteria_hit": list(kills),
        "trainable_parameters": int(trainable_parameters),
        "claim_class": claim_class,
    }
    pops = metrics_cfg.populations
    for spec in metrics_cfg.aggregates:
        out[str(spec["id"])] = _run_aggregate(
            spec, rows, wilson_z=metrics_cfg.wilson_z, populations=pops
        )
    # Derived metrics that do not need control (ratio of aggregates, wilson from rows).
    for spec in metrics_cfg.derived:
        kind = str(spec["type"])
        mid = str(spec["id"])
        if kind == "ratio":
            num = out.get(str(spec["numerator"]))
            den = out.get(str(spec["denominator"]))
            floor = spec.get("denominator_floor")
            if num is None and "if_null_numerator" in spec:
                out[mid] = spec.get("if_null_numerator")
                continue
            den_f = float(den or 0.0)
            if floor is not None:
                den_f = max(den_f, float(floor))
            if den_f == 0:
                out[mid] = spec.get("if_zero_denominator")
            else:
                out[mid] = float(num or 0.0) / den_f
        elif kind == "wilson_ci":
            out[mid] = _run_aggregate(
                spec, rows, wilson_z=metrics_cfg.wilson_z, populations=pops
            )
        # control-relative types applied later
        elif kind in {
            "minus_control",
            "ratio_to_control",
            "delta_over_extra_ms",
            "dominates_control",
        }:
            continue
        else:
            raise ValueError(f"unsupported derived type {kind!r}")
    return out


def apply_control_relative_metrics(
    metrics_cfg: SFFMetricsConfig,
    arm_rows: list[dict[str, Any]],
    *,
    control_arm_id: str,
) -> None:
    """Fill control-relative derived metrics in place on arm rows."""

    control = next(r for r in arm_rows if r["arm_id"] == control_arm_id)
    numerics = metrics_cfg.numerics
    zero_eps = float(numerics.get("ratio_zero_eps", 1e-12))
    acc_eps = float(numerics.get("dominance_acc_eps", 1e-12))
    ms_eps = float(numerics.get("dominance_ms_eps", 1e-9))
    for row in arm_rows:
        for spec in metrics_cfg.derived:
            kind = str(spec["type"])
            mid = str(spec["id"])
            ctrl_id = str(spec.get("control_arm_id") or control_arm_id)
            if ctrl_id != control_arm_id:
                ctrl = next(r for r in arm_rows if r["arm_id"] == ctrl_id)
            else:
                ctrl = control
            if kind == "minus_control":
                metric = str(spec["metric"])
                a = row.get(metric)
                b = ctrl.get(metric)
                row[mid] = None if a is None or b is None else float(a) - float(b)
            elif kind == "ratio_to_control":
                metric = str(spec["metric"])
                a = float(row.get(metric) or 0.0)
                b = float(ctrl.get(metric) or 0.0)
                row[mid] = (a / b) if b > 0 else None
            elif kind == "delta_over_extra_ms":
                q = str(spec["quality_metric"])
                rt = str(spec["runtime_metric"])
                a_q, b_q = row.get(q), ctrl.get(q)
                a_ms, b_ms = float(row.get(rt) or 0.0), float(ctrl.get(rt) or 0.0)
                if a_q is None or b_q is None:
                    row[mid] = None
                    continue
                delta = float(a_q) - float(b_q)
                extra = a_ms - b_ms
                if abs(extra) < zero_eps:
                    row[mid] = None if delta == 0 else float("inf")
                else:
                    row[mid] = delta / extra
            elif kind == "dominates_control":
                q = str(spec["quality_metric"])
                rt = str(spec["runtime_metric"])
                a_q, b_q = row.get(q), ctrl.get(q)
                a_ms, b_ms = float(row.get(rt) or 0.0), float(ctrl.get(rt) or 0.0)
                if a_q is None or b_q is None or b_ms <= 0:
                    row[mid] = False
                    continue
                delta = float(a_q) - float(b_q)
                ratio = a_ms / b_ms
                faster_or_equal = ratio <= 1.0 + ms_eps
                better = delta > acc_eps
                same = abs(delta) <= acc_eps
                faster = ratio < 1.0 - ms_eps
                row[mid] = bool((better and faster_or_equal) or (same and faster))
