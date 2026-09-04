"""Deciding whether a metric movement is a win, a loss, or a tradeoff.

One responsibility: the latency/quality tradeoff rule -- the regression budget,
the minimum primary-metric rate that buys a latency win, and the timeout band a
measurement must sit in for the comparison to mean anything.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from scripts.autotrain_measurement import (
    EPS,
)
from scripts.autotrain_metrics import (
    finite_metric,
)

MIN_MPR_FOR_LATENCY_WIN = 1.0 / 3.0 - 1e-9

LATENCY_REGRESSION_BUDGET = 0.15

LATENCY_REGRESSION_ABS_MS = 750.0

TIMEOUT_BAND_LO_MS = 11900.0

TIMEOUT_BAND_HI_MS = 12150.0


def in_timeout_band(latency_ms: float | None) -> bool:
    return (
        latency_ms is not None
        and TIMEOUT_BAND_LO_MS <= latency_ms <= TIMEOUT_BAND_HI_MS
    )


def classify_metric_tradeoff(
    *,
    control: dict[str, float | None],
    candidate: dict[str, float | None],
    primary_metric: str,
    minimum_efficiency_gain_fraction: float,
) -> tuple[bool, list[str]]:
    """Score control vs candidate with quality/latency tradeoffs.

    A pure latency improvement with empty meaning is **not** positive. A quality
    improvement that spends a bounded latency budget **is** positive even when
    the declared primary is latency. Efficiency (mpr / latency) is also a win
    path when meaning stays above the smoke floor.
    """
    reasons: list[str] = []
    positive = False
    metric_leaf = primary_metric.split(".")[-1]

    c_lat = finite_metric(control.get("latency_ms_p50"))
    t_lat = finite_metric(candidate.get("latency_ms_p50"))
    c_pr = finite_metric(control.get("parse_rate"))
    t_pr = finite_metric(candidate.get("parse_rate"))
    c_mpr = finite_metric(control.get("meaningful_program_rate"))
    t_mpr = finite_metric(candidate.get("meaningful_program_rate"))

    from slm_training.autoresearch.hillclimb import invalid_grammar_reasons

    grammar_fail = invalid_grammar_reasons(
        control, arm="control"
    ) + invalid_grammar_reasons(candidate, arm="candidate")
    reasons.extend(grammar_fail)
    parse_perfect = not grammar_fail
    parse_held = parse_perfect and (t_pr is None or c_pr is None or t_pr + EPS >= c_pr)
    mpr_held = t_mpr is None or c_mpr is None or t_mpr + EPS >= c_mpr
    mpr_improved = t_mpr is not None and c_mpr is not None and t_mpr > c_mpr + EPS
    lat_improved = t_lat is not None and c_lat is not None and t_lat + EPS < c_lat
    if t_lat is not None and c_lat is not None and c_lat > 0:
        lat_within_tradeoff = (
            t_lat <= c_lat * (1.0 + LATENCY_REGRESSION_BUDGET)
            or t_lat <= c_lat + LATENCY_REGRESSION_ABS_MS
        )
    else:
        # Missing latency must not veto a quality win.
        lat_within_tradeoff = True

    both_timeout_band = in_timeout_band(c_lat) and in_timeout_band(t_lat)

    # Path 1: latency primary win — only with held quality and non-empty meaning.
    if metric_leaf == "latency_ms_p50":
        no_metrics = (
            c_lat is None
            and t_lat is None
            and c_pr is None
            and t_pr is None
            and c_mpr is None
            and t_mpr is None
        )
        if no_metrics:
            # Incomplete stage/decode walls are not model quality failures.
            reasons.append("measurement_incomplete:no_smoke_metrics")
        elif c_lat is None or t_lat is None:
            reasons.append("primary_metric_unavailable")
        elif lat_improved and parse_held and mpr_held:
            if both_timeout_band:
                reasons.append(
                    "latency_win_rejected_timeout_band:"
                    f"control={c_lat} candidate={t_lat}"
                )
            elif t_mpr is None:
                reasons.append("latency_win_rejected_unmeasured_mpr")
            elif t_mpr + EPS < MIN_MPR_FOR_LATENCY_WIN:
                reasons.append(
                    "latency_win_rejected_low_mpr:"
                    f"mpr={t_mpr}<{MIN_MPR_FOR_LATENCY_WIN + 1e-9:g}"
                )
            else:
                positive = True
                reasons.append(f"primary_metric_win:{primary_metric}:{c_lat}->{t_lat}")
                reasons.append(f"quality_held:parse={t_pr} mpr={t_mpr}")
        else:
            reasons.append(
                f"primary_metric_null_or_worse:{primary_metric}:"
                f"control={c_lat} candidate={t_lat} "
                f"parse={c_pr}->{t_pr} mpr={c_mpr}->{t_mpr}"
            )

    # Path 2: quality win may spend a bounded latency budget (even under a
    # latency primary). Prevents "naive latency primary" from failing better
    # meaning at a small latency cost.
    if mpr_improved and parse_held and lat_within_tradeoff:
        positive = True
        reasons.append(
            "quality_metric_win:meaningful_program_rate:"
            f"{c_mpr}->{t_mpr}:lat={c_lat}->{t_lat}"
        )
    elif mpr_improved and parse_held and not lat_within_tradeoff:
        reasons.append(
            "quality_win_rejected_latency_budget:"
            f"mpr={c_mpr}->{t_mpr} lat={c_lat}->{t_lat}"
        )

    # Path 3: efficiency (meaningful programs per ms) with a meaning floor.
    # Still respects the latency tradeoff budget so a 2× slowdown cannot mint a
    # free win from mpr 0→ε alone.
    if (
        t_mpr is not None
        and c_mpr is not None
        and t_lat is not None
        and c_lat is not None
        and t_lat > 0
        and c_lat > 0
        and parse_held
        and lat_within_tradeoff
        and t_mpr + EPS >= MIN_MPR_FOR_LATENCY_WIN
        and not both_timeout_band
    ):
        c_eff = c_mpr / c_lat
        t_eff = t_mpr / t_lat
        efficiency_gain_fraction = (t_eff / c_eff - 1.0) if c_eff > 0 else None
        if not mpr_held and t_eff > c_eff + EPS:
            reasons.append(
                "efficiency_win_rejected_mpr_regression:"
                f"mpr={c_mpr}->{t_mpr}:mpr_per_ms={c_eff:.8g}->{t_eff:.8g}"
            )
        elif (
            efficiency_gain_fraction is not None
            and efficiency_gain_fraction + EPS >= minimum_efficiency_gain_fraction
        ):
            positive = True
            reasons.append(
                f"efficiency_win:mpr_per_ms:{c_eff:.8g}->{t_eff:.8g}:"
                f"gain_fraction={efficiency_gain_fraction:.8g}:"
                f"minimum={minimum_efficiency_gain_fraction:.8g}"
            )
            reasons.append(f"quality_held:parse={t_pr} mpr={t_mpr}")
        elif efficiency_gain_fraction is not None and efficiency_gain_fraction > EPS:
            reasons.append(
                f"efficiency_win_rejected_min_effect:mpr_per_ms:"
                f"{c_eff:.8g}->{t_eff:.8g}:"
                f"gain_fraction={efficiency_gain_fraction:.8g}<"
                f"{minimum_efficiency_gain_fraction:.8g}"
            )

    if grammar_fail:
        positive = False
    return positive, reasons
