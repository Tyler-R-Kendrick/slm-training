"""Numeric ingestion and display projection for the existing climb policy.

No thresholds, promotion decisions or statistical tests are owned here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def leaf(metric: str) -> str:
    return metric.split(".")[-1]


def metric_from_map(metrics: Mapping[str, Any], metric_id: str) -> float | None:
    name = leaf(metric_id)
    keys = [metric_id, name, *(key for key in metrics if str(key).endswith(f".{name}"))]
    raw = next((metrics[key] for key in keys if key in metrics), None)
    if type(raw) not in (int, float) or not math.isfinite(raw):
        return None
    rates = {
        "parse_rate",
        "meaningful_program_rate",
        "structural_similarity",
        "component_type_recall",
        "ast_beq_rate",
        "canonical_beq_rate",
        "placeholder_fidelity",
        "slot_f1",
        "binder_f1",
        "reference_f1",
    }
    if name in rates and not 0 <= raw <= 1:
        return None
    return float(raw)


def scientific_status(reasons, positive, paired):
    if any(reason.startswith("measurement_incomplete:") for reason in reasons):
        return "incomplete"
    if positive:
        return "supported_diagnostic_benefit"
    if paired is not None and paired["verdict"] == "inconclusive":
        return "inconclusive"
    return "no_supported_benefit"


def paired_primary_test(
    policy: Any,
    paired_records: Mapping[str, Any] | None,
    *,
    direction: str,
    minimum_effect: float,
) -> dict[str, Any] | None:
    """Policy paired test on per-record maps; ``None`` when either arm lacks them."""

    from .climb_policy import SCREENING_NLL_PAIRED_DECIDABILITY_FLOOR, ClimbPolicyError

    if not isinstance(paired_records, Mapping):
        return None
    control = paired_records.get("control")
    candidate = paired_records.get("candidate")
    if not isinstance(control, Mapping) or not isinstance(candidate, Mapping):
        return None
    from slm_training.autoresearch.paired_stats import (
        DEFAULT_ALPHA,
        DEFAULT_MIN_NONTIED_PAIRS,
        PairedSelection,
        paired_record_screening,
    )

    cfg = policy.measurement.get("paired_test")
    if not isinstance(cfg, Mapping):
        cfg = {}
    kind = str(cfg.get("kind") or "wilcoxon_signed_rank")
    if kind not in {"wilcoxon_signed_rank", "sign_test"}:
        raise ClimbPolicyError(f"measurement.paired_test.kind unsupported: {kind!r}")
    result = paired_record_screening(
        control,
        candidate,
        direction="decrease" if direction == "decrease" else "increase",
        alpha=cfg.get("alpha") or DEFAULT_ALPHA,
        min_nontied_pairs=int(
            cfg.get("min_nontied_pairs") or DEFAULT_MIN_NONTIED_PAIRS
        ),
        kind=kind,  # type: ignore[arg-type]
        minimum_effect=float(minimum_effect),
        selection=PairedSelection(
            paired_records["selected_record_ids"], paired_records.get("root_ids")
        )
        if paired_records.get("selected_record_ids") is not None
        else None,
    )
    result["claim_class"] = "diagnostic"
    result["decidability_floor"] = SCREENING_NLL_PAIRED_DECIDABILITY_FLOOR
    return result
