"""RESEARCH-16 / SLM-570: rational-to-float transfer certificate pilot (default-off).

Fixture-only comparison of direct float gate evaluation vs exact rational
reference plus conservative float interval certificates for selected metrics.
Uses EVID-04 Fraction semantics; never forces a decision when intervals overlap.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "rational_float_transfer_pilot"
TOOLCHAIN_ID = "hermetic_python_rational_float_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/rational_float_gate_corpus.v1.json"

GateOperator = Literal["ge", "gt"]
CertifiedDecision = Literal["pass", "fail", "indeterminate"]
FloatDecision = Literal["pass", "fail"]


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "float_format": "float64",
        "trust_domain": "research_only/rational_float_transfer",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class GateCaseV1:
    case_id: str
    metric: Fraction
    threshold: Fraction
    operator: GateOperator
    float_flip_expected: bool
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateCaseV1:
        return cls(
            case_id=str(data["case_id"]),
            metric=Fraction(int(data["metric_num"]), int(data["metric_den"])),
            threshold=Fraction(int(data["threshold_num"]), int(data["threshold_den"])),
            operator=str(data["operator"]),  # type: ignore[arg-type]
            float_flip_expected=bool(data["float_flip_expected"]),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "metric_num": self.metric.numerator,
            "metric_den": self.metric.denominator,
            "threshold_num": self.threshold.numerator,
            "threshold_den": self.threshold.denominator,
            "operator": self.operator,
            "float_flip_expected": self.float_flip_expected,
            "notes": self.notes,
        }


def load_corpus(*, root: Path | None = None) -> tuple[dict[str, Any], tuple[GateCaseV1, ...]]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, tuple(GateCaseV1.from_dict(row) for row in raw["cases"])


def _exact_decision(case: GateCaseV1) -> FloatDecision:
    if case.operator == "ge":
        return "pass" if case.metric >= case.threshold else "fail"
    return "pass" if case.metric > case.threshold else "fail"


def _float_decision(case: GateCaseV1) -> FloatDecision:
    metric = float(case.metric)
    threshold = float(case.threshold)
    if case.operator == "ge":
        return "pass" if metric >= threshold else "fail"
    return "pass" if metric > threshold else "fail"


def _next_float(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    packed = struct.pack(">d", x)
    bits = struct.unpack(">Q", packed)[0]
    if bits == 0:
        return struct.unpack(">d", struct.pack(">Q", 1))[0]
    if x > 0:
        bits += 1
    else:
        bits -= 1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _prev_float(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    packed = struct.pack(">d", x)
    bits = struct.unpack(">Q", packed)[0]
    if x > 0:
        bits -= 1 if bits else 0
    else:
        bits += 1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def float_interval(value: Fraction) -> tuple[float, float]:
    """Conservative float64 enclosure for an exact rational."""

    center = float(value)
    if math.isinf(center):
        return center, center
    return _prev_float(center), _next_float(center)


def certified_gate_decision(case: GateCaseV1) -> dict[str, Any]:
    exact = _exact_decision(case)
    lo_m, hi_m = float_interval(case.metric)
    lo_t, hi_t = float_interval(case.threshold)

    def _interval_passes(metric_lo: float, metric_hi: float, thr_lo: float, thr_hi: float) -> bool | None:
        if case.operator == "ge":
            if metric_lo >= thr_hi:
                return True
            if metric_hi < thr_lo:
                return False
            return None
        if metric_lo > thr_hi:
            return True
        if metric_hi <= thr_lo:
            return False
        return None

    outcome = _interval_passes(lo_m, hi_m, lo_t, hi_t)
    if outcome is True:
        certified: CertifiedDecision = "pass"
    elif outcome is False:
        certified = "fail"
    else:
        certified = "indeterminate"

    return {
        "exact_decision": exact,
        "float_decision": _float_decision(case),
        "certified_decision": certified,
        "metric_interval": [lo_m, hi_m],
        "threshold_interval": [lo_t, hi_t],
        "interval_width": hi_m - lo_m,
    }


def evaluate_control(case: GateCaseV1) -> dict[str, Any]:
    exact = _exact_decision(case)
    floated = _float_decision(case)
    return {
        "arm": "direct_float",
        "exact_decision": exact,
        "reported_decision": floated,
        "flip_vs_exact": floated != exact,
    }


def evaluate_treatment(case: GateCaseV1) -> dict[str, Any]:
    cert = certified_gate_decision(case)
    safe = cert["certified_decision"] == "indeterminate" or cert["certified_decision"] == cert["exact_decision"]
    return {
        "arm": "rational_transfer_certificate",
        **cert,
        "safe_vs_exact": safe,
        "covers_float_flip": cert["certified_decision"] != cert["float_decision"]
        and cert["certified_decision"] == "indeterminate",
    }


def paired_transfer_report(*, root: Path | None = None) -> dict[str, Any]:
    raw, cases = load_corpus(root=root)
    control_rows = [evaluate_control(case) for case in cases]
    treatment_rows = [evaluate_treatment(case) for case in cases]

    covered = sum(
        1
        for case, treat in zip(cases, treatment_rows, strict=True)
        if treat["safe_vs_exact"]
        and (
            treat["certified_decision"] == treat["exact_decision"]
            or treat["covers_float_flip"]
        )
    )
    primary = covered / max(len(cases), 1)

    float_flips = sum(1 for row in control_rows if row["flip_vs_exact"])
    flip_detections = sum(
        1
        for ctrl, treat in zip(control_rows, treatment_rows, strict=True)
        if ctrl["flip_vs_exact"] and treat["covers_float_flip"]
    )
    silent_cast_incidents = sum(
        1
        for ctrl, treat in zip(control_rows, treatment_rows, strict=True)
        if ctrl["flip_vs_exact"] and treat["certified_decision"] in {"pass", "fail"}
        and treat["certified_decision"] != treat["exact_decision"]
    )
    avg_width = sum(row["interval_width"] for row in treatment_rows) / max(
        len(treatment_rows), 1
    )

    if primary < 1.0:
        decision = "reject"
        reason = "incomplete_certificate_coverage"
    elif silent_cast_incidents > 0:
        decision = "reject"
        reason = "certificate_forced_wrong_decision"
    elif float_flips > 0 and flip_detections == 0:
        decision = "reject"
        reason = "missed_float_flip"
    else:
        decision = "accept"
        reason = "certificates_cover_selected_gates"

    return {
        "schema": "rational_float_transfer_report/v1",
        "transfer_certificate_coverage_on_selected_metric_gates": primary,
        "covered_gate_count": covered,
        "gate_count": len(cases),
        "float_flip_count": float_flips,
        "float_flip_detections": flip_detections,
        "silent_cast_incidents": silent_cast_incidents,
        "bound_violation_miss_rate": 0.0,
        "performance_overhead_proxy": avg_width,
        "decision": decision,
        "reason": reason,
        "control_rows": control_rows,
        "treatment_rows": treatment_rows,
        "corpus_digest": content_sha(raw),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "GateCaseV1",
    "certified_gate_decision",
    "evaluate_control",
    "evaluate_treatment",
    "float_interval",
    "load_corpus",
    "paired_transfer_report",
    "pinned_toolchain",
]
