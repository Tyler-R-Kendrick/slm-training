"""RESEARCH-19 / SLM-558: Lean-checked Rademacher bound pilot (default-off).

Hermetic validation of a frozen Lean-exported toy generalization certificate for a
deliberately tiny hypothesis class. Never grants ship-gate or promotion authority.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "lean_rademacher_bound_pilot"
TOOLCHAIN_ID = "hermetic_python_lean_rademacher_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/lean_rademacher_corpus.v1.json"
VACUITY_THRESHOLD = 1.0

ArmKind = Literal["control_narrative", "treatment_lean_certificate"]
SplitKind = Literal["train", "eval"]
REQUIRED_ASSUMPTIONS = frozenset({"iid", "bounded_loss_0_1", "confidence_delta_declared"})


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "lean_package": "research_only/hermetic_certificate",
        "trust_domain": "research_only/lean_rademacher_bound",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class BoundCaseV1:
    case_id: str
    arm: ArmKind
    split: SplitKind
    hypothesis_class_size: int
    sample_n: int
    confidence_delta: float
    empirical_risk: float
    true_risk: float
    certificate: Mapping[str, Any] | None
    expected_valid: bool
    expected_useful: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundCaseV1:
        cert = data.get("certificate")
        return cls(
            case_id=str(data["case_id"]),
            arm=str(data["arm"]),  # type: ignore[arg-type]
            split=str(data["split"]),  # type: ignore[arg-type]
            hypothesis_class_size=int(data["hypothesis_class_size"]),
            sample_n=int(data["sample_n"]),
            confidence_delta=float(data["confidence_delta"]),
            empirical_risk=float(data["empirical_risk"]),
            true_risk=float(data["true_risk"]),
            certificate=None if cert is None else dict(cert),
            expected_valid=bool(data["expected_valid"]),
            expected_useful=bool(data["expected_useful"]),
        )


def load_corpus(*, root: Path | None = None) -> tuple[dict[str, Any], tuple[BoundCaseV1, ...]]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, tuple(BoundCaseV1.from_dict(row) for row in raw["cases"])


def default_cases(*, root: Path | None = None) -> tuple[BoundCaseV1, ...]:
    _, cases = load_corpus(root=root)
    return cases


def rademacher_bound(*, class_size: int, sample_n: int, confidence_delta: float) -> float:
    if sample_n <= 0 or class_size <= 0:
        raise ValueError("invalid bound parameters")
    return math.sqrt(2.0 * math.log(2.0 * class_size) / sample_n) + math.sqrt(
        math.log(1.0 / confidence_delta) / (2.0 * sample_n)
    )


def evaluate_control(case: BoundCaseV1) -> dict[str, Any]:
    gap = abs(case.empirical_risk - case.true_risk)
    return {
        "case_id": case.case_id,
        "arm": "control_narrative",
        "certificate_valid": False,
        "bound_value": None,
        "generalization_gap": gap,
        "decision_useful": False,
        "ship_authority_claimed": True,
    }


def evaluate_treatment(case: BoundCaseV1) -> dict[str, Any]:
    cert = case.certificate or {}
    assumptions = frozenset(cert.get("assumptions", []))
    missing = sorted(REQUIRED_ASSUMPTIONS - assumptions)
    if missing:
        return {
            "case_id": case.case_id,
            "arm": "treatment_lean_certificate",
            "certificate_valid": False,
            "bound_value": None,
            "generalization_gap": abs(case.empirical_risk - case.true_risk),
            "decision_useful": False,
            "missing_assumptions": missing,
        }

    recomputed = rademacher_bound(
        class_size=case.hypothesis_class_size,
        sample_n=case.sample_n,
        confidence_delta=case.confidence_delta,
    )
    declared = float(cert.get("rademacher_bound", -1.0))
    bound_ok = declared + 1e-9 >= recomputed
    population_upper = case.empirical_risk + declared
    gap = abs(case.empirical_risk - case.true_risk)
    vacuous = declared >= VACUITY_THRESHOLD
    decision_useful = bound_ok and not vacuous and population_upper >= case.true_risk
    digest_ok = cert.get("certificate_sha256") == content_sha(
        {
            "assumptions": sorted(assumptions),
            "hypothesis_class_size": case.hypothesis_class_size,
            "sample_n": case.sample_n,
            "confidence_delta": case.confidence_delta,
            "empirical_risk": case.empirical_risk,
            "rademacher_bound": declared,
            "lean_theorem": cert.get("lean_theorem"),
        }
    )
    valid = bound_ok and digest_ok and not vacuous
    return {
        "case_id": case.case_id,
        "arm": "treatment_lean_certificate",
        "certificate_valid": valid,
        "bound_value": declared,
        "recomputed_bound": recomputed,
        "population_risk_upper": population_upper,
        "generalization_gap": gap,
        "decision_useful": decision_useful,
        "bound_vacuous": vacuous,
        "missing_assumptions": [],
    }


def _rate(matches: Sequence[bool]) -> float:
    if not matches:
        return 1.0
    return sum(1 for m in matches if m) / len(matches)


def paired_bound_report(*, root: Path | None = None) -> dict[str, Any]:
    cases = default_cases(root=root)
    control_rows = [evaluate_control(c) for c in cases if c.arm == "control_narrative"]
    treatment_cases = [c for c in cases if c.arm == "treatment_lean_certificate"]
    treatment_rows = [evaluate_treatment(c) for c in treatment_cases]
    eval_pairs = [
        (c, r)
        for c, r in zip(treatment_cases, treatment_rows, strict=True)
        if c.split == "eval"
    ]

    validity = _rate([r["certificate_valid"] == c.expected_valid for c, r in eval_pairs])
    usefulness = _rate([r["decision_useful"] == c.expected_useful for c, r in eval_pairs])
    ship_smuggling = sum(
        1
        for r in treatment_rows
        if r.get("ship_authority_claimed") and not r["certificate_valid"]
    )
    vacuity = sum(
        1
        for c, r in eval_pairs
        if c.expected_valid and r.get("bound_vacuous")
    )

    if ship_smuggling > 0:
        decision = "reject"
        reason = "ship_gate_authority_smuggling"
    elif vacuity > 0:
        decision = "reject"
        reason = "bound_vacuity"
    elif validity < 1.0:
        decision = "reject"
        reason = "certificate_invalid"
    elif usefulness < 1.0:
        decision = "reject"
        reason = "bound_not_decision_useful"
    else:
        decision = "accept"
        reason = "lean_checked_toy_bound_is_decision_useful"

    return {
        "schema": "lean_rademacher_bound_report/v1",
        "lean_checked_toy_bound_certificate_validity": validity,
        "decision_usefulness_rate": usefulness,
        "ship_gate_authority_smuggling": ship_smuggling,
        "bound_vacuity": vacuity,
        "assumption_under_declaration": sum(
            1 for r in treatment_rows if r.get("missing_assumptions")
        ),
        "eval_case_count": len(eval_pairs),
        "decision": decision,
        "reason": reason,
        "control_rows": control_rows,
        "treatment_rows": treatment_rows,
        "corpus_digest": content_sha({"cases": [c.case_id for c in cases]}),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "BoundCaseV1",
    "TOOLCHAIN_ID",
    "default_cases",
    "evaluate_control",
    "evaluate_treatment",
    "load_corpus",
    "paired_bound_report",
    "pinned_toolchain",
    "rademacher_bound",
]
