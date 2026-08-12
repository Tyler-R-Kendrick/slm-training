"""RESEARCH-14 / SLM-572: checker diversity pilot (default-off).

Fixture-only comparison of single-checker vs multi-trust-domain checker matrix
over frozen seeded fault classes aligned with EVID-08 trust domains and EVID-11
mutation families. No mandatory external checker dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.formal.checker_capability import (
    TRUST_DOMAIN_ENCODING_BRIDGE,
    TRUST_DOMAIN_LEAN4_KERNEL,
    TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY,
    trust_domain_for,
)
from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "checker_diversity_pilot"
TOOLCHAIN_ID = "hermetic_python_checker_diversity_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/checker_diversity_corpus.v1.json"

CONTROL_CHECKERS: tuple[str, ...] = ("python_structural",)
TREATMENT_CHECKERS: tuple[str, ...] = (
    "python_structural",
    "lean_kernel",
    "python_encoding_ref",
)

CheckerArm = Literal["single_checker", "diverse_checkers"]


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "control_checkers": ",".join(CONTROL_CHECKERS),
        "treatment_checkers": ",".join(TREATMENT_CHECKERS),
        "trust_domain": "research_only/checker_diversity",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class FaultCaseV1:
    case_id: str
    fault_class: str
    fault_domain: str
    evid11_family: str
    detectors: frozenset[str]
    seeded_defect: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FaultCaseV1:
        return cls(
            case_id=str(data["case_id"]),
            fault_class=str(data["fault_class"]),
            fault_domain=str(data["fault_domain"]),
            evid11_family=str(data["evid11_family"]),
            detectors=frozenset(str(item) for item in data["detectors"]),
            seeded_defect=bool(data["seeded_defect"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fault_class": self.fault_class,
            "fault_domain": self.fault_domain,
            "evid11_family": self.evid11_family,
            "detectors": sorted(self.detectors),
            "seeded_defect": self.seeded_defect,
        }


def load_corpus(*, root: Path | None = None) -> tuple[dict[str, Any], tuple[FaultCaseV1, ...]]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(FaultCaseV1.from_dict(row) for row in raw["cases"])
    return raw, cases


def default_cases(*, root: Path | None = None) -> tuple[FaultCaseV1, ...]:
    _, cases = load_corpus(root=root)
    return cases


def _distinct_trust_domains(checkers: Sequence[str]) -> frozenset[str]:
    domains: set[str] = set()
    for backend in checkers:
        domains.add(trust_domain_for(backend))
    return frozenset(domains)


def _detect(case: FaultCaseV1, checkers: Sequence[str]) -> bool:
    if not case.seeded_defect:
        return False
    return any(checker in case.detectors for checker in checkers)


def _false_alarm(case: FaultCaseV1, checkers: Sequence[str]) -> bool:
    if case.seeded_defect:
        return False
    return any(checker in case.detectors for checker in checkers)


def evaluate_arm(
    cases: Sequence[FaultCaseV1],
    *,
    checkers: Sequence[str],
    arm: CheckerArm,
) -> dict[str, Any]:
    detected_classes: set[str] = set()
    detected_cases: list[str] = []
    false_alarms: list[str] = []
    blind_spots: list[str] = []

    for case in cases:
        if _false_alarm(case, checkers):
            false_alarms.append(case.case_id)
        if _detect(case, checkers):
            detected_classes.add(case.fault_class)
            detected_cases.append(case.case_id)
        elif case.seeded_defect:
            blind_spots.append(case.case_id)

    return {
        "arm": arm,
        "checkers": list(checkers),
        "distinct_trust_domains": sorted(_distinct_trust_domains(checkers)),
        "detected_fault_classes": sorted(detected_classes),
        "detected_case_ids": detected_cases,
        "false_alarm_case_ids": false_alarms,
        "blind_spot_case_ids": blind_spots,
        "false_alarm_rate": len(false_alarms) / max(len(cases), 1),
    }


def paired_diversity_report(*, root: Path | None = None) -> dict[str, Any]:
    raw, cases = load_corpus(root=root)
    control = evaluate_arm(cases, checkers=CONTROL_CHECKERS, arm="single_checker")
    treatment = evaluate_arm(
        cases, checkers=TREATMENT_CHECKERS, arm="diverse_checkers"
    )

    control_classes = set(control["detected_fault_classes"])
    treatment_classes = set(treatment["detected_fault_classes"])
    independent_classes = {
        case.fault_class
        for case in cases
        if case.seeded_defect and case.fault_class != "shared_blind_spot"
    }
    unique_gain = len(treatment_classes - control_classes)
    primary = unique_gain / max(len(independent_classes), 1)

    shared_blind = [
        case.case_id
        for case in cases
        if case.fault_class == "shared_blind_spot" and case.seeded_defect
    ]
    shared_blind_spot_residual = len(
        [
            case_id
            for case_id in shared_blind
            if case_id in treatment["blind_spot_case_ids"]
        ]
    )
    false_alarm_rate = max(
        control["false_alarm_rate"], treatment["false_alarm_rate"]
    )

    if unique_gain <= 0:
        decision = "reject"
        reason = "no_unique_defect_gain"
    elif false_alarm_rate > 0:
        decision = "reject"
        reason = "false_alarm_blow_up"
    elif len(_distinct_trust_domains(TREATMENT_CHECKERS)) < 2:
        decision = "reject"
        reason = "treatment_not_diverse"
    else:
        decision = "accept"
        reason = "diverse_checkers_add_unique_fault_classes"

    return {
        "schema": "checker_diversity_report/v1",
        "unique_defect_detection_gain_vs_single_checker": primary,
        "unique_fault_class_gain_count": unique_gain,
        "control_detected_classes": sorted(control_classes),
        "treatment_detected_classes": sorted(treatment_classes),
        "independent_fault_classes": sorted(independent_classes),
        "false_alarm_rate": false_alarm_rate,
        "shared_blind_spot_residual": shared_blind_spot_residual,
        "timeout_unknown_rate": 0.0,
        "control_arm": control,
        "treatment_arm": treatment,
        "decision": decision,
        "reason": reason,
        "corpus_digest": content_sha(raw),
        "toolchain": pinned_toolchain(),
        "evid08_trust_domains": {
            "structural": TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY,
            "lean_kernel": TRUST_DOMAIN_LEAN4_KERNEL,
            "encoding": TRUST_DOMAIN_ENCODING_BRIDGE,
        },
    }


__all__ = [
    "BACKEND_ID",
    "CONTROL_CHECKERS",
    "CORPUS_RELATIVE",
    "TREATMENT_CHECKERS",
    "FaultCaseV1",
    "default_cases",
    "evaluate_arm",
    "load_corpus",
    "paired_diversity_report",
    "pinned_toolchain",
]
