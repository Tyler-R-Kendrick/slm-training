"""RESEARCH-17 / SLM-557: proof-mined quantitative bounds pilot (default-off).

Fixture-only comparison of qualitative-only theorem statements vs HARN-07
proof-mining extraction over frozen revmath quantitative-bound fixtures.
Never promotes empirical timings; nonextractable theorems must refuse bounds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.quantitative_bound import (
    extract_quantitative_bound,
    parse_quantitative_bound_meta,
)
from slm_training.harnesses.reasoning.revmath.schemas import parse_revmath_task

BACKEND_ID = "proof_mined_quantitative_bounds_pilot"
TOOLCHAIN_ID = "hermetic_python_proof_mined_bounds_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/proof_mined_theorems.v1.json"
FIXTURES_RELATIVE = "src/slm_training/resources/revmath/fixtures"


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "fixtures_relpath": FIXTURES_RELATIVE,
        "trust_domain": "research_only/proof_mined_quantitative_bounds",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class TheoremRecordV1:
    theorem_id: str
    fixture_stem: str
    expected_actionable: bool
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TheoremRecordV1:
        return cls(
            theorem_id=str(data["theorem_id"]),
            fixture_stem=str(data["fixture_stem"]),
            expected_actionable=bool(data["expected_actionable"]),
            notes=str(data.get("notes", "")),
        )


def load_corpus(*, root: Path | None = None) -> dict[str, Any]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    return json.loads(path.read_text(encoding="utf-8"))


def default_theorems(*, root: Path | None = None) -> tuple[TheoremRecordV1, ...]:
    raw = load_corpus(root=root)
    return tuple(TheoremRecordV1.from_dict(t) for t in raw["theorems"])


def _load_fixture_pair(stem: str, *, root: Path) -> tuple[Any, Any]:
    fixtures = root / FIXTURES_RELATIVE
    task = parse_revmath_task(
        json.loads((fixtures / f"{stem}.task.json").read_text(encoding="utf-8"))
    )
    meta = parse_quantitative_bound_meta(
        json.loads((fixtures / f"{stem}.meta.json").read_text(encoding="utf-8"))
    )
    return task, meta


def evaluate_control(record: TheoremRecordV1, *, root: Path) -> dict[str, Any]:
    _task, meta = _load_fixture_pair(record.fixture_stem, root=root)
    return {
        "arm": "qualitative_only",
        "theorem_id": record.theorem_id,
        "qualitative_statement": meta.qualitative_statement,
        "bound_value": None,
        "actionable": False,
        "certificate_sha256": None,
    }


def evaluate_treatment(record: TheoremRecordV1, *, root: Path) -> dict[str, Any]:
    task, meta = _load_fixture_pair(record.fixture_stem, root=root)
    report = extract_quantitative_bound(task, meta)
    actionable = (
        report.outcome == "extracted"
        and report.validated_via_bound_ast
        and report.certificate_sha256 is not None
        and report.theorem_derived_bound
        and not report.empirical_timing_claimed
        and report.bound_value is not None
    )
    spurious = actionable and report.bound_value in {"0", "1"} and record.expected_actionable
    return {
        "arm": "proof_mined_extraction",
        "theorem_id": record.theorem_id,
        "outcome": report.outcome,
        "bound_value": report.bound_value,
        "bound_ast_id": report.bound_ast_id,
        "actionable": actionable and not spurious,
        "spurious_constant": spurious,
        "certificate_sha256": report.certificate_sha256,
        "validated_via_bound_ast": report.validated_via_bound_ast,
        "missing_assumptions": list(report.missing_assumptions),
        "report_digest": report.report_digest(),
    }


def paired_mining_report(*, root: Path | None = None) -> dict[str, Any]:
    base = root or _repo_root()
    raw = load_corpus(root=base)
    records = default_theorems(root=base)

    control_rows = [evaluate_control(r, root=base) for r in records]
    treatment_rows = [evaluate_treatment(r, root=base) for r in records]

    expected_actionable = [r for r in records if r.expected_actionable]
    actionable_success = sum(
        1
        for rec, treat in zip(records, treatment_rows, strict=True)
        if rec.expected_actionable and treat["actionable"]
    )
    primary = actionable_success / max(len(expected_actionable), 1)

    spurious = sum(1 for row in treatment_rows if row.get("spurious_constant"))
    false_positive = sum(
        1
        for rec, treat in zip(records, treatment_rows, strict=True)
        if not rec.expected_actionable and treat["actionable"]
    )
    missed = sum(
        1
        for rec, treat in zip(records, treatment_rows, strict=True)
        if rec.expected_actionable and not treat["actionable"]
    )

    if false_positive > 0:
        decision = "reject"
        reason = "bound_extracted_on_nonextractable_theorem"
    elif spurious > 0:
        decision = "reject"
        reason = "spurious_constant_bound"
    elif missed > 0:
        decision = "reject"
        reason = "failed_to_extract_expected_bounds"
    elif primary < 1.0:
        decision = "reject"
        reason = "incomplete_actionable_extraction"
    else:
        decision = "accept"
        reason = "proof_mining_yields_replayable_bounds"

    return {
        "schema": "proof_mined_quantitative_bounds_report/v1",
        "actionable_bound_extraction_success_with_replayable_certificate": primary,
        "actionable_success_count": actionable_success,
        "expected_actionable_count": len(expected_actionable),
        "spurious_constant_rate": spurious / max(len(records), 1),
        "bound_uselessness_rate": false_positive / max(len(records), 1),
        "proof_mining_failures_as_unknown": sum(
            1 for row in treatment_rows if row["outcome"] == "unknown"
        ),
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
    "TheoremRecordV1",
    "default_theorems",
    "evaluate_control",
    "evaluate_treatment",
    "load_corpus",
    "paired_mining_report",
    "pinned_toolchain",
]
