"""RESEARCH-12 / SLM-567: successor-theorem semantic validation pilot (default-off).

Fixture-only comparison of compile-only acceptance vs successor-theorem,
proposition-fingerprint, and counterexample validation on frozen generated
theorem records. Never production authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "successor_theorem_semantic_validation_pilot"
TOOLCHAIN_ID = "hermetic_python_successor_semantic_v1"
CORPUS_RELATIVE = (
    "src/slm_training/resources/formal/successor_theorem_corpus.v1.json"
)


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "trust_domain": "research_only/successor_theorem_semantic_validation",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class SuccessorObligationV1:
    obligation_id: str
    holds: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SuccessorObligationV1:
        return cls(
            obligation_id=str(data["obligation_id"]),
            holds=bool(data["holds"]),
        )


@dataclass(frozen=True)
class TheoremRecordV1:
    record_id: str
    compiles: bool
    statement_sha256: str
    expected_statement_sha256: str
    successor_obligations: tuple[SuccessorObligationV1, ...]
    counterexample_refutes: bool
    ground_truth_meaning_changing: bool
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TheoremRecordV1:
        return cls(
            record_id=str(data["record_id"]),
            compiles=bool(data["compiles"]),
            statement_sha256=str(data["statement_sha256"]),
            expected_statement_sha256=str(data["expected_statement_sha256"]),
            successor_obligations=tuple(
                SuccessorObligationV1.from_dict(o)
                for o in data.get("successor_obligations") or ()
            ),
            counterexample_refutes=bool(data.get("counterexample_refutes", False)),
            ground_truth_meaning_changing=bool(data["ground_truth_meaning_changing"]),
            notes=str(data.get("notes", "")),
        )


def load_corpus(*, root: Path | None = None) -> dict[str, Any]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    return json.loads(path.read_text(encoding="utf-8"))


def default_records(*, root: Path | None = None) -> tuple[TheoremRecordV1, ...]:
    raw = load_corpus(root=root)
    return tuple(TheoremRecordV1.from_dict(r) for r in raw["records"])


def _fingerprint_matches(record: TheoremRecordV1) -> bool:
    return record.statement_sha256 == record.expected_statement_sha256


def _successors_hold(record: TheoremRecordV1) -> bool:
    return all(o.holds for o in record.successor_obligations)


def evaluate_control(record: TheoremRecordV1) -> dict[str, Any]:
    return {
        "arm": "compile_only",
        "record_id": record.record_id,
        "accepts": record.compiles,
        "compiles": record.compiles,
    }


def evaluate_fingerprint_counterexample(record: TheoremRecordV1) -> dict[str, Any]:
    accepts = (
        record.compiles
        and _fingerprint_matches(record)
        and not record.counterexample_refutes
    )
    return {
        "arm": "fingerprint_counterexample",
        "record_id": record.record_id,
        "accepts": accepts,
        "fingerprint_matches": _fingerprint_matches(record),
        "counterexample_refutes": record.counterexample_refutes,
    }


def evaluate_treatment(record: TheoremRecordV1) -> dict[str, Any]:
    fp = evaluate_fingerprint_counterexample(record)
    accepts = fp["accepts"] and _successors_hold(record)
    return {
        "arm": "successor_semantic_validation",
        "record_id": record.record_id,
        "accepts": accepts,
        "fingerprint_matches": fp["fingerprint_matches"],
        "counterexample_refutes": record.counterexample_refutes,
        "successors_hold": _successors_hold(record),
    }


def paired_validation_report(*, root: Path | None = None) -> dict[str, Any]:
    base = root or _repo_root()
    raw = load_corpus(root=base)
    records = default_records(root=base)

    control_rows = [evaluate_control(r) for r in records]
    fp_rows = [evaluate_fingerprint_counterexample(r) for r in records]
    treatment_rows = [evaluate_treatment(r) for r in records]

    faithful = [r for r in records if not r.ground_truth_meaning_changing]

    smuggled = [
        (rec, ctrl, treat, fp)
        for rec, ctrl, treat, fp in zip(
            records, control_rows, treatment_rows, fp_rows, strict=True
        )
        if rec.ground_truth_meaning_changing and ctrl["accepts"]
    ]
    caught = sum(1 for _rec, _ctrl, treat, _fp in smuggled if not treat["accepts"])
    catch_rate = caught / max(len(smuggled), 1)

    false_positive = sum(
        1
        for rec, treat in zip(records, treatment_rows, strict=True)
        if not rec.ground_truth_meaning_changing and treat["accepts"] is False
    )
    false_positive_rate = false_positive / max(len(faithful), 1)

    successor_incremental = sum(
        1
        for _rec, _ctrl, treat, fp in smuggled
        if fp["accepts"] and not treat["accepts"]
    )

    compile_only_pass_smuggling = len(smuggled)

    if false_positive > 0:
        decision = "reject"
        reason = "false_positive_semantic_block"
    elif len(smuggled) == 0:
        decision = "reject"
        reason = "no_compile_smuggling_cases_in_corpus"
    elif catch_rate < 1.0:
        decision = "reject"
        reason = "incomplete_meaning_change_detection"
    elif successor_incremental == 0:
        decision = "reject"
        reason = "successor_adds_no_detection_beyond_fingerprint_counterexample"
    else:
        decision = "accept"
        reason = "successor_validation_catches_subtle_meaning_change"

    return {
        "schema": "successor_theorem_semantic_validation_report/v1",
        "semantic_validation_catch_rate_on_meaning_changing_successors": catch_rate,
        "meaning_changing_compile_smuggling_count": compile_only_pass_smuggling,
        "meaning_changing_caught_count": caught,
        "successor_incremental_catch_count": successor_incremental,
        "false_positive_block_rate": false_positive_rate,
        "compile_only_pass_smuggling": compile_only_pass_smuggling,
        "unknown_rate": 0.0,
        "decision": decision,
        "reason": reason,
        "control_rows": control_rows,
        "fingerprint_counterexample_rows": fp_rows,
        "treatment_rows": treatment_rows,
        "corpus_digest": content_sha(raw),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "TheoremRecordV1",
    "default_records",
    "evaluate_control",
    "evaluate_fingerprint_counterexample",
    "evaluate_treatment",
    "load_corpus",
    "paired_validation_report",
    "pinned_toolchain",
]
