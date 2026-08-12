"""RESEARCH-04 / SLM-550: Reverse Mathematics Zoo classification benchmark pilot.

Frozen non-Big-Five zoo principles plus Big-Five controls exercise the revmath
labeling harness (HARN-08) without widening production authority.

Default-off / research-only. Every zoo label binds to cited lineage; uncertain
literature stays ``unclassified`` rather than forcing a Big-Five class.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.labeling import (
    KNOWN_RM_SUBSYSTEMS,
    parse_label_claim,
    validate_label_claim,
)

BACKEND_ID = "rm_zoo_benchmark_pilot"
TOOLCHAIN_ID = "hermetic_python_rm_zoo_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/rm_zoo_corpus.v1.json"

ArmKind = Literal["control_big_five", "treatment_zoo"]
SplitKind = Literal["train", "eval"]


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "label_validator": "harn08.validate_label_claim",
        "trust_domain": "research_only/rm_zoo_benchmark",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


def corpus_path(*, root: Path | None = None) -> Path:
    return (root or _repo_root()) / CORPUS_RELATIVE


@dataclass(frozen=True)
class ZooBenchmarkEntryV1:
    entry_id: str
    arm: ArmKind
    split: SplitKind
    root_family_id: str
    principle_id: str
    task_kind: str
    expected_label_accepted: bool
    claim: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ZooBenchmarkEntryV1:
        return cls(
            entry_id=str(data["entry_id"]),
            arm=str(data["arm"]),  # type: ignore[arg-type]
            split=str(data["split"]),  # type: ignore[arg-type]
            root_family_id=str(data["root_family_id"]),
            principle_id=str(data["principle_id"]),
            task_kind=str(data["task_kind"]),
            expected_label_accepted=bool(data["expected_label_accepted"]),
            claim=dict(data["claim"]),
        )

    def is_big_five(self) -> bool:
        return self.principle_id in KNOWN_RM_SUBSYSTEMS

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "arm": self.arm,
            "split": self.split,
            "root_family_id": self.root_family_id,
            "principle_id": self.principle_id,
            "task_kind": self.task_kind,
            "expected_label_accepted": self.expected_label_accepted,
            "claim": dict(self.claim),
        }


def load_corpus(*, root: Path | None = None) -> tuple[dict[str, Any], tuple[ZooBenchmarkEntryV1, ...]]:
    path = corpus_path(root=root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = tuple(ZooBenchmarkEntryV1.from_dict(row) for row in raw["entries"])
    return raw, entries


def default_zoo_corpus(*, root: Path | None = None) -> tuple[ZooBenchmarkEntryV1, ...]:
    _, entries = load_corpus(root=root)
    return entries


def evaluate_entry(entry: ZooBenchmarkEntryV1) -> dict[str, Any]:
    claim = parse_label_claim(entry.claim)
    verdict = validate_label_claim(claim)
    accepted = bool(verdict.accepted)
    return {
        "entry_id": entry.entry_id,
        "arm": entry.arm,
        "split": entry.split,
        "principle_id": entry.principle_id,
        "task_kind": entry.task_kind,
        "expected_label_accepted": entry.expected_label_accepted,
        "observed_label_accepted": accepted,
        "labeling_state": verdict.labeling_state,
        "rejection_reasons": list(verdict.rejection_reasons),
        "exact_match": accepted == entry.expected_label_accepted,
        "is_big_five": entry.is_big_five(),
    }


def _rate(matches: Sequence[bool]) -> float:
    if not matches:
        return 1.0
    return sum(1 for m in matches if m) / len(matches)


def paired_benchmark_report(
    entries: Sequence[ZooBenchmarkEntryV1] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Paired control (Big-Five-only train) vs treatment (zoo eval) benchmark."""

    rows = list(entries or default_zoo_corpus(root=root))
    evaluated = [evaluate_entry(e) for e in rows]

    eval_rows = [r for r in evaluated if r["split"] == "eval"]
    train_control = [r for r in evaluated if r["arm"] == "control_big_five"]
    zoo_eval = [r for r in evaluated if r["arm"] == "treatment_zoo"]

    eval_exact = _rate([r["exact_match"] for r in eval_rows])
    train_exact = _rate([r["exact_match"] for r in train_control])

    overclaim_attempts = [
        r for r in zoo_eval if r["expected_label_accepted"] is False
    ]
    overclaim_accepted = sum(
        1 for r in overclaim_attempts if r["observed_label_accepted"]
    )
    overclaim_rate = (
        overclaim_accepted / len(overclaim_attempts) if overclaim_attempts else 0.0
    )

    unknown_rows = [r for r in zoo_eval if r["task_kind"] == "unknown"]
    unknown_ok = _rate([r["exact_match"] for r in unknown_rows]) if unknown_rows else 1.0

    zoo_non_bf = [r for r in zoo_eval if not r["is_big_five"]]
    zoo_discriminates = len(zoo_non_bf) >= 1 and _rate([r["exact_match"] for r in zoo_non_bf]) == 1.0
    library_retrieval_only = not zoo_discriminates

    correctness_ok = eval_exact == 1.0 and overclaim_rate == 0.0
    if correctness_ok and zoo_discriminates:
        decision = "accept"
        reason = "zoo_benchmark_pass"
    elif overclaim_rate > 0:
        decision = "reject"
        reason = "overclaim_detected"
    elif eval_exact < 1.0:
        decision = "reject"
        reason = "eval_mismatch"
    elif library_retrieval_only:
        decision = "reject"
        reason = "library_retrieval_only"
    else:
        decision = "reject"
        reason = "zoo_benchmark_inconclusive"

    return {
        "schema": "rm_zoo_benchmark_report/v1",
        "exact_classification_reversal_success_dependency_disjoint": eval_exact,
        "train_control_exact_rate": train_exact,
        "overclaim_rate": overclaim_rate,
        "unknown_calibration": unknown_ok,
        "library_retrieval_only": library_retrieval_only,
        "zoo_discriminates": zoo_discriminates,
        "eval_entry_count": len(eval_rows),
        "zoo_eval_count": len(zoo_eval),
        "correctness_ok": correctness_ok,
        "decision": decision,
        "reason": reason,
        "corpus_digest": content_sha({"entries": [e.to_dict() for e in rows]}),
        "entries": evaluated,
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "TOOLCHAIN_ID",
    "ZooBenchmarkEntryV1",
    "corpus_path",
    "default_zoo_corpus",
    "evaluate_entry",
    "load_corpus",
    "paired_benchmark_report",
    "pinned_toolchain",
]
