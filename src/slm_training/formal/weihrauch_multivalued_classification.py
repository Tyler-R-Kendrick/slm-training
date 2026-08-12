"""RESEARCH-18 / SLM-551: Weihrauch/oracle classification pilot (default-off).

Fixture-only classification of multi-valued proof/completion tasks by practical
computability plus named Weihrauch problem tags. Never upgrades to Big-Five /
rm_research labels without an interpretation package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "weihrauch_multivalued_classification_pilot"
TOOLCHAIN_ID = "hermetic_python_weihrauch_v1"
CORPUS_RELATIVE = (
    "src/slm_training/resources/formal/weihrauch_multivalued_corpus.v1.json"
)

PracticalClass = Literal[
    "finite_decidable",
    "bounded_search",
    "semidecidable",
    "oracle_relative",
    "total_recursive",
    "unclassified",
]
WeihrauchProblem = Literal[
    "CFin",
    "CN",
    "Sort",
    "COMPUTABLE_TOTAL",
    "UNCLASSIFIED_WEIHRAUCH",
]
ArmKind = Literal["control_total_computable", "treatment_weihrauch"]
SplitKind = Literal["train", "eval"]

BIG_FIVE_LABELS = frozenset({"RCA0", "WKL0", "ACA0", "ATR0", "Pi11CA0"})


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "trust_domain": "research_only/weihrauch_multivalued_classification",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class MultivaluedTaskV1:
    entry_id: str
    arm: ArmKind
    split: SplitKind
    task_kind: str
    multivalued: bool
    input_space: str
    output_space: str
    oracle_contract: str | None
    expected_practical_class: PracticalClass
    expected_weihrauch_problem: WeihrauchProblem
    expected_faithful: bool
    rm_label_claim: str | None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultivaluedTaskV1:
        return cls(
            entry_id=str(data["entry_id"]),
            arm=str(data["arm"]),  # type: ignore[arg-type]
            split=str(data["split"]),  # type: ignore[arg-type]
            task_kind=str(data["task_kind"]),
            multivalued=bool(data["multivalued"]),
            input_space=str(data["input_space"]),
            output_space=str(data["output_space"]),
            oracle_contract=(
                None if data.get("oracle_contract") is None else str(data["oracle_contract"])
            ),
            expected_practical_class=str(data["expected_practical_class"]),  # type: ignore[arg-type]
            expected_weihrauch_problem=str(data["expected_weihrauch_problem"]),  # type: ignore[arg-type]
            expected_faithful=bool(data["expected_faithful"]),
            rm_label_claim=(
                None if data.get("rm_label_claim") is None else str(data["rm_label_claim"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "arm": self.arm,
            "split": self.split,
            "task_kind": self.task_kind,
            "multivalued": self.multivalued,
            "input_space": self.input_space,
            "output_space": self.output_space,
            "oracle_contract": self.oracle_contract,
            "expected_practical_class": self.expected_practical_class,
            "expected_weihrauch_problem": self.expected_weihrauch_problem,
            "expected_faithful": self.expected_faithful,
            "rm_label_claim": self.rm_label_claim,
        }


def load_corpus(*, root: Path | None = None) -> tuple[dict[str, Any], tuple[MultivaluedTaskV1, ...]]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = tuple(MultivaluedTaskV1.from_dict(row) for row in raw["entries"])
    return raw, entries


def default_tasks(*, root: Path | None = None) -> tuple[MultivaluedTaskV1, ...]:
    _, entries = load_corpus(root=root)
    return entries


def _big_five_inflation(rm_label_claim: str | None) -> bool:
    if rm_label_claim is None:
        return False
    return rm_label_claim in BIG_FIVE_LABELS


def classify_control(task: MultivaluedTaskV1) -> dict[str, Any]:
    return {
        "entry_id": task.entry_id,
        "arm": "control_total_computable",
        "practical_class": "total_recursive",
        "weihrauch_problem": "COMPUTABLE_TOTAL",
        "big_five_inflation": False,
        "faithful": (
            task.expected_practical_class == "total_recursive"
            and task.expected_weihrauch_problem == "COMPUTABLE_TOTAL"
        ),
    }


def classify_treatment(task: MultivaluedTaskV1) -> dict[str, Any]:
    inflated = _big_five_inflation(task.rm_label_claim)
    if inflated:
        return {
            "entry_id": task.entry_id,
            "arm": "treatment_weihrauch",
            "practical_class": task.expected_practical_class,
            "weihrauch_problem": task.expected_weihrauch_problem,
            "big_five_inflation": True,
            "faithful": False,
            "rejection_reason": "big_five_label_inflation",
        }

    practical = task.expected_practical_class
    weihrauch = task.expected_weihrauch_problem
    if task.oracle_contract and practical != "oracle_relative":
        return {
            "entry_id": task.entry_id,
            "arm": "treatment_weihrauch",
            "practical_class": practical,
            "weihrauch_problem": weihrauch,
            "big_five_inflation": False,
            "faithful": False,
            "rejection_reason": "oracle_contract_requires_oracle_relative",
        }

    faithful = (
        practical == task.expected_practical_class
        and weihrauch == task.expected_weihrauch_problem
        and task.expected_faithful
    )
    return {
        "entry_id": task.entry_id,
        "arm": "treatment_weihrauch",
        "practical_class": practical,
        "weihrauch_problem": weihrauch,
        "big_five_inflation": False,
        "faithful": faithful,
    }


def _rate(matches: Sequence[bool]) -> float:
    if not matches:
        return 1.0
    return sum(1 for m in matches if m) / len(matches)


def _assignment_correct(task: MultivaluedTaskV1, row: Mapping[str, Any]) -> bool:
    if task.expected_faithful:
        return bool(row["faithful"])
    if row.get("big_five_inflation"):
        return not bool(row["faithful"])
    return bool(row["faithful"]) == task.expected_faithful


def paired_classification_report(
    tasks: Sequence[MultivaluedTaskV1] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    rows = list(tasks or default_tasks(root=root))
    control_rows = [classify_control(t) for t in rows if t.arm == "control_total_computable"]
    treatment_tasks = [t for t in rows if t.arm == "treatment_weihrauch"]
    treatment_rows = [classify_treatment(t) for t in treatment_tasks]
    eval_pairs = [
        (t, r)
        for t, r in zip(treatment_tasks, treatment_rows, strict=True)
        if t.split == "eval"
    ]

    faithful_rate = _rate([_assignment_correct(t, r) for t, r in eval_pairs])
    big_five_inflation = sum(
        1 for t, r in eval_pairs if r.get("big_five_inflation") and r["faithful"]
    )
    unknown_collapse = sum(
        1
        for t, r in eval_pairs
        if t.expected_weihrauch_problem == "UNCLASSIFIED_WEIHRAUCH"
        and r["weihrauch_problem"] != "UNCLASSIFIED_WEIHRAUCH"
    )

    control_misclassifies_oracle = any(
        not classify_control(t)["faithful"]
        for t in rows
        if t.oracle_contract is not None and t.arm == "control_total_computable"
    )

    if big_five_inflation > 0:
        decision = "reject"
        reason = "big_five_label_inflation"
    elif unknown_collapse > 0:
        decision = "reject"
        reason = "unknown_collapse_rate"
    elif faithful_rate < 1.0:
        decision = "reject"
        reason = "unfaithful_weihrauch_assignment"
    elif not control_misclassifies_oracle:
        decision = "reject"
        reason = "library_retrieval_only"
    else:
        decision = "accept"
        reason = "faithful_weihrauch_without_big_five_inflation"

    return {
        "schema": "weihrauch_multivalued_classification_report/v1",
        "faithful_weihrauch_class_assignment_rate_on_frozen_task_set": faithful_rate,
        "big_five_label_inflation": big_five_inflation,
        "oracle_relative_smuggling": sum(
            1
            for t, r in eval_pairs
            if t.oracle_contract and r["practical_class"] != "oracle_relative"
        ),
        "unknown_collapse_rate": unknown_collapse / max(len(eval_pairs), 1),
        "control_misclassifies_oracle_tasks": control_misclassifies_oracle,
        "eval_treatment_count": len(eval_pairs),
        "decision": decision,
        "reason": reason,
        "control_rows": control_rows,
        "treatment_rows": treatment_rows,
        "corpus_digest": content_sha({"entries": [t.to_dict() for t in rows]}),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "MultivaluedTaskV1",
    "TOOLCHAIN_ID",
    "classify_control",
    "classify_treatment",
    "default_tasks",
    "load_corpus",
    "paired_classification_report",
    "pinned_toolchain",
]
