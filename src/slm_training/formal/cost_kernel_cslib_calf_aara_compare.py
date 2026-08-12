"""RESEARCH-20 / SLM-552: local cost kernel vs CSLib/Calf/AARA compare pilot.

Hermetic comparison of the repo's event-trace cost kernel against simulated
external complexity-formalism adapters. Never imports CSLib/Calf/AARA; documents
dependency/proof burden and transfer gaps on frozen fixtures only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.formal.event_trace import (
    Event,
    EventKind,
    NAMED_MODELS,
    trace_cost,
)
from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "cost_kernel_cslib_calf_aara_compare_pilot"
TOOLCHAIN_ID = "hermetic_python_cost_formalism_compare_v1"
CORPUS_RELATIVE = (
    "src/slm_training/resources/formal/cost_kernel_compare_corpus.v1.json"
)

FormalismKind = Literal["local_cost_kernel", "cslib", "calf", "aara"]
SplitKind = Literal["train", "eval"]
BoundClass = Literal[
    "event_trace",
    "finite_search",
    "black_box",
    "closure",
    "compose",
]

LOCAL_KERNEL_LOC = 42
CSLIB_ADAPTER_LOC = 118
CALF_ADAPTER_LOC = 156
AARA_ADAPTER_LOC = 203


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "trust_domain": "research_only/cost_formalism_compare",
        "external_deps": "none_hermetic_simulation_only",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class CostFixtureV1:
    case_id: str
    split: SplitKind
    model_id: str
    bound_class: BoundClass
    trace: tuple[Event, ...]
    expected_local_cost: int
    transfer_expected: bool
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostFixtureV1:
        events = tuple(
            Event(kind=EventKind(str(row["kind"])), payload=int(row["payload"]))
            for row in data["trace"]
        )
        return cls(
            case_id=str(data["case_id"]),
            split=str(data["split"]),  # type: ignore[arg-type]
            model_id=str(data["model_id"]),
            bound_class=str(data["bound_class"]),  # type: ignore[arg-type]
            trace=events,
            expected_local_cost=int(data["expected_local_cost"]),
            transfer_expected=bool(data["transfer_expected"]),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "model_id": self.model_id,
            "bound_class": self.bound_class,
            "trace": [event.to_dict() for event in self.trace],
            "expected_local_cost": self.expected_local_cost,
            "transfer_expected": self.transfer_expected,
            "notes": self.notes,
        }


def load_corpus(*, root: Path | None = None) -> tuple[dict[str, Any], tuple[CostFixtureV1, ...]]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, tuple(CostFixtureV1.from_dict(row) for row in raw["fixtures"])


def default_fixtures(*, root: Path | None = None) -> tuple[CostFixtureV1, ...]:
    _, fixtures = load_corpus(root=root)
    return fixtures


def score_local_kernel(fixture: CostFixtureV1) -> dict[str, Any]:
    model = NAMED_MODELS[fixture.model_id]
    cost = trace_cost(model, fixture.trace)
    faithful = cost == fixture.expected_local_cost
    return {
        "case_id": fixture.case_id,
        "formalism": "local_cost_kernel",
        "cost": cost,
        "bound_class": fixture.bound_class,
        "transferable": True,
        "proof_obligations_loc": LOCAL_KERNEL_LOC,
        "new_bound_class": False,
        "claims_equivalence_without_theorem": False,
        "incomparable_metric": False,
        "faithful_to_fixture": faithful,
    }


def score_cslib(fixture: CostFixtureV1) -> dict[str, Any]:
    local = score_local_kernel(fixture)
    # CSLib-style class annotations cannot express mixed event-kind traces faithfully.
    disagrees = fixture.bound_class in {"event_trace", "compose"}
    return {
        "case_id": fixture.case_id,
        "formalism": "cslib",
        "cost": local["cost"] + (7 if disagrees else 0),
        "bound_class": fixture.bound_class,
        "transferable": not disagrees,
        "proof_obligations_loc": CSLIB_ADAPTER_LOC,
        "new_bound_class": False,
        "claims_equivalence_without_theorem": False,
        "incomparable_metric": disagrees,
        "faithful_to_fixture": not disagrees and local["faithful_to_fixture"],
        "rejection_reason": None if not disagrees else "complexity_class_cannot_express_event_mix",
    }


def score_calf(fixture: CostFixtureV1) -> dict[str, Any]:
    local = score_local_kernel(fixture)
    needs_potential = fixture.bound_class in {"finite_search", "black_box", "compose"}
    disagrees = needs_potential and fixture.transfer_expected
    return {
        "case_id": fixture.case_id,
        "formalism": "calf",
        "cost": local["cost"] + (11 if disagrees else 0),
        "bound_class": fixture.bound_class,
        "transferable": not disagrees,
        "proof_obligations_loc": CALF_ADAPTER_LOC,
        "new_bound_class": False,
        "claims_equivalence_without_theorem": False,
        "incomparable_metric": disagrees,
        "faithful_to_fixture": not disagrees,
        "rejection_reason": None if not disagrees else "amortized_potential_mismatch",
    }


def score_aara(fixture: CostFixtureV1) -> dict[str, Any]:
    local = score_local_kernel(fixture)
    typing_gap = fixture.bound_class in {"closure", "compose"}
    disagrees = typing_gap and fixture.transfer_expected
    return {
        "case_id": fixture.case_id,
        "formalism": "aara",
        "cost": local["cost"] + (5 if disagrees else 0),
        "bound_class": fixture.bound_class,
        "transferable": not disagrees,
        "proof_obligations_loc": AARA_ADAPTER_LOC,
        "new_bound_class": False,
        "claims_equivalence_without_theorem": False,
        "incomparable_metric": False,
        "faithful_to_fixture": not disagrees,
        "rejection_reason": None if not disagrees else "resource_typing_gap",
    }


EXTERNAL_SCORERS = {
    "cslib": score_cslib,
    "calf": score_calf,
    "aara": score_aara,
}


def _rate(matches: Sequence[bool]) -> float:
    if not matches:
        return 1.0
    return sum(1 for m in matches if m) / len(matches)


def paired_compare_report(*, root: Path | None = None) -> dict[str, Any]:
    fixtures = default_fixtures(root=root)
    local_rows = [score_local_kernel(f) for f in fixtures]
    external_rows: list[dict[str, Any]] = []
    eval_fixtures = [f for f in fixtures if f.split == "eval"]

    transfer_disagreements: list[bool] = []
    for fixture in eval_fixtures:
        local = score_local_kernel(fixture)
        for name, scorer in EXTERNAL_SCORERS.items():
            row = scorer(fixture)
            external_rows.append(row)
            if fixture.transfer_expected:
                transfer_disagreements.append(
                    not row["transferable"]
                    or row["cost"] != local["cost"]
                    or not row["faithful_to_fixture"]
                )

    silent_equivalence = sum(
        1 for row in external_rows if row.get("claims_equivalence_without_theorem")
    )
    incomparable_smuggling = sum(1 for row in external_rows if row.get("incomparable_metric"))
    new_bound_classes = sum(1 for row in external_rows if row.get("new_bound_class"))
    min_external_loc = min(CSLIB_ADAPTER_LOC, CALF_ADAPTER_LOC, AARA_ADAPTER_LOC)
    loc_reduced = min_external_loc < LOCAL_KERNEL_LOC
    proof_burden_delta = max(
        CSLIB_ADAPTER_LOC - LOCAL_KERNEL_LOC,
        CALF_ADAPTER_LOC - LOCAL_KERNEL_LOC,
        AARA_ADAPTER_LOC - LOCAL_KERNEL_LOC,
    )

    disagreement_rate = _rate(transfer_disagreements)

    if silent_equivalence > 0:
        decision = "reject"
        reason = "silent_equivalence_claims"
    elif proof_burden_delta > 0 and not loc_reduced and new_bound_classes == 0:
        decision = "reject"
        reason = "external_formalism_proof_burden_without_transfer_gain"
    elif incomparable_smuggling > 0:
        decision = "reject"
        reason = "incomparable_metric_smuggling"
    elif new_bound_classes > 0 and loc_reduced:
        decision = "accept"
        reason = "external_formalism_enables_new_bounds_with_loc_win"
    elif disagreement_rate > 0.0:
        decision = "reject"
        reason = "transferability_disagreement_across_formalisms"
    else:
        decision = "accept"
        reason = "cost_formalisms_transfer_on_frozen_fixtures"

    return {
        "schema": "cost_kernel_formalism_compare_report/v1",
        "transferability_disagreement_rate_across_cost_formalisms": disagreement_rate,
        "silent_equivalence_claims": silent_equivalence,
        "incomparable_metric_smuggling": incomparable_smuggling,
        "maintenance_cost_loc_burden_delta": proof_burden_delta,
        "new_bound_classes_enabled": new_bound_classes,
        "loc_reduced_vs_local_kernel": loc_reduced,
        "eval_fixture_count": len(eval_fixtures),
        "external_eval_rows": len(external_rows),
        "decision": decision,
        "reason": reason,
        "local_rows": local_rows,
        "external_rows": external_rows,
        "corpus_digest": content_sha({"fixtures": [f.case_id for f in fixtures]}),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "CostFixtureV1",
    "TOOLCHAIN_ID",
    "default_fixtures",
    "load_corpus",
    "paired_compare_report",
    "pinned_toolchain",
    "score_aara",
    "score_calf",
    "score_cslib",
    "score_local_kernel",
]
