"""Rate targets shared by exact finite graphs and OpenUI legal-edit bridges."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from slm_training.data.flow.bridge_corpus import LegalEditBridgeRowV1
from slm_training.flow.reference.row import FlowTargetRowV1

TargetFidelity = Literal[
    "exact_finite_graph",
    "exact_bridge_dag",
    "adapted_path_approximation",
    "surrogate_rate_weight",
]


@dataclass(frozen=True)
class LegalEditRateTargetV1:
    """One nonnegative rate target over an exact live candidate set."""

    row_id: str
    candidate_ids: tuple[str, ...]
    edge_rates: tuple[float, ...]
    total_hazard: float
    positive_candidate_ids: tuple[str, ...]
    supervised_candidate_ids: tuple[str, ...]
    hazard_supervised: bool
    terminal_probability: float
    time: float
    fidelity: TargetFidelity
    schema: str = "LegalEditRateTargetV1"

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != len(self.edge_rates):
            raise ValueError("candidate and edge-rate counts differ")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("duplicate candidate ID")
        if any(not math.isfinite(rate) or rate < 0.0 for rate in self.edge_rates):
            raise ValueError("edge rates must be nonnegative")
        if not math.isfinite(self.total_hazard):
            raise ValueError("total hazard must be finite")
        if abs(sum(self.edge_rates) - self.total_hazard) > 1e-6:
            raise ValueError("total hazard must equal the live edge-rate sum")
        if self.candidate_ids and self.total_hazard <= 0.0:
            raise ValueError("a live candidate set requires positive total hazard")
        if not set(self.positive_candidate_ids) <= set(self.candidate_ids):
            raise ValueError("positive candidate is absent from the live set")
        if len(set(self.positive_candidate_ids)) != len(self.positive_candidate_ids):
            raise ValueError("duplicate positive candidate")
        if len(set(self.supervised_candidate_ids)) != len(
            self.supervised_candidate_ids
        ):
            raise ValueError("duplicate supervised candidate")
        if not set(self.supervised_candidate_ids) <= set(self.candidate_ids):
            raise ValueError("supervised candidate is absent from the live set")
        if not set(self.positive_candidate_ids) <= set(self.supervised_candidate_ids):
            raise ValueError("positive candidates must be supervised")
        positive_rate_ids = {
            candidate
            for candidate, rate in zip(
                self.candidate_ids, self.edge_rates, strict=True
            )
            if rate > 0.0 and candidate in self.supervised_candidate_ids
        }
        if positive_rate_ids != set(self.positive_candidate_ids):
            raise ValueError("positive IDs must match supervised positive-rate edges")
        if not 0.0 <= self.terminal_probability <= 1.0:
            raise ValueError("terminal probability must lie in [0, 1]")
        if not math.isfinite(self.time) or self.time < 0.0:
            raise ValueError("time must be nonnegative")

    def rate_dict(self) -> dict[str, float]:
        return dict(zip(self.candidate_ids, self.edge_rates, strict=True))


def from_exact_rows(
    rows: Sequence[FlowTargetRowV1],
) -> tuple[LegalEditRateTargetV1, ...]:
    """Adapt exact SLM-190 rows without changing their rates or membership."""
    targets: list[LegalEditRateTargetV1] = []
    for row in rows:
        candidate_ids = tuple(row.exact_live_candidates)
        if set(candidate_ids) != set(row.target_rates):
            raise ValueError("exact row rate keys do not match live candidates")
        targets.append(
            LegalEditRateTargetV1(
                row_id=row.row_id,
                candidate_ids=candidate_ids,
                edge_rates=tuple(float(row.target_rates[item]) for item in candidate_ids),
                total_hazard=float(row.total_hazard),
                positive_candidate_ids=tuple(
                    item for item in candidate_ids if row.target_rates[item] > 0.0
                ),
                supervised_candidate_ids=candidate_ids,
                hazard_supervised=True,
                terminal_probability=float(not candidate_ids),
                time=float(row.time),
                fidelity="exact_finite_graph",
            )
        )
    return tuple(targets)


def from_bridge_rows(
    rows: Sequence[LegalEditBridgeRowV1],
    *,
    fidelity: TargetFidelity = "adapted_path_approximation",
) -> tuple[LegalEditRateTargetV1, ...]:
    """Build explicit surrogate rates from exhaustive bridge candidate sets.

    The bridge corpus certifies candidate membership and target-consistent
    positives, but it does not identify physical CTMC holding times. Therefore
    the default is intentionally labelled an adapted path approximation.
    """
    if fidelity not in {"adapted_path_approximation", "surrogate_rate_weight"}:
        raise ValueError(
            "bridge rows lack holding-time evidence for an exact fidelity label"
        )
    targets: list[LegalEditRateTargetV1] = []
    for row in rows:
        candidate_ids = tuple(sorted(row.complete_candidate_ids))
        positives = tuple(sorted(row.positive_candidate_ids))
        if not positives:
            raise ValueError("a nonterminal bridge row needs a positive edit")
        # Unit total hazard keeps rate and normalized-policy supervision
        # distinguishable without inventing unobserved holding-time evidence.
        rate = 1.0 / len(positives)
        rate_by_id = {item: rate for item in positives}
        targets.append(
            LegalEditRateTargetV1(
                row_id=row.row_id,
                candidate_ids=candidate_ids,
                edge_rates=tuple(rate_by_id.get(item, 0.0) for item in candidate_ids),
                total_hazard=1.0,
                positive_candidate_ids=positives,
                supervised_candidate_ids=tuple(
                    sorted(
                        set(row.supported_candidate_ids)
                        | set(row.unsupported_candidate_ids)
                    )
                ),
                hazard_supervised=False,
                terminal_probability=0.0,
                time=min(1.0, row.step_index / max(1, row.bridge_length)),
                fidelity=fidelity,
            )
        )
    return tuple(targets)
