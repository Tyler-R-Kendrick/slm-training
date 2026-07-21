"""FlowTargetRowV1: exact target row consumable by production loss interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FlowTargetRowV1:
    """One target-rate row for an exact CTMC reference trajectory.

    Mirrors the production target interface so that approximate loss
    functions can consume reference rows without a separate code path.
    """

    schema: str = "FlowTargetRowV1"
    row_id: str = ""
    source_fingerprint: str = ""
    target_fingerprint: str = ""
    time: float = 0.0
    state_fingerprint: str = ""
    exact_live_candidates: tuple[str, ...] = ()
    target_rates: dict[str, float] = field(default_factory=dict)
    total_hazard: float = 0.0
    next_state_fingerprints: tuple[str, ...] = ()
    endpoint_class: str = ""
    planner_version: str = "ffe2-02-v1"
    coupling_version: str = "exact_ctmc_v1"
    certificate_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "time": self.time,
            "state_fingerprint": self.state_fingerprint,
            "exact_live_candidates": list(self.exact_live_candidates),
            "target_rates": dict(self.target_rates),
            "total_hazard": self.total_hazard,
            "next_state_fingerprints": list(self.next_state_fingerprints),
            "endpoint_class": self.endpoint_class,
            "planner_version": self.planner_version,
            "coupling_version": self.coupling_version,
            "certificate_ids": list(self.certificate_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowTargetRowV1":
        return cls(
            schema=str(data.get("schema", "FlowTargetRowV1")),
            row_id=str(data.get("row_id", "")),
            source_fingerprint=str(data.get("source_fingerprint", "")),
            target_fingerprint=str(data.get("target_fingerprint", "")),
            time=float(data.get("time", 0.0)),
            state_fingerprint=str(data.get("state_fingerprint", "")),
            exact_live_candidates=tuple(data.get("exact_live_candidates", ())),
            target_rates=dict(data.get("target_rates", {})),
            total_hazard=float(data.get("total_hazard", 0.0)),
            next_state_fingerprints=tuple(data.get("next_state_fingerprints", ())),
            endpoint_class=str(data.get("endpoint_class", "")),
            planner_version=str(data.get("planner_version", "ffe2-02-v1")),
            coupling_version=str(data.get("coupling_version", "exact_ctmc_v1")),
            certificate_ids=tuple(data.get("certificate_ids", ())),
        )

    def normalized_next_edit_probs(self) -> dict[str, float]:
        """Return normalized next-edit probabilities (objective comparison #1)."""
        total = sum(self.target_rates.values())
        if total <= 0.0:
            return {}
        return {k: v / total for k, v in self.target_rates.items()}

    def edge_rate_dict(self) -> dict[str, float]:
        """Return direct edge rates (objective comparison #3)."""
        return dict(self.target_rates)

    def posterior_parameterization(self) -> dict[str, Any]:
        """Return denoising/posterior parameterization (objective comparison #4)."""
        return {
            "state_fingerprint": self.state_fingerprint,
            "time": self.time,
            "total_hazard": self.total_hazard,
            "normalized_next_edit": self.normalized_next_edit_probs(),
            "endpoint_class": self.endpoint_class,
        }
