"""Normalized data models for multi-farm GPU offers and launches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Offer:
    farm: str
    offer_id: str
    gpu_type: str
    price_per_hr: float
    spot: bool = False
    vram_gb: float | None = None
    availability: str = "unknown"
    raw_ref: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FarmListResult:
    farm: str
    offers: list[Offer] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "farm": self.farm,
            "offers": [o.to_dict() for o in self.offers],
            "error": self.error,
        }


@dataclass
class LaunchResult:
    pod_id: str
    farm: str
    estimated_cost_per_hr: float
    status: str
    ssh_command: str | None = None
    connect_url: str | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FarmCostEstimate:
    farm: str
    price_per_hr: float | None
    hours: int
    compute_cost: float | None
    overhead_multiplier: float
    total_cost: float | None
    gpu_type: str
    available: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
