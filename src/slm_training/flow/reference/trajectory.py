"""Replay/trajectory schemas for exact CTMC reference samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FlowTrajectoryV1:
    """One sampled CTMC path through compiler-certified legal edits."""

    schema: str = "FlowTrajectoryV1"
    trajectory_id: str = ""
    source_fingerprint: str = ""
    terminal_fingerprint: str = ""
    states: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    holding_times: tuple[float, ...] = ()
    wall_times: tuple[float, ...] = ()
    certificates: tuple[str, ...] = ()
    total_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trajectory_id": self.trajectory_id,
            "source_fingerprint": self.source_fingerprint,
            "terminal_fingerprint": self.terminal_fingerprint,
            "states": list(self.states),
            "actions": list(self.actions),
            "holding_times": list(self.holding_times),
            "wall_times": list(self.wall_times),
            "certificates": list(self.certificates),
            "total_time": self.total_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowTrajectoryV1":
        return cls(
            schema=str(data.get("schema", "FlowTrajectoryV1")),
            trajectory_id=str(data.get("trajectory_id", "")),
            source_fingerprint=str(data.get("source_fingerprint", "")),
            terminal_fingerprint=str(data.get("terminal_fingerprint", "")),
            states=tuple(data.get("states", ())),
            actions=tuple(data.get("actions", ())),
            holding_times=tuple(data.get("holding_times", ())),
            wall_times=tuple(data.get("wall_times", ())),
            certificates=tuple(data.get("certificates", ())),
            total_time=float(data.get("total_time", 0.0)),
        )


@dataclass(frozen=True)
class FlowSampleV1:
    """Empirical distribution produced by repeated CTMC sampling."""

    schema: str = "FlowSampleV1"
    source_fingerprint: str = ""
    n_samples: int = 0
    empirical_terminal_distribution: dict[str, float] = field(default_factory=dict)
    trajectories: tuple[FlowTrajectoryV1, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_fingerprint": self.source_fingerprint,
            "n_samples": self.n_samples,
            "empirical_terminal_distribution": dict(self.empirical_terminal_distribution),
            "trajectories": [t.to_dict() for t in self.trajectories],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowSampleV1":
        return cls(
            schema=str(data.get("schema", "FlowSampleV1")),
            source_fingerprint=str(data.get("source_fingerprint", "")),
            n_samples=int(data.get("n_samples", 0)),
            empirical_terminal_distribution=dict(data.get("empirical_terminal_distribution", {})),
            trajectories=tuple(
                FlowTrajectoryV1.from_dict(t) for t in data.get("trajectories", ())
            ),
        )
