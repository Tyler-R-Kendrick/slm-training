"""Finite-domain solver state and adapters."""

from __future__ import annotations

from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleDomain, HoleId
from slm_training.dsl.solver.topology_adapter import (
    TopologyAction,
    TopologyAdapterConfig,
    TopologyEdit,
    TopologyHole,
    derive_topology_holes,
    legal_topology_productions,
)

__all__ = [
    "DomainValue",
    "FiniteDomainState",
    "HoleDomain",
    "HoleId",
    "TopologyAction",
    "TopologyAdapterConfig",
    "TopologyEdit",
    "TopologyHole",
    "derive_topology_holes",
    "legal_topology_productions",
]
