"""Torch-free finite-domain support-lattice primitives."""

from slm_training.dsl.solver.adapters import (
    TopologyDomainAdapter,
    completion_forest_state,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    JsonScalar,
    SolverBounds,
    SupportVerdict,
)

__all__ = [
    "DomainValue",
    "FiniteDomainState",
    "HoleDomain",
    "HoleId",
    "JsonScalar",
    "SolverBounds",
    "SupportVerdict",
    "TopologyDomainAdapter",
    "completion_forest_state",
]
