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
from slm_training.dsl.solver.support import (
    EnumerativeSupportOracle,
    ExpandStatus,
    ExpandStep,
    ProblemExpander,
    ReplayResult,
    SearchCounters,
    SupportCertificate,
    SupportOracle,
    SupportQuery,
    SupportResult,
    VerifyOutcome,
    VerifyStatus,
    Verifier,
    replay_support_certificate,
)

__all__ = [
    "DomainValue",
    "EnumerativeSupportOracle",
    "ExpandStatus",
    "ExpandStep",
    "FiniteDomainState",
    "HoleDomain",
    "HoleId",
    "JsonScalar",
    "ProblemExpander",
    "ReplayResult",
    "SearchCounters",
    "SolverBounds",
    "SupportCertificate",
    "SupportOracle",
    "SupportQuery",
    "SupportResult",
    "SupportVerdict",
    "TopologyDomainAdapter",
    "Verifier",
    "VerifyOutcome",
    "VerifyStatus",
    "completion_forest_state",
    "replay_support_certificate",
]
