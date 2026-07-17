"""Torch-free finite-domain support lattice for the verified scope solver.

Implements the state contract in ``docs/design/verified-scope-solver.md``
(VSS0-01 / SLM-57), required by VSS0-03 (SLM-59). This package is
model-independent: importing it pulls in no ``torch`` and performs no model
inference. The compiler-forest adapter is loaded lazily via ``__getattr__`` so
``import slm_training.dsl.solver`` stays light and never eagerly imports the
grammar/compiler machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainProjection,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    JsonScalar,
    SolverBounds,
    SupportVerdict,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_training.dsl.solver.adapters import (
        CompletionForestProjection,
        completion_forest_state,
    )

_LAZY_EXPORTS = {"CompletionForestProjection", "completion_forest_state"}

__all__ = [
    "CompletionForestProjection",
    "DomainValue",
    "FiniteDomainProjection",
    "FiniteDomainState",
    "HoleDomain",
    "HoleId",
    "JsonScalar",
    "SolverBounds",
    "SupportVerdict",
    "completion_forest_state",
]


def __getattr__(name: str) -> Any:
    """Load the compiler-forest adapter only when explicitly requested."""
    if name in _LAZY_EXPORTS:
        from slm_training.dsl.solver import adapters

        return getattr(adapters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
