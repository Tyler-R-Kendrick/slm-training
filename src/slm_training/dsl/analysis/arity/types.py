"""Immutable data contracts for the arity analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SupportVerdict(Enum):
    """Verdict for a bounded support query."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AnalysisBounds:
    """Declared bounded frame for arity analysis."""

    max_ast_nodes: int
    max_ast_depth: int | None = None
    max_live_bindings: int = 0
    template_classes: tuple[str, ...] = ()
    result_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_ast_nodes < 0:
            raise ValueError("max_ast_nodes must be non-negative")
        if self.max_ast_depth is not None and self.max_ast_depth < 0:
            raise ValueError("max_ast_depth must be non-negative")
        if self.max_live_bindings < 0:
            raise ValueError("max_live_bindings must be non-negative")


@dataclass(frozen=True)
class StateAtom:
    """One canonical atom in a state signature."""

    kind: str
    payload: tuple[Any, ...] = ()

    @staticmethod
    def placeholder(index: int) -> StateAtom:
        return StateAtom("placeholder", (index,))

    @staticmethod
    def literal(value: Any) -> StateAtom:
        return StateAtom("literal", (value,))

    @staticmethod
    def ref(index: int) -> StateAtom:
        return StateAtom("ref", (index,))

    @staticmethod
    def component(type_name: str, props: tuple[tuple[str, StateAtom], ...]) -> StateAtom:
        return StateAtom("component", (type_name, props))

    @staticmethod
    def list(items: tuple[StateAtom, ...]) -> StateAtom:
        return StateAtom("list", items)

    @staticmethod
    def hole(rule: str) -> StateAtom:
        return StateAtom("hole", (rule,))

    def __repr__(self) -> str:
        if self.kind == "component":
            name, props = self.payload
            props_repr = ", ".join(f"{k}={v}" for k, v in props)
            return f"{name}({props_repr})"
        if self.kind == "list":
            return f"[{', '.join(repr(a) for a in self.payload)}]"
        return f"{self.kind}({self.payload[0]!r})"


@dataclass(frozen=True)
class StateSignature:
    """Canonical, hashable state signature under a declared frame."""

    version: str
    generation_order: str
    atoms: tuple[StateAtom, ...]
    atom_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_count", len(self.atoms))

    def fingerprint(self) -> str:
        """Stable string fingerprint for caching and reports."""
        from hashlib import sha256

        return sha256(repr(self.atoms).encode("utf-8"), usedforsecurity=False).hexdigest()[:32]


@dataclass(frozen=True)
class SupportQuery:
    """Query to a support oracle."""

    state_fingerprint: str
    hole_id: str
    candidate: StateAtom


@dataclass(frozen=True)
class SupportResult:
    """Result from a support oracle."""

    verdict: SupportVerdict
    certificate: dict[str, Any]
    witness: str | None = None


class SupportOracle(Protocol):
    """Protocol for bounded support decisions."""

    def check(self, state: StateSignature, query: SupportQuery) -> SupportResult:
        ...
