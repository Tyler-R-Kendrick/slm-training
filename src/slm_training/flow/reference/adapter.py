"""Graph/state adapter protocol for exact finite-state CTMC reference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class StateRef:
    """A hashable, JSON-serializable reference to a CTMC state.

    The reference wraps an arbitrary immutable value and exposes a stable
    string fingerprint.  Equality and hashing are delegated to the wrapped
    value so the same logical state produced by different adapters or
    enumeration passes is comparable.
    """

    __slots__ = ("value", "fingerprint")

    def __init__(self, value: Any, fingerprint: str) -> None:
        self.value = value
        self.fingerprint = fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateRef):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"StateRef({self.fingerprint!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint, "value": self.value}


class ActionRef:
    """A hashable, JSON-serializable reference to a legal action/edit."""

    __slots__ = ("value", "action_id")

    def __init__(self, value: Any, action_id: str) -> None:
        self.value = value
        self.action_id = action_id

    def __hash__(self) -> int:
        return hash(self.action_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionRef):
            return NotImplemented
        return self.action_id == other.action_id

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"ActionRef({self.action_id!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "value": self.value}


class StateAdapter(ABC):
    """Abstract adapter that makes a domain enumerable for exact CTMC."""

    @property
    @abstractmethod
    def domain_id(self) -> str:
        """Stable domain identifier, e.g. ``toy_layout_01``."""

    @abstractmethod
    def initial_states(self) -> list[StateRef]:
        """Return the source/starting states for the domain."""

    @abstractmethod
    def is_terminal(self, state: StateRef) -> bool:
        """Whether ``state`` is a canonical terminal or endpoint class."""

    @abstractmethod
    def terminal_class(self, state: StateRef) -> str:
        """Return a terminal class fingerprint, or the state fingerprint if none."""

    @abstractmethod
    def legal_actions(self, state: StateRef) -> list[ActionRef]:
        """Return all legal actions available in ``state``."""

    @abstractmethod
    def apply(
        self, state: StateRef, action: ActionRef
    ) -> StateRef | None:
        """Apply ``action`` to ``state`` and return the successor, or None."""

    @abstractmethod
    def state_fingerprint(self, state_value: Any) -> str:
        """Stable fingerprint of a raw state value."""

    def all_states(self, max_states: int = 10_000) -> list[StateRef]:
        """Default exhaustive enumeration using BFS over legal actions."""
        visited: dict[str, StateRef] = {}
        frontier = list(self.initial_states())
        for state in frontier:
            visited[state.fingerprint] = state
        while frontier and len(visited) < max_states:
            current = frontier.pop(0)
            for action in self.legal_actions(current):
                nxt = self.apply(current, action)
                if nxt is None:
                    continue
                if nxt.fingerprint not in visited:
                    visited[nxt.fingerprint] = nxt
                    frontier.append(nxt)
        return list(visited.values())


class RateFn(Protocol):
    """Callable that assigns a non-negative rate to a legal transition."""

    def __call__(
        self,
        source: StateRef,
        action: ActionRef,
        target: StateRef,
        graph: Any,
    ) -> float:
        ...
