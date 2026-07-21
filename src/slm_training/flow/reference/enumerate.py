"""Exact state-graph enumeration for finite CTMC domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slm_training.flow.reference.adapter import ActionRef, StateAdapter, StateRef


@dataclass(frozen=True)
class Transition:
    source: StateRef
    action: ActionRef
    target: StateRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fingerprint": self.source.fingerprint,
            "action_id": self.action.action_id,
            "target_fingerprint": self.target.fingerprint,
        }


@dataclass(frozen=True)
class StateGraph:
    """Finite, directed, multi-edge state graph produced by exact enumeration."""

    domain_id: str
    states: tuple[StateRef, ...]
    transitions: tuple[Transition, ...]
    initial_states: tuple[StateRef, ...]
    terminal_states: tuple[StateRef, ...]
    state_index: dict[str, int] = field(repr=False, compare=False)
    outgoing: dict[int, list[tuple[int, ActionRef]]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.state_index:
            object.__setattr__(
                self,
                "state_index",
                {s.fingerprint: i for i, s in enumerate(self.states)},
            )
        if not self.outgoing:
            out: dict[int, list[tuple[int, ActionRef]]] = {}
            for t in self.transitions:
                src = self.state_index[t.source.fingerprint]
                tgt = self.state_index[t.target.fingerprint]
                out.setdefault(src, []).append((tgt, t.action))
            object.__setattr__(self, "outgoing", out)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "n_states": self.n_states,
            "n_transitions": self.n_transitions,
            "states": [s.to_dict() for s in self.states],
            "transitions": [t.to_dict() for t in self.transitions],
            "initial_states": [s.fingerprint for s in self.initial_states],
            "terminal_states": [s.fingerprint for s in self.terminal_states],
        }


class ExactEnumerator:
    """Enumerate all reachable states and legal transitions for an adapter."""

    def __init__(self, adapter: StateAdapter, max_states: int = 10_000) -> None:
        self.adapter = adapter
        self.max_states = max_states

    def enumerate(self) -> StateGraph:
        """BFS over legal actions until the state space is closed or bounded."""
        seen: dict[str, StateRef] = {}
        transitions: list[Transition] = []
        frontier = list(self.adapter.initial_states())
        for s in frontier:
            seen[s.fingerprint] = s

        while frontier and len(seen) < self.max_states:
            current = frontier.pop(0)
            for action in self.adapter.legal_actions(current):
                nxt = self.adapter.apply(current, action)
                if nxt is None:
                    continue
                transitions.append(Transition(current, action, nxt))
                if nxt.fingerprint not in seen:
                    seen[nxt.fingerprint] = nxt
                    frontier.append(nxt)

        states = tuple(seen.values())
        terminal = tuple(s for s in states if self.adapter.is_terminal(s))
        return StateGraph(
            domain_id=self.adapter.domain_id,
            states=states,
            transitions=tuple(transitions),
            initial_states=tuple(self.adapter.initial_states()),
            terminal_states=terminal,
            state_index={s.fingerprint: i for i, s in enumerate(states)},
            outgoing={},
        )

    def enumerate_from(self, source: StateRef) -> StateGraph:
        """Enumerate the reachable subgraph starting from a single source."""

        class SourceAdapter(StateAdapter):
            def __init__(self, base: StateAdapter, src: StateRef) -> None:
                self._base = base
                self._src = src

            @property
            def domain_id(self) -> str:
                return f"{self._base.domain_id}__from_{self._src.fingerprint[:16]}"

            def initial_states(self) -> list[StateRef]:
                return [self._src]

            def is_terminal(self, state: StateRef) -> bool:
                return self._base.is_terminal(state)

            def terminal_class(self, state: StateRef) -> str:
                return self._base.terminal_class(state)

            def legal_actions(self, state: StateRef) -> list[ActionRef]:
                return self._base.legal_actions(state)

            def apply(
                self, state: StateRef, action: ActionRef
            ) -> StateRef | None:
                return self._base.apply(state, action)

            def state_fingerprint(self, state_value: Any) -> str:
                return self._base.state_fingerprint(state_value)

        return ExactEnumerator(SourceAdapter(self.adapter, source), self.max_states).enumerate()
