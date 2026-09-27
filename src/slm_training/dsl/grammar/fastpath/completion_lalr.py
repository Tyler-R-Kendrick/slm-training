"""Canonical LALR control adapter and sound acceptance-length bounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_UNREACHABLE_MIN_TERMINALS = -1


def _control_rows(
    tensors: dict[str, list[int]],
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """Map canonical state color -> ``(symbol_index, action_kind, target)`` rows."""

    offsets = tensors["lalr_state_offsets"]
    rows: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for index, color in enumerate(tensors["lalr_state_colors"]):
        row = tuple(
            (
                tensors["lalr_edge_symbols"][slot],
                tensors["lalr_action_kinds"][slot],
                tensors["lalr_action_targets"][slot],
            )
            for slot in range(offsets[index], offsets[index + 1])
        )
        previous = rows.setdefault(int(color), row)
        if previous != row:
            raise ValueError("canonical color collision with differing control rows")
    return rows


def _state_min_terminals(
    tensors: dict[str, list[int]], metadata: dict[str, Any]
) -> list[int]:
    """Sound LOWER bound on terminals still needed before the parse can accept.

    Acceptance is a property of the whole stack, not of the top state, so an
    exact per-state table does not exist.  This computes a *relaxation*: a
    reduction may expose any state that carries a goto on the reduced rule's
    origin, not only the state the real stack would expose.  The relaxed
    transition relation is a superset of every real one and reductions consume
    no terminals, therefore the fixpoint is ``<=`` the true stack-dependent
    minimum for every stack topped by that state — a lower bound, usable only
    for negative-direction reasoning ("fewer than this many tokens cannot
    finish").  Production uses this bound only to prune impossible completions.
    """

    rows = _control_rows(tensors)
    symbols = list(metadata["symbols"])
    rule_origins = tensors["rule_origins"]
    end_state = int(metadata["end_states"]["start"])

    goto_targets: dict[int, set[int]] = {}
    for row in rows.values():
        for symbol, kind, target in row:
            if kind == 0 and not symbols[symbol].isupper():
                goto_targets.setdefault(symbol, set()).add(int(target))

    infinity = len(rows) + 1
    cost = {color: infinity for color in rows}
    cost[end_state] = 0
    for _ in range(len(rows) + 2):
        changed = False
        for color, row in rows.items():
            best = cost[color]
            for symbol, kind, target in row:
                name = symbols[symbol]
                if kind == 0:
                    if not name.isupper() or name == "$END":
                        continue  # goto edges are traversed through reductions
                    candidate = cost[int(target)] + 1
                else:
                    origin = rule_origins[int(target)]
                    candidate = min(
                        (cost[state] for state in goto_targets.get(origin, ())),
                        default=infinity,
                    )
                best = min(best, candidate)
            changed |= best < cost[color]
            cost[color] = best
        if not changed:
            break
    else:  # pragma: no cover - monotone fixpoint converges in <= state count
        raise RuntimeError("acceptance-length fixpoint did not converge")

    return [
        (
            _UNREACHABLE_MIN_TERMINALS
            if cost[int(color)] >= infinity
            else int(cost[int(color)])
        )
        for color in tensors["lalr_state_colors"]
    ]


@dataclass(frozen=True)
class StaticLalrAdapter:
    """Executable shift/reduce over the certified control arrays.

    This is a *checked* adapter, never an authority: it is usable only after a
    lockstep certification against the live Lark ``InteractiveParser`` passes
    (see ``static_control_domain.require_certified_static_lalr``).  The decode
    path may consume ``min_terminals`` as a **negative-direction** prune inside
    ``completion_kernel.terminal_witness`` (reject when room is strictly below
    the certified lower bound); Lark remains the live parser authority.
    ``$END`` follows Lark's end handling: the reduction chain runs until the
    end state is exposed.
    """

    symbols: tuple[str, ...]
    rows: dict[int, tuple[tuple[int, int, int], ...]]
    rule_origins: tuple[int, ...]
    rule_lengths: tuple[int, ...]
    start_state: int
    end_state: int
    state_min_terminals: tuple[int, ...]
    live_state_colors: dict[int, int] = field(
        default_factory=dict, repr=False, compare=False
    )

    END_SYMBOL = "$END"

    @classmethod
    def from_arrays(
        cls,
        tensors: dict[str, list[int]],
        metadata: dict[str, Any],
        *,
        start: str = "start",
        live_state_colors: dict[int, int] | None = None,
    ) -> StaticLalrAdapter:
        rows = _control_rows(tensors)
        colors = tensors["lalr_state_colors"]
        minimums = tensors["state_min_terminals"]
        by_color: dict[int, int] = {}
        for color, value in zip(colors, minimums, strict=True):
            if by_color.setdefault(int(color), int(value)) != int(value):
                raise ValueError("state_min_terminals disagrees within a color")
        if sorted(by_color) != list(range(len(by_color))):
            raise ValueError("canonical state colors are not densely numbered")
        return cls(
            symbols=tuple(str(name) for name in metadata["symbols"]),
            rows=rows,
            rule_origins=tuple(int(value) for value in tensors["rule_origins"]),
            rule_lengths=tuple(int(value) for value in tensors["rule_lengths"]),
            start_state=int(metadata["start_states"][start]),
            end_state=int(metadata["end_states"][start]),
            state_min_terminals=tuple(by_color[color] for color in sorted(by_color)),
            live_state_colors=dict(live_state_colors or {}),
        )

    def start_stack(self) -> tuple[int, ...]:
        return (int(self.start_state),)

    def _action(self, state: int, terminal: str) -> tuple[int, int] | None:
        for symbol, kind, target in self.rows[int(state)]:
            if self.symbols[symbol] == terminal:
                return int(kind), int(target)
        return None

    def step(
        self, stack: tuple[int, ...], terminal_name: str
    ) -> tuple[int, ...] | None:
        """Advance the stack over one terminal, or ``None`` when rejected."""

        working = [int(state) for state in stack]
        if not working:
            return None
        is_end = terminal_name == self.END_SYMBOL
        for _ in range(len(self.rows) * len(self.rule_lengths) + 8):
            action = self._action(working[-1], terminal_name)
            if action is None:
                return None
            kind, target = action
            if kind == 0:  # Shift
                # Lark never shifts $END; never invent acceptance for it.
                return None if is_end else (*working, target)
            if not 0 <= target < len(self.rule_lengths):
                raise ValueError(f"reduce action targets unknown rule id {target}")
            size = self.rule_lengths[target]
            if size:
                if len(working) <= size:
                    return None
                del working[-size:]
            goto = self._action(working[-1], self.symbols[self.rule_origins[target]])
            if goto is None or goto[0] != 0:
                return None
            working.append(goto[1])
            if is_end and working[-1] == self.end_state:
                return tuple(working)
        raise RuntimeError("static LALR adapter reduction chain did not terminate")

    def accepts(self, stack: tuple[int, ...]) -> frozenset[str]:
        """Terminals the control table actually advances on from ``stack``.

        Mirrors ``InteractiveParser.accepts``: a terminal counts only when its
        whole reduction chain succeeds, not merely when the top state's row
        carries an entry for it.
        """

        names = {
            self.symbols[symbol] for symbol, _kind, _target in self.rows[stack[-1]]
        }
        return frozenset(
            name
            for name in names
            if name.isupper() and self.step(stack, name) is not None
        )

    def min_terminals(self, state: int) -> int:
        """Lower bound on remaining terminals; ``-1`` when no bound is known."""

        return int(self.state_min_terminals[int(state)])

    def min_terminals_for_live_state(self, state: int) -> int:
        """Translate a live Lark state ID before reading canonical bounds."""

        color = self.live_state_colors.get(int(state))
        return -1 if color is None else self.min_terminals(color)


