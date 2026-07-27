"""PartialStateClassV1 (DSH1-07 / SLM-359): COMPLETE / FRAGMENT / PREFIX / INVALID.

Distinguishes complete programs, named valid fragments, and incomplete
prefixes with at least one compiler-proven accepting continuation, so
partial training examples align with states the constrained decoder can
actually encounter (``docs/design/verified-scope-solver.md``).

Torch-free and problem-independent: :func:`classify_partial_state` decides
from facts the caller has already established (which ``OutputKind`` a live
fragment parser accepted, and/or what a :class:`~slm_training.dsl.grammar.fastpath.compiler_draft.CompletionForest`
reports) rather than parsing or building a forest itself. This mirrors the
injected-fact style of :mod:`slm_training.dsl.solver.support`
(``ProblemExpander``/``Verifier``) and keeps this module reusable outside the
OpenUI pack. The OpenUI wiring lives in
:mod:`slm_training.dsl.solver.openui_partial_state`.

No decode-path integration, runtime dependency, experiment, or checkpoint is
added by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

_COVERAGE_VALUES = {"complete", "partial", "none"}


class PartialKind(str, Enum):
    """Which of the four CAP0 partial-state classes a span/prefix occupies."""

    COMPLETE = "complete"
    FRAGMENT = "fragment"
    PREFIX = "prefix"
    INVALID = "invalid"


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"partial state {field} must be non-empty text")
    return value


def _normalize_str_tuple(values: Any, *, field: str) -> tuple[str, ...]:
    normalized = tuple(values)
    for value in normalized:
        if not isinstance(value, str) or not value:
            raise ValueError(f"partial state {field} entries must be non-empty text")
    # Deterministic, de-duplicated, order-independent recording.
    return tuple(sorted(set(normalized)))


def _normalize_position_tuple(values: Any, *, field: str) -> tuple[int, ...]:
    normalized = tuple(values)
    for value in normalized:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"partial state {field} entries must be non-negative integers"
            )
    return tuple(sorted(set(int(value) for value in normalized)))


@dataclass(frozen=True)
class PartialStateClassV1:
    """One classified span/prefix under the DSH1-07 CAP0 partial-state contract.

    * ``COMPLETE`` — a full program; no ``start_symbol``, no open ``frontier``.
    * ``FRAGMENT`` — a named valid fragment; ``start_symbol`` names the grammar
      rule it parsed under (e.g. an ``OutputKind`` such as ``"statement"``).
    * ``PREFIX`` — an incomplete prefix certified by the compiler to have at
      least one drafted, grammar/schema-admissible continuation; ``frontier``
      names the open next-terminal set and ``min_completion_cost`` is the
      shortest drafted continuation length observed among the enumerated
      compiler paths (a lower-bound proxy from ``CompletionForest``, not an
      exhaustively certified shortest path to a verified terminal — that
      stronger claim requires the separate
      :class:`~slm_training.dsl.solver.support.EnumerativeSupportOracle`).
    * ``INVALID`` — none of the above: no accepted kind and no compiler-proven
      continuation (a certified dead end, at the declared bounds).
    """

    kind: PartialKind
    start_symbol: str | None = None
    frontier: tuple[str, ...] = ()
    open_nonterminals: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    singleton_positions: tuple[int, ...] = ()
    support_coverage: str = "none"
    min_completion_cost: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PartialKind):
            raise ValueError(f"partial state kind must be a PartialKind, got {self.kind!r}")
        if self.support_coverage not in _COVERAGE_VALUES:
            raise ValueError(
                "partial state support_coverage must be complete, partial, or none"
            )
        object.__setattr__(
            self, "frontier", _normalize_str_tuple(self.frontier, field="frontier")
        )
        object.__setattr__(
            self,
            "open_nonterminals",
            _normalize_str_tuple(self.open_nonterminals, field="open_nonterminals"),
        )
        object.__setattr__(
            self,
            "required_roles",
            _normalize_str_tuple(self.required_roles, field="required_roles"),
        )
        object.__setattr__(
            self,
            "singleton_positions",
            _normalize_position_tuple(
                self.singleton_positions, field="singleton_positions"
            ),
        )
        if self.min_completion_cost is not None and (
            isinstance(self.min_completion_cost, bool)
            or not isinstance(self.min_completion_cost, int)
            or self.min_completion_cost < 0
        ):
            raise ValueError(
                "partial state min_completion_cost must be a non-negative integer or None"
            )

        if self.kind is PartialKind.COMPLETE:
            if self.start_symbol is not None:
                raise ValueError("a COMPLETE partial state has no start_symbol")
            if self.frontier:
                raise ValueError("a COMPLETE partial state has no open frontier")
        elif self.kind is PartialKind.FRAGMENT:
            _require_text(self.start_symbol, field="start_symbol")
            if self.frontier:
                raise ValueError("a FRAGMENT partial state has no open frontier")
        elif self.kind is PartialKind.PREFIX:
            if self.start_symbol is not None:
                raise ValueError("a PREFIX partial state has no start_symbol")
            if not self.frontier:
                raise ValueError("a PREFIX partial state requires a non-empty frontier")
            if self.support_coverage == "none":
                raise ValueError(
                    "a PREFIX partial state requires complete or partial compiler coverage"
                )
            if self.min_completion_cost is None:
                raise ValueError(
                    "a PREFIX partial state requires a certified minimum completion cost"
                )
        else:  # INVALID
            if self.start_symbol is not None:
                raise ValueError("an INVALID partial state has no start_symbol")
            if self.frontier:
                raise ValueError("an INVALID partial state has no open frontier")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "start_symbol": self.start_symbol,
            "frontier": list(self.frontier),
            "open_nonterminals": list(self.open_nonterminals),
            "required_roles": list(self.required_roles),
            "singleton_positions": list(self.singleton_positions),
            "support_coverage": self.support_coverage,
            "min_completion_cost": self.min_completion_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartialStateClassV1":
        expected = {
            "kind",
            "start_symbol",
            "frontier",
            "open_nonterminals",
            "required_roles",
            "singleton_positions",
            "support_coverage",
            "min_completion_cost",
        }
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise ValueError(
                f"PartialStateClassV1 fields mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(
            kind=PartialKind(data["kind"]),
            start_symbol=data["start_symbol"],
            frontier=tuple(data["frontier"]),
            open_nonterminals=tuple(data["open_nonterminals"]),
            required_roles=tuple(data["required_roles"]),
            singleton_positions=tuple(data["singleton_positions"]),
            support_coverage=data["support_coverage"],
            min_completion_cost=data["min_completion_cost"],
        )


def classify_partial_state(
    *,
    completed_kind: str | None,
    forest_has_paths: bool,
    forest_coverage: str,
    frontier: tuple[str, ...] = (),
    open_nonterminals: tuple[str, ...] = (),
    required_roles: tuple[str, ...] = (),
    singleton_positions: tuple[int, ...] = (),
    min_completion_cost: int | None = None,
    cut_is_lexical_boundary: bool = True,
) -> PartialStateClassV1:
    """Classify one candidate span/prefix under the CAP0 partial-state contract.

    ``completed_kind`` is the grammar start rule (e.g. an ``OutputKind`` such
    as ``"document"``/``"statement"``/``"expression"``) that a live fragment
    parser already accepted for this exact text, decided by the caller so this
    function stays parser-independent. ``"document"`` yields ``COMPLETE``; any
    other accepted kind yields ``FRAGMENT(start_symbol=completed_kind)``. When
    ``completed_kind`` is ``None``, the ``CompletionForest`` facts
    (``forest_has_paths``/``forest_coverage``) decide between ``PREFIX`` (at
    least one drafted, grammar/schema-admissible continuation exists under
    ``complete`` or ``partial`` coverage) and ``INVALID`` (none does).

    ``cut_is_lexical_boundary`` must be ``True`` (checked by the caller, e.g.
    via ``literal_frame_is_open``) — a cut that splits a lexical token or an
    opaque marker is refused outright, never silently reclassified.
    """
    if not cut_is_lexical_boundary:
        raise ValueError(
            "partial-state classification refused: the candidate cut splits a "
            "lexical token or opaque marker"
        )
    if forest_coverage not in _COVERAGE_VALUES:
        raise ValueError("forest_coverage must be complete, partial, or none")

    if completed_kind == "document":
        return PartialStateClassV1(
            kind=PartialKind.COMPLETE,
            support_coverage=forest_coverage,
            min_completion_cost=0,
        )
    if completed_kind is not None:
        return PartialStateClassV1(
            kind=PartialKind.FRAGMENT,
            start_symbol=completed_kind,
            open_nonterminals=open_nonterminals,
            required_roles=required_roles,
            support_coverage=forest_coverage,
            min_completion_cost=0,
        )
    if forest_has_paths and forest_coverage in {"complete", "partial"}:
        if min_completion_cost is None:
            raise ValueError(
                "a PREFIX verdict requires a certified minimum completion cost"
            )
        resolved_frontier = frontier or open_nonterminals
        if not resolved_frontier:
            raise ValueError(
                "a PREFIX verdict requires a non-empty frontier or open_nonterminals"
            )
        return PartialStateClassV1(
            kind=PartialKind.PREFIX,
            frontier=resolved_frontier,
            open_nonterminals=open_nonterminals,
            required_roles=required_roles,
            singleton_positions=singleton_positions,
            support_coverage=forest_coverage,
            min_completion_cost=min_completion_cost,
        )
    return PartialStateClassV1(kind=PartialKind.INVALID, support_coverage=forest_coverage)


__all__ = ["PartialKind", "PartialStateClassV1", "classify_partial_state"]
