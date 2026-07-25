"""State/branch/request-bound exact selector builder and resolver (DSH5-01).

``SelectorRefV1`` lets one model-visible argument denote a bounded, exact,
finite set of AST targets without a free-form query language and without
weakening the DSH3-03 typed-reference / permutation invariants. This module
never accepts a raw predicate string: a selector's semantics are always one
of the closed :class:`SelectorKind` predicate classes declared in
``operators.references``, and its exact target set is always validated
against entries already bound to the owning :class:`ReferenceTableV1` — never
an unchecked, gold-derived, or externally supplied fingerprint list.

Two building blocks:

* :func:`build_selector_descriptor` / :func:`build_selector` /
  :func:`attach_selector` commit to one exact finite target set and allocate
  its opaque ``SelectorRef`` through the same permutation-safe reference-table
  path used for every other reference kind.
* :func:`build_selector_from_scope` adds a bounded scan budget over the
  table's own entries; a truncated scan is always ``UNKNOWN`` and carries no
  descriptor, so it can never be forced into a committed selector.
* :class:`SelectorContextV1` is the only way to turn a resolved selector back
  into its compiler-private target references, and always returns a fresh,
  independent tuple rather than a live view into table storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from slm_training.dsl.operators.contracts import (
    OperatorRef,
    SelectorRef,
    _fingerprint,
)
from slm_training.dsl.operators.references import (
    ReferenceEntryV1,
    ReferenceResolutionError,
    ReferenceTableV1,
    SelectorDescriptorV1,
    SelectorFact,
    SelectorKind,
    attach_selectors,
)


class SelectorBuildVerdict(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SelectorBuildResultV1:
    """Outcome of one bounded selector build attempt.

    ``UNKNOWN`` is the only truncation outcome: it carries no descriptor, so
    a caller cannot force a partially scanned candidate set into a committed
    ``SelectorRef``. A ``RESOLVED`` result always covered every candidate.
    """

    verdict: SelectorBuildVerdict
    evaluated_candidates: int
    total_candidates: int
    descriptor: SelectorDescriptorV1 | None = None

    def __post_init__(self) -> None:
        if self.evaluated_candidates < 0 or self.total_candidates < 0:
            raise ValueError("selector build candidate counts must be non-negative")
        if self.evaluated_candidates > self.total_candidates:
            raise ValueError(
                "selector build evaluated candidates exceed the declared total"
            )
        if self.verdict is SelectorBuildVerdict.RESOLVED:
            if self.descriptor is None:
                raise ValueError("a resolved selector build requires a descriptor")
            if self.evaluated_candidates != self.total_candidates:
                raise ValueError(
                    "a resolved selector build requires complete candidate coverage"
                )
        elif self.descriptor is not None:
            raise ValueError("an unknown selector build cannot carry a descriptor")


def _semantic_fingerprint(
    selector_kind: SelectorKind,
    scope_fingerprint: str,
    target_fingerprints: tuple[str, ...],
) -> str:
    return _fingerprint(
        {
            "schema": "selector_semantic/v1",
            "selector_kind": selector_kind.value,
            "scope_fingerprint": scope_fingerprint,
            "target_fingerprints": list(target_fingerprints),
        }
    )


def build_selector_descriptor(
    *,
    table: ReferenceTableV1,
    selector_kind: SelectorKind,
    scope_fingerprint: str,
    matching_refs: tuple[OperatorRef, ...],
    max_fanout: int,
    compiler_facts: tuple[SelectorFact, ...] = (),
) -> SelectorDescriptorV1:
    """Commit to one exact finite target set drawn from ``table.entries``.

    Every ref in ``matching_refs`` must already be a member of ``table`` —
    this is the structural guard against gold-derived or otherwise invented
    selectors. Input order carries no semantics: targets are re-sorted by
    descriptor fingerprint before the semantic fingerprint is computed.
    """
    if len(set(matching_refs)) != len(matching_refs):
        raise ReferenceResolutionError("selector.duplicate_target")
    by_ref = {entry.ref: entry.descriptor.fingerprint for entry in table.entries}
    try:
        target_fingerprints = tuple(
            sorted(by_ref[ref] for ref in matching_refs)
        )
    except KeyError as exc:
        raise ReferenceResolutionError("selector.member_missing") from exc
    return SelectorDescriptorV1(
        selector_kind=selector_kind,
        semantic_fingerprint=_semantic_fingerprint(
            selector_kind, scope_fingerprint, target_fingerprints
        ),
        scope_fingerprint=scope_fingerprint,
        target_fingerprints=target_fingerprints,
        cardinality=len(target_fingerprints),
        max_fanout=max_fanout,
        compiler_facts=compiler_facts,
    )


def attach_selector(
    table: ReferenceTableV1, descriptor: SelectorDescriptorV1, *, seed: int
) -> ReferenceTableV1:
    """Add one selector to ``table`` without disturbing any existing ref.

    Thin wrapper over :func:`references.attach_selectors` — the same
    permutation-safe allocation formula every other reference kind goes
    through — so a caller who already holds node/role/... refs on ``table``
    keeps resolving them after a selector is added.
    """
    return attach_selectors(table, (descriptor,), seed=seed)


def build_selector(
    table: ReferenceTableV1,
    *,
    selector_kind: SelectorKind,
    scope_fingerprint: str,
    matching_refs: tuple[OperatorRef, ...],
    max_fanout: int,
    seed: int,
    compiler_facts: tuple[SelectorFact, ...] = (),
) -> tuple[ReferenceTableV1, SelectorRef]:
    """Build one exact selector descriptor and allocate it in one call."""
    descriptor = build_selector_descriptor(
        table=table,
        selector_kind=selector_kind,
        scope_fingerprint=scope_fingerprint,
        matching_refs=matching_refs,
        max_fanout=max_fanout,
        compiler_facts=compiler_facts,
    )
    new_table = attach_selector(table, descriptor, seed=seed)
    ref = next(
        entry.ref for entry in new_table.selectors if entry.descriptor == descriptor
    )
    return new_table, ref


def build_selector_from_scope(
    table: ReferenceTableV1,
    *,
    selector_kind: SelectorKind,
    scope_fingerprint: str,
    predicate: Callable[[ReferenceEntryV1], bool],
    max_fanout: int,
    max_candidates_scanned: int,
    compiler_facts: tuple[SelectorFact, ...] = (),
) -> SelectorBuildResultV1:
    """Scan a bounded, canonically ordered prefix of ``table.entries``.

    ``predicate`` classifies already compiler-owned, table-bound entries; it
    is never a free-form query string. If the budget truncates the scan
    before every entry has been checked, the result is ``UNKNOWN`` and
    carries no descriptor — a truncated scan can never be forced into a
    committed selector.
    """
    if max_candidates_scanned < 0:
        raise ValueError("max_candidates_scanned must be non-negative")
    ordered = tuple(
        sorted(table.entries, key=lambda entry: entry.descriptor.fingerprint)
    )
    total = len(ordered)
    scanned = ordered[:max_candidates_scanned]
    evaluated = len(scanned)
    if evaluated < total:
        return SelectorBuildResultV1(
            verdict=SelectorBuildVerdict.UNKNOWN,
            evaluated_candidates=evaluated,
            total_candidates=total,
        )
    matching_refs = tuple(entry.ref for entry in scanned if predicate(entry))
    descriptor = build_selector_descriptor(
        table=table,
        selector_kind=selector_kind,
        scope_fingerprint=scope_fingerprint,
        matching_refs=matching_refs,
        max_fanout=max_fanout,
        compiler_facts=compiler_facts,
    )
    return SelectorBuildResultV1(
        verdict=SelectorBuildVerdict.RESOLVED,
        evaluated_candidates=evaluated,
        total_candidates=total,
        descriptor=descriptor,
    )


@dataclass(frozen=True)
class SelectorContextV1:
    """Read-only resolver from a validated selector membership to target refs.

    Compiler-private target payloads are exposed only through
    :meth:`resolve_members`, which always returns a freshly built,
    independent tuple — never a live view into the owning table's storage.
    """

    table: ReferenceTableV1

    def resolve_members(
        self, descriptor: SelectorDescriptorV1
    ) -> tuple[OperatorRef, ...]:
        by_fingerprint = {
            entry.descriptor.fingerprint: entry.ref for entry in self.table.entries
        }
        members: list[OperatorRef] = []
        for target in descriptor.target_fingerprints:
            try:
                members.append(by_fingerprint[target])
            except KeyError as exc:
                raise ReferenceResolutionError("selector.member_missing") from exc
        return tuple(members)


__all__ = [
    "SelectorBuildResultV1",
    "SelectorBuildVerdict",
    "SelectorContextV1",
    "attach_selector",
    "build_selector",
    "build_selector_descriptor",
    "build_selector_from_scope",
]
