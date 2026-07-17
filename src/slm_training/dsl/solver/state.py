"""Torch-free finite-domain support lattice for the verified scope solver.

This module implements the model-independent state representation specified by
``docs/design/verified-scope-solver.md`` (VSS0-01 / SLM-57) and required by
VSS0-03 (SLM-59). The design document is the source of truth for the semantics
below; the docstrings here reference it rather than duplicating it.

What this is
------------
Immutable, JSON-safe dataclasses (:class:`HoleId`, :class:`SolverBounds`,
:class:`HoleDomain`, :class:`FiniteDomainState`) plus the :class:`SupportVerdict`
enum. Every mutation-like operation returns a *new* validated state; nothing here
mutates in place. Finite bounds and uncertainty are explicit, and non-monotone
(candidate-adding) updates are rejected.

What this is not
----------------
No recursive support search, no proof checker, no decode/config integration, and
no model or energy score (see the "Non-goals" of SLM-59). Soft candidate scores
are deliberately *not* stored here; a separate ranker keys scores by
``(state_fingerprint, hole_id, value)`` later. This module imports no ``torch``
and performs no model inference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# A deterministic JSON scalar. ``bool`` is intentionally included (valid JSON and
# a subclass of ``int``); nested containers are not scalars.
JsonScalar = str | int | float | bool | None

__all__ = [
    "DomainValue",
    "FiniteDomainProjection",
    "FiniteDomainState",
    "HoleDomain",
    "HoleId",
    "JsonScalar",
    "SolverBounds",
    "SupportVerdict",
]


# --------------------------------------------------------------------------- #
# Canonicalisation helpers (shared by every fingerprint and dedup path).
# --------------------------------------------------------------------------- #
def _canonical_json(value: Any) -> str:
    """Order-insensitive canonical JSON text used by every fingerprint/dedup key."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_json_scalar(value: Any) -> bool:
    # ``bool`` is a subclass of ``int`` and is a valid JSON scalar.
    return value is None or isinstance(value, (str, int, float))


def _first_duplicate(items: Sequence[str]) -> str | None:
    seen: set[str] = set()
    for item in items:
        if item in seen:
            return item
        seen.add(item)
    return None


def _validate_scalar_pairs(
    pairs: tuple[tuple[str, JsonScalar], ...], *, owner: str
) -> None:
    if not isinstance(pairs, tuple):
        raise ValueError(f"{owner} must be a tuple of (key, scalar) pairs")
    seen: set[str] = set()
    for entry in pairs:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise ValueError(f"{owner} entries must be (key, value) pairs, got {entry!r}")
        key, value = entry
        if not isinstance(key, str):
            raise ValueError(f"{owner} keys must be str, got {key!r}")
        if key in seen:
            raise ValueError(f"{owner} has a duplicate key {key!r}")
        seen.add(key)
        if not _is_json_scalar(value):
            raise ValueError(f"{owner}[{key!r}] must be a JSON scalar, got {type(value).__name__}")


# --------------------------------------------------------------------------- #
# Core value types.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, order=True)
class HoleId:
    """Stable identifier for one unresolved decision site (see the design doc).

    ``path`` elements are ``str`` or ``int`` (never ``bool``) so identities such
    as ``("root", 0)`` survive a JSON round-trip with their element types intact.
    """

    namespace: str
    path: tuple[str | int, ...]
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str):
            raise ValueError(f"HoleId.namespace must be str, got {type(self.namespace).__name__}")
        if not isinstance(self.kind, str):
            raise ValueError(f"HoleId.kind must be str, got {type(self.kind).__name__}")
        if not isinstance(self.path, tuple):
            raise ValueError(f"HoleId.path must be a tuple, got {type(self.path).__name__}")
        for part in self.path:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise ValueError(
                    f"HoleId.path elements must be str|int, got {part!r} "
                    f"in namespace {self.namespace!r}"
                )

    @property
    def canonical_key(self) -> str:
        """Type-safe total-order key (JSON text) used for sorting/lookup.

        Preferred over the generated ``order`` comparison, which raises on
        heterogeneously typed ``path`` tuples.
        """
        return _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"namespace": self.namespace, "path": list(self.path), "kind": self.kind}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HoleId:
        # JSON preserves int-vs-str for list elements, so ``tuple(...)`` keeps types.
        return cls(
            namespace=str(data["namespace"]),
            path=tuple(data["path"]),
            kind=str(data["kind"]),
        )


@dataclass(frozen=True)
class SolverBounds:
    """Declared finite budgets. Finiteness within these bounds is a precondition
    for exact closure (see the design doc); negative bounds are rejected."""

    max_tokens: int
    max_nodes: int
    max_depth: int
    max_backtracks: int
    max_verifier_calls: int

    _FIELDS = ("max_tokens", "max_nodes", "max_depth", "max_backtracks", "max_verifier_calls")

    def __post_init__(self) -> None:
        for name in self._FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"SolverBounds.{name} must be a non-negative int, got {value!r}")
            if value < 0:
                raise ValueError(f"SolverBounds.{name} must be non-negative, got {value}")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SolverBounds:
        return cls(**{name: int(data[name]) for name in cls._FIELDS})


class SupportVerdict(str, Enum):
    """Reference support semantics from the design doc. ``UNKNOWN`` never permits
    candidate removal and no API may translate it to ``UNSUPPORTED``."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DomainValue:
    """A deterministic, tagged, JSON-safe candidate filling for a hole.

    Identity is by value, never by object identity. ``tag`` discriminates value
    families (``"token_path"`` for the completion-forest adapter today; structured
    action tags later). ``token_ids`` carries a *full* compiler path — not only its
    first token — so grammar-forced suffixes stay distinguishable. ``kind`` is the
    semantic decision kind. ``attributes`` is a canonical (sorted, unique-key)
    tuple of extra JSON scalars reserved for future structured action values.
    """

    tag: str
    token_ids: tuple[int, ...] = ()
    kind: str | None = None
    attributes: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError(f"DomainValue.tag must be a non-empty str, got {self.tag!r}")
        if not isinstance(self.token_ids, tuple):
            raise ValueError(f"DomainValue.token_ids must be a tuple of ints (tag={self.tag!r})")
        for token in self.token_ids:
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError(
                    f"DomainValue.token_ids must be ints, got {token!r} (tag={self.tag!r})"
                )
        if self.kind is not None and not isinstance(self.kind, str):
            raise ValueError(f"DomainValue.kind must be str|None, got {self.kind!r}")
        _validate_scalar_pairs(self.attributes, owner=f"DomainValue(tag={self.tag!r}).attributes")
        canonical = tuple(sorted(self.attributes, key=lambda kv: kv[0]))
        if canonical != self.attributes:
            object.__setattr__(self, "attributes", canonical)

    @property
    def canonical_key(self) -> str:
        return _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "token_ids": list(self.token_ids),
            "kind": self.kind,
            "attributes": [list(kv) for kv in self.attributes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DomainValue:
        return cls(
            tag=str(data["tag"]),
            token_ids=tuple(int(token) for token in data.get("token_ids") or ()),
            kind=None if data.get("kind") is None else str(data["kind"]),
            attributes=tuple((str(key), value) for key, value in data.get("attributes") or ()),
        )


@dataclass(frozen=True)
class HoleDomain:
    """The finite domain (candidate set) for one :class:`HoleId`.

    Values are deduplicated (duplicates are rejected, not silently merged) and
    canonically ordered on construction so the enclosing state's fingerprint is
    order-insensitive. ``metadata`` carries JSON scalars such as the compiler
    ``coverage`` guarantee; it never carries soft scores.
    """

    hole_id: HoleId
    values: tuple[DomainValue, ...]
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.hole_id, HoleId):
            raise ValueError("HoleDomain.hole_id must be a HoleId")
        if not isinstance(self.values, tuple):
            raise ValueError(f"HoleDomain.values must be a tuple for hole {self.hole_id}")
        for value in self.values:
            if not isinstance(value, DomainValue):
                raise ValueError(
                    f"HoleDomain.values entries must be DomainValue for hole {self.hole_id}, "
                    f"got {type(value).__name__}"
                )
        keys = [value.canonical_key for value in self.values]
        duplicate = _first_duplicate(keys)
        if duplicate is not None:
            raise ValueError(f"duplicate domain value in hole {self.hole_id}: {duplicate}")
        ordered = tuple(value for _, value in sorted(zip(keys, self.values), key=lambda pair: pair[0]))
        if ordered != self.values:
            object.__setattr__(self, "values", ordered)
        _validate_scalar_pairs(self.metadata, owner=f"HoleDomain({self.hole_id}).metadata")
        canonical_meta = tuple(sorted(self.metadata, key=lambda kv: kv[0]))
        if canonical_meta != self.metadata:
            object.__setattr__(self, "metadata", canonical_meta)

    @property
    def is_empty(self) -> bool:
        """``⊥`` (bottom): a hole with no legal candidate — a local contradiction."""
        return len(self.values) == 0

    @property
    def is_singleton(self) -> bool:
        return len(self.values) == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id.to_dict(),
            "values": [value.to_dict() for value in self.values],
            "metadata": [list(kv) for kv in self.metadata],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HoleDomain:
        return cls(
            hole_id=HoleId.from_dict(data["hole_id"]),
            values=tuple(DomainValue.from_dict(value) for value in data.get("values") or ()),
            metadata=tuple((str(key), value) for key, value in data.get("metadata") or ()),
        )


@dataclass(frozen=True)
class FiniteDomainState:
    """A bounded, model-independent solver state over a set of finite domains.

    See ``docs/design/verified-scope-solver.md``. Holes are deduplicated and
    canonically ordered on construction. ``decision_level`` and
    ``parent_fingerprint`` record reversible-search lineage and are deliberately
    excluded from :attr:`fingerprint` (they are trail bookkeeping, not hard
    state). Every mutation-like method returns a new validated state.
    """

    problem_id: str
    pack_id: str
    constraint_version: str
    bounds: SolverBounds
    holes: tuple[HoleDomain, ...]
    decision_level: int = 0
    parent_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("problem_id", "pack_id", "constraint_version"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"FiniteDomainState.{name} must be str")
        if not isinstance(self.bounds, SolverBounds):
            raise ValueError("FiniteDomainState.bounds must be a SolverBounds")
        if not isinstance(self.holes, tuple):
            raise ValueError("FiniteDomainState.holes must be a tuple")
        for hole in self.holes:
            if not isinstance(hole, HoleDomain):
                raise ValueError(
                    f"FiniteDomainState.holes entries must be HoleDomain, got {type(hole).__name__}"
                )
        if isinstance(self.decision_level, bool) or not isinstance(self.decision_level, int):
            raise ValueError(f"FiniteDomainState.decision_level must be an int, got {self.decision_level!r}")
        if self.decision_level < 0:
            raise ValueError(f"FiniteDomainState.decision_level must be non-negative, got {self.decision_level}")
        if self.parent_fingerprint is not None and not isinstance(self.parent_fingerprint, str):
            raise ValueError("FiniteDomainState.parent_fingerprint must be str|None")
        keys = [hole.hole_id.canonical_key for hole in self.holes]
        duplicate = _first_duplicate(keys)
        if duplicate is not None:
            raise ValueError(f"duplicate hole id in problem {self.problem_id!r}: {duplicate}")
        ordered = tuple(hole for _, hole in sorted(zip(keys, self.holes), key=lambda pair: pair[0]))
        if ordered != self.holes:
            object.__setattr__(self, "holes", ordered)

    # -- structural predicates -------------------------------------------- #
    @property
    def is_bottom(self) -> bool:
        """True when any hole domain is empty (a certified local contradiction)."""
        return any(hole.is_empty for hole in self.holes)

    @property
    def is_structurally_solved(self) -> bool:
        """Every domain is a singleton and the state is not bottom.

        Structural only: semantic verification has *not* been applied, so this is
        never a ``SUPPORTED`` claim about the whole program.
        """
        return not self.is_bottom and all(hole.is_singleton for hole in self.holes)

    # -- identity / fingerprint ------------------------------------------- #
    def _fingerprint_payload(self) -> dict[str, Any]:
        # Hard state and configuration only: problem/pack/constraint identity,
        # declared bounds, and the (canonically ordered) holes. Excludes
        # decision_level and parent_fingerprint (trail lineage), and by
        # construction excludes logits, scores, timestamps, PIDs, and caches.
        return {
            "problem_id": self.problem_id,
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "bounds": self.bounds.to_dict(),
            "holes": [hole.to_dict() for hole in self.holes],
        }

    @property
    def fingerprint(self) -> str:
        """Stable, order-insensitive SHA-256 over hard state/configuration only."""
        return _sha256_hex(self._fingerprint_payload())

    def _identity_mismatch(self, other: FiniteDomainState) -> str | None:
        for name in ("problem_id", "pack_id", "constraint_version"):
            if getattr(self, name) != getattr(other, name):
                return f"{name} ({getattr(self, name)!r} vs {getattr(other, name)!r})"
        if self.bounds != other.bounds:
            return "bounds"
        return None

    # -- lookup ----------------------------------------------------------- #
    def has_hole(self, hole_id: HoleId) -> bool:
        key = hole_id.canonical_key
        return any(hole.hole_id.canonical_key == key for hole in self.holes)

    def domain(self, hole_id: HoleId) -> HoleDomain:
        """Return the :class:`HoleDomain` for ``hole_id`` or raise ``KeyError``."""
        key = hole_id.canonical_key
        for hole in self.holes:
            if hole.hole_id.canonical_key == key:
                return hole
        raise KeyError(f"unknown hole {hole_id} in problem {self.problem_id!r}")

    # -- monotone transitions --------------------------------------------- #
    def _reduced_holes(
        self, hole_id: HoleId, retained_values: Sequence[DomainValue]
    ) -> tuple[HoleDomain, ...]:
        key = hole_id.canonical_key
        target = next((hole for hole in self.holes if hole.hole_id.canonical_key == key), None)
        if target is None:
            raise ValueError(f"cannot refine unknown hole {hole_id} in problem {self.problem_id!r}")
        retained = tuple(retained_values)
        for value in retained:
            if not isinstance(value, DomainValue):
                raise ValueError(
                    f"retained_values must be DomainValue for hole {hole_id}, got {type(value).__name__}"
                )
        current_keys = {value.canonical_key for value in target.values}
        added = sorted({value.canonical_key for value in retained} - current_keys)
        if added:
            raise ValueError(
                f"refine on hole {hole_id} in problem {self.problem_id!r} would add values "
                f"outside the current domain (monotonicity violated): {added}"
            )
        reduced = HoleDomain(target.hole_id, retained, target.metadata)
        return tuple(reduced if hole.hole_id.canonical_key == key else hole for hole in self.holes)

    def _with_holes(
        self,
        holes: tuple[HoleDomain, ...],
        *,
        decision_level: int,
        parent_fingerprint: str | None,
    ) -> FiniteDomainState:
        return FiniteDomainState(
            problem_id=self.problem_id,
            pack_id=self.pack_id,
            constraint_version=self.constraint_version,
            bounds=self.bounds,
            holes=holes,
            decision_level=decision_level,
            parent_fingerprint=parent_fingerprint,
        )

    def refine(
        self,
        hole_id: HoleId,
        retained_values: Sequence[DomainValue],
        *,
        certificate_ref: str | None = None,
    ) -> FiniteDomainState:
        """Monotonically reduce ``hole_id`` to ``retained_values`` (a subset).

        Rejects added values and unknown holes. Leaves ``decision_level`` and
        ``parent_fingerprint`` unchanged: a certified deduction is not a search
        decision. ``certificate_ref`` is a forward-compatibility hook for the
        proof-carrying loop; certificate persistence and replay are a later issue
        (VSS1-04) and this method does not store it (keeping it out of the
        fingerprint so two states reduced to the same domain remain
        interchangeable for replay).
        """
        if certificate_ref is not None and not isinstance(certificate_ref, str):
            raise ValueError("certificate_ref must be a str reference or None")
        holes = self._reduced_holes(hole_id, retained_values)
        return self._with_holes(
            holes, decision_level=self.decision_level, parent_fingerprint=self.parent_fingerprint
        )

    def with_decision(
        self, hole_id: HoleId, retained_values: Sequence[DomainValue]
    ) -> FiniteDomainState:
        """Record a reversible search decision that commits ``hole_id`` to a subset.

        Increments ``decision_level`` and records this state's fingerprint as the
        child's ``parent_fingerprint``. Claims no proof: no certificate is
        required or stored. The reduction is still monotone (added values and
        unknown holes are rejected).
        """
        holes = self._reduced_holes(hole_id, retained_values)
        return self._with_holes(
            holes, decision_level=self.decision_level + 1, parent_fingerprint=self.fingerprint
        )

    def meet(self, other: FiniteDomainState) -> FiniteDomainState:
        """Greatest-lower-bound over the shared problem identity.

        Intersects the domains of holes present in both states and keeps holes
        present in only one (an absent hole is unconstrained/top). An empty
        intersection yields an empty domain, i.e. a bottom state. Rejected across
        different problem/pack/constraint/bounds identity. Commutative up to
        lineage: ``decision_level`` is ``max`` of both, ``parent_fingerprint`` is
        cleared (a meet is not a single-parent decision).
        """
        if not isinstance(other, FiniteDomainState):
            raise ValueError("meet requires another FiniteDomainState")
        mismatch = self._identity_mismatch(other)
        if mismatch is not None:
            raise ValueError(
                f"cannot meet states across different {mismatch} "
                f"(problem {self.problem_id!r})"
            )
        self_map = {hole.hole_id.canonical_key: hole for hole in self.holes}
        other_map = {hole.hole_id.canonical_key: hole for hole in other.holes}
        merged: list[HoleDomain] = []
        for key in self_map.keys() | other_map.keys():
            left = self_map.get(key)
            right = other_map.get(key)
            if left is not None and right is not None:
                right_keys = {value.canonical_key for value in right.values}
                intersection = tuple(value for value in left.values if value.canonical_key in right_keys)
                merged.append(HoleDomain(left.hole_id, intersection, _merge_metadata(left, right)))
            else:
                merged.append(left if left is not None else right)  # type: ignore[arg-type]
        return self._with_holes(
            tuple(merged),
            decision_level=max(self.decision_level, other.decision_level),
            parent_fingerprint=None,
        )

    # -- metrics ---------------------------------------------------------- #
    def summary(self) -> dict[str, Any]:
        """Compact metrics: hole/candidate counts, domain sizes, and flags."""
        sizes = [len(hole.values) for hole in self.holes]
        total = sum(sizes)
        return {
            "hole_count": len(self.holes),
            "unresolved_count": sum(1 for size in sizes if size != 1),
            "total_candidates": total,
            "max_domain_size": max(sizes) if sizes else 0,
            "mean_domain_size": (total / len(sizes)) if sizes else 0.0,
            "is_bottom": self.is_bottom,
            "is_structurally_solved": self.is_structurally_solved,
        }

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "bounds": self.bounds.to_dict(),
            "holes": [hole.to_dict() for hole in self.holes],
            "decision_level": self.decision_level,
            "parent_fingerprint": self.parent_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FiniteDomainState:
        return cls(
            problem_id=str(data["problem_id"]),
            pack_id=str(data["pack_id"]),
            constraint_version=str(data["constraint_version"]),
            bounds=SolverBounds.from_dict(data["bounds"]),
            holes=tuple(HoleDomain.from_dict(hole) for hole in data.get("holes") or ()),
            decision_level=int(data.get("decision_level", 0)),
            parent_fingerprint=(
                None if data.get("parent_fingerprint") is None else str(data["parent_fingerprint"])
            ),
        )


def _merge_metadata(left: HoleDomain, right: HoleDomain) -> tuple[tuple[str, JsonScalar], ...]:
    merged = dict(left.metadata)
    for key, value in right.metadata:
        if key in merged and merged[key] != value:
            raise ValueError(
                f"meet metadata conflict on hole {left.hole_id} key {key!r}: "
                f"{merged[key]!r} vs {value!r}"
            )
        merged[key] = value
    return tuple(sorted(merged.items(), key=lambda kv: kv[0]))


@runtime_checkable
class FiniteDomainProjection(Protocol):
    """Seam for projecting a model-independent source into a state.

    The completion-forest adapter is the only implementation in this issue (see
    ``slm_training.dsl.solver.adapters``). Future topology-node domains implement
    the same ``finite_domain_state()`` method. Implementations MUST remain
    Torch-free at import time; model inference belongs behind a later feature gate
    (see ``docs/design/verified-scope-solver.md``).
    """

    def finite_domain_state(self) -> FiniteDomainState: ...
