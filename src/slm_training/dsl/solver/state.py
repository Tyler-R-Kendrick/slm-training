"""Torch-free finite-domain state for bounded verified synthesis.

The semantics and fingerprint exclusions are owned by
``docs/design/verified-scope-solver.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from functools import total_ordering
from typing import Any, Iterable, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_json(value: Any, *, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item, context=context)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} object keys must be strings")
            _validate_json(item, context=context)
        return
    raise ValueError(f"{context} is not JSON-safe: {type(value).__name__}")


def _require_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires a non-empty {field}")
    return value


def _strict_fields(data: dict[str, Any], expected: set[str], *, context: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown or missing:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


class SupportVerdict(str, Enum):
    """Bounded support result; ``UNKNOWN`` never licenses candidate removal."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@total_ordering
@dataclass(frozen=True)
class HoleId:
    """Stable identity for one unresolved semantic decision site."""

    namespace: str
    path: tuple[str | int, ...]
    kind: str

    def __post_init__(self) -> None:
        context = f"hole {self.namespace!r}/{self.kind!r}"
        _require_text(self.namespace, field="namespace", context=context)
        _require_text(self.kind, field="kind", context=context)
        normalized: list[str | int] = []
        for part in self.path:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise ValueError(
                    f"{context} path entries must be strings or integers"
                )
            normalized.append(part)
        object.__setattr__(self, "path", tuple(normalized))

    @property
    def sort_key(self) -> str:
        return _canonical_json(self.to_dict())

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HoleId):
            return NotImplemented
        return self.sort_key < other.sort_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "path": list(self.path),
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoleId:
        _strict_fields(data, {"namespace", "path", "kind"}, context="HoleId")
        path = data["path"]
        if not isinstance(path, list):
            raise ValueError("HoleId path must be a JSON array")
        return cls(
            namespace=data["namespace"],
            path=tuple(path),
            kind=data["kind"],
        )


@dataclass(frozen=True)
class SolverBounds:
    """Finite resource bounds that participate in solver-state identity."""

    max_tokens: int
    max_nodes: int
    max_depth: int
    max_backtracks: int
    max_verifier_calls: int

    def __post_init__(self) -> None:
        for field, value in self.to_dict().items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"solver bounds require non-negative {field}")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_tokens": self.max_tokens,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "max_backtracks": self.max_backtracks,
            "max_verifier_calls": self.max_verifier_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverBounds:
        expected = {
            "max_tokens",
            "max_nodes",
            "max_depth",
            "max_backtracks",
            "max_verifier_calls",
        }
        _strict_fields(data, expected, context="SolverBounds")
        return cls(**{field: data[field] for field in expected})


@dataclass(frozen=True, order=True)
class DomainValue:
    """A tagged JSON value stored as immutable canonical JSON text."""

    tag: str
    payload_json: str

    def __post_init__(self) -> None:
        _require_text(self.tag, field="tag", context="domain value")
        if not isinstance(self.payload_json, str):
            raise ValueError(f"domain value {self.tag!r} payload_json must be text")
        try:
            payload = json.loads(self.payload_json)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"domain value {self.tag!r} has invalid JSON") from exc
        _validate_json(payload, context=f"domain value {self.tag!r}")
        object.__setattr__(self, "payload_json", _canonical_json(payload))

    @classmethod
    def create(cls, tag: str, payload: Any) -> DomainValue:
        _validate_json(payload, context=f"domain value {tag!r}")
        return cls(tag=tag, payload_json=_canonical_json(payload))

    @property
    def payload(self) -> Any:
        return json.loads(self.payload_json)

    def to_dict(self) -> dict[str, Any]:
        return {"tag": self.tag, "value": self.payload}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainValue:
        _strict_fields(data, {"tag", "value"}, context="DomainValue")
        return cls.create(data["tag"], data["value"])


def _normalize_metadata(
    metadata: Iterable[tuple[str, JsonScalar]], *, context: str
) -> tuple[tuple[str, JsonScalar], ...]:
    rows = tuple(metadata)
    keys: set[str] = set()
    normalized: list[tuple[str, JsonScalar]] = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(f"{context} metadata entries must be key/value tuples")
        key, value = row
        _require_text(key, field="metadata key", context=context)
        if key in keys:
            raise ValueError(f"{context} has duplicate metadata key {key!r}")
        _validate_json(value, context=f"{context} metadata {key!r}")
        if isinstance(value, (list, tuple, dict)):
            raise ValueError(f"{context} metadata {key!r} must be a JSON scalar")
        keys.add(key)
        normalized.append((key, value))
    return tuple(sorted(normalized, key=lambda row: row[0]))


@dataclass(frozen=True, eq=False)
class HoleDomain:
    """Canonical finite values and epistemic metadata for one hole."""

    hole_id: HoleId
    values: tuple[DomainValue, ...]
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.hole_id, HoleId):
            raise ValueError("hole domain requires a HoleId")
        values = tuple(self.values)
        if any(not isinstance(value, DomainValue) for value in values):
            raise ValueError(f"hole {self.hole_id!r} contains a non-DomainValue")
        if len(set(values)) != len(values):
            raise ValueError(f"hole {self.hole_id!r} contains duplicate values")
        object.__setattr__(self, "values", tuple(sorted(values)))
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata, context=f"hole {self.hole_id!r}"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id.to_dict(),
            "values": [value.to_dict() for value in self.values],
            "metadata": [[key, value] for key, value in self.metadata],
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HoleDomain):
            return NotImplemented
        return _canonical_json(self.to_dict()) == _canonical_json(other.to_dict())

    def __hash__(self) -> int:
        return hash(_canonical_json(self.to_dict()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoleDomain:
        _strict_fields(data, {"hole_id", "values", "metadata"}, context="HoleDomain")
        values = data["values"]
        metadata = data["metadata"]
        if not isinstance(values, list) or not isinstance(metadata, list):
            raise ValueError("HoleDomain values and metadata must be JSON arrays")
        return cls(
            hole_id=HoleId.from_dict(data["hole_id"]),
            values=tuple(DomainValue.from_dict(value) for value in values),
            metadata=tuple(tuple(row) for row in metadata),
        )


@dataclass(frozen=True)
class FiniteDomainState:
    """Canonical hard state; soft scores and proof artifacts live elsewhere."""

    problem_id: str
    pack_id: str
    constraint_version: str
    bounds: SolverBounds
    holes: tuple[HoleDomain, ...]
    decision_level: int = 0
    parent_fingerprint: str | None = None

    def __post_init__(self) -> None:
        context = f"problem {self.problem_id!r}"
        _require_text(self.problem_id, field="problem_id", context=context)
        _require_text(self.pack_id, field="pack_id", context=context)
        _require_text(
            self.constraint_version, field="constraint_version", context=context
        )
        if not isinstance(self.bounds, SolverBounds):
            raise ValueError(f"{context} requires SolverBounds")
        if (
            isinstance(self.decision_level, bool)
            or not isinstance(self.decision_level, int)
            or self.decision_level < 0
        ):
            raise ValueError(f"{context} requires a non-negative decision_level")
        if self.parent_fingerprint is not None and (
            not isinstance(self.parent_fingerprint, str)
            or len(self.parent_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.parent_fingerprint)
        ):
            raise ValueError(f"{context} parent_fingerprint must be a SHA-256 hex digest")
        holes = tuple(self.holes)
        if any(not isinstance(hole, HoleDomain) for hole in holes):
            raise ValueError(f"{context} contains a non-HoleDomain")
        ids = [hole.hole_id for hole in holes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{context} contains duplicate hole IDs")
        object.__setattr__(self, "holes", tuple(sorted(holes, key=lambda h: h.hole_id)))

    def _hard_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "bounds": self.bounds.to_dict(),
            "holes": [hole.to_dict() for hole in self.holes],
        }

    @property
    def fingerprint(self) -> str:
        """Full SHA-256 over hard state/config, excluding search lineage."""
        return hashlib.sha256(_canonical_json(self._hard_dict()).encode()).hexdigest()

    @property
    def is_bottom(self) -> bool:
        return any(not hole.values for hole in self.holes)

    @property
    def is_structurally_solved(self) -> bool:
        return not self.is_bottom and all(len(hole.values) == 1 for hole in self.holes)

    def domain(self, hole_id: HoleId) -> HoleDomain:
        for hole in self.holes:
            if hole.hole_id == hole_id:
                return hole
        raise LookupError(f"problem {self.problem_id!r} has no hole {hole_id!r}")

    def refine(
        self,
        hole_id: HoleId,
        retained_values: Iterable[DomainValue],
        *,
        certificate_ref: str | None = None,
    ) -> FiniteDomainState:
        """Return a monotone subset; certificate persistence starts in VSS0-04."""
        if certificate_ref is not None and (
            not isinstance(certificate_ref, str) or not certificate_ref
        ):
            raise ValueError(
                f"problem {self.problem_id!r} certificate_ref must be non-empty"
            )
        current = self.domain(hole_id)
        retained = tuple(retained_values)
        if any(not isinstance(value, DomainValue) for value in retained):
            raise ValueError(
                f"problem {self.problem_id!r} hole {hole_id!r} refinement "
                "requires DomainValue candidates"
            )
        current_values = set(current.values)
        added = [value for value in retained if value not in current_values]
        if added:
            raise ValueError(
                f"problem {self.problem_id!r} hole {hole_id!r} refinement "
                "cannot add candidates"
            )
        replacement = HoleDomain(hole_id, retained, current.metadata)
        holes = tuple(replacement if hole.hole_id == hole_id else hole for hole in self.holes)
        return replace(self, holes=holes)

    def meet(self, other: FiniteDomainState) -> FiniteDomainState:
        """Intersect corresponding domains without inventing missing-hole semantics."""
        if not isinstance(other, FiniteDomainState):
            raise ValueError(f"problem {self.problem_id!r} can meet only another state")
        identity = (self.problem_id, self.pack_id, self.constraint_version, self.bounds)
        other_identity = (
            other.problem_id,
            other.pack_id,
            other.constraint_version,
            other.bounds,
        )
        if identity != other_identity:
            raise ValueError(f"problem {self.problem_id!r} cannot meet mismatched identity")
        if {hole.hole_id for hole in self.holes} != {
            hole.hole_id for hole in other.holes
        }:
            raise ValueError(f"problem {self.problem_id!r} cannot meet mismatched holes")
        domains: list[HoleDomain] = []
        for left in self.holes:
            right = other.domain(left.hole_id)
            if _canonical_json(left.to_dict()["metadata"]) != _canonical_json(
                right.to_dict()["metadata"]
            ):
                raise ValueError(
                    f"problem {self.problem_id!r} hole {left.hole_id!r} "
                    "cannot meet mismatched metadata"
                )
            live = set(right.values)
            domains.append(
                HoleDomain(
                    left.hole_id,
                    tuple(value for value in left.values if value in live),
                    left.metadata,
                )
            )
        return FiniteDomainState(
            problem_id=self.problem_id,
            pack_id=self.pack_id,
            constraint_version=self.constraint_version,
            bounds=self.bounds,
            holes=tuple(domains),
        )

    def with_decision(self, hole_id: HoleId, value: DomainValue) -> FiniteDomainState:
        """Choose one live value reversibly without claiming proof."""
        refined = self.refine(hole_id, (value,))
        return replace(
            refined,
            decision_level=self.decision_level + 1,
            parent_fingerprint=self.fingerprint,
        )

    def summary(self) -> dict[str, int | float | bool]:
        sizes = [len(hole.values) for hole in self.holes]
        total = sum(sizes)
        return {
            "hole_count": len(sizes),
            "unresolved_count": sum(size != 1 for size in sizes),
            "total_candidate_count": total,
            "max_domain_size": max(sizes, default=0),
            "mean_domain_size": total / len(sizes) if sizes else 0.0,
            "is_bottom": self.is_bottom,
            "is_structurally_solved": self.is_structurally_solved,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._hard_dict(),
            "decision_level": self.decision_level,
            "parent_fingerprint": self.parent_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FiniteDomainState:
        expected = {
            "problem_id",
            "pack_id",
            "constraint_version",
            "bounds",
            "holes",
            "decision_level",
            "parent_fingerprint",
        }
        _strict_fields(data, expected, context="FiniteDomainState")
        holes = data["holes"]
        if not isinstance(holes, list):
            raise ValueError("FiniteDomainState holes must be a JSON array")
        return cls(
            problem_id=data["problem_id"],
            pack_id=data["pack_id"],
            constraint_version=data["constraint_version"],
            bounds=SolverBounds.from_dict(data["bounds"]),
            holes=tuple(HoleDomain.from_dict(hole) for hole in holes),
            decision_level=data["decision_level"],
            parent_fingerprint=data["parent_fingerprint"],
        )
