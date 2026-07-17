"""Immutable finite-domain solver state types."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SupportVerdict(str, Enum):
    """Bounded support verdict for a domain value."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HoleId:
    """Stable identity for a semantic hole."""

    namespace: str
    path: tuple[str | int, ...]
    kind: str

    def __str__(self) -> str:
        return f"{self.namespace}:{self.kind}:" + "/".join(str(p) for p in self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "path": list(self.path),
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoleId:
        return cls(
            namespace=str(data["namespace"]),
            path=tuple(data.get("path") or ()),
            kind=str(data["kind"]),
        )


@dataclass(frozen=True)
class DomainValue:
    """One tagged value in a finite domain."""

    tag: str
    payload: tuple[Any, ...]

    @staticmethod
    def topology_edit(action: str, production_id: int, arity: int, slot_id: int) -> DomainValue:
        return DomainValue(
            "topology_edit",
            (action, production_id, arity, slot_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"tag": self.tag, "payload": list(self.payload)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainValue:
        return cls(tag=str(data["tag"]), payload=tuple(data.get("payload") or ()))


@dataclass(frozen=True)
class HoleDomain:
    """Finite domain for one hole."""

    hole_id: HoleId
    values: tuple[DomainValue, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.values)) != len(self.values):
            raise ValueError("hole domain values must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id.to_dict(),
            "values": [v.to_dict() for v in self.values],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoleDomain:
        return cls(
            hole_id=HoleId.from_dict(data["hole_id"]),
            values=tuple(DomainValue.from_dict(v) for v in data.get("values", [])),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class FiniteDomainState:
    """Hard carrier mapping hole IDs to finite domains."""

    domains: dict[str, HoleDomain]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint({hid: domain.to_dict() for hid, domain in self.domains.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": {hid: domain.to_dict() for hid, domain in self.domains.items()},
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FiniteDomainState:
        return cls(
            domains={
                hid: HoleDomain.from_dict(d)
                for hid, d in (data.get("domains") or {}).items()
            }
        )

    def is_empty(self) -> bool:
        return not self.domains or all(not domain.values for domain in self.domains.values())


def _fingerprint(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()[:32]
