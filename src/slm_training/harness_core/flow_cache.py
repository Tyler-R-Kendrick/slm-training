"""Bit-exact content-addressed caches for flow states, candidates, closure, and bridges.

Provides a small, deterministic cache layer that can be used directly where
``exact_closure`` expects ``dict[str, SupportResult]`` and can also back bridge
plan / canonical fingerprint caches.  The cache is fail-closed: corrupted,
version-mismatched, or unparseable entries are treated as misses.

Wiring/fixture only; no model, GPU, or ship claim.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harness_core.lineage.store import _atomic_write

CACHE_SCHEMA_VERSION = 1


class FlowCacheMode(str, Enum):
    """Cache access mode."""

    OFF = "off"
    READ = "read"
    READ_WRITE = "read_write"
    REFRESH = "refresh"


@dataclass(frozen=True)
class FlowCacheKey:
    """Stable content-addressed key for a flow cache entry.

    All identity fields participate in the fingerprint.  Callers must include
    every dependency that can change the exact result: state fingerprint,
    hole/candidate/edit identity, bounds, version pins, backend profile, output
    kind, and namespace schema version.
    """

    namespace: str
    fingerprint: str
    schema_version: int = CACHE_SCHEMA_VERSION
    component_versions: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        payload = {
            "namespace": self.namespace,
            "fingerprint": self.fingerprint,
            "schema_version": self.schema_version,
            "component_versions": self.component_versions,
            "extra": self.extra,
        }
        return content_sha(payload)


@dataclass(frozen=True)
class CacheEntry:
    """On-disk cache entry envelope."""

    schema_version: int
    digest: str
    created_at: str
    dependencies: dict[str, Any]
    payload: dict[str, Any]
    payload_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digest": self.digest,
            "created_at": self.created_at,
            "dependencies": _safe_json(self.dependencies),
            "payload": _safe_json(self.payload),
            "payload_checksum": self.payload_checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            schema_version=int(data["schema_version"]),
            digest=str(data["digest"]),
            created_at=str(data["created_at"]),
            dependencies=data["dependencies"],
            payload=data["payload"],
            payload_checksum=str(data["payload_checksum"]),
        )


class FlowCacheCounters:
    """Mutable counters for cache bookkeeping."""

    __slots__ = ("hits", "misses", "writes", "evictions", "bytes_stored", "bytes_read")

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.evictions = 0
        self.bytes_stored = 0
        self.bytes_read = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "evictions": self.evictions,
            "bytes_stored": self.bytes_stored,
            "bytes_read": self.bytes_read,
        }


class FlowCache(ABC):
    """Abstract bit-exact content-addressed cache."""

    def __init__(self, mode: FlowCacheMode = FlowCacheMode.READ_WRITE) -> None:
        self.mode = mode
        self.counters = FlowCacheCounters()

    @abstractmethod
    def _get_payload(self, key: FlowCacheKey) -> dict[str, Any] | None: ...

    @abstractmethod
    def _put_payload(
        self, key: FlowCacheKey, payload: dict[str, Any], dependencies: dict[str, Any] | None
    ) -> None: ...

    @abstractmethod
    def _remove(self, key: FlowCacheKey) -> bool: ...

    @abstractmethod
    @abstractmethod
    def __len__(self) -> int: ...

    def reset(self) -> None: ...

    def get(self, key: FlowCacheKey) -> dict[str, Any] | None:
        if self.mode is FlowCacheMode.OFF:
            return None
        payload = self._get_payload(key)
        if payload is None:
            self.counters.misses += 1
            return None
        self.counters.hits += 1
        return payload

    def put(
        self,
        key: FlowCacheKey,
        payload: dict[str, Any],
        dependencies: dict[str, Any] | None = None,
    ) -> None:
        if self.mode in (FlowCacheMode.OFF, FlowCacheMode.READ):
            raise RuntimeError(f"cache mode {self.mode.value!r} does not permit writes")
        self._put_payload(key, payload, dependencies)
        self.counters.writes += 1

    def invalidate(self, key: FlowCacheKey) -> bool:
        return self._remove(key)

    def _compute_checksum(self, payload: Any) -> str:
        return content_sha(_safe_json(payload))


class InMemoryFlowCache(FlowCache):
    """LRU memory cache with optional max entry bound and byte accounting."""

    def __init__(
        self,
        mode: FlowCacheMode = FlowCacheMode.READ_WRITE,
        max_entries: int | None = None,
    ) -> None:
        super().__init__(mode=mode)
        self.max_entries = max_entries
        # Store (payload, checksum) so corruption is detected like DiskFlowCache.
        self._store: OrderedDict[str, tuple[dict[str, Any], str]] = OrderedDict()

    def _get_payload(self, key: FlowCacheKey) -> dict[str, Any] | None:
        digest = key.digest()
        if digest not in self._store:
            return None
        self._store.move_to_end(digest)
        payload, checksum = self._store[digest]
        if checksum != self._compute_checksum(payload):
            return None
        self.counters.bytes_read += _approximate_bytes(payload)
        return payload

    def _put_payload(
        self, key: FlowCacheKey, payload: dict[str, Any], dependencies: dict[str, Any] | None
    ) -> None:
        digest = key.digest()
        checksum = self._compute_checksum(payload)
        if digest in self._store:
            self._store.move_to_end(digest)
            old_payload, _old_checksum = self._store[digest]
            self.counters.bytes_stored -= _approximate_bytes(old_payload)
            self._store[digest] = (payload, checksum)
            self.counters.bytes_stored += _approximate_bytes(payload)
            return
        if self.max_entries is not None and len(self._store) >= self.max_entries:
            _digest, evicted = self._store.popitem(last=False)
            self.counters.evictions += 1
            self.counters.bytes_stored -= _approximate_bytes(evicted[0])
        self._store[digest] = (payload, checksum)
        self.counters.bytes_stored += _approximate_bytes(payload)

    def _remove(self, key: FlowCacheKey) -> bool:
        digest = key.digest()
        if digest in self._store:
            payload, _checksum = self._store.pop(digest)
            self.counters.bytes_stored -= _approximate_bytes(payload)
            return True
        return False

    def reset(self) -> None:
        self._store.clear()
        self.counters = FlowCacheCounters()

    def __len__(self) -> int:
        return len(self._store)


class DiskFlowCache(FlowCache):
    """Disk-backed content-addressed cache with atomic writes and checksums."""

    def __init__(
        self,
        root: Path | str,
        mode: FlowCacheMode = FlowCacheMode.READ_WRITE,
        schema_version: int = CACHE_SCHEMA_VERSION,
    ) -> None:
        super().__init__(mode=mode)
        self.root = Path(root)
        self.schema_version = schema_version
        self.root.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest[2:]}.json"

    def _get_payload(self, key: FlowCacheKey) -> dict[str, Any] | None:
        path = self._entry_path(key.digest())
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry.from_dict(data)
        except Exception:  # noqa: BLE001 - cache read must never break callers
            return None
        if entry.schema_version != self.schema_version:
            return None
        if entry.digest != key.digest():
            return None
        if entry.payload_checksum != self._compute_checksum(entry.payload):
            return None
        self.counters.bytes_read += _approximate_bytes(entry.payload)
        return entry.payload

    def _put_payload(
        self, key: FlowCacheKey, payload: dict[str, Any], dependencies: dict[str, Any] | None
    ) -> None:
        safe_payload = _safe_json(payload)
        digest = key.digest()
        entry = CacheEntry(
            schema_version=self.schema_version,
            digest=digest,
            created_at=datetime.now(timezone.utc).isoformat(),
            dependencies=_safe_json(dependencies or {}),
            payload=safe_payload,
            payload_checksum=self._compute_checksum(safe_payload),
        )
        path = self._entry_path(digest)
        _atomic_write(path, entry.to_dict())
        self.counters.bytes_stored += _approximate_bytes(safe_payload)

    def _remove(self, key: FlowCacheKey) -> bool:
        path = self._entry_path(key.digest())
        if path.exists():
            path.unlink()
            return True
        return False

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json"))

    def reset(self) -> None:
        if self.root.exists():
            for path in self.root.rglob("*.json"):
                path.unlink()
        self.counters = FlowCacheCounters()


def _approximate_bytes(value: Any) -> int:
    """Best-effort byte estimate for memory accounting."""
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except Exception:  # noqa: BLE001
        return 0


def _safe_json(value: Any) -> Any:
    """Drop non-finite floats so JSON serialization never fails."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "FlowCacheMode",
    "FlowCacheKey",
    "CacheEntry",
    "FlowCacheCounters",
    "FlowCache",
    "InMemoryFlowCache",
    "DiskFlowCache",
]
