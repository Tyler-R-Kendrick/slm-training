"""Content-addressed evaluation cache for SDE3-01.

Provides a disk-backed, schema-versioned cache with atomic writes and explicit
dependency fingerprints.  It is intentionally generic over "layers" (L2
generation, L3 metrics, L4 judge results, or a full suite result) so the same
store can serve all evaluation caching needs without adding a new dependency.

The cache is fail-closed: corrupted, partial, schema-mismatched, or
validation-failed entries are rejected and recomputed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harness_core.lineage.store import _atomic_write

CACHE_SCHEMA_VERSION = 1


class EvalCacheMode(str, Enum):
    """Cache access mode."""

    OFF = "off"
    READ = "read"
    READ_WRITE = "read_write"
    REFRESH = "refresh"


@dataclass(frozen=True)
class EvalCacheConfig:
    """Configuration for an evaluation cache store."""

    mode: EvalCacheMode = EvalCacheMode.OFF
    root: Path = field(default_factory=lambda: Path("outputs/eval_cache"))
    schema_version: int = CACHE_SCHEMA_VERSION


@dataclass(frozen=True)
class EvalCacheKey:
    """Stable key for a content-addressed cache entry.

    All fields participate in the fingerprint.  Callers must include every
    dependency that would change the result: checkpoint hash, request hash,
    decode/eval policy, component versions, etc.
    """

    layer: str
    checkpoint_sha256: str | None = None
    request_sha256: str | None = None
    policy: dict[str, Any] = field(default_factory=dict)
    suite: str | None = None
    eval_limit: int | None = None
    component_versions: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = {
            "layer": self.layer,
            "checkpoint_sha256": self.checkpoint_sha256,
            "request_sha256": self.request_sha256,
            "policy": self.policy,
            "suite": self.suite,
            "eval_limit": self.eval_limit,
            "component_versions": self.component_versions,
            "extra": self.extra,
        }
        return content_sha(payload)


@dataclass(frozen=True)
class CacheEntry:
    """On-disk cache entry envelope."""

    schema_version: int
    fingerprint: str
    created_at: str
    dependencies: dict[str, Any]
    payload: dict[str, Any]
    payload_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "dependencies": _safe_json(self.dependencies),
            "payload": _safe_json(self.payload),
            "payload_checksum": self.payload_checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            schema_version=int(data["schema_version"]),
            fingerprint=str(data["fingerprint"]),
            created_at=str(data["created_at"]),
            dependencies=data["dependencies"],
            payload=data["payload"],
            payload_checksum=str(data["payload_checksum"]),
        )


class EvalCache:
    """Content-addressed disk cache for evaluation artifacts."""

    def __init__(self, config: EvalCacheConfig | None = None) -> None:
        self.config = config or EvalCacheConfig()
        self.root = self.config.root
        self.root.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, fingerprint: str) -> Path:
        # Two-level prefix to avoid huge flat directories.
        return self.root / fingerprint[:2] / f"{fingerprint[2:]}.json"

    def _compute_checksum(self, payload: Any) -> str:
        return content_sha(_safe_json(payload))

    def get(self, key: EvalCacheKey) -> dict[str, Any] | None:
        """Return a validated cached payload, or None on miss/invalid."""
        if self.config.mode is EvalCacheMode.OFF:
            return None
        path = self._entry_path(key.fingerprint())
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry.from_dict(data)
        except Exception:  # noqa: BLE001 - cache read must never break eval
            return None
        if entry.schema_version != self.config.schema_version:
            return None
        if entry.fingerprint != key.fingerprint():
            return None
        if entry.payload_checksum != self._compute_checksum(entry.payload):
            return None
        return entry.payload

    def put(
        self,
        key: EvalCacheKey,
        payload: dict[str, Any],
        dependencies: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically store a payload under its content-addressed key.

        Raises when ``mode`` does not permit writes.
        """
        if self.config.mode in (EvalCacheMode.OFF, EvalCacheMode.READ):
            raise RuntimeError(
                f"cache mode {self.config.mode.value!r} does not permit writes"
            )
        fingerprint = key.fingerprint()
        safe_payload = _safe_json(payload)
        entry = CacheEntry(
            schema_version=self.config.schema_version,
            fingerprint=fingerprint,
            created_at=datetime.now(timezone.utc).isoformat(),
            dependencies=_safe_json(dependencies or {}),
            payload=safe_payload,
            payload_checksum=self._compute_checksum(safe_payload),
        )
        path = self._entry_path(fingerprint)
        _atomic_write(path, entry.to_dict())
        return path

    def invalidate(self, key: EvalCacheKey) -> bool:
        """Remove a single entry. Returns True if an entry was deleted."""
        path = self._entry_path(key.fingerprint())
        if path.exists():
            path.unlink()
            return True
        return False


def suite_result_key(
    *,
    suite: str,
    checkpoint_sha256: str | None,
    eval_data_manifest_sha: str | None,
    eval_suite_manifest_sha: str | None,
    eval_limit: int | None,
    evaluation_policy: dict[str, Any],
    component_versions: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> EvalCacheKey:
    """Build a content-addressed key for a full suite evaluation result."""
    return EvalCacheKey(
        layer="suite_result",
        checkpoint_sha256=checkpoint_sha256,
        suite=suite,
        eval_limit=eval_limit,
        policy=_safe_json(evaluation_policy),
        component_versions=component_versions,
        extra={
            "eval_data_manifest_sha": eval_data_manifest_sha,
            "eval_suite_manifest_sha": eval_suite_manifest_sha,
            **(extra or {}),
        },
    )


def request_generation_key(
    *,
    checkpoint_sha256: str,
    request_sha256: str,
    evaluation_policy: dict[str, Any],
    component_versions: dict[str, str],
    attempt_index: int = 0,
) -> EvalCacheKey:
    """Build a content-addressed key for a single generation attempt."""
    return EvalCacheKey(
        layer="generation",
        checkpoint_sha256=checkpoint_sha256,
        request_sha256=request_sha256,
        policy=_safe_json(evaluation_policy),
        component_versions=component_versions,
        extra={"attempt_index": attempt_index},
    )


def metric_result_key(
    *,
    prediction_sha256: str,
    source_record_sha256: str,
    evaluation_policy: dict[str, Any],
    component_versions: dict[str, str],
    metric_name: str = "all",
) -> EvalCacheKey:
    """Build a content-addressed key for deterministic validation/metrics."""
    return EvalCacheKey(
        layer="metric",
        checkpoint_sha256=None,
        request_sha256=prediction_sha256,
        policy=_safe_json(evaluation_policy),
        component_versions=component_versions,
        extra={
            "source_record_sha256": source_record_sha256,
            "metric_name": metric_name,
        },
    )


def _safe_json(value: Any) -> Any:
    """Drop non-finite floats so JSON serialization never fails."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value
