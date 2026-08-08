"""Hermetic replay bundles over existing decision/identity evidence (PCT-002).

A ``ReplayBundleV1`` composes evidence this repository already owns --
``DecodeIdentityV1`` (``models/decode_stats.py``) and solver-transition
events validated by ``solver_replay_violations``
(``dsl/solver/replay.py``) -- into one compact, content-bound, tamper-evident
envelope that can be replayed offline (re-validated from its own recorded
fields) without re-running any model. It introduces no new decode/verifier
authority: ``replayable``/``non_replayable_reasons`` only ever report what
the existing identity and solver-replay checks already decided.

Observability stays optional and non-authoritative: ``mirror_bundle_best_effort``
never raises, matching the existing OTLP-mirror contract in
``runtime/telemetry/trace.py``, so an export failure can never influence a
decode/evaluation decision. Persistence reuses whatever append-only,
resume-safe store the caller already has (e.g.
``harnesses/distill/trace_store.py::TraceStore``) via a minimal duck-typed
protocol -- this module never opens its own store.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Protocol

from slm_training.dsl.solver.replay import SOLVER_EVENT_KINDS, solver_replay_violations
from slm_training.models.decode_stats import DecodeIdentityV1

REPLAY_BUNDLE_SCHEMA = "replay_bundle/v1"

_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^\s,;]+)"),
)
_MAX_FIELD_CHARS = 240


def _redact_string(text: str) -> str:
    """Strip secret-shaped substrings and bound length before a value enters
    a replay bundle. A lightweight, independently owned copy of the same
    pattern used in ``evals/semantic_failure.py::_scrub_detail`` -- that
    helper is private/unexported and diagnoses a different evidence type."""
    scrubbed = text
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)
    if len(scrubbed) > _MAX_FIELD_CHARS:
        scrubbed = f"{scrubbed[:_MAX_FIELD_CHARS]}...[truncated]"
    return scrubbed


def _redact_shallow(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact top-level string values only -- bounded, not a general recursive
    scrubber. Callers owning nested structures redact before passing them in."""
    return {
        key: (_redact_string(value) if isinstance(value, str) else value)
        for key, value in payload.items()
    }


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _bundle_digest(bundle: "ReplayBundleV1") -> str:
    payload = bundle.to_dict()
    payload.pop("bundle_hash", None)
    return _canonical_digest(payload)


@dataclass(frozen=True)
class ReplayBundleV1:
    """Immutable, tamper-evident, offline-replayable evidence envelope.

    Produced only by ``build_replay_bundle``. Every identity/domain field
    mirrors the no-fabrication contract of ``DecodeIdentityV1``/
    ``DecodeStatsRecordV1`` -- absent evidence stays explicit ``None`` rather
    than being invented.
    """

    schema_version: str
    request_id: str
    seed: int | None
    identity: dict[str, Any]
    decoder_choices: tuple[dict[str, Any], ...]
    legal_domain_size: int | None
    legal_domain_status: str
    legal_domain_digest: str | None
    artifact_state: dict[str, Any]
    final_evidence: dict[str, Any]
    replayable: bool
    non_replayable_reasons: tuple[str, ...]
    bundle_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "seed": self.seed,
            "identity": dict(self.identity),
            "decoder_choices": [dict(c) for c in self.decoder_choices],
            "legal_domain_size": self.legal_domain_size,
            "legal_domain_status": self.legal_domain_status,
            "legal_domain_digest": self.legal_domain_digest,
            "artifact_state": dict(self.artifact_state),
            "final_evidence": dict(self.final_evidence),
            "replayable": self.replayable,
            "non_replayable_reasons": list(self.non_replayable_reasons),
            "bundle_hash": self.bundle_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayBundleV1":
        if data.get("schema_version") != REPLAY_BUNDLE_SCHEMA:
            raise ValueError(
                f"unsupported replay bundle schema: {data.get('schema_version')!r}"
            )
        bundle = cls(
            schema_version=data.get("schema_version", REPLAY_BUNDLE_SCHEMA),
            request_id=data.get("request_id", ""),
            seed=data.get("seed"),
            identity=dict(data.get("identity", {})),
            decoder_choices=tuple(
                dict(c) for c in data.get("decoder_choices", [])
            ),
            legal_domain_size=data.get("legal_domain_size"),
            legal_domain_status=data.get("legal_domain_status", "unknown"),
            legal_domain_digest=data.get("legal_domain_digest"),
            artifact_state=dict(data.get("artifact_state", {})),
            final_evidence=dict(data.get("final_evidence", {})),
            replayable=bool(data.get("replayable", False)),
            non_replayable_reasons=tuple(data.get("non_replayable_reasons", [])),
            bundle_hash=data.get("bundle_hash", ""),
        )
        if _bundle_digest(bundle) != bundle.bundle_hash:
            raise ValueError(
                "replay bundle hash mismatch (tampered or corrupted payload)"
            )
        return bundle


def replay_bundle_integrity_ok(bundle: ReplayBundleV1) -> bool:
    """Recompute the bundle hash and compare; used by replay/tamper checks."""
    return _bundle_digest(bundle) == bundle.bundle_hash


def build_replay_bundle(
    *,
    request_id: str,
    identity: DecodeIdentityV1 | None = None,
    seed: int | None = None,
    decoder_choices: tuple[dict[str, Any], ...] = (),
    legal_domain_size: int | None = None,
    legal_domain_status: str = "unknown",
    legal_domain_digest: str | None = None,
    artifact_state: dict[str, Any] | None = None,
    final_evidence: dict[str, Any] | None = None,
) -> ReplayBundleV1:
    """Compose an offline-replayable bundle around already-decided evidence.

    Pure and stateless: never calls into telemetry (``get_telemetry``/
    ``current_trace``) or any decoder, so observability being bound or not
    can never change the resulting bundle for identical inputs. Fails
    closed -- a missing request id, missing checkpoint identity, or a
    solver-replay violation in ``decoder_choices`` marks the bundle
    ``replayable=False`` with an explicit reason instead of silently
    treating the sequence as reconstructible.
    """
    reasons: list[str] = []
    if not request_id:
        reasons.append("missing request_id")
    ident = identity or DecodeIdentityV1()
    if ident.checkpoint_sha256 is None:
        reasons.append("missing checkpoint identity")

    solver_events = [
        event for event in decoder_choices if event.get("kind") in SOLVER_EVENT_KINDS
    ]
    if solver_events:
        reasons.extend(
            f"solver replay violation: {violation}"
            for violation in solver_replay_violations(solver_events)
        )

    bundle = ReplayBundleV1(
        schema_version=REPLAY_BUNDLE_SCHEMA,
        request_id=request_id,
        seed=seed,
        identity=ident.to_dict(),
        decoder_choices=tuple(decoder_choices),
        legal_domain_size=legal_domain_size,
        legal_domain_status=legal_domain_status,
        legal_domain_digest=legal_domain_digest,
        artifact_state=_redact_shallow(artifact_state or {}),
        final_evidence=_redact_shallow(final_evidence or {}),
        replayable=not reasons,
        non_replayable_reasons=tuple(reasons),
        bundle_hash="",
    )
    return replace(bundle, bundle_hash=_bundle_digest(bundle))


def mirror_bundle_best_effort(
    bundle: ReplayBundleV1, sink: Callable[[dict[str, Any]], None] | None
) -> bool:
    """Best-effort optional export; any sink failure is swallowed.

    Mirrors ``runtime/telemetry/trace.py``'s OTLP-mirror contract: observability
    is additive and non-authoritative, so a broken/misconfigured/offline sink
    must never raise into the caller's decode/evaluation path. Returns
    whether the mirror call itself completed without error.
    """
    if sink is None:
        return False
    try:
        sink(bundle.to_dict())
        return True
    except Exception:  # noqa: BLE001 - export failure must never propagate
        return False


class ReplayBundleStore(Protocol):
    """Minimal duck-typed subset of ``TraceStore`` this module depends on.

    Any append-only, resume-safe store exposing this shape (``TraceStore``
    already does) works -- this module never opens or owns a store itself.
    """

    def append(self, row: dict[str, Any]) -> str: ...

    def iter_kind(self, kind: str) -> Iterator[dict[str, Any]]: ...


_REPLAY_BUNDLE_KIND = "replay_bundle"


def append_replay_bundle(store: ReplayBundleStore, bundle: ReplayBundleV1) -> str:
    """Append one bundle to an existing append-only store, unmodified."""
    return store.append({"kind": _REPLAY_BUNDLE_KIND, "bundle": bundle.to_dict()})


def iter_replay_bundles(store: ReplayBundleStore) -> Iterator[ReplayBundleV1]:
    """Read replay bundles back from an existing append-only store."""
    for row in store.iter_kind(_REPLAY_BUNDLE_KIND):
        yield ReplayBundleV1.from_dict(row["bundle"])
