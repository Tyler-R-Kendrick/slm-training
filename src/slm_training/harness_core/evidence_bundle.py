"""Content-addressed experiment evidence and preregistered claim bars (SLM-291).

`EvidenceBundleV1` is the fail-closed provenance envelope for one experiment
claim: source commit and dirty-tree status, full config digest, seeds,
environment digest, corpus and suite hashes, checkpoint digest/size/location,
raw result envelopes, evaluator and gate versions, and the claim class. A
non-fixture claim whose evidence cannot be resolved and hash-verified is not
publishable — the same honesty bar as
:mod:`slm_training.harness_core.checkpoint_reference`, generalized from a
single checkpoint to a whole experiment result.

Design rules:

* Provenance is never inferred. Unknown fields stay :data:`UNKNOWN` / ``None``
  and block non-fixture publication.
* ``fixture`` claims may reference local paths honestly; they can never be
  treated as production evidence.
* Serialization is deterministic (canonical JSON), so every bundle has a
  stable content hash usable as its address in a :class:`LocalEvidenceStore`.

The module is Torch-free and import-light so verifiers load from an empty
environment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from slm_training.harness_core.checkpoint_reference import (
    DURABLE_CLAIM_CLASSES,
    UNKNOWN,
    ClaimClass,
)
from slm_training.harness_core.lineage.records import content_sha

__all__ = [
    "SCHEMA_VERSION",
    "BAR_SCHEMA_VERSION",
    "VERIFIER_VERSION",
    "UNKNOWN",
    "ArtifactRef",
    "EvidenceBundleV1",
    "EvidenceStore",
    "LocalEvidenceStore",
    "PreregisteredClaimBarV1",
    "claim_bar_from_dict",
    "validate_claim_bar",
    "require_valid_claim_bar",
    "verify_evidence_bundle",
    "bundle_from_dict",
    "CAS_SCHEME",
]

#: Schema identifier embedded in every serialized bundle.
SCHEMA_VERSION = "evidence_bundle/v1"

#: Schema identifier for the preregistered evidence bar.
BAR_SCHEMA_VERSION = "preregistered_claim_bar/v1"

#: Version of this verifier, embedded in verification reports.
VERIFIER_VERSION = "evidence_bundle_verifier/v1"

#: URI scheme for objects in a content-addressed store.
CAS_SCHEME = "cas://sha256/"

# URI prefixes that count as durable, digest-pinned remotes. A bare ``hf://``
# URI is mutable without a recorded digest, so durability additionally
# requires the artifact's sha256 field to be set.
_DURABLE_SCHEMES = ("hf://", "cas://")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value == UNKNOWN
    if isinstance(value, int) and not isinstance(value, bool):
        return value < 0
    return False


@dataclass(frozen=True)
class ArtifactRef:
    """One hash-addressed artifact (checkpoint, suite file, raw envelope)."""

    name: str
    sha256: str
    size_bytes: int = -1
    uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "uri": self.uri,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            name=str(data["name"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data.get("size_bytes", -1)),
            uri=data.get("uri"),
        )

    @property
    def has_digest(self) -> bool:
        return not _is_missing(self.sha256) and len(self.sha256) == 64

    @property
    def durable(self) -> bool:
        """True only for a digest-pinned artifact at a durable location."""
        return (
            self.has_digest
            and self.uri is not None
            and self.uri.startswith(_DURABLE_SCHEMES)
        )


@dataclass(frozen=True)
class EvidenceBundleV1:
    """Fail-closed provenance envelope for one experiment claim."""

    run_id: str
    claim_class: ClaimClass
    source_commit: str = UNKNOWN
    source_dirty: bool | None = None
    config_sha256: str = UNKNOWN
    seeds: tuple[int, ...] = ()
    environment_digest: str = UNKNOWN
    corpus_sha256: str = UNKNOWN
    suite_hashes: Mapping[str, ArtifactRef] = field(default_factory=dict)
    checkpoint: ArtifactRef | None = None
    raw_envelopes: tuple[ArtifactRef, ...] = ()
    evaluator_version: str = UNKNOWN
    gate_version: str = UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "claim_class": self.claim_class,
            "source_commit": self.source_commit,
            "source_dirty": self.source_dirty,
            "config_sha256": self.config_sha256,
            "seeds": list(self.seeds),
            "environment_digest": self.environment_digest,
            "corpus_sha256": self.corpus_sha256,
            "suite_hashes": {
                name: ref.to_dict() for name, ref in sorted(self.suite_hashes.items())
            },
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "raw_envelopes": [ref.to_dict() for ref in self.raw_envelopes],
            "evaluator_version": self.evaluator_version,
            "gate_version": self.gate_version,
            "metadata": dict(self.metadata),
        }

    @property
    def sha(self) -> str:
        return content_sha(self)

    def blocking_reasons(self) -> list[str]:
        """Fail-closed publication bar for non-fixture claim classes."""
        if self.claim_class not in DURABLE_CLAIM_CLASSES:
            return []
        reasons: list[str] = []
        if _is_missing(self.source_commit):
            reasons.append("source commit is required for a non-fixture claim")
        if self.source_dirty is not False:
            reasons.append(
                "a clean source tree is required for a non-fixture claim "
                f"(source_dirty={self.source_dirty!r})"
            )
        if _is_missing(self.config_sha256):
            reasons.append("full config digest is required")
        if not self.seeds:
            reasons.append("at least one declared seed is required")
        if _is_missing(self.environment_digest):
            reasons.append("environment/package-lock digest is required")
        if _is_missing(self.corpus_sha256):
            reasons.append("corpus hash is required")
        if not self.suite_hashes:
            reasons.append("suite hashes are required")
        if self.checkpoint is not None and not self.checkpoint.durable:
            reasons.append(
                "checkpoint must be digest-pinned at a durable URI "
                f"(got uri={self.checkpoint.uri!r}, sha256={self.checkpoint.sha256!r})"
            )
        if _is_missing(self.evaluator_version):
            reasons.append("a pinned evaluator version is required")
        if _is_missing(self.gate_version):
            reasons.append("a pinned gate version is required")
        return reasons

    @property
    def is_publishable(self) -> bool:
        return not self.blocking_reasons()

    def require_publishable(self) -> None:
        reasons = self.blocking_reasons()
        if reasons:
            raise ValueError(
                f"evidence bundle {self.run_id!r} is not publishable: "
                + "; ".join(reasons)
            )


def bundle_from_dict(data: Mapping[str, Any]) -> EvidenceBundleV1:
    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"not an {SCHEMA_VERSION} payload: {data.get('schema')!r}")
    suites = data.get("suite_hashes") or {}
    checkpoint = data.get("checkpoint")
    dirty = data.get("source_dirty")
    return EvidenceBundleV1(
        run_id=str(data["run_id"]),
        claim_class=data["claim_class"],
        source_commit=str(data.get("source_commit", UNKNOWN)),
        source_dirty=None if dirty is None else bool(dirty),
        config_sha256=str(data.get("config_sha256", UNKNOWN)),
        seeds=tuple(int(s) for s in data.get("seeds", ())),
        environment_digest=str(data.get("environment_digest", UNKNOWN)),
        corpus_sha256=str(data.get("corpus_sha256", UNKNOWN)),
        suite_hashes={
            str(name): ArtifactRef.from_dict(ref) for name, ref in suites.items()
        },
        checkpoint=ArtifactRef.from_dict(checkpoint) if checkpoint else None,
        raw_envelopes=tuple(
            ArtifactRef.from_dict(ref) for ref in data.get("raw_envelopes", ())
        ),
        evaluator_version=str(data.get("evaluator_version", UNKNOWN)),
        gate_version=str(data.get("gate_version", UNKNOWN)),
        metadata=data.get("metadata") or {},
    )


def _resolve_artifact_uri(uri: str, artifact_root: Path) -> Path | None:
    """Map a recorded URI to a local file, or None when not locally resolvable."""
    if uri.startswith(_DURABLE_SCHEMES):
        return None
    candidate = Path(uri)
    if not candidate.is_absolute():
        candidate = artifact_root / candidate
    return candidate


def _verify_artifact(
    ref: ArtifactRef,
    *,
    kind: str,
    artifact_root: Path | None,
    store: "EvidenceStore | None",
) -> list[str]:
    """Hash-check one artifact's bytes when resolvable; fail with reasons."""
    if not ref.has_digest:
        return [f"{kind} {ref.name!r} has no SHA-256 digest"]
    data: bytes | None = None
    if ref.uri is not None and ref.uri.startswith(CAS_SCHEME):
        if store is None:
            return [f"{kind} {ref.name!r} is store-addressed but no store was provided"]
        if not store.exists(ref.sha256):
            return [f"{kind} {ref.name!r} is missing from the evidence store"]
        data = store.get(ref.sha256)
    elif ref.uri is not None and artifact_root is not None:
        path = _resolve_artifact_uri(ref.uri, artifact_root)
        if path is not None:
            if not path.is_file():
                return [f"{kind} {ref.name!r} is missing at {ref.uri!r}"]
            data = path.read_bytes()
    if data is None:
        # Durable remote without a local copy: digest-pinned by construction;
        # byte verification happened at sync time (checkpoint_bucket).
        if ref.durable:
            return []
        return [f"{kind} {ref.name!r} bytes are not resolvable for verification"]
    actual = hashlib.sha256(data).hexdigest()
    if actual != ref.sha256:
        return [
            f"{kind} {ref.name!r} bytes were altered: expected sha256 "
            f"{ref.sha256}, got {actual}"
        ]
    if ref.size_bytes >= 0 and len(data) != ref.size_bytes:
        return [
            f"{kind} {ref.name!r} size mismatch: expected {ref.size_bytes}, "
            f"got {len(data)}"
        ]
    return []


def verify_evidence_bundle(
    bundle: EvidenceBundleV1,
    *,
    artifact_root: Path | str | None = None,
    store: "EvidenceStore | None",
) -> list[str]:
    """Deterministic, no-write verification. Returns failure reasons ([] = ok).

    Applies the publication bar first, then hash-checks every resolvable
    artifact: checkpoint bytes, suite bytes, and raw result envelopes.
    """
    root = Path(artifact_root) if artifact_root is not None else None
    reasons = list(bundle.blocking_reasons())
    if bundle.checkpoint is not None:
        reasons.extend(
            _verify_artifact(
                bundle.checkpoint, kind="checkpoint", artifact_root=root, store=store
            )
        )
    for suite, ref in sorted(bundle.suite_hashes.items()):
        reasons.extend(
            _verify_artifact(
                ref, kind=f"suite {suite!r}", artifact_root=root, store=store
            )
        )
    for ref in bundle.raw_envelopes:
        reasons.extend(
            _verify_artifact(
                ref, kind="raw envelope", artifact_root=root, store=store
            )
        )
    return reasons


@runtime_checkable
class EvidenceStore(Protocol):
    """Content-addressed byte store. Backends are pluggable behind this API."""

    def put(self, data: bytes) -> str:
        """Store bytes; returns the SHA-256 hex address."""
        ...

    def get(self, sha256: str) -> bytes:
        ...

    def exists(self, sha256: str) -> bool:
        ...

    def uri(self, sha256: str) -> str:
        """Stable, hash-checked URI for the object."""
        ...


class LocalEvidenceStore:
    """Filesystem content-addressed store (``objects/<aa>/<sha256>``).

    Writes are create-only and digest-verified; a write whose bytes do not
    match their computed address raises rather than storing. A durable
    backend (bucket, remote CAS) plugs in by implementing the same protocol.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _path(self, sha256: str) -> Path:
        return self._root / "objects" / sha256[:2] / sha256

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        if path.exists():
            existing = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing != digest:
                raise ValueError(f"digest mismatch at {path}: store is corrupt")
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
        return digest

    def get(self, sha256: str) -> bytes:
        path = self._path(sha256)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise ValueError(f"digest mismatch at {path}: expected {sha256}")
        return data

    def exists(self, sha256: str) -> bool:
        return self._path(sha256).is_file()

    def uri(self, sha256: str) -> str:
        return f"{CAS_SCHEME}{sha256}"


@dataclass(frozen=True)
class PreregisteredClaimBarV1:
    """Versioned preregistered evidence bar for one experiment claim.

    Encodes what must be true *before* outcomes are visible: target metrics
    with thresholds, minimum sample size, declared seeds, the interval
    confidence level and required lower bound, the stop rule, the entry gate,
    and the falsifier that would reject the claim.
    """

    bar_id: str
    target_metrics: Mapping[str, float]
    min_sample_size: int
    seeds: tuple[int, ...]
    confidence_level: float = 0.95
    required_lower_bound: float | None = None
    stop_rule: str = ""
    entry_gate: str = ""
    falsifier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BAR_SCHEMA_VERSION,
            "bar_id": self.bar_id,
            "target_metrics": dict(sorted(self.target_metrics.items())),
            "min_sample_size": self.min_sample_size,
            "seeds": list(self.seeds),
            "confidence_level": self.confidence_level,
            "required_lower_bound": self.required_lower_bound,
            "stop_rule": self.stop_rule,
            "entry_gate": self.entry_gate,
            "falsifier": self.falsifier,
        }

    @property
    def sha(self) -> str:
        return content_sha(self)


def claim_bar_from_dict(data: Mapping[str, Any]) -> PreregisteredClaimBarV1:
    if data.get("schema") != BAR_SCHEMA_VERSION:
        raise ValueError(
            f"not a {BAR_SCHEMA_VERSION} payload: {data.get('schema')!r}"
        )
    return PreregisteredClaimBarV1(
        bar_id=str(data["bar_id"]),
        target_metrics={str(k): float(v) for k, v in data["target_metrics"].items()},
        min_sample_size=int(data["min_sample_size"]),
        seeds=tuple(int(s) for s in data["seeds"]),
        confidence_level=float(data.get("confidence_level", 0.95)),
        required_lower_bound=data.get("required_lower_bound"),
        stop_rule=str(data.get("stop_rule", "")),
        entry_gate=str(data.get("entry_gate", "")),
        falsifier=str(data.get("falsifier", "")),
    )


def validate_claim_bar(bar: PreregisteredClaimBarV1) -> list[str]:
    """Fail-closed sanity check; LAR harnesses must require this before running."""
    reasons: list[str] = []
    if not bar.bar_id.strip():
        reasons.append("bar_id is required")
    if not bar.target_metrics:
        reasons.append("at least one target metric with a threshold is required")
    for name, threshold in sorted(bar.target_metrics.items()):
        if not name.strip():
            reasons.append("target metric names must be non-empty")
        if not (threshold == threshold):  # NaN
            reasons.append(f"target metric {name!r} threshold is not finite")
    if bar.min_sample_size < 1:
        reasons.append("min_sample_size must be >= 1")
    if not bar.seeds:
        reasons.append("at least one declared seed is required")
    if len(set(bar.seeds)) != len(bar.seeds):
        reasons.append("seeds must be unique")
    if not (0.0 < bar.confidence_level < 1.0):
        reasons.append("confidence_level must be in (0, 1)")
    if bar.required_lower_bound is not None and not (
        0.0 <= bar.required_lower_bound <= 1.0
    ):
        reasons.append("required_lower_bound must be in [0, 1]")
    if not bar.stop_rule.strip():
        reasons.append("an explicit stop rule is required")
    if not bar.entry_gate.strip():
        reasons.append("an explicit entry gate is required")
    if not bar.falsifier.strip():
        reasons.append("an explicit falsifier is required")
    return reasons


def require_valid_claim_bar(bar: PreregisteredClaimBarV1) -> None:
    reasons = validate_claim_bar(bar)
    if reasons:
        raise ValueError(
            f"preregistered claim bar {bar.bar_id!r} is invalid: "
            + "; ".join(reasons)
        )
