"""Canonical, fail-closed provenance for a synced checkpoint.

`CheckpointReferenceV1` is the single repository-owned schema that makes a
checkpoint cited by a campaign, frontier comparison, README/model-card row, or
follow-on issue resolvable and verifiable **from a fresh clone**. It is a
description of an already-persisted artifact, not a second persistence
subsystem: the bucket sync in :mod:`checkpoint_bucket` produces and uploads it.

Design rules (see ``docs/design/checkpoint-provenance.md`` and the EFS
experiment execution contract):

* Provenance is **never inferred from a filename**. Any field whose value is
  unknown stays the explicit sentinel :data:`UNKNOWN` (or ``None``) and blocks
  ``frontier``/``ship_candidate`` publication.
* ``fixture`` and ``diagnostic`` references may stay local-only (no durable
  remote is required) as long as they are honestly classified. They can never
  be treated as a frontier or ship claim.
* Serialization is deterministic (canonical JSON), so the reference has a
  stable :pyattr:`~CheckpointReferenceV1.sha` and a byte-stable on-disk form
  that round-trips exactly.

The type is intentionally Torch-free and import-light so the resolver/audit can
load it from an empty environment.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from slm_training.harness_core.lineage.records import canonical_json, content_sha

__all__ = [
    "SCHEMA_VERSION",
    "VERIFIER_VERSION",
    "UNKNOWN",
    "ClaimClass",
    "DURABLE_CLAIM_CLASSES",
    "FileArtifact",
    "CheckpointReferenceV1",
    "sha256_file",
    "file_artifact",
    "reference_filename",
    "MANIFEST_FILENAME",
    "build_references",
    "reference_manifest",
    "write_reference_sidecars",
]

#: Schema identifier embedded in every serialized reference.
SCHEMA_VERSION = "checkpoint_reference/v1"

#: Version of the sync-time remote verifier that stamps ``verification_*``.
VERIFIER_VERSION = "checkpoint_reference_verifier/v1"

#: Aggregate manifest filename written alongside per-checkpoint sidecars.
MANIFEST_FILENAME = "checkpoint_references.json"

#: Explicit sentinel for provenance that is not (yet) known. Never guessed.
UNKNOWN = "UNKNOWN"

#: The claim classes a reference can carry, ordered by evidentiary strength.
ClaimClass = Literal["fixture", "diagnostic", "frontier", "ship_candidate"]

#: Claim classes that require a durable, verified, fresh-clone-resolvable
#: artifact. ``fixture``/``diagnostic`` are intentionally excluded.
DURABLE_CLAIM_CLASSES: frozenset[str] = frozenset({"frontier", "ship_candidate"})

# Provenance fields that must be concretely present before a ``frontier`` or
# ``ship_candidate`` reference may be published. Maps field name -> the reason
# the field is required, surfaced verbatim in blocking errors.
_REQUIRED_FOR_DURABLE: dict[str, str] = {
    "remote_uri": "durable remote location is required to resolve the checkpoint",
    "bucket_id": "durable bucket id is required to resolve the checkpoint",
    "checkpoint_filename": "checkpoint filename is required",
    "size_bytes": "byte size is required to verify the checkpoint",
    "sha256": "SHA-256 is required to verify checkpoint integrity",
    "training_source_commit": "training source commit is required for lineage",
    "evaluation_source_commit": "evaluation source commit is required for lineage",
    "model_config_hash": "model config hash is required for identity",
    "tokenizer_hash": "tokenizer hash is required for identity",
    "output_codec_hash": "output codec hash is required for identity",
    "corpus_manifest_hash": "corpus manifest hash is required for data lineage",
    "data_version": "data version is required for data lineage",
    "verification_timestamp": "remote verification timestamp is required",
    "verifier_version": "verifier version is required",
}


def _is_missing(value: Any) -> bool:
    """True when ``value`` is an unfilled provenance slot."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value == UNKNOWN
    if isinstance(value, int) and not isinstance(value, bool):
        return value < 0
    return False


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256 hex digest (safe for multi-GB ``.pt`` weights)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileArtifact:
    """One uploaded companion file with its verifiable size and digest."""

    name: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileArtifact":
        return cls(
            name=str(data["name"]),
            size_bytes=int(data["size_bytes"]),
            sha256=str(data["sha256"]),
        )


def file_artifact(path: Path | str) -> FileArtifact:
    """Build a :class:`FileArtifact` by hashing a local file."""
    p = Path(path)
    return FileArtifact(name=p.name, size_bytes=p.stat().st_size, sha256=sha256_file(p))


def reference_filename(checkpoint_filename: str) -> str:
    """Canonical companion filename for a checkpoint's reference sidecar."""
    return f"{checkpoint_filename}.ref.json"


@dataclass(frozen=True)
class CheckpointReferenceV1:
    """Fail-closed provenance record for one checkpoint artifact.

    Only ``run_id``, ``claim_class``, ``checkpoint_role`` and
    ``checkpoint_filename`` are mandatory to construct. Every provenance field
    defaults to :data:`UNKNOWN`/``None`` so a caller can only *fill* provenance
    it actually has — it can never silently fabricate it. Whether the resulting
    record is publishable as a durable claim is decided by
    :meth:`blocking_reasons`.
    """

    run_id: str
    claim_class: ClaimClass
    checkpoint_role: str
    checkpoint_filename: str

    # Integrity of the primary checkpoint artifact.
    size_bytes: int | None = None
    sha256: str = UNKNOWN

    # Durable location.
    remote_uri: str = UNKNOWN
    bucket_id: str = UNKNOWN

    # Source lineage.
    training_source_commit: str = UNKNOWN
    evaluation_source_commit: str = UNKNOWN
    parent_uri: str | None = None
    parent_sha256: str | None = None

    # Model / decode identity.
    model_config_hash: str = UNKNOWN
    tokenizer_hash: str = UNKNOWN
    output_codec_hash: str = UNKNOWN
    context_tower_id: str = UNKNOWN

    # Data lineage.
    corpus_manifest_hash: str = UNKNOWN
    split_hashes: tuple[tuple[str, str], ...] = ()
    data_version: str = UNKNOWN

    # Training exposure.
    train_steps: int | None = None
    train_tokens: int | None = None
    seed: int | None = None

    # Sync / verification.
    sync_timestamp: str | None = None
    verification_timestamp: str | None = None
    verifier_version: str | None = None

    # Uploaded companion files (tokenizer/config/meta/train summary/...).
    companion_files: tuple[FileArtifact, ...] = ()

    schema_version: str = SCHEMA_VERSION
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.claim_class not in {"fixture", "diagnostic", "frontier", "ship_candidate"}:
            raise ValueError(f"unknown claim_class {self.claim_class!r}")
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.checkpoint_filename:
            raise ValueError("checkpoint_filename is required")
        # Provenance must never be back-derived from the filename.
        if self.sha256 != UNKNOWN and len(self.sha256) != 64:
            raise ValueError(
                f"sha256 must be a 64-char hex digest or {UNKNOWN!r}, got {self.sha256!r}"
            )

    # -- provenance completeness -------------------------------------------------

    @property
    def requires_durable(self) -> bool:
        return self.claim_class in DURABLE_CLAIM_CLASSES

    def blocking_reasons(self) -> tuple[tuple[str, str], ...]:
        """Fields blocking durable publication, as ``(field, reason)`` pairs.

        Empty for ``fixture``/``diagnostic`` references (they never claim a
        durable artifact) and for a fully-provenanced durable reference.
        """
        if not self.requires_durable:
            return ()
        reasons: list[tuple[str, str]] = []
        for name, reason in _REQUIRED_FOR_DURABLE.items():
            if _is_missing(getattr(self, name)):
                reasons.append((name, reason))
        return tuple(reasons)

    @property
    def is_publishable(self) -> bool:
        """True when this reference may back a claim of its declared class."""
        return not self.blocking_reasons()

    def require_publishable(self) -> None:
        """Raise if a durable claim is missing required provenance (fail closed)."""
        reasons = self.blocking_reasons()
        if reasons:
            detail = "; ".join(f"{name}: {why}" for name, why in reasons)
            raise ValueError(
                f"checkpoint reference for run_id={self.run_id!r} "
                f"role={self.checkpoint_role!r} cannot be published as "
                f"{self.claim_class!r}: {detail}"
            )

    # -- serialization -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointReferenceV1":
        version = str(data.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint reference schema_version {version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        companions = tuple(
            FileArtifact.from_dict(entry) for entry in data.get("companion_files", ())
        )
        split_hashes = tuple(
            (str(k), str(v)) for k, v in (data.get("split_hashes") or ())
        )
        metadata = tuple((str(k), str(v)) for k, v in (data.get("metadata") or ()))
        size = data.get("size_bytes")
        return cls(
            run_id=str(data["run_id"]),
            claim_class=str(data["claim_class"]),  # type: ignore[arg-type]
            checkpoint_role=str(data["checkpoint_role"]),
            checkpoint_filename=str(data["checkpoint_filename"]),
            size_bytes=None if size is None else int(size),
            sha256=str(data.get("sha256", UNKNOWN)),
            remote_uri=str(data.get("remote_uri", UNKNOWN)),
            bucket_id=str(data.get("bucket_id", UNKNOWN)),
            training_source_commit=str(data.get("training_source_commit", UNKNOWN)),
            evaluation_source_commit=str(data.get("evaluation_source_commit", UNKNOWN)),
            parent_uri=_opt_str(data.get("parent_uri")),
            parent_sha256=_opt_str(data.get("parent_sha256")),
            model_config_hash=str(data.get("model_config_hash", UNKNOWN)),
            tokenizer_hash=str(data.get("tokenizer_hash", UNKNOWN)),
            output_codec_hash=str(data.get("output_codec_hash", UNKNOWN)),
            context_tower_id=str(data.get("context_tower_id", UNKNOWN)),
            corpus_manifest_hash=str(data.get("corpus_manifest_hash", UNKNOWN)),
            split_hashes=split_hashes,
            data_version=str(data.get("data_version", UNKNOWN)),
            train_steps=_opt_int(data.get("train_steps")),
            train_tokens=_opt_int(data.get("train_tokens")),
            seed=_opt_int(data.get("seed")),
            sync_timestamp=_opt_str(data.get("sync_timestamp")),
            verification_timestamp=_opt_str(data.get("verification_timestamp")),
            verifier_version=_opt_str(data.get("verifier_version")),
            companion_files=companions,
            schema_version=SCHEMA_VERSION,
            metadata=metadata,
        )

    def to_json(self) -> str:
        """Deterministic, human-readable JSON (stable key order)."""
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)

    def canonical_json(self) -> str:
        """Compact canonical JSON used for hashing / byte-stability checks."""
        return canonical_json(self.to_dict())

    def write_json(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p

    @classmethod
    def load_json(cls, path: Path | str) -> "CheckpointReferenceV1":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- construction helpers ----------------------------------------------------

    @classmethod
    def for_local_file(
        cls,
        checkpoint_path: Path | str,
        *,
        run_id: str,
        claim_class: ClaimClass,
        checkpoint_role: str,
        companions: tuple[Path | str, ...] = (),
        **provenance: Any,
    ) -> "CheckpointReferenceV1":
        """Build a reference from a local checkpoint, hashing it and companions.

        Only integrity (size/sha256) and the file inventory are derived from
        disk. Every other provenance value must be supplied explicitly via
        ``provenance``; anything omitted stays :data:`UNKNOWN`.
        """
        p = Path(checkpoint_path)
        companion_files = tuple(file_artifact(c) for c in companions)
        return cls(
            run_id=run_id,
            claim_class=claim_class,
            checkpoint_role=checkpoint_role,
            checkpoint_filename=p.name,
            size_bytes=p.stat().st_size,
            sha256=sha256_file(p),
            companion_files=companion_files,
            **provenance,
        )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


# --- reference/manifest builders (used by the bucket sync path) ----------------

# Provenance a caller may attach to a synced reference. Anything not supplied
# stays UNKNOWN — never back-filled from a filename.
_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "training_source_commit",
        "evaluation_source_commit",
        "parent_uri",
        "parent_sha256",
        "model_config_hash",
        "tokenizer_hash",
        "output_codec_hash",
        "context_tower_id",
        "corpus_manifest_hash",
        "split_hashes",
        "data_version",
        "train_steps",
        "train_tokens",
        "seed",
    }
)


def _filter_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if not provenance:
        return {}
    return {
        key: value
        for key, value in provenance.items()
        if key in _PROVENANCE_KEYS and value is not None
    }


def build_references(
    *,
    staged_dir: Path | str,
    checkpoint_names: tuple[str, ...],
    run_id: str,
    remote_uri: str,
    bucket_id: str,
    claim_class: ClaimClass,
    sync_timestamp: str | None = None,
    verification_timestamp: str | None = None,
    verifier_version: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> list[CheckpointReferenceV1]:
    """Build one reference per checkpoint file in ``staged_dir``.

    Each checkpoint's role is its filename stem. Companion files are attached by
    stem prefix (e.g. ``last.tokenizer.json`` -> role ``last``); files that do
    not match any checkpoint stem (e.g. ``train_summary.json``) are shared
    across every reference. Reference sidecars/manifests are never treated as
    companions. Size/SHA-256 are computed from disk; every other field comes
    from ``provenance`` or stays UNKNOWN.
    """
    staged = Path(staged_dir)
    prov = _filter_provenance(provenance)
    checkpoint_stems = {Path(name).stem for name in checkpoint_names}
    all_files = sorted(
        path
        for path in staged.iterdir()
        if path.is_file()
        and not path.name.endswith(".ref.json")
        and path.name != MANIFEST_FILENAME
    )
    refs: list[CheckpointReferenceV1] = []
    for name in sorted(checkpoint_names):
        path = staged / name
        stem = Path(name).stem
        companions = tuple(
            f
            for f in all_files
            if f.name != name
            and (
                f.name.startswith(stem + ".")
                or Path(f.name).stem not in checkpoint_stems
            )
        )
        refs.append(
            CheckpointReferenceV1.for_local_file(
                path,
                run_id=run_id,
                claim_class=claim_class,
                checkpoint_role=stem,
                companions=companions,
                remote_uri=f"{remote_uri.rstrip('/')}/{name}",
                bucket_id=bucket_id,
                sync_timestamp=sync_timestamp,
                verification_timestamp=verification_timestamp,
                verifier_version=verifier_version,
                **prov,
            )
        )
    return refs


def reference_manifest(
    references: list[CheckpointReferenceV1],
    *,
    run_id: str,
    remote_uri: str,
    bucket_id: str,
    claim_class: ClaimClass,
    sync_timestamp: str | None = None,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate manifest describing every checkpoint reference for a run."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "remote_uri": remote_uri,
        "bucket_id": bucket_id,
        "claim_class": claim_class,
        "sync_timestamp": sync_timestamp,
        "verification": dict(verification) if verification is not None else None,
        "references": [ref.to_dict() for ref in references],
    }


def write_reference_sidecars(
    target_dir: Path | str,
    references: list[CheckpointReferenceV1],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> list[str]:
    """Write ``<checkpoint>.ref.json`` sidecars (+ optional manifest) into a dir."""
    import json

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for ref in references:
        sidecar = target / reference_filename(ref.checkpoint_filename)
        ref.write_json(sidecar)
        written.append(sidecar.name)
    if manifest is not None:
        manifest_path = target / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(dict(manifest), sort_keys=True, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        written.append(MANIFEST_FILENAME)
    return sorted(written)
