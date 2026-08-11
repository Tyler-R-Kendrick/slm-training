"""Authority extraction — repository IDs remain canonical (EVID-12).

Importing an external projection never upgrades semantic authority. Callers
that need certified FormalObject / campaign construction must use the native
constructors and checkers, not these projections.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from slm_training.evidence_interop.profile import (
    AUTHORITY_FIELDS,
    KIND_KEY,
    PROFILE_ID,
    SEMANTIC_AUTHORITY_DEFAULT,
)

__all__ = [
    "AuthorityRecord",
    "ProjectionView",
    "UnsupportedFieldError",
    "AuthorityRoundtripError",
    "extract_authority",
    "authority_ids_equal",
]


class UnsupportedFieldError(ValueError):
    """Projection carries predicates outside the thin interop profile."""


class AuthorityRoundtripError(ValueError):
    """Round-trip dropped or mutated an authority-relevant identifier."""


def _freeze(value: Any) -> Any:
    """Recursively freeze mappings/sequences so AuthorityRecord stays immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze(item) for item in value]
    return deepcopy(value) if isinstance(value, (dict, list)) else value


@dataclass(frozen=True)
class AuthorityRecord:
    """Flat authority-relevant identifiers extracted from a native object."""

    kind: str
    authority_ids: Mapping[str, Any]
    source_schema: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "authority_ids", _freeze(self.authority_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "authority_ids": _unfreeze(self.authority_ids),
            "source_schema": self.source_schema,
            "profile": PROFILE_ID,
            "semantic_authority": SEMANTIC_AUTHORITY_DEFAULT,
        }


@dataclass(frozen=True)
class ProjectionView:
    """Result of validating an external projection.

    ``semantic_authority`` is **always** False. A provenance record alone never
    certifies a FormalObject, campaign lock, or ship-gate claim.
    """

    format: str
    kind: str
    authority_ids: dict[str, Any]
    unsupported_fields: tuple[str, ...]
    semantic_authority: bool = False
    profile_id: str = PROFILE_ID

    def __post_init__(self) -> None:
        if self.semantic_authority:
            raise ValueError(
                "importing a projection must never set semantic_authority=True"
            )


def _digest_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "unknown":
        return None
    return text


def extract_authority(obj: Any) -> AuthorityRecord:
    """Extract authority-relevant identifiers from a native repo object."""
    # Late imports keep this module import-light for verifiers.
    from slm_training.formal.objects import FormalObjectV1
    from slm_training.harness_core.evidence_bundle import EvidenceBundleV1
    from slm_training.harness_core.lineage.records import DataSnapshot

    if isinstance(obj, FormalObjectV1):
        ids = {
            "object_id": obj.object_id,
            "content_digest": obj.content_digest,
            "required_checkers": list(obj.required_checkers),
        }
        return AuthorityRecord(
            kind="formal_object",
            authority_ids=ids,
            source_schema=obj.schema,
        )

    if isinstance(obj, EvidenceBundleV1):
        suite_hashes = {
            name: ref.sha256 for name, ref in sorted(obj.suite_hashes.items())
        }
        raw = [ref.sha256 for ref in obj.raw_envelopes]
        ids = {
            "run_id": obj.run_id,
            "config_sha256": obj.config_sha256,
            "corpus_sha256": obj.corpus_sha256,
            "checkpoint_sha256": (
                obj.checkpoint.sha256 if obj.checkpoint is not None else None
            ),
            "suite_hashes": suite_hashes,
            "raw_envelope_sha256": raw,
        }
        return AuthorityRecord(
            kind="evidence_bundle",
            authority_ids=ids,
            source_schema="evidence_bundle/v1",
        )

    if isinstance(obj, DataSnapshot):
        ids = {
            "snapshot_id": obj.snapshot_id,
            "records_sha": obj.records_sha,
            "annotations_sha": _digest_or_none(obj.annotations_sha),
        }
        return AuthorityRecord(
            kind="data_snapshot",
            authority_ids=ids,
            source_schema="data_snapshot",
        )

    # ExperimentCampaignV1 is a pydantic StrictModel — duck-type by fields.
    if hasattr(obj, "campaign_id") and hasattr(obj, "source_commit"):
        schema = getattr(obj, "schema_version", "ExperimentCampaignV1")
        locked = getattr(obj, "locked_eval_manifest_sha256", None)
        ids = {
            "campaign_id": str(obj.campaign_id),
            "locked_eval_manifest_sha256": locked,
            "source_commit": str(obj.source_commit),
        }
        return AuthorityRecord(
            kind="experiment_campaign",
            authority_ids=ids,
            source_schema=str(schema),
        )

    if isinstance(obj, Mapping):
        kind = str(obj.get(KIND_KEY) or obj.get("kind") or "")
        if kind in AUTHORITY_FIELDS and "authority_ids" in obj:
            return AuthorityRecord(
                kind=kind,
                authority_ids=dict(obj["authority_ids"]),
                source_schema=str(obj.get("source_schema", "mapping")),
            )

    raise TypeError(
        f"unsupported native object for authority extraction: {type(obj)!r}"
    )


def authority_ids_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Structural equality with list/tuple normalization."""
    return _normalize(left) == _normalize(right)


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value
