"""OCI subject / referrer descriptor projection (thin, local-only)."""

from __future__ import annotations

from typing import Any

from slm_training.evidence_interop._envelope import base_envelope, canonical_json
from slm_training.evidence_interop.authority import extract_authority
from slm_training.evidence_interop.profile import OCI_MEDIA_TYPES

FORMAT = "oci-referrer"


def export_oci_referrer(obj: Any) -> dict[str, Any]:
    """Project authority digests as an OCI subject + referrer descriptor.

    Useful for attaching proof/evidence blobs to an artifact subject. Does
    **not** prove registry presence, signature validity, or semantic authority.
    No registry client is invoked.
    """
    import hashlib

    authority = extract_authority(obj)
    primary = _primary_id(authority.authority_ids)
    subject_digest = _best_digest(authority.authority_ids) or "0" * 64
    subject_digest = _strip_algo(subject_digest)

    authority_bytes = canonical_json(authority.to_dict()).encode("utf-8")
    authority_sha = hashlib.sha256(authority_bytes).hexdigest()

    subject = {
        "mediaType": OCI_MEDIA_TYPES["descriptor"],
        "digest": f"sha256:{subject_digest}",
        "size": 0,
        "annotations": {
            "org.opencontainers.image.title": primary,
            "slm.kind": authority.kind,
        },
    }
    referrer = {
        "mediaType": OCI_MEDIA_TYPES["slm_authority"],
        "digest": f"sha256:{authority_sha}",
        "size": len(authority_bytes),
        "artifactType": OCI_MEDIA_TYPES["slm_authority"],
        "annotations": {
            "slm.profile": "slm-evidence-interop/v1",
            "slm.semantic_authority": "false",
        },
    }

    payload = base_envelope(
        kind=authority.kind, authority=authority, format_name=FORMAT
    )
    payload.update(
        {
            "schema": "oci_referrer_projection/v1",
            "subject": subject,
            "referrers": [referrer],
        }
    )
    return payload


def _primary_id(ids: dict[str, Any]) -> str:
    for key in ("object_id", "run_id", "campaign_id", "snapshot_id"):
        value = ids.get(key)
        if value:
            return str(value)
    return "unknown"


def _best_digest(ids: dict[str, Any]) -> str | None:
    for key in (
        "content_digest",
        "config_sha256",
        "records_sha",
        "locked_eval_manifest_sha256",
        "corpus_sha256",
        "source_commit",
    ):
        value = ids.get(key)
        if value:
            return str(value)
    return None


def _strip_algo(digest: str) -> str:
    if digest.startswith("sha256:"):
        return digest[7:]
    return digest
