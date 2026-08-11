"""OCI subject / referrer descriptor projection (thin, local-only)."""

from __future__ import annotations

from typing import Any, Mapping

from slm_training.evidence_interop._envelope import base_envelope, canonical_json
from slm_training.evidence_interop.authority import extract_authority
from slm_training.evidence_interop.profile import OCI_MEDIA_TYPES

FORMAT = "oci-referrer"


def export_oci_referrer(obj: Any) -> dict[str, Any]:
    """Project authority digests as an OCI subject + referrer descriptor.

    Useful for attaching proof/evidence blobs to an artifact subject. Does
    **not** prove registry presence, signature validity, or semantic authority.
    No registry client is invoked.

    The subject descriptor hashes the canonical authority payload itself so
    digest and size always describe real bytes (no synthetic zero digest).
    """
    import hashlib

    authority = extract_authority(obj)
    primary = _primary_id(authority.authority_ids)
    authority_bytes = canonical_json(authority.to_dict()).encode("utf-8")
    authority_sha = hashlib.sha256(authority_bytes).hexdigest()
    media = OCI_MEDIA_TYPES["slm_authority"]

    subject = {
        "mediaType": media,
        "digest": f"sha256:{authority_sha}",
        "size": len(authority_bytes),
        "annotations": {
            "org.opencontainers.image.title": primary,
            "slm.kind": authority.kind,
        },
    }
    # Referrer annotates the same authority blob (local projection only).
    referrer = {
        "mediaType": media,
        "digest": f"sha256:{authority_sha}",
        "size": len(authority_bytes),
        "artifactType": media,
        "annotations": {
            "slm.profile": "slm-evidence-interop/v1",
            "slm.semantic_authority": "false",
            "slm.role": "authority-referrer",
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
            "subject_bytes_sha256": authority_sha,
        }
    )
    return payload


def _primary_id(ids: Mapping[str, Any]) -> str:
    for key in ("object_id", "run_id", "campaign_id", "snapshot_id"):
        value = ids.get(key)
        if value:
            return str(value)
    return "unknown"
