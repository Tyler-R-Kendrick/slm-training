"""RO-Crate metadata projection (thin, deterministic)."""

from __future__ import annotations

from typing import Any

from slm_training.evidence_interop._envelope import base_envelope
from slm_training.evidence_interop.authority import extract_authority
from slm_training.evidence_interop.profile import RO_CRATE_MAPPING

FORMAT = "ro-crate"


def export_ro_crate(obj: Any) -> dict[str, Any]:
    """Project a native object to a minimal RO-Crate 1.1 metadata document.

    Portable packaging metadata only — does **not** prove experiment honesty,
    publishability, or formal loop closure.
    """
    authority = extract_authority(obj)
    primary = _primary_id(authority.authority_ids)
    root_id = "./"
    file_id = f"./{authority.kind}/{primary}.json"

    graph = [
        {
            "@id": root_id,
            "@type": RO_CRATE_MAPPING["root"],
            "name": f"slm-{authority.kind}",
            "description": "Projection-only research object; not semantic authority.",
            "hasPart": [{"@id": file_id}],
        },
        {
            "@id": file_id,
            "@type": RO_CRATE_MAPPING["file"],
            "name": primary,
            "encodingFormat": "application/json",
            "identifier": primary,
            "sha256": _best_digest(authority.authority_ids),
        },
        {
            "@id": "#exporter",
            "@type": RO_CRATE_MAPPING["software"],
            "name": "slm-training-evidence-interop",
            "version": "v1",
        },
    ]

    payload = base_envelope(
        kind=authority.kind, authority=authority, format_name=FORMAT
    )
    payload.update(
        {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": graph,
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
        "corpus_sha256",
        "records_sha",
        "locked_eval_manifest_sha256",
        "checkpoint_sha256",
        "source_commit",
    ):
        value = ids.get(key)
        if value:
            return str(value)
    return None
