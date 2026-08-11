"""SPDX 3 thin projection for model/data/build metadata."""

from __future__ import annotations

from typing import Any

from slm_training.evidence_interop._envelope import base_envelope
from slm_training.evidence_interop.authority import extract_authority
from slm_training.evidence_interop.profile import SPDX_MAPPING

FORMAT = "spdx3"


def export_spdx(obj: Any) -> dict[str, Any]:
    """Project native metadata to a thin SPDX-3-shaped JSON document.

    Documents identity and checksums only. Does **not** prove Lean kernel
    checks, ship gates, or campaign lock integrity.
    """
    authority = extract_authority(obj)
    spdx_type = SPDX_MAPPING.get(authority.kind, "software_Package")
    primary = _primary_id(authority.authority_ids)
    digest = _best_digest(authority.authority_ids)
    spdx_id = f"SPDXRef-{authority.kind}-{_safe(primary)}"

    element: dict[str, Any] = {
        "SPDXID": spdx_id,
        "@type": spdx_type,
        "name": primary,
        "authority_ids": dict(authority.authority_ids),
    }
    if digest:
        element["checksums"] = [
            {"algorithm": "SHA256", "checksumValue": _strip_algo(digest)}
        ]

    payload = base_envelope(
        kind=authority.kind, authority=authority, format_name=FORMAT
    )
    payload.update(
        {
            "spdxVersion": "SPDX-3.0",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"slm-{authority.kind}-projection",
            "documentNamespace": (
                f"https://slm-training.invalid/spdx3/{authority.kind}/{primary}"
            ),
            "creationInfo": {
                "created": "1970-01-01T00:00:00Z",
                "creators": ["Tool: slm-training-evidence-interop"],
            },
            "documentDescribes": [spdx_id],
            "elements": [element],
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


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)[:120]
