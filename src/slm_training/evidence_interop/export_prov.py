"""W3C PROV-O JSON-LD projection (thin, deterministic)."""

from __future__ import annotations

from typing import Any

from slm_training.evidence_interop._envelope import base_envelope
from slm_training.evidence_interop.authority import extract_authority
from slm_training.evidence_interop.profile import PROV_O_MAPPING

FORMAT = "prov-o-jsonld"


def export_prov_o(obj: Any) -> dict[str, Any]:
    """Project a native object to a PROV-O JSON-LD document.

    Entities carry authority digests; activities/agents are documentary only.
    Does **not** prove checker passage, ship gates, or Lean kernel success.
    """
    authority = extract_authority(obj)
    entity_id = f"urn:slm:{authority.kind}:{_primary_id(authority.authority_ids)}"
    activity_id = f"{entity_id}#export"
    agent_id = "urn:slm:agent:evidence-interop-exporter"

    graph = [
        {
            "@id": entity_id,
            "@type": PROV_O_MAPPING["entity"],
            "prov:label": authority.kind,
            "slm:authority_ids": dict(authority.authority_ids),
        },
        {
            "@id": activity_id,
            "@type": PROV_O_MAPPING["activity"],
            "prov:label": "project-to-prov-o",
        },
        {
            "@id": agent_id,
            "@type": PROV_O_MAPPING["agent"],
            "prov:label": "slm-training-evidence-interop",
        },
        {
            "@id": f"{entity_id}#links",
            PROV_O_MAPPING["generation"]: {"@id": activity_id},
            PROV_O_MAPPING["attribution"]: {"@id": agent_id},
            PROV_O_MAPPING["derivation"]: {"@id": entity_id},
        },
    ]

    payload = base_envelope(
        kind=authority.kind, authority=authority, format_name=FORMAT
    )
    payload.update(
        {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "slm": "https://slm-training.invalid/ns/evidence-interop#",
            },
            "@graph": graph,
        }
    )
    return payload


def _primary_id(ids: dict[str, Any]) -> str:
    for key in (
        "object_id",
        "run_id",
        "campaign_id",
        "snapshot_id",
        "content_digest",
    ):
        value = ids.get(key)
        if value:
            return str(value)
    return "unknown"
