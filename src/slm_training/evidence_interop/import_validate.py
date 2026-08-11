"""Import validation for evidence interop projections (fail closed)."""

from __future__ import annotations

from typing import Any, Mapping

from slm_training.evidence_interop.authority import (
    ProjectionView,
    UnsupportedFieldError,
)
from slm_training.evidence_interop.profile import (
    AUTHORITY_BLOCK_KEY,
    KIND_KEY,
    KNOWN_EXTERNAL_PREDICATES,
    PROFILE_BLOCK_KEY,
    PROFILE_ID,
    SEMANTIC_AUTHORITY_KEY,
    SUPPORTED_FORMATS,
)

__all__ = [
    "import_projection",
    "import_prov_o",
    "import_ro_crate",
    "import_spdx",
    "import_intoto",
    "import_oci_referrer",
]


def import_projection(
    document: Mapping[str, Any],
    *,
    expected_format: str | None = None,
    raise_on_unsupported: bool = False,
) -> ProjectionView:
    """Validate a projection and extract authority ids only.

    Always returns ``semantic_authority=False``. Never constructs a certified
    FormalObject, EvidenceBundle, or ExperimentCampaign from the document.
    """
    if not isinstance(document, Mapping):
        raise TypeError("projection document must be a mapping")

    profile = document.get(PROFILE_BLOCK_KEY)
    if profile is not None and profile != PROFILE_ID:
        raise ValueError(f"unsupported interop profile: {profile!r}")

    fmt = str(document.get("slm:format") or expected_format or _infer_format(document))
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported projection format: {fmt!r}")
    if expected_format is not None and fmt != expected_format:
        raise ValueError(
            f"format mismatch: expected {expected_format!r}, got {fmt!r}"
        )

    # Hard law: even if a malicious document claims True, force False.
    _ = document.get(SEMANTIC_AUTHORITY_KEY, False)

    authority_block = document.get(AUTHORITY_BLOCK_KEY)
    if not isinstance(authority_block, Mapping):
        raise ValueError("projection missing slm:authority block")
    authority_ids = authority_block.get("authority_ids")
    if not isinstance(authority_ids, Mapping):
        raise ValueError("slm:authority.authority_ids must be a mapping")
    kind = str(
        document.get(KIND_KEY) or authority_block.get("kind") or "unknown"
    )

    unsupported = _collect_unsupported(document)
    if raise_on_unsupported and unsupported:
        raise UnsupportedFieldError(
            "unsupported projection fields: " + ", ".join(unsupported)
        )

    return ProjectionView(
        format=fmt,
        kind=kind,
        authority_ids=dict(authority_ids),
        unsupported_fields=tuple(unsupported),
        semantic_authority=False,
        profile_id=str(profile or PROFILE_ID),
    )


def import_prov_o(document: Mapping[str, Any], **kwargs: Any) -> ProjectionView:
    return import_projection(document, expected_format="prov-o-jsonld", **kwargs)


def import_ro_crate(document: Mapping[str, Any], **kwargs: Any) -> ProjectionView:
    return import_projection(document, expected_format="ro-crate", **kwargs)


def import_spdx(document: Mapping[str, Any], **kwargs: Any) -> ProjectionView:
    return import_projection(document, expected_format="spdx3", **kwargs)


def import_intoto(document: Mapping[str, Any], **kwargs: Any) -> ProjectionView:
    return import_projection(document, expected_format="intoto", **kwargs)


def import_oci_referrer(document: Mapping[str, Any], **kwargs: Any) -> ProjectionView:
    return import_projection(document, expected_format="oci-referrer", **kwargs)


def _infer_format(document: Mapping[str, Any]) -> str:
    if document.get("payloadType") == "application/vnd.in-toto+json":
        return "intoto"
    if document.get("schema") == "oci_referrer_projection/v1":
        return "oci-referrer"
    if document.get("spdxVersion"):
        return "spdx3"
    ctx = document.get("@context")
    if ctx == "https://w3id.org/ro/crate/1.1/context":
        return "ro-crate"
    if isinstance(ctx, Mapping) and "prov" in ctx:
        return "prov-o-jsonld"
    return "unknown"


def _collect_unsupported(document: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    _walk(document, (), found)
    return sorted(set(found))


def _walk(node: Any, path: tuple[str, ...], found: list[str]) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_s = str(key)
            if path == () and key_s not in KNOWN_EXTERNAL_PREDICATES:
                if not key_s.startswith("slm:"):
                    found.append(key_s)
            elif key_s.startswith("prov:") and key_s not in KNOWN_EXTERNAL_PREDICATES:
                found.append(key_s)
            elif key_s.startswith("spdx:") and key_s not in KNOWN_EXTERNAL_PREDICATES:
                found.append(key_s)
            _walk(value, path + (key_s,), found)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _walk(item, path, found)
