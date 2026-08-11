"""Round-trip helpers that fail closed on authority drift."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from slm_training.evidence_interop.authority import (
    AuthorityRoundtripError,
    ProjectionView,
    authority_ids_equal,
    extract_authority,
)
from slm_training.evidence_interop.export_intoto import export_intoto
from slm_training.evidence_interop.export_oci import export_oci_referrer
from slm_training.evidence_interop.export_prov import export_prov_o
from slm_training.evidence_interop.export_rocrate import export_ro_crate
from slm_training.evidence_interop.export_spdx import export_spdx
from slm_training.evidence_interop.import_validate import import_projection

Exporter = Callable[[Any], dict[str, Any]]

EXPORTERS: dict[str, Exporter] = {
    "prov-o-jsonld": export_prov_o,
    "ro-crate": export_ro_crate,
    "spdx3": export_spdx,
    "intoto": export_intoto,
    "oci-referrer": export_oci_referrer,
}


def roundtrip(
    obj: Any,
    *,
    format: str,
) -> ProjectionView:
    """Export then import; preserve authority ids or raise."""
    if format not in EXPORTERS:
        raise ValueError(f"unknown format: {format!r}")
    expected = extract_authority(obj)
    document = EXPORTERS[format](obj)
    view = import_projection(document, expected_format=format)
    if view.semantic_authority:
        raise AuthorityRoundtripError("import upgraded semantic_authority")
    if not authority_ids_equal(view.authority_ids, expected.authority_ids):
        raise AuthorityRoundtripError(
            f"authority ids drifted under {format}: "
            f"expected={expected.authority_ids!r} got={view.authority_ids!r}"
        )
    if view.kind != expected.kind:
        raise AuthorityRoundtripError(
            f"kind drifted under {format}: {expected.kind!r} -> {view.kind!r}"
        )
    return view


def roundtrip_all(obj: Any) -> dict[str, ProjectionView]:
    """Round-trip every supported format; fail closed on first drift."""
    return {name: roundtrip(obj, format=name) for name in sorted(EXPORTERS)}
