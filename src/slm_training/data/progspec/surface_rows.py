"""Surface-realization training rows for VSS3-05 (SLM-73).

Derives ``ExampleRecord`` rows from verified ``ProgramSpec`` roots after split
assignment. Each row represents one surface slot realization target. Split and
group identity are inherited from the parent spec so leakage checks remain
intact.
"""

from __future__ import annotations

import hashlib
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.dsl.surface import (
    SurfaceAuthority,
    SurfaceRealizer,
    realize_surface_and_verify,
    resolve_surface_slot_extractor,
)


def _semantic_fingerprint(spec: ProgramSpec) -> str:
    """Stable fingerprint of the solved semantic IR."""
    payload = f"{spec.program_family_id}:{spec.lineage_id}:{spec.canonical_openui}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _surface_prompt(spec: ProgramSpec, slot: Any) -> str:
    """Short prompt describing the slot for training/conditioning."""
    parts = [
        f"kind={slot.kind.value}",
        f"symbol={slot.semantic_symbol_id or ''}",
        f"slot={slot.slot_id}",
    ]
    return " ".join(parts)


def derive_surface_realization_records(
    spec: ProgramSpec,
    *,
    realizer: SurfaceRealizer | None = None,
    opaque_bindings: dict[str, Any] | None = None,
    include_authorities: frozenset[str] | None = None,
) -> tuple[Any, ...]:
    """Derive one ``ExampleRecord`` per realized surface slot.

    The deterministic baseline is used when ``realizer`` is ``None``. Only slots
    whose authority is in ``include_authorities`` are emitted (default:
    ``SURFACE_ONLY`` and ``OPAQUE_USER_VALUE``).
    """
    from slm_training.dsl.pack import get_pack

    include_authorities = include_authorities or frozenset(
        {SurfaceAuthority.SURFACE_ONLY.value, SurfaceAuthority.OPAQUE_USER_VALUE.value}
    )
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        spec,
        pack=pack,
        realizer=realizer,
        opaque_bindings=opaque_bindings or {},
        semantic_ir_fingerprint=_semantic_fingerprint(spec),
        prior_status="verified",
    )
    if result.status not in {"solved", "verified"}:
        return ()

    assignment_by_slot = {a.slot_id: a for a in result.assignments}
    # Main resolves the OpenUI extractor via the pack_id registry; the pack does
    # not expose a ``surface_slot_extractor`` attribute directly.
    extractor = resolve_surface_slot_extractor(pack)
    if extractor is None:
        return ()
    slots = extractor(spec.canonical_openui)
    records: list[Any] = []
    for slot in slots:
        if slot.authority.value not in include_authorities:
            continue
        assignment = assignment_by_slot.get(slot.slot_id)
        if assignment is None:
            continue
        prompt = _surface_prompt(spec, slot)
        record_id = f"{spec.id}_surface_{slot.slot_id}_{hashlib.sha256(assignment.value.encode()).hexdigest()[:8]}"
        record = emit_record(
            spec,
            prompt=prompt,
            task="surface_realization",
            openui=result.source,
            record_id=record_id,
            source="surface_realization",
            determinacy="deterministic",
            tier="Silver",
            provenance={
                "surface_slot_id": slot.slot_id,
                "surface_assignment": assignment.to_dict(),
                "surface_result_status": result.status,
            },
        )
        records.append(record)
    return tuple(records)


__all__ = ["derive_surface_realization_records"]
