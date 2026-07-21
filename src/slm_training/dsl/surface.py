"""Typed late-surface-realization slots and deterministic baseline for VSS3-04.

The surface boundary separates solved semantic IR from late surface filling.
Fields classified as surface-only can be realized after solving; all other
fields remain semantic and must pass through the solver/verifier.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class SurfaceSlotKind(str, Enum):
    """Kind of surface field awaiting realization."""

    INTERNAL_IDENTIFIER = "internal_identifier"
    DECORATIVE_TEXT = "decorative_text"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    STRUCTURED_STRING = "structured_string"
    EXTERNALLY_OBSERVABLE_NAME = "externally_observable_name"


class SurfaceAuthority(str, Enum):
    """Who is allowed to fill this surface slot."""

    SURFACE_ONLY = "surface_only"
    SEMANTIC = "semantic"
    OPAQUE_USER_VALUE = "opaque_user_value"


@dataclass(frozen=True)
class SurfaceConstraint:
    """Per-slot validation contract."""

    pattern: str | None = None
    max_bytes: int | None = None
    reserved: tuple[str, ...] = ()
    must_be_unique_within: str | None = None
    preserve_case: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "max_bytes": self.max_bytes,
            "reserved": list(self.reserved),
            "must_be_unique_within": self.must_be_unique_within,
            "preserve_case": self.preserve_case,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SurfaceConstraint:
        return cls(
            pattern=_optional_str(data.get("pattern")),
            max_bytes=data.get("max_bytes") if data.get("max_bytes") is not None else None,
            reserved=tuple(str(v) for v in data.get("reserved", ())),
            must_be_unique_within=_optional_str(data.get("must_be_unique_within")),
            preserve_case=bool(data.get("preserve_case", False)),
        )


@dataclass(frozen=True)
class SurfaceSlot:
    """One classifiable surface slot on a solved program."""

    slot_id: str
    kind: SurfaceSlotKind
    authority: SurfaceAuthority
    ast_path: tuple[str | int, ...]
    semantic_symbol_id: str | None
    opaque_region_id: str | None
    constraints: SurfaceConstraint
    current_value_digest: str | None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "kind": self.kind.value,
            "authority": self.authority.value,
            "ast_path": list(self.ast_path),
            "semantic_symbol_id": self.semantic_symbol_id,
            "opaque_region_id": self.opaque_region_id,
            "constraints": self.constraints.to_dict(),
            "current_value_digest": self.current_value_digest,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SurfaceSlot:
        return cls(
            slot_id=str(data["slot_id"]),
            kind=SurfaceSlotKind(data["kind"]),
            authority=SurfaceAuthority(data["authority"]),
            ast_path=tuple(data.get("ast_path", ())),
            semantic_symbol_id=_optional_str(data.get("semantic_symbol_id")),
            opaque_region_id=_optional_str(data.get("opaque_region_id")),
            constraints=SurfaceConstraint.from_dict(data.get("constraints", {})),
            current_value_digest=_optional_str(data.get("current_value_digest")),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class SurfaceRealizationRequest:
    """Input to a SurfaceRealizer."""

    pack_id: str
    constraint_version: str
    semantic_ir_fingerprint: str
    slots: tuple[SurfaceSlot, ...]
    context: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "constraint_version": self.constraint_version,
            "semantic_ir_fingerprint": self.semantic_ir_fingerprint,
            "slots": [slot.to_dict() for slot in self.slots],
            "context": self.context,
        }


@dataclass(frozen=True)
class SurfaceAssignment:
    """One realized surface value."""

    slot_id: str
    value: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "value": self.value,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SurfaceRealizationResult:
    """Outcome of ``realize_surface_and_verify``."""

    status: str
    source: str | None
    ast: dict[str, Any] | None
    verifier_report: Any | None
    assignments: tuple[SurfaceAssignment, ...]
    source_map: dict[str, tuple[int, int]]
    semantic_equivalence: dict[str, Any] | None
    fallback_counters: dict[str, int]
    diagnostics: dict[str, Any] | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "ast": self.ast,
            "verifier_report": self.verifier_report,
            "assignments": [a.to_dict() for a in self.assignments],
            "source_map": {k: list(v) for k, v in self.source_map.items()},
            "semantic_equivalence": self.semantic_equivalence,
            "fallback_counters": dict(self.fallback_counters),
            "diagnostics": self.diagnostics,
            "errors": list(self.errors),
        }


@runtime_checkable
class SurfaceRealizer(Protocol):
    """Pluggable surface realization strategy."""

    def realize(self, request: SurfaceRealizationRequest) -> tuple[SurfaceAssignment, ...]: ...


# Conservative OpenUI identifier grammar: lowercase start, alphanumeric/underscore.
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-zA-Z0-9_]*$")
# Reserved words / names that must not be used for surface identifiers.
_RESERVED_IDENTIFIERS = frozenset(
    {
        "true",
        "false",
        "null",
        "if",
        "else",
        "for",
        "while",
        "return",
        "root",
    }
)
_DEFAULT_MAX_IDENTIFIER_BYTES = 64


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _valid_identifier(value: str, constraints: SurfaceConstraint) -> tuple[bool, str]:
    """Check a candidate identifier against grammar and constraints."""
    if not value:
        return False, "identifier must be non-empty"
    if constraints.max_bytes is not None and len(value.encode("utf-8")) > constraints.max_bytes:
        return False, f"identifier exceeds max_bytes={constraints.max_bytes}"
    if value in constraints.reserved or value in _RESERVED_IDENTIFIERS:
        return False, f"identifier {value!r} is reserved"
    if not _IDENTIFIER_RE.match(value):
        return False, f"identifier {value!r} does not match [a-z_][A-Za-z0-9_]*"
    return True, ""


def _canonical_name(index: int) -> str:
    """Deterministic canonical binder name: v0, v1, ..."""
    return f"v{index}"


def _repair_collision(
    name: str,
    used: set[str],
    constraints: SurfaceConstraint,
    counter: int,
) -> tuple[str, int]:
    """Deterministic collision repair by appending a numeric suffix."""
    candidate = name
    while True:
        if candidate not in used and _valid_identifier(candidate, constraints)[0]:
            return candidate, counter
        counter += 1
        candidate = f"{name}_{counter}"


class DeterministicSurfaceRealizer:
    """Model-free surface realizer.

    * ``INTERNAL_IDENTIFIER`` slots receive deterministic canonical names.
    * ``OPAQUE_USER_VALUE`` slots are skipped; the caller supplies validated
      bindings through ``realize_surface_and_verify``.
    * ``STRUCTURED_STRING`` / ``EXTERNALLY_OBSERVABLE_NAME`` slots are rejected
      if presented as freely assignable.
    * ``COMMENT`` / ``DOCSTRING`` slots are unsupported for OpenUI V1.
    * ``DECORATIVE_TEXT`` slots receive a pack-declared neutral default or the
      existing canonical digest, if one is supplied; otherwise they are rejected.
    """

    def realize(self, request: SurfaceRealizationRequest) -> tuple[SurfaceAssignment, ...]:
        assignments: list[SurfaceAssignment] = []
        used_names: set[str] = set()
        id_index = 0

        for slot in request.slots:
            if slot.authority is SurfaceAuthority.SEMANTIC:
                raise ValueError(
                    f"slot {slot.slot_id!r} is semantic and cannot be realized on the surface"
                )
            if slot.authority is SurfaceAuthority.OPAQUE_USER_VALUE:
                # Caller supplies validated bindings via realize_surface_and_verify.
                continue
            if slot.kind is SurfaceSlotKind.INTERNAL_IDENTIFIER:
                if slot.authority is not SurfaceAuthority.SURFACE_ONLY:
                    raise ValueError(
                        f"internal identifier slot {slot.slot_id!r} must be SURFACE_ONLY"
                    )
                candidate = _canonical_name(id_index)
                constraints = SurfaceConstraint(
                    max_bytes=slot.constraints.max_bytes or _DEFAULT_MAX_IDENTIFIER_BYTES,
                    reserved=slot.constraints.reserved,
                )
                candidate, _ = _repair_collision(candidate, used_names, constraints, 0)
                used_names.add(candidate)
                id_index += 1
                # Reject obvious overlong outputs even after repair.
                ok, reason = _valid_identifier(candidate, constraints)
                if not ok:
                    raise ValueError(f"slot {slot.slot_id!r}: {reason}")
                assignments.append(
                    SurfaceAssignment(
                        slot_id=slot.slot_id,
                        value=candidate,
                        provenance="deterministic:canonical_name",
                    )
                )
            elif slot.kind is SurfaceSlotKind.DECORATIVE_TEXT:
                neutral = (
                    request.context.get("decorative_default")
                    if isinstance(request.context, Mapping)
                    else None
                )
                if neutral is None:
                    raise ValueError(
                        f"decorative slot {slot.slot_id!r} has no pack-declared neutral default"
                    )
                assignments.append(
                    SurfaceAssignment(
                        slot_id=slot.slot_id,
                        value=str(neutral),
                        provenance="deterministic:neutral_default",
                    )
                )
            elif slot.kind in (SurfaceSlotKind.COMMENT, SurfaceSlotKind.DOCSTRING):
                raise ValueError(
                    f"slot {slot.slot_id!r} ({slot.kind.value}) is unsupported in this pack version"
                )
            else:
                raise ValueError(
                    f"slot {slot.slot_id!r} ({slot.kind.value}) cannot be freely realized"
                )

        return tuple(assignments)


def _extract_source(source_or_spec: Any) -> tuple[str, dict[str, Any] | None]:
    """Normalize a ProgramSpec or raw source into source + optional AST dict."""
    from slm_training.data.progspec.schema import ProgramSpec

    if isinstance(source_or_spec, ProgramSpec):
        return source_or_spec.canonical_openui, dict(source_or_spec.ast)
    if isinstance(source_or_spec, str):
        return source_or_spec, None
    if isinstance(source_or_spec, Mapping):
        return str(source_or_spec.get("canonical_openui", source_or_spec.get("source", ""))), dict(
            source_or_spec.get("ast", {}) or {}
        )
    return str(source_or_spec), None


def _validate_assignments(
    slots: tuple[SurfaceSlot, ...],
    assignments: tuple[SurfaceAssignment, ...],
) -> list[str]:
    """Check coverage, duplicates, unknown slots, and missing required values."""
    errors: list[str] = []
    slot_by_id = {slot.slot_id: slot for slot in slots}
    seen: set[str] = set()
    for assignment in assignments:
        if assignment.slot_id in seen:
            errors.append(f"duplicate assignment for {assignment.slot_id}")
        seen.add(assignment.slot_id)
        if assignment.slot_id not in slot_by_id:
            errors.append(f"unknown slot {assignment.slot_id}")
            continue
        slot = slot_by_id[assignment.slot_id]
        if slot.kind is SurfaceSlotKind.INTERNAL_IDENTIFIER:
            ok, reason = _valid_identifier(
                assignment.value,
                SurfaceConstraint(
                    max_bytes=slot.constraints.max_bytes or _DEFAULT_MAX_IDENTIFIER_BYTES,
                    reserved=slot.constraints.reserved,
                ),
            )
            if not ok:
                errors.append(f"slot {slot.slot_id}: {reason}")
    assigned_ids = {a.slot_id for a in assignments}
    for slot in slots:
        if slot.required and slot.slot_id not in assigned_ids:
            errors.append(f"missing required slot {slot.slot_id}")
    return errors


def _apply_binder_substitutions(
    source: str,
    slots: tuple[SurfaceSlot, ...],
    assignments: tuple[SurfaceAssignment, ...],
) -> tuple[str, dict[str, tuple[int, int]], list[str]]:
    """Substitute internal identifier values for binder names, whole-word only."""
    assignment_by_id = {a.slot_id: a.value for a in assignments}
    binder_map: dict[str, str] = {}
    for slot in slots:
        if slot.kind is SurfaceSlotKind.INTERNAL_IDENTIFIER and slot.semantic_symbol_id is not None:
            value = assignment_by_id.get(slot.slot_id)
            if value is not None:
                binder_map[slot.semantic_symbol_id] = value

    source_map: dict[str, tuple[int, int]] = {}
    current = source
    # Apply longest-old-name first to avoid shadowing prefixes.
    for old_name in sorted(binder_map, key=len, reverse=True):
        new_name = binder_map[old_name]
        # Find the first definition span to report in the source map.
        # Whole-word only, and never inside a placeholder (e.g. :hero.title).
        pattern = re.compile(rf"(?<![:.\\w]){re.escape(old_name)}(?![\\w:.])")
        first = pattern.search(current)
        if first is not None:
            current = pattern.sub(new_name, current)
            # Report the first occurrence's new span.
            new_start = first.start()
            new_end = new_start + len(new_name)
            # Locate the slot that owns this old_name.
            for slot in slots:
                if slot.semantic_symbol_id == old_name:
                    source_map[slot.slot_id] = (new_start, new_end)
                    break

    return current, source_map, []


def _opaque_bindings_from_assignments(
    slots: tuple[SurfaceSlot, ...],
    assignments: tuple[SurfaceAssignment, ...],
) -> dict[str, Any]:
    """Build OpaqueRegionBinding values for OPAQUE_USER_VALUE slots."""
    from slm_training.dsl.opaque_regions import OpaqueRegionBinding

    slot_by_id = {slot.slot_id: slot for slot in slots}
    bindings: dict[str, Any] = {}
    for assignment in assignments:
        slot = slot_by_id.get(assignment.slot_id)
        if slot is None or slot.opaque_region_id is None:
            continue
        bindings[slot.opaque_region_id] = OpaqueRegionBinding(
            region_id=slot.opaque_region_id,
            scalar_value=assignment.value,
        )
    return bindings


def _openui_surface_slot_extractor(source: str) -> tuple[SurfaceSlot, ...]:
    """Classify late-realization slots for OpenUI V1.

    * Binder definitions (left-hand side of ``name = ...``) are internal
      identifiers: their surface spelling is not externally observable and the
      canonicalizer normalizes them to ``v0, v1, ...``.
    * Content placeholders for user-facing string props (``text``, ``label``,
      etc.) are opaque user values realized through the VSS2-04 opaque-region
      splice path.
    * Everything else is left out of the surface slot set and therefore remains
      semantic by default.
    """
    import hashlib

    from slm_training.data.contract import _BINDER_RE
    from slm_training.dsl.placeholders import CONTENT_PROPS, extract_placeholders

    slots: list[SurfaceSlot] = []
    seen_binders: set[str] = set()
    for index, match in enumerate(_BINDER_RE.finditer(source)):
        name = match.group(1)
        # The program root binder is syntactically required to be spelled "root";
        # it is therefore externally observable and remains semantic.
        if name == "root" or name in seen_binders:
            continue
        seen_binders.add(name)
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        slots.append(
            SurfaceSlot(
                slot_id=f"openui:binder:{name}",
                kind=SurfaceSlotKind.INTERNAL_IDENTIFIER,
                authority=SurfaceAuthority.SURFACE_ONLY,
                ast_path=("statement", index),
                semantic_symbol_id=name,
                opaque_region_id=None,
                constraints=SurfaceConstraint(),
                current_value_digest=digest,
            )
        )

    for placeholder in extract_placeholders(source):
        prop = placeholder.lstrip(":").split(".")[-1]
        if prop not in CONTENT_PROPS:
            continue
        digest = hashlib.sha256(placeholder.encode("utf-8")).hexdigest()[:16]
        slots.append(
            SurfaceSlot(
                slot_id=f"openui:content:{placeholder}",
                kind=SurfaceSlotKind.DECORATIVE_TEXT,
                authority=SurfaceAuthority.OPAQUE_USER_VALUE,
                ast_path=("placeholder", placeholder),
                semantic_symbol_id=None,
                opaque_region_id=f"openui:content:{placeholder}",
                constraints=SurfaceConstraint(),
                current_value_digest=digest,
            )
        )

    return tuple(slots)


# Built-in surface-slot extractors keyed by ``pack.pack_id``. Canonical ``main``
# packs do not (yet) declare a ``surface_slot_extractor`` slot, so surface
# realization owns the OpenUI extractor here; a pack that later declares the slot
# directly takes precedence via ``resolve_surface_slot_extractor``.
_SURFACE_SLOT_EXTRACTORS: dict[str, Any] = {
    "openui": _openui_surface_slot_extractor,
}


def resolve_surface_slot_extractor(pack: Any) -> Any | None:
    """Resolve the surface-slot extractor for ``pack``.

    A pack may declare a ``surface_slot_extractor`` slot directly; otherwise a
    built-in extractor is looked up by ``pack.pack_id``. Packs that provide
    neither (for example ``toy-layout``) resolve to ``None`` and callers must
    fail closed.
    """
    declared = getattr(pack, "surface_slot_extractor", None)
    if declared is not None:
        return declared
    return _SURFACE_SLOT_EXTRACTORS.get(getattr(pack, "pack_id", None))


def _oracle_failure(verifier_report: Any) -> str | None:
    """Return the authoritative failure marker for dict or typed reports."""
    if verifier_report is None:
        return "missing_oracle_report"
    if isinstance(verifier_report, Mapping):
        failing_gate = verifier_report.get("failing_gate")
        ok = verifier_report.get("ok")
    else:
        failing_gate = getattr(verifier_report, "failing_gate", None)
        ok = getattr(verifier_report, "ok", None)
    if failing_gate is not None:
        return getattr(failing_gate, "value", str(failing_gate))
    if ok is False:
        return "oracle_rejected"
    return None


def resolve_verified_template_bindings(
    template: str,
    request: Any,
    caller_bindings: tuple[Any, ...],
    *,
    pack: Any,
) -> Any:
    """Verify a canonical template and bind content out of band.

    OpenUI's content policy requires placeholders, so caller values remain in a
    typed envelope and are never spliced into source by this boundary.
    """
    from slm_training.data.contract import (
        BoundGenerationResult,
        ResolvedContentBinding,
        bound_generation_fingerprint,
    )
    from slm_training.dsl.placeholders import PLACEHOLDER_RE

    mode = "template_plus_bindings"
    materialized_verification = "not_materialized_placeholder_policy"

    def finish(
        *,
        status: str,
        canonical: str | None,
        template_verification: str,
        template_fingerprint: str | None,
        resolved: tuple[Any, ...] = (),
        diagnostics: dict[str, Any] | None = None,
        errors: tuple[str, ...] = (),
    ) -> Any:
        safe_payload = {
            "status": status,
            "template_verification": template_verification,
            "template_fingerprint": template_fingerprint,
            "bindings": [binding.evidence_dict() for binding in resolved],
            "materialized_verification": materialized_verification,
            "realization_mode": mode,
            "diagnostics": diagnostics or {},
            "errors": list(errors),
        }
        return BoundGenerationResult(
            status=status,
            canonical_template=canonical,
            template_verification=template_verification,
            template_fingerprint=template_fingerprint,
            bindings=resolved,
            materialized_source=None,
            materialized_verification=materialized_verification,
            realization_mode=mode,
            fingerprint=bound_generation_fingerprint(safe_payload),
            diagnostics=diagnostics or {},
            errors=errors,
        )

    try:
        canonical = pack.canonicalize(template) if pack.canonicalize is not None else template
    except Exception:  # noqa: BLE001 - diagnostics must not echo generated/user text
        return finish(
            status="error",
            canonical=None,
            template_verification="canonicalization_failed",
            template_fingerprint=None,
            errors=("generated template failed canonicalization",),
        )
    template_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if pack.oracle is None:
        return finish(
            status="error",
            canonical=canonical,
            template_verification="oracle_unavailable",
            template_fingerprint=template_fingerprint,
            errors=("pack oracle unavailable",),
        )
    try:
        verifier_report = pack.oracle(canonical)
    except Exception:  # noqa: BLE001 - diagnostics must not echo generated/user text
        return finish(
            status="error",
            canonical=canonical,
            template_verification="oracle_error",
            template_fingerprint=template_fingerprint,
            errors=("pack oracle failed",),
        )
    failure = _oracle_failure(verifier_report)
    if failure is not None:
        return finish(
            status="error",
            canonical=canonical,
            template_verification="rejected",
            template_fingerprint=template_fingerprint,
            diagnostics={"failure_category": failure},
            errors=("generated template was rejected by the pack oracle",),
        )

    occurrences = Counter(PLACEHOLDER_RE.findall(canonical))
    opaque_by_placeholder = {
        placeholder: f"openui:content:{placeholder}" for placeholder in occurrences
    }

    declared: dict[str, tuple[int, str, str]] = {}
    declaration_errors: list[str] = []
    for index, placeholder in enumerate(request.slot_contract):
        external_key = placeholder[1:]
        if external_key in declared:
            declaration_errors.append(f"duplicate declared key {external_key}")
        declared[external_key] = (index, placeholder, f":slot_{index}")
    internal_to_external = {
        internal: external_key
        for external_key, (_index, _external, internal) in declared.items()
    }
    for placeholder in opaque_by_placeholder:
        if placeholder not in internal_to_external:
            declaration_errors.append(f"undeclared model slot {placeholder}")
    if declaration_errors:
        return finish(
            status="error",
            canonical=canonical,
            template_verification="pack_verified",
            template_fingerprint=template_fingerprint,
            errors=tuple(declaration_errors),
        )

    supplied: dict[str, Any] = {}
    binding_errors: list[str] = []
    template_keys = {
        internal_to_external[placeholder]
        for placeholder in opaque_by_placeholder
    }
    for binding in caller_bindings:
        key = binding.external_key
        if key in supplied:
            binding_errors.append(f"duplicate binding for {key}")
        supplied[key] = binding
        if key not in template_keys:
            binding_errors.append(f"unknown binding {key}")
    for key in sorted(template_keys - supplied.keys()):
        binding_errors.append(f"missing required binding {key}")
    if binding_errors:
        return finish(
            status="error",
            canonical=canonical,
            template_verification="pack_verified",
            template_fingerprint=template_fingerprint,
            errors=tuple(binding_errors),
        )

    runtime_symbols = {
        symbol.surface: symbol for symbol in request.effective_runtime_symbols()
    }
    resolved: list[Any] = []
    for key, (internal_slot, external_placeholder, placeholder) in sorted(
        declared.items(), key=lambda item: item[1][0]
    ):
        opaque_slot_id = opaque_by_placeholder.get(placeholder)
        binding = supplied.get(key)
        if opaque_slot_id is None or binding is None:
            continue
        value_bytes = binding.value.encode("utf-8")
        symbol = runtime_symbols.get(external_placeholder)
        resolved.append(
            ResolvedContentBinding(
                external_key=key,
                internal_slot=internal_slot,
                opaque_slot_id=opaque_slot_id,
                value=binding.value,
                value_digest=hashlib.sha256(value_bytes).hexdigest(),
                value_bytes=len(value_bytes),
                occurrence_count=occurrences[placeholder],
                semantic_type=getattr(symbol, "semantic_type", None),
            )
        )
    diagnostics = {
        "binding_count": len(resolved),
        "declared_slot_count": len(declared),
        "literal_materialization_supported": False,
        "repeated_occurrences": sum(max(0, binding.occurrence_count - 1) for binding in resolved),
    }
    return finish(
        status="resolved",
        canonical=canonical,
        template_verification="pack_verified",
        template_fingerprint=template_fingerprint,
        resolved=tuple(resolved),
        diagnostics=diagnostics,
    )


def realize_surface_and_verify(
    solved_program: Any,
    *,
    pack: Any,
    realizer: SurfaceRealizer | None = None,
    opaque_bindings: Mapping[str, Any] | None = None,
    semantic_ir_fingerprint: str | None = None,
    prior_status: str | None = None,
) -> SurfaceRealizationResult:
    """Realize surface slots on a solved program and re-verify the result.

    Required order:
      1. Assert the input carries a solved semantic-IR fingerprint and prior
         solver/global-verifier status.
      2. Extract/classify surface slots via ``pack.surface_slot_extractor``.
      3. Obtain assignments from the realizer.
      4. Validate exact coverage and reject unknown/duplicate/missing slots.
      5. Validate every assignment against pack and generic constraints.
      6. Apply identifier changes by semantic symbol ID (definitions + refs).
      7. Apply opaque content through the VSS2-04 splice path.
      8. Serialize/canonicalize and run the pack oracle.
      9. Return an honest status; a failed verifier yields no certified result.
    """
    from slm_training.dsl.pack import PackSlotUnavailable

    if not semantic_ir_fingerprint:
        return SurfaceRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            assignments=(),
            source_map={},
            semantic_equivalence=None,
            fallback_counters={},
            diagnostics=None,
            errors=("missing semantic_ir_fingerprint",),
        )
    if prior_status not in {"solved", "verified"}:
        return SurfaceRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            assignments=(),
            source_map={},
            semantic_equivalence=None,
            fallback_counters={},
            diagnostics=None,
            errors=(f"prior_status must be 'solved' or 'verified', got {prior_status!r}",),
        )

    extractor = resolve_surface_slot_extractor(pack)
    if extractor is None:
        return SurfaceRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            assignments=(),
            source_map={},
            semantic_equivalence=None,
            fallback_counters={},
            diagnostics=None,
            errors=(f"pack {pack.pack_id!r} has no surface_slot_extractor",),
        )

    source, ast = _extract_source(solved_program)
    try:
        slots = extractor(source)
    except Exception as exc:  # noqa: BLE001
        return SurfaceRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            assignments=(),
            source_map={},
            semantic_equivalence=None,
            fallback_counters={},
            diagnostics=None,
            errors=(f"surface slot extraction failed: {exc}",),
        )

    # Detect duplicate slot IDs / paths or ambiguous authority.
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_paths: set[tuple[str | int, ...]] = set()
    for slot in slots:
        if slot.slot_id in seen_ids:
            errors.append(f"duplicate slot_id {slot.slot_id!r}")
        seen_ids.add(slot.slot_id)
        if slot.ast_path in seen_paths:
            errors.append(f"duplicate ast_path for slot {slot.slot_id!r}")
        seen_paths.add(slot.ast_path)
        if slot.authority is SurfaceAuthority.SEMANTIC:
            errors.append(f"slot {slot.slot_id!r} has ambiguous SEMANTIC authority")

    if errors:
        return SurfaceRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            assignments=(),
            source_map={},
            semantic_equivalence=None,
            fallback_counters={},
            diagnostics=None,
            errors=tuple(errors),
        )

    # Separate opaque-user-value slots from surface-only slots.
    surface_only_slots = tuple(
        slot for slot in slots if slot.authority is SurfaceAuthority.SURFACE_ONLY
    )
    opaque_slots = tuple(
        slot for slot in slots if slot.authority is SurfaceAuthority.OPAQUE_USER_VALUE
    )

    # Build the realization request for the pluggable realizer.
    request = SurfaceRealizationRequest(
        pack_id=pack.pack_id,
        constraint_version="vss3-04/deterministic-v1",
        semantic_ir_fingerprint=semantic_ir_fingerprint,
        slots=surface_only_slots,
        context={"opaque_bindings": dict(opaque_bindings or {})},
    )

    try:
        if realizer is None:
            realized = DeterministicSurfaceRealizer().realize(request)
        else:
            realized = realizer.realize(request)
    except Exception as exc:  # noqa: BLE001
        return SurfaceRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            assignments=(),
            source_map={},
            semantic_equivalence=None,
            fallback_counters={},
            diagnostics=None,
            errors=(f"realizer failed: {exc}",),
        )

    # Merge caller-supplied opaque bindings into OPAQUE_USER_VALUE assignments.
    opaque_bindings = opaque_bindings or {}
    opaque_assignments: list[SurfaceAssignment] = []
    for slot in opaque_slots:
        binding = opaque_bindings.get(slot.opaque_region_id or slot.slot_id)
        if binding is not None:
            value: str | None = None
            if hasattr(binding, "scalar_value") and binding.scalar_value is not None:
                value = str(binding.scalar_value)
            elif hasattr(binding, "source_fragment") and binding.source_fragment is not None:
                value = binding.source_fragment
            if value is None:
                errors.append(f"opaque slot {slot.slot_id!r} has no usable value")
                continue
            opaque_assignments.append(
                SurfaceAssignment(
                    slot_id=slot.slot_id,
                    value=value,
                    provenance="caller:opaque_binding",
                )
            )

    all_assignments = tuple([*realized, *opaque_assignments])
    validation_errors = _validate_assignments(slots, all_assignments)
    if validation_errors:
        return SurfaceRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            assignments=all_assignments,
            source_map={},
            semantic_equivalence=None,
            fallback_counters={},
            diagnostics=None,
            errors=tuple(validation_errors),
        )

    # 6. Apply internal identifier substitutions.
    current_source, id_source_map, _ = _apply_binder_substitutions(
        source, surface_only_slots, all_assignments
    )

    # 7. Apply opaque content through the VSS2-04 splice path.
    opaque_binding_map = _opaque_bindings_from_assignments(slots, all_assignments)
    if opaque_binding_map:
        from slm_training.dsl.opaque_regions import realize_opaque_regions

        try:
            opaque_result = realize_opaque_regions(
                current_source, opaque_binding_map, pack=pack
            )
        except (ValueError, PackSlotUnavailable) as exc:
            return SurfaceRealizationResult(
                status="error",
                source=None,
                ast=None,
                verifier_report=None,
                assignments=all_assignments,
                source_map=id_source_map,
                semantic_equivalence=None,
                fallback_counters={},
                diagnostics=None,
                errors=(f"opaque splicing failed: {exc}",),
            )
        if opaque_result.status == "error":
            return SurfaceRealizationResult(
                status="error",
                source=None,
                ast=None,
                verifier_report=opaque_result.verifier_report,
                assignments=all_assignments,
                source_map=id_source_map,
                semantic_equivalence=None,
                fallback_counters={},
                diagnostics=None,
                errors=opaque_result.errors,
            )
        current_source = opaque_result.source or current_source
        ast = opaque_result.ast or ast
        id_source_map.update(opaque_result.source_map)

    # 8. Canonicalize and verify.
    canonical = current_source
    if pack.canonicalize is not None:
        try:
            canonical = pack.canonicalize(current_source)
        except Exception as exc:  # noqa: BLE001
            return SurfaceRealizationResult(
                status="error",
                source=None,
                ast=ast,
                verifier_report=None,
                assignments=all_assignments,
                source_map=id_source_map,
                semantic_equivalence=None,
                fallback_counters={},
                diagnostics=None,
                errors=(f"canonicalization failed: {exc}",),
            )

    verifier_report = None
    if pack.oracle is not None:
        try:
            verifier_report = pack.oracle(canonical)
        except Exception as exc:  # noqa: BLE001
            return SurfaceRealizationResult(
                status="error",
                source=None,
                ast=ast,
                verifier_report=None,
                assignments=all_assignments,
                source_map=id_source_map,
                semantic_equivalence=None,
                fallback_counters={},
                diagnostics=None,
                errors=(f"verification failed: {exc}",),
            )

    failing_gate = _oracle_failure(verifier_report) if verifier_report is not None else None
    if failing_gate is not None:
        status = "rejected"
    elif verifier_report is not None:
        status = "solved"
    else:
        status = "unknown"

    # Semantic equivalence evidence.
    before_fp = _digest(canonicalize_input(source, pack))
    after_fp = _digest(canonical)
    semantic_equivalence = {
        "before_canonical_fingerprint": before_fp,
        "after_canonical_fingerprint": after_fp,
        "alpha_equivalent": before_fp == after_fp,
    }

    diagnostics = {
        "total_slots": len(slots),
        "surface_only_slots": len(surface_only_slots),
        "opaque_user_value_slots": len(opaque_slots),
        "semantic_slots": sum(
            1 for slot in slots if slot.authority is SurfaceAuthority.SEMANTIC
        ),
        "internal_identifier_assignments": sum(
            1
            for a in all_assignments
            if any(
                s.slot_id == a.slot_id
                and s.kind is SurfaceSlotKind.INTERNAL_IDENTIFIER
                for s in slots
            )
        ),
        "opaque_user_value_assignments": sum(
            1
            for a in all_assignments
            if any(
                s.slot_id == a.slot_id
                and s.authority is SurfaceAuthority.OPAQUE_USER_VALUE
                for s in slots
            )
        ),
    }

    return SurfaceRealizationResult(
        status=status,
        source=canonical,
        ast=ast,
        verifier_report=verifier_report,
        assignments=all_assignments,
        source_map=id_source_map,
        semantic_equivalence=semantic_equivalence,
        fallback_counters={},
        diagnostics=diagnostics,
        errors=(),
    )


def canonicalize_input(source: str, pack: Any) -> str:
    """Best-effort canonical form of the pre-realization source for equivalence."""
    if pack.canonicalize is not None:
        try:
            return pack.canonicalize(source)
        except Exception:  # noqa: BLE001
            pass
    return source


__all__ = [
    "DeterministicSurfaceRealizer",
    "SurfaceAssignment",
    "SurfaceAuthority",
    "SurfaceConstraint",
    "SurfaceRealizationRequest",
    "SurfaceRealizationResult",
    "SurfaceRealizer",
    "SurfaceSlot",
    "SurfaceSlotKind",
    "canonicalize_input",
    "realize_surface_and_verify",
    "resolve_verified_template_bindings",
    "resolve_surface_slot_extractor",
]
