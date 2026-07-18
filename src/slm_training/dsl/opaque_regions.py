"""Opaque-region model and hygienic splicing for VSS2-04 (SLM-68).

User-defined/template regions are first-class semantic objects. The structural
solver operates on stable region IDs and conservative summaries; this module
splices supplied values/fragments back into the assembled program through the
pack AST/IR, resolves bindings hygienically, canonicalizes, and re-runs the
applicable global verifier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from slm_training.dsl.pack import DslPack, PackSlotUnavailable


class OpaqueRegionKind(str, Enum):
    """Kind of opaque region the solver does not expand semantically."""

    CONTENT_VALUE = "content_value"
    IDENTIFIER = "identifier"
    EXPRESSION = "expression"
    STATEMENT_BLOCK = "statement_block"
    AST_SUBTREE = "ast_subtree"
    COMMENT = "comment"


@dataclass(frozen=True)
class OpaqueRegionSummary:
    """Conservative contract summarizing an opaque region's interface."""

    input_bindings: tuple[str, ...] = ()
    output_bindings: tuple[str, ...] = ()
    type_or_schema: Any | None = None
    effects: tuple[str, ...] = ("unknown",)
    exceptions: tuple[str, ...] = ("unknown",)
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    conservative: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_bindings": list(self.input_bindings),
            "output_bindings": list(self.output_bindings),
            "type_or_schema": self.type_or_schema,
            "effects": list(self.effects),
            "exceptions": list(self.exceptions),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "conservative": self.conservative,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OpaqueRegionSummary:
        return cls(
            input_bindings=tuple(str(v) for v in data.get("input_bindings", ())),
            output_bindings=tuple(str(v) for v in data.get("output_bindings", ())),
            type_or_schema=data.get("type_or_schema"),
            effects=tuple(str(v) for v in data.get("effects", ("unknown",))),
            exceptions=tuple(str(v) for v in data.get("exceptions", ("unknown",))),
            preconditions=tuple(str(v) for v in data.get("preconditions", ())),
            postconditions=tuple(str(v) for v in data.get("postconditions", ())),
            conservative=bool(data.get("conservative", True)),
        )


@dataclass(frozen=True)
class OpaqueRegion:
    """A stripped user/template region represented by ID and summary."""

    region_id: str
    kind: OpaqueRegionKind
    ast_path: tuple[str | int, ...] = ()
    placeholder: str | None = None
    source_digest: str = ""
    raw_source_ref: str | None = None
    summary: OpaqueRegionSummary = field(default_factory=OpaqueRegionSummary)
    required: bool = True

    def __post_init__(self) -> None:
        if not str(self.region_id).strip():
            raise ValueError("region_id must be non-empty")
        if not isinstance(self.kind, OpaqueRegionKind):
            raise ValueError(f"kind must be an OpaqueRegionKind, got {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": self.kind.value,
            "ast_path": list(self.ast_path),
            "placeholder": self.placeholder,
            "source_digest": self.source_digest,
            "raw_source_ref": self.raw_source_ref,
            "summary": self.summary.to_dict(),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OpaqueRegion:
        return cls(
            region_id=str(data["region_id"]),
            kind=OpaqueRegionKind(data["kind"]),
            ast_path=tuple(data.get("ast_path", ())),
            placeholder=data.get("placeholder"),
            source_digest=str(data.get("source_digest", "")),
            raw_source_ref=data.get("raw_source_ref"),
            summary=OpaqueRegionSummary.from_dict(data.get("summary", {})),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class OpaqueRegionBinding:
    """One concrete value/fragment supplied for an opaque region."""

    region_id: str
    scalar_value: Any | None = None
    ast_fragment: Any | None = None
    source_fragment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "scalar_value": self.scalar_value,
            "ast_fragment": self.ast_fragment,
            "source_fragment": self.source_fragment,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OpaqueRegionBinding:
        return cls(
            region_id=str(data["region_id"]),
            scalar_value=data.get("scalar_value"),
            ast_fragment=data.get("ast_fragment"),
            source_fragment=data.get("source_fragment"),
        )


@dataclass(frozen=True)
class OpaqueRealizationResult:
    """Result of splicing opaque regions back into a solved program."""

    status: str
    source: str | None
    ast: dict[str, Any] | None
    verifier_report: Any | None
    source_map: dict[str, tuple[int, int]]
    region_digests: dict[str, str]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "ast": self.ast,
            "verifier_report": self.verifier_report,
            "source_map": {k: list(v) for k, v in self.source_map.items()},
            "region_digests": self.region_digests,
            "errors": list(self.errors),
        }


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _validate_bindings(
    regions: tuple[OpaqueRegion, ...],
    bindings: Mapping[str, OpaqueRegionBinding],
) -> list[str]:
    """Check coverage, duplicates, and missing/wrong-kind bindings."""
    errors: list[str] = []
    region_by_id = {region.region_id: region for region in regions}
    seen: set[str] = set()
    for key, binding in bindings.items():
        region_id = binding.region_id
        if region_id in seen:
            errors.append(f"duplicate binding for {region_id}")
        seen.add(region_id)
        if region_id not in region_by_id:
            errors.append(f"unknown region {region_id}")
            continue
        region = region_by_id[region_id]
        if region.kind is OpaqueRegionKind.CONTENT_VALUE:
            if binding.scalar_value is None and binding.source_fragment is None:
                errors.append(
                    f"region {region_id} ({region.kind.value}) requires scalar_value or source_fragment"
                )
        else:
            if binding.ast_fragment is None and binding.source_fragment is None:
                errors.append(
                    f"region {region_id} ({region.kind.value}) requires ast_fragment or source_fragment"
                )
    bound_region_ids = {binding.region_id for binding in bindings.values()}
    for region in regions:
        if region.required and region.region_id not in bound_region_ids:
            errors.append(f"missing required region {region.region_id}")
    return errors


def _splice_content_value(source: str, placeholder: str, value: Any) -> tuple[str, tuple[int, int]]:
    """Replace a content placeholder with its scalar value and report the span."""
    text = str(value)
    # Escape quotes inside a string literal context when the placeholder is quoted.
    # This is conservative: only double-quote escaping for OpenUI string props.
    if '"' in text:
        text = text.replace('"', '\\"')
    start = source.find(placeholder)
    if start == -1:
        raise ValueError(f"placeholder {placeholder!r} not found in source")
    end = start + len(placeholder)
    new_source = source[:start] + text + source[end:]
    new_end = start + len(text)
    return new_source, (start, new_end)


def _splice_with_pack(
    source: str,
    ast: dict[str, Any] | None,
    region: OpaqueRegion,
    binding: OpaqueRegionBinding,
    pack: DslPack,
) -> tuple[str, dict[str, Any] | None, tuple[int, int]]:
    """Splice one region using pack hooks when available."""
    if region.kind is OpaqueRegionKind.CONTENT_VALUE:
        if binding.scalar_value is None and binding.source_fragment is not None:
            value = binding.source_fragment
        elif binding.scalar_value is not None:
            value = binding.scalar_value
        else:
            raise ValueError(f"no value supplied for content region {region.region_id}")
        if not region.placeholder:
            raise ValueError(f"content region {region.region_id} has no placeholder")
        new_source, span = _splice_content_value(source, region.placeholder, value)
        return new_source, ast, span

    splicer = pack.capsule_materializer or getattr(pack, "region_splicer", None)
    if splicer is None:
        raise PackSlotUnavailable(
            f"pack {pack.pack_id!r} does not provide region splicing for {region.kind.value}"
        )
    if binding.ast_fragment is not None:
        fragment = binding.ast_fragment
    elif binding.source_fragment is not None:
        parser = getattr(pack, "fragment_parser", None)
        if parser is None:
            raise PackSlotUnavailable(
                f"pack {pack.pack_id!r} does not provide fragment parsing"
            )
        fragment = parser(binding.source_fragment, kind=region.kind.value)
    else:
        raise ValueError(f"no fragment supplied for region {region.region_id}")
    return splicer(source=source, ast=ast, region=region, fragment=fragment)


def realize_opaque_regions(
    program_or_spec: Any,
    bindings: Mapping[str, OpaqueRegionBinding],
    *,
    pack: DslPack,
) -> OpaqueRealizationResult:
    """Splice opaque regions into a solved program and re-verify the result.

    Accepts a ``ProgramSpec`` or a source string. Returns an honest status:
    ``"solved"`` only when the pack oracle accepts the assembled source.
    """
    from slm_training.data.progspec.schema import ProgramSpec

    if isinstance(program_or_spec, ProgramSpec):
        source = program_or_spec.canonical_openui
        ast = dict(program_or_spec.ast)
        regions = program_or_spec.opaque_regions
    else:
        source = str(program_or_spec)
        ast = None
        extractor = getattr(pack, "opaque_region_extractor", None)
        if extractor is None:
            return OpaqueRealizationResult(
                status="error",
                source=None,
                ast=None,
                verifier_report=None,
                source_map={},
                region_digests={},
                errors=(f"pack {pack.pack_id!r} has no opaque_region_extractor",),
            )
        regions = extractor(source)

    errors = _validate_bindings(regions, bindings)
    if errors:
        return OpaqueRealizationResult(
            status="error",
            source=None,
            ast=None,
            verifier_report=None,
            source_map={},
            region_digests={},
            errors=tuple(errors),
        )

    region_map = {region.region_id: region for region in regions}
    current_source = source
    current_ast = ast
    source_map: dict[str, tuple[int, int]] = {}
    region_digests: dict[str, str] = {}

    # Deterministic order by region_id.
    for region_id in sorted(bindings):
        region = region_map[region_id]
        binding = bindings[region_id]
        region_digests[region_id] = _digest(binding.to_dict())
        try:
            current_source, current_ast, span = _splice_with_pack(
                current_source, current_ast, region, binding, pack
            )
        except (ValueError, PackSlotUnavailable) as exc:
            return OpaqueRealizationResult(
                status="error",
                source=None,
                ast=None,
                verifier_report=None,
                source_map=source_map,
                region_digests=region_digests,
                errors=(f"splice failed for {region_id}: {exc}",),
            )
        source_map[region_id] = span

    if pack.canonicalize is not None:
        try:
            canonical = pack.canonicalize(current_source)
        except Exception as exc:  # noqa: BLE001
            return OpaqueRealizationResult(
                status="error",
                source=None,
                ast=None,
                verifier_report=None,
                source_map=source_map,
                region_digests=region_digests,
                errors=(f"canonicalization failed: {exc}",),
            )
    else:
        canonical = current_source

    verifier_report = None
    if pack.oracle is not None:
        try:
            verifier_report = pack.oracle(canonical)
        except Exception as exc:  # noqa: BLE001
            return OpaqueRealizationResult(
                status="error",
                source=None,
                ast=None,
                verifier_report=None,
                source_map=source_map,
                region_digests=region_digests,
                errors=(f"verification failed: {exc}",),
            )

    failing_gate = (
        verifier_report.get("failing_gate") if isinstance(verifier_report, dict) else None
    )
    if failing_gate is not None:
        status = "rejected"
    elif verifier_report is not None:
        status = "solved"
    else:
        status = "unknown"

    return OpaqueRealizationResult(
        status=status,
        source=canonical,
        ast=current_ast,
        verifier_report=verifier_report,
        source_map=source_map,
        region_digests=region_digests,
        errors=(),
    )


__all__ = [
    "OpaqueRealizationResult",
    "OpaqueRegion",
    "OpaqueRegionBinding",
    "OpaqueRegionKind",
    "OpaqueRegionSummary",
    "realize_opaque_regions",
]
