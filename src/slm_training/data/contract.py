"""Generation request contract and canonical example-record normalization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from slm_training.data.structure import strip_style_literals
from slm_training.dsl.lang_core import library_schema
from slm_training.dsl.placeholders import extract_placeholders, merge_placeholders
from slm_training.dsl.schema import OUTPUT_KINDS, ExampleRecord, OutputKind

_BINDER_RE = re.compile(r"(?m)^([a-z_][A-Za-z0-9_]*)\s*=")
_TYPED_AUTHORITY_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")


@dataclass(frozen=True)
class RuntimeSymbol:
    """A request-visible symbol whose surface form is not global vocabulary."""

    surface: str
    role: Literal["alpha_binder", "external_entity", "state", "fresh_binder"]
    namespace: tuple[str, ...] = ()
    semantic_type: str | None = None
    semantic_role: str | None = None
    scope: str | None = None
    signature: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.surface.strip():
            raise ValueError("runtime symbol surface must be non-empty")
        if self.role not in {
            "alpha_binder",
            "external_entity",
            "state",
            "fresh_binder",
        }:
            raise ValueError(f"unknown runtime symbol role {self.role!r}")
        if self.role == "external_entity" and not self.surface.startswith(":"):
            raise ValueError("external_entity surfaces must start with ':'")
        if self.role == "state" and not self.surface.startswith("$"):
            raise ValueError("state surfaces must start with '$'")
        for field_name in ("semantic_type", "semantic_role"):
            value = getattr(self, field_name)
            if value is not None and not _TYPED_AUTHORITY_RE.fullmatch(value):
                raise ValueError(
                    f"{field_name} must be a declared lowercase typed identifier"
                )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "surface": self.surface,
            "role": self.role,
            "namespace": list(self.namespace),
        }
        for key in (
            "semantic_type",
            "semantic_role",
            "scope",
            "signature",
            "description",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeSymbol:
        return cls(
            surface=str(data["surface"]),
            role=str(data["role"]),  # type: ignore[arg-type]
            namespace=tuple(str(part) for part in data.get("namespace") or ()),
            semantic_type=_optional_str(data.get("semantic_type")),
            semantic_role=_optional_str(data.get("semantic_role")),
            scope=_optional_str(data.get("scope")),
            signature=_optional_str(data.get("signature")),
            description=_optional_str(data.get("description")),
        )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class GenerationRequest:
    """Inputs available to the model in production."""

    prompt: str
    slot_contract: tuple[str, ...] = ()
    schema: str | None = None
    design_md: str | None = None
    runtime_symbols: tuple[RuntimeSymbol, ...] = ()
    output_kind: OutputKind = "document"
    output_category: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if self.output_kind not in OUTPUT_KINDS:
            raise ValueError(f"invalid output kind {self.output_kind!r}")
        for slot in self.slot_contract:
            if not slot.startswith(":"):
                raise ValueError(
                    f"slot_contract entries must start with ':', got {slot!r}"
                )
        seen: set[str] = set()
        for symbol in self.runtime_symbols:
            if symbol.surface in seen:
                raise ValueError(
                    f"duplicate or conflicting runtime symbol {symbol.surface!r}"
                )
            seen.add(symbol.surface)
            if (
                symbol.surface in self.slot_contract
                and symbol.role != "external_entity"
            ):
                raise ValueError(
                    f"slot_contract surface {symbol.surface!r} conflicts with role {symbol.role!r}"
                )

    def effective_runtime_symbols(self) -> tuple[RuntimeSymbol, ...]:
        """Merge typed symbols with the legacy placeholder inventory."""
        explicit = {symbol.surface: symbol for symbol in self.runtime_symbols}
        return (
            *self.runtime_symbols,
            *(
                RuntimeSymbol(surface=slot, role="external_entity")
                for slot in self.slot_contract
                if slot not in explicit
            ),
        )

    @classmethod
    def from_record(
        cls,
        record: ExampleRecord,
        *,
        schema: str | None = None,
        normalize: bool = True,
        include_design_md: bool = True,
    ) -> GenerationRequest:
        if normalize:
            record = normalize_example_record(record)
        design_md = None
        if include_design_md and record.design_md and str(record.design_md).strip():
            design_md = str(record.design_md).strip()
        return cls(
            prompt=record.prompt.strip(),
            slot_contract=canonical_slot_contract(
                record.openui,
                declared=record.placeholders,
            ),
            schema=schema,
            design_md=design_md,
            output_kind=record.target_kind,
            output_category=record.target_category,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "prompt": self.prompt,
            "slot_contract": list(self.slot_contract),
        }
        if self.schema is not None:
            data["schema"] = self.schema
        if self.design_md is not None:
            data["design_md"] = self.design_md
        if self.runtime_symbols:
            data["runtime_symbols"] = [
                symbol.to_dict() for symbol in self.runtime_symbols
            ]
        if self.output_kind != "document":
            data["output_kind"] = self.output_kind
        if self.output_category is not None:
            data["output_category"] = self.output_category
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationRequest:
        return cls(
            prompt=str(data["prompt"]),
            slot_contract=tuple(data.get("slot_contract") or ()),
            schema=None if data.get("schema") is None else str(data["schema"]),
            design_md=None if data.get("design_md") is None else str(data["design_md"]),
            runtime_symbols=tuple(
                RuntimeSymbol.from_dict(item)
                for item in data.get("runtime_symbols") or ()
            ),
            output_kind=str(data.get("output_kind") or "document"),  # type: ignore[arg-type]
            output_category=_optional_str(data.get("output_category")),
        )


@dataclass(frozen=True)
class CallerContentBinding:
    """One caller-owned value keyed by its external, non-model identity."""

    external_key: str
    value: str

    def __post_init__(self) -> None:
        if not self.external_key or self.external_key.startswith(":"):
            raise ValueError("external_key must be an unprefixed declared key")
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
            self.external_key,
        ):
            raise ValueError(f"invalid external_key {self.external_key!r}")
        if not isinstance(self.value, str):
            raise TypeError("binding value must be a string")


@dataclass(frozen=True)
class ResolvedContentBinding:
    """Deterministic request-local slot assignment for one caller binding."""

    external_key: str
    internal_slot: int
    opaque_slot_id: str
    value: str
    value_digest: str
    value_bytes: int
    occurrence_count: int
    semantic_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_key": self.external_key,
            "internal_slot": self.internal_slot,
            "opaque_slot_id": self.opaque_slot_id,
            "value": self.value,
            "value_digest": self.value_digest,
            "value_bytes": self.value_bytes,
            "occurrence_count": self.occurrence_count,
            "semantic_type": self.semantic_type,
        }

    def evidence_dict(self) -> dict[str, Any]:
        """Sanitized metadata safe for telemetry and generation evidence."""
        return {key: value for key, value in self.to_dict().items() if key != "value"}


@dataclass(frozen=True)
class BoundGenerationResult:
    """Verified template plus separately transported caller-owned values."""

    status: str
    canonical_template: str | None
    template_verification: str
    template_fingerprint: str | None
    bindings: tuple[ResolvedContentBinding, ...]
    materialized_source: None
    materialized_verification: str
    realization_mode: str
    fingerprint: str
    diagnostics: dict[str, Any]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "canonical_template": self.canonical_template,
            "template_verification": self.template_verification,
            "template_fingerprint": self.template_fingerprint,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "materialized_source": self.materialized_source,
            "materialized_verification": self.materialized_verification,
            "realization_mode": self.realization_mode,
            "fingerprint": self.fingerprint,
            "diagnostics": dict(self.diagnostics),
            "errors": list(self.errors),
        }

    def evidence_dict(self) -> dict[str, Any]:
        """Return result evidence without raw caller content or template bytes."""
        return {
            "status": self.status,
            "template_verification": self.template_verification,
            "template_fingerprint": self.template_fingerprint,
            "bindings": [binding.evidence_dict() for binding in self.bindings],
            "materialized_verification": self.materialized_verification,
            "realization_mode": self.realization_mode,
            "fingerprint": self.fingerprint,
            "diagnostics": dict(self.diagnostics),
            "error_count": len(self.errors),
        }


@dataclass(frozen=True)
class ChoiceGenerationResult:
    """Verified source materialized from one exact model choice stream."""

    status: str
    choice_ids: tuple[int, ...]
    choice_tokens: tuple[str, ...]
    opaque_slot_contract: tuple[str, ...]
    slot_projection: tuple[tuple[str, str], ...]
    canonical_source: str
    verification: str
    source_fingerprint: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "choice_ids": list(self.choice_ids),
            "choice_tokens": list(self.choice_tokens),
            "opaque_slot_contract": list(self.opaque_slot_contract),
            "slot_projection": [list(pair) for pair in self.slot_projection],
            "canonical_source": self.canonical_source,
            "verification": self.verification,
            "source_fingerprint": self.source_fingerprint,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChoiceGenerationResult":
        return cls(
            status=str(data["status"]),
            choice_ids=tuple(int(value) for value in data["choice_ids"]),
            choice_tokens=tuple(str(value) for value in data["choice_tokens"]),
            opaque_slot_contract=tuple(
                str(value) for value in data["opaque_slot_contract"]
            ),
            slot_projection=tuple(
                (str(pair[0]), str(pair[1])) for pair in data["slot_projection"]
            ),
            canonical_source=str(data["canonical_source"]),
            verification=str(data["verification"]),
            source_fingerprint=str(data["source_fingerprint"]),
            fingerprint=str(data["fingerprint"]),
        )


def choice_generation_fingerprint(payload: dict[str, Any]) -> str:
    """Fingerprint exact choice evidence and its deterministic materialization."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bound_generation_fingerprint(payload: dict[str, Any]) -> str:
    """Full deterministic fingerprint for a terminal binding result."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_slot_contract(
    openui: str,
    *,
    declared: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Ordered placeholder inventory used by production codec slot pointers."""
    extracted = extract_placeholders(openui or "")
    if declared:
        return tuple(merge_placeholders(declared, extracted))
    return tuple(extracted)


def normalize_example_record(record: ExampleRecord) -> ExampleRecord:
    """
    Canonical training/eval record shape: style-stripped OpenUI, aligned placeholders,
    and component signatures normalized for SwitchItem / Slider.
    """
    from slm_training.dsl.parser import ParseError, validate

    if record.target_kind != "document":
        return ExampleRecord(
            id=record.id,
            prompt=record.prompt.strip(),
            openui=record.openui.strip(),
            placeholders=list(
                canonical_slot_contract(
                    record.openui,
                    declared=record.placeholders,
                )
            ),
            split=record.split,
            source=record.source,
            meta={**dict(record.meta), "schema_normalized": True},
            design_md=record.design_md,
            target_kind=record.target_kind,
            target_category=record.target_category,
            accepted_outputs=list(record.accepted_outputs),
        )

    scrubbed = strip_style_literals(record.openui or "")
    scrubbed = _normalize_component_signatures(scrubbed)
    try:
        program = validate(scrubbed)
        openui = strip_style_literals(program.serialized or scrubbed.strip())
        openui = _normalize_component_signatures(openui)
        placeholders = canonical_slot_contract(
            openui,
            declared=merge_placeholders(
                record.placeholders or [],
                program.placeholders or [],
            ),
        )
    except (ParseError, ValueError, RuntimeError):
        openui = scrubbed.strip()
        placeholders = canonical_slot_contract(
            openui,
            declared=record.placeholders,
        )

    placeholders = canonical_slot_contract(openui, declared=placeholders)

    return ExampleRecord(
        id=record.id,
        prompt=record.prompt.strip(),
        openui=openui,
        placeholders=list(placeholders),
        split=record.split,
        source=record.source,
        meta={**dict(record.meta), "schema_normalized": True},
        design_md=record.design_md,
        target_kind=record.target_kind,
        target_category=record.target_category,
        accepted_outputs=list(record.accepted_outputs),
    )


def _normalize_component_signatures(openui: str) -> str:
    """Rewrite common SwitchItem / Slider fixture drift to canonical prop order."""
    lines = (openui or "").splitlines()
    out: list[str] = []
    for line in lines:
        if re.search(r"\bSwitchItem\s*\(", line):
            out.append(_normalize_switchitem_line(line))
            continue
        if re.search(r"\bSlider\s*\(", line):
            out.append(_normalize_slider_line(line))
            continue
        out.append(line)
    return "\n".join(out)


def _split_top_level_args(inner: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(inner[start:i].strip())
            start = i + 1
    tail = inner[start:].strip()
    if tail:
        args.append(tail)
    return args


def generated_enum_literal(component: str, prop: str, value: str | None = None) -> str:
    """Preserve a legal enum source literal or select the generated default."""
    spec = library_schema()["$defs"][component]["properties"][prop]
    if value is not None:
        try:
            if json.loads(value) in spec["enum"]:
                return value
        except (json.JSONDecodeError, TypeError):
            pass
    return json.dumps(spec["enum"][0])


def generated_value_literal(component: str, prop: str, value: str) -> str:
    """Adapt one source literal to the generated property's container shape."""
    spec = library_schema()["$defs"][component]["properties"][prop]
    if spec.get("type") == "array" and not value.lstrip().startswith("["):
        return f"[{value}]"
    return value


def _normalize_switchitem_line(line: str) -> str:
    m = re.match(r"^(\s*)(\w+)\s*=\s*SwitchItem\((.*)\)\s*$", line)
    if not m:
        return line
    indent, name, inner = m.groups()
    args = _split_top_level_args(inner)
    if len(args) < 3:
        return line
    label, description, third = args[0], args[1], args[2]
    if third in {"true", "false"}:
        third = json.dumps(name)
    elif third.startswith('"') and third.endswith('"') and not third.startswith('":'):
        pass
    elif not third.startswith('"'):
        third = json.dumps(third)
    rest = ", ".join(args[3:]) if len(args) > 3 else ""
    body = f"{label}, {description}, {third}"
    if rest:
        body = f"{body}, {rest}"
    return f"{indent}{name} = SwitchItem({body})"


def _normalize_slider_line(line: str) -> str:
    m = re.match(r"^(\s*)(\w+)\s*=\s*Slider\((.*)\)\s*$", line)
    if not m:
        return line
    indent, name, inner = m.groups()
    args = _split_top_level_args(inner)
    if len(args) >= 7:
        args[1] = generated_enum_literal("Slider", "variant", args[1])
        args[5] = generated_value_literal("Slider", "defaultValue", args[5])
        return f"{indent}{name} = Slider({', '.join(args)})"
    if len(args) == 4 and args[0].startswith('":'):
        label = args[0]
        min_v, max_v, default_v = args[1], args[2], args[3]
        variant = generated_enum_literal("Slider", "variant")
        default_v = generated_value_literal("Slider", "defaultValue", default_v)
        return (
            f'{indent}{name} = Slider("{name}", {variant}, {min_v}, {max_v}, 1, '
            f"{default_v}, {label})"
        )
    if len(args) == 3 and args[0].startswith('":'):
        label, min_v, max_v = args
        variant = generated_enum_literal("Slider", "variant")
        default_v = generated_value_literal("Slider", "defaultValue", "50")
        return (
            f'{indent}{name} = Slider("{name}", {variant}, {min_v}, {max_v}, 1, '
            f"{default_v}, {label})"
        )
    return line


def binders_in_source(openui: str) -> list[str]:
    """Return binder names in first-appearance order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _BINDER_RE.finditer(openui or ""):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def load_generation_requests(
    path: Path | str,
    *,
    schema: str | None = None,
    normalize: bool = True,
) -> list[GenerationRequest]:
    from slm_training.dsl.schema import load_jsonl

    return [
        GenerationRequest.from_record(record, schema=schema, normalize=normalize)
        for record in load_jsonl(path)
    ]


__all__ = [
    "BoundGenerationResult",
    "CallerContentBinding",
    "ChoiceGenerationResult",
    "GenerationRequest",
    "ResolvedContentBinding",
    "bound_generation_fingerprint",
    "choice_generation_fingerprint",
    "binders_in_source",
    "canonical_slot_contract",
    "load_generation_requests",
    "normalize_example_record",
]
