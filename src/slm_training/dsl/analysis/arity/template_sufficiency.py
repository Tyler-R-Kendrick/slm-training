"""Template-abstraction sufficiency audit for CAP1-05 (SLM-85).

Determines whether replacing user values with canonical template slots discards
information that the compiler/grammar actually needs for structural decisions.
The audit operates on production-token literals (strings, numbers, booleans) and
compares the structural choice stream before and after a controlled value change.
It is Torch-free and uses only existing pack/codec utilities.

This is a wiring harness and counterexample finder, not a ship claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from slm_training.data.leakage import normalize_openui_structure
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.production_codec import (
    LIT_PREFIX,
    decode_productions,
    encode_choices,
    encode_openui,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl

# User-facing string props that must be placeholders in the OpenUI pack.
CONTENT_PROPS = frozenset(
    {
        "text",
        "label",
        "title",
        "body",
        "content",
        "placeholder",
        "alt",
        "hint",
        "description",
        "trigger",
    }
)

ValueKind = Literal["string", "number", "boolean", "null"]
ChangeKind = Literal[
    "empty_string",
    "nonempty_string",
    "longer_string",
    "zero",
    "nonzero",
    "negative",
    "positive",
    "integer",
    "fractional",
    "flip_bool",
]


@dataclass(frozen=True)
class ValueClass:
    """One canonicalized value class in the template contract."""

    class_id: str
    value_kind: ValueKind
    slot_representation: str
    information_retained: tuple[str, ...]
    information_discarded: tuple[str, ...]
    structural_decisions: tuple[str, ...] = ()
    pack_constraints: tuple[str, ...] = ()
    late_realization_owner: str = ""
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        def _hash(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

        return {
            "class_id": self.class_id,
            "value_kind": self.value_kind,
            "slot_representation": self.slot_representation,
            "information_retained": list(self.information_retained),
            "information_discarded": list(self.information_discarded),
            "structural_decisions": list(self.structural_decisions),
            "pack_constraints": list(self.pack_constraints),
            "late_realization_owner": self.late_realization_owner,
            "example_fingerprints": [_hash(e) for e in self.examples],
        }


@dataclass(frozen=True)
class TemplateContractInventory:
    """Machine-readable inventory of every canonicalized value class."""

    version: str
    value_classes: tuple[ValueClass, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "value_classes": [vc.to_dict() for vc in self.value_classes],
        }


@dataclass(frozen=True)
class TemplateVariant:
    """One paired example: original vs controlled value change."""

    record_id: str
    value_class_id: str
    change_kind: ChangeKind
    original_literal: str
    variant_literal: str
    original_openui: str
    variant_openui: str
    template_projection: str
    original_choice_stream: tuple[str, ...]
    variant_choice_stream: tuple[str, ...]
    is_violation: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        def _hash(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

        return {
            "record_id": self.record_id,
            "value_class_id": self.value_class_id,
            "change_kind": self.change_kind,
            "original_literal_fingerprint": _hash(self.original_literal),
            "variant_literal_fingerprint": _hash(self.variant_literal),
            "template_projection": self.template_projection,
            "original_choice_stream": list(_sanitize_choice_stream(self.original_choice_stream)),
            "variant_choice_stream": list(_sanitize_choice_stream(self.variant_choice_stream)),
            "is_violation": self.is_violation,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RefinementCandidate:
    """Proposed bounded summary that may remove a class of violations."""

    value_class_id: str
    retained_attributes: tuple[str, ...]
    estimated_added_bits: float
    removes_violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_class_id": self.value_class_id,
            "retained_attributes": list(self.retained_attributes),
            "estimated_added_bits": self.estimated_added_bits,
            "removes_violations": list(self.removes_violations),
        }


@dataclass
class TemplateSufficiencyReport:
    """Aggregate report for a template-sufficiency audit."""

    inventory: TemplateContractInventory
    variants: list[TemplateVariant]
    violations: list[TemplateVariant]
    refinements: list[RefinementCandidate]
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory": self.inventory.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "violations": [v.to_dict() for v in self.violations],
            "refinements": [r.to_dict() for r in self.refinements],
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Literal token helpers
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _decode_literal_payload(payload: str) -> tuple[str, ValueKind]:
    """Decode a production literal token payload back to raw source text + kind."""
    if payload == "null":
        return "null", "null"
    if payload in {"true", "false"}:
        return payload, "boolean"
    if _NUM_RE.fullmatch(payload):
        return payload, "number"
    # JSON-encoded string -> raw string value.
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return payload, "string"
    if isinstance(value, bool):
        return str(value).lower(), "boolean"
    if isinstance(value, (int, float)):
        return str(value), "number"
    if value is None:
        return "null", "null"
    return value, "string"


def _encode_literal_payload(source_text: str, kind: ValueKind) -> str:
    """Encode a source text into a production literal token payload."""
    if kind == "null":
        return "null"
    if kind == "boolean":
        return source_text.strip().lower()
    if kind == "number":
        return source_text.strip()
    return json.dumps(source_text)


def _literal_kind(payload: str) -> ValueKind:
    if payload == "null":
        return "null"
    if payload in {"true", "false"}:
        return "boolean"
    if _NUM_RE.fullmatch(payload):
        return "number"
    return "string"


def _template_projection(openui: str) -> str:
    """Collapse all literal values so only structural/template shape remains."""
    text = normalize_openui_structure(openui)
    # Replace quoted string literals with a generic marker.
    text = re.sub(r'"[^"]*"', '"__LIT__"', text)
    # Replace numeric and boolean literals outside quotes.
    text = re.sub(r"\b-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\b", "0", text)
    text = re.sub(r"\b(true|false)\b", "__BOOL__", text)
    return text.strip()


def _sanitize_choice_stream(tokens: Sequence[str]) -> tuple[str, ...]:
    """Replace literal payloads with fingerprints so raw user text is not stored."""

    def _sanitize(token: str) -> str:
        if not token.startswith(LIT_PREFIX):
            return token
        payload = token[len(LIT_PREFIX) :]
        kind = _literal_kind(payload)
        fp = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{LIT_PREFIX}{kind}:{fp}"

    return tuple(_sanitize(t) for t in tokens)


def _class_id_for_literal(kind: ValueKind, context_name: str | None) -> str:
    """Stable class identifier from literal kind and optional grammar context."""
    ctx = re.sub(r"[^A-Za-z0-9_]+", "_", (context_name or "value").lower()).strip("_")
    return f"{kind}_{ctx}"


def _variant_literal(kind: ValueKind, original: str, change: ChangeKind) -> str | None:
    """Return a variant literal value (raw source text) or None if not applicable."""
    if kind == "string":
        decoded = original
        if change == "empty_string":
            return "" if decoded != "" else None
        if change == "nonempty_string":
            return "x" if decoded == "" else None
        if change == "longer_string":
            return decoded + " extra" if len(decoded) < 80 else None
        return None
    if kind == "number":
        try:
            num = float(original)
        except ValueError:
            return None
        if change == "zero":
            return "0" if num != 0.0 else None
        if change == "nonzero":
            return "1" if num == 0.0 else None
        if change == "negative":
            return str(-abs(num)) if num >= 0 else None
        if change == "positive":
            return str(abs(num)) if num < 0 else None
        if change == "integer":
            if "." in original.lower() or "e" in original.lower():
                return str(int(num))
            return None
        if change == "fractional":
            if "." not in original.lower() and "e" not in original.lower():
                return f"{num}.5"
            return None
        return None
    if kind == "boolean":
        if change == "flip_bool":
            return "false" if original == "true" else "true"
        return None
    return None


# ---------------------------------------------------------------------------
# Inventory extraction
# ---------------------------------------------------------------------------


def _context_name_for_literal(
    tokens: Sequence[str], idx: int, slot_contract: Sequence[str]
) -> str | None:
    """Best-effort grammar context name for a literal (preceding property key)."""
    # Walk backwards to find the most recent NAME token, which is likely a prop.
    for j in range(idx - 1, max(-1, idx - 8), -1):
        tok = tokens[j]
        if tok.startswith("n:"):
            return tok[2:]
    return None


def extract_value_classes(
    records: Iterable[ExampleRecord],
    *,
    max_examples_per_class: int = 10,
    version: str = "cap1-05.v1",
) -> TemplateContractInventory:
    """Build a machine-readable inventory of canonicalized value classes."""
    classes: dict[str, dict[str, Any]] = {}
    example_counts: Counter[str] = Counter()

    for record in records:
        if not record.openui:
            continue
        try:
            program = encode_openui(record.openui)
        except ParseError:
            continue
        for idx, tok in enumerate(program.tokens):
            if not tok.startswith(LIT_PREFIX):
                continue
            payload = tok[len(LIT_PREFIX) :]
            kind = _literal_kind(payload)
            if kind == "null":
                continue
            src_text, _ = _decode_literal_payload(payload)
            context = _context_name_for_literal(
                program.tokens, idx, program.slot_contract
            )
            class_id = _class_id_for_literal(kind, context)
            if class_id not in classes:
                retained, discarded = _default_retained_discarded(kind, context)
                classes[class_id] = {
                    "class_id": class_id,
                    "value_kind": kind,
                    "slot_representation": f"{kind.upper()}_SLOT",
                    "information_retained": retained,
                    "information_discarded": discarded,
                    "structural_decisions": _default_structural_decisions(context),
                    "pack_constraints": _default_pack_constraints(kind, context),
                    "late_realization_owner": _late_realization_owner(context),
                    "examples": [],
                }
            if example_counts[class_id] < max_examples_per_class:
                classes[class_id]["examples"].append(src_text)
                example_counts[class_id] += 1

    value_classes = tuple(
        ValueClass(
            class_id=c["class_id"],
            value_kind=c["value_kind"],
            slot_representation=c["slot_representation"],
            information_retained=tuple(c["information_retained"]),
            information_discarded=tuple(c["information_discarded"]),
            structural_decisions=tuple(c["structural_decisions"]),
            pack_constraints=tuple(c["pack_constraints"]),
            late_realization_owner=c["late_realization_owner"],
            examples=tuple(c["examples"]),
        )
        for c in classes.values()
    )
    return TemplateContractInventory(version=version, value_classes=value_classes)


def _default_retained_discarded(
    kind: ValueKind, context: str | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if kind == "string":
        return ("token_kind",), (
            "exact_text",
            "length",
            "line_breaks",
            "locale",
            "role",
        )
    if kind == "number":
        return ("token_kind",), (
            "sign",
            "magnitude",
            "range_bin",
            "unit",
            "sentinel",
        )
    if kind == "boolean":
        return ("token_kind",), ("polarity",)
    return ("token_kind",), ("value",)


def _default_structural_decisions(context: str | None) -> tuple[str, ...]:
    if context in {"text", "label", "title", "body", "content"}:
        return ("component_presence", "layout_wrap", "multiline")
    if context in {"m", "margin", "gap", "width", "height", "padding"}:
        return ("dimension", "direction", "overflow")
    if context in {"visible", "enabled", "editable", "readonly"}:
        return ("branch",)
    return ("property_assignment",)


def _default_pack_constraints(kind: ValueKind, context: str | None) -> tuple[str, ...]:
    if context in CONTENT_PROPS:
        return ("must_be_placeholder",)
    if kind == "number" and context in {"m", "margin", "gap", "width", "height"}:
        return ("finite_numeric",)
    return ("finite_literal",)


def _late_realization_owner(context: str | None) -> str:
    if context in CONTENT_PROPS:
        return "surface_realization"
    if context in {"identifier", "name", "id", "ref"}:
        return "opaque_region"
    return "semantic_decoder"


# ---------------------------------------------------------------------------
# Variant generation and oracle
# ---------------------------------------------------------------------------


def _choice_stream(openui: str) -> tuple[str, ...]:
    """Encode OpenUI into the pure grammar-choice token stream."""
    return encode_choices(openui).tokens


def _structural_choice_stream(openui: str) -> tuple[str, ...]:
    """Choice stream with literal payloads collapsed so only structure is compared."""
    tokens = _choice_stream(openui)

    def _collapse(token: str) -> str:
        if not token.startswith(LIT_PREFIX):
            return token
        payload = token[len(LIT_PREFIX) :]
        return f"{LIT_PREFIX}{_literal_kind(payload)}"

    return tuple(_collapse(t) for t in tokens)


def generate_variants(
    records: Iterable[ExampleRecord],
    inventory: TemplateContractInventory,
    *,
    max_per_record: int = 20,
) -> list[TemplateVariant]:
    """Generate paired examples identical under template projection."""
    variants: list[TemplateVariant] = []
    class_ids = {vc.class_id for vc in inventory.value_classes}

    for record in records:
        if not record.openui:
            continue
        try:
            program = encode_openui(record.openui)
        except ParseError:
            continue
        template_projection = _template_projection(record.openui)
        original_choice = _choice_stream(record.openui)

        generated = 0
        for idx, tok in enumerate(program.tokens):
            if not tok.startswith(LIT_PREFIX):
                continue
            payload = tok[len(LIT_PREFIX) :]
            kind = _literal_kind(payload)
            if kind == "null":
                continue
            src_text, _ = _decode_literal_payload(payload)
            context = _context_name_for_literal(
                program.tokens, idx, program.slot_contract
            )
            class_id = _class_id_for_literal(kind, context)
            if class_id not in class_ids:
                continue
            changes: list[ChangeKind] = []
            if kind == "string":
                changes = ["empty_string", "nonempty_string", "longer_string"]
            elif kind == "number":
                changes = ["zero", "nonzero", "negative", "positive"]
            elif kind == "boolean":
                changes = ["flip_bool"]
            for change in changes:
                if generated >= max_per_record:
                    break
                variant_src = _variant_literal(kind, src_text, change)
                if variant_src is None:
                    continue
                variant_payload = _encode_literal_payload(variant_src, kind)
                variant_tok = f"{LIT_PREFIX}{variant_payload}"
                variant_tokens = list(program.tokens)
                variant_tokens[idx] = variant_tok
                try:
                    variant_openui = decode_productions(
                        variant_tokens, program.slot_contract
                    )
                    validate(variant_openui)
                except (ParseError, ValueError):
                    continue
                # Template projection must stay identical.
                if _template_projection(variant_openui) != template_projection:
                    continue
                variant_choice = _choice_stream(variant_openui)
                is_violation = (
                    _structural_choice_stream(variant_openui)
                    != _structural_choice_stream(record.openui)
                )
                reason = (
                    "value change altered structural grammar choice stream"
                    if is_violation
                    else "value change preserved structural grammar choice stream"
                )
                variants.append(
                    TemplateVariant(
                        record_id=record.id,
                        value_class_id=class_id,
                        change_kind=change,
                        original_literal=src_text,
                        variant_literal=variant_src,
                        original_openui=record.openui,
                        variant_openui=variant_openui,
                        template_projection=template_projection,
                        original_choice_stream=original_choice,
                        variant_choice_stream=variant_choice,
                        is_violation=is_violation,
                        reason=reason,
                    )
                )
                generated += 1
            if generated >= max_per_record:
                break
    return variants


# ---------------------------------------------------------------------------
# Refinement candidates
# ---------------------------------------------------------------------------


def propose_refinements(
    inventory: TemplateContractInventory,
    violations: Sequence[TemplateVariant],
) -> list[RefinementCandidate]:
    """Propose bounded summaries that could remove observed violations."""
    by_class: dict[str, list[TemplateVariant]] = {}
    for v in violations:
        by_class.setdefault(v.value_class_id, []).append(v)

    candidates: list[RefinementCandidate] = []
    for vc in inventory.value_classes:
        cls_violations = by_class.get(vc.class_id, [])
        if not cls_violations:
            continue
        change_kinds = {v.change_kind for v in cls_violations}
        retained: list[str] = []
        if vc.value_kind == "string":
            if any(k in change_kinds for k in ("empty_string", "nonempty_string")):
                retained.append("empty_nonempty")
            if "longer_string" in change_kinds:
                retained.append("length_bin")
        elif vc.value_kind == "number":
            if any(k in change_kinds for k in ("negative", "positive")):
                retained.append("sign")
            if any(k in change_kinds for k in ("zero", "nonzero")):
                retained.append("zero_nonzero")
            if any(k in change_kinds for k in ("integer", "fractional")):
                retained.append("integer_fractional")
        elif vc.value_kind == "boolean":
            retained.append("polarity")
        estimated_bits = len(retained) * 0.5  # coarse wiring-only estimate
        candidates.append(
            RefinementCandidate(
                value_class_id=vc.class_id,
                retained_attributes=tuple(retained),
                estimated_added_bits=estimated_bits,
                removes_violations=tuple(sorted(change_kinds)),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Audit entry point
# ---------------------------------------------------------------------------


def audit_template_sufficiency(
    records: Iterable[ExampleRecord],
    *,
    max_examples_per_class: int = 10,
    max_per_record: int = 20,
) -> TemplateSufficiencyReport:
    """Run the CAP1-05 template-sufficiency audit."""
    records = list(records)
    inventory = extract_value_classes(
        records, max_examples_per_class=max_examples_per_class
    )
    variants = generate_variants(records, inventory, max_per_record=max_per_record)
    violations = [v for v in variants if v.is_violation]
    refinements = propose_refinements(inventory, violations)

    by_class = Counter(v.value_class_id for v in variants)
    violation_by_class = Counter(v.value_class_id for v in violations)
    metrics: dict[str, Any] = {
        "records_audited": len(records),
        "value_classes": len(inventory.value_classes),
        "variants_generated": len(variants),
        "violations": len(violations),
        "variants_by_class": dict(by_class),
        "violations_by_class": dict(violation_by_class),
    }
    notes = [
        "Audit covers literal values only (strings, numbers, booleans).",
        "Identifier/component-reference abstraction is inherited from structural fingerprinting.",
        "Estimated added bits are coarse wiring placeholders, not measured state counts.",
    ]
    return TemplateSufficiencyReport(
        inventory=inventory,
        variants=variants,
        violations=violations,
        refinements=refinements,
        metrics=metrics,
        notes=notes,
    )


def load_records(path: Path | str) -> list[ExampleRecord]:
    return list(load_jsonl(path))
