"""Generation request contract and canonical example-record normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from slm_training.data.structure import strip_style_literals
from slm_training.dsl.placeholders import extract_placeholders, merge_placeholders
from slm_training.dsl.schema import ExampleRecord

_BINDER_RE = re.compile(r"(?m)^([a-z_][A-Za-z0-9_]*)\s*=")


@dataclass(frozen=True)
class GenerationRequest:
    """Inputs available to the model in production."""

    prompt: str
    slot_contract: tuple[str, ...] = ()
    schema: str | None = None
    design_md: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        for slot in self.slot_contract:
            if not slot.startswith(":"):
                raise ValueError(f"slot_contract entries must start with ':', got {slot!r}")

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
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationRequest:
        return cls(
            prompt=str(data["prompt"]),
            slot_contract=tuple(data.get("slot_contract") or ()),
            schema=None if data.get("schema") is None else str(data["schema"]),
            design_md=None if data.get("design_md") is None else str(data["design_md"]),
        )


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
        return line
    if len(args) == 4 and args[0].startswith('":'):
        label = args[0]
        min_v, max_v, default_v = args[1], args[2], args[3]
        return (
            f'{indent}{name} = Slider("{name}", "default", {min_v}, {max_v}, 1, '
            f"{default_v}, {label})"
        )
    if len(args) == 3 and args[0].startswith('":'):
        label, min_v, max_v = args
        return (
            f'{indent}{name} = Slider("{name}", "default", {min_v}, {max_v}, 1, '
            f"50, {label})"
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
    "GenerationRequest",
    "binders_in_source",
    "canonical_slot_contract",
    "load_generation_requests",
    "normalize_example_record",
]
