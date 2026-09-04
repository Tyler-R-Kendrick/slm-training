"""Slot admission and the evidence a choice phase leaves.

One responsibility: whether a required-slot margin position will accept a given
slot, which components own which slots, and the evidence recorded for a choice
phase.

Extracted from ``TwoTowerModel``. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from typing import Any

from slm_training.models.twotower_schema import (
    schema_can_reach_visible_slot,
)


def slot_component_owners(source: str) -> dict[str, str]:
    from slm_training.dsl.lang_core import parse

    owners: dict[str, str] = {}

    def walk(value: object, component: str | None = None) -> None:
        if isinstance(value, dict):
            owner = (
                str(value["typeName"])
                if value.get("type") == "element" and value.get("typeName")
                else component
            )
            for child in value.values():
                walk(child, owner)
        elif isinstance(value, list):
            for child in value:
                walk(child, component)
        elif isinstance(value, str) and value.startswith(":") and component is not None:
            owners.setdefault(value, component)

    walk(parse(source).root)
    return owners


def required_slot_margin_position_accepts_slot(state: Any) -> bool:
    """Whether the active argument position is schema-tagged for a slot.

    E630: mirrors ``_schema_role_slot_bias``'s own ``accepts_slot`` gate
    exactly -- a choice decoder ``component`` frame's current argument must carry
    ``x-openui-placeholder`` (the content properties: ``label``, ``text``,
    ``title``, ...), and an ``object`` frame's current property schema
    must be able to reach a visible slot at all. Any other frame kind
    (``variadic``, ``fixed``, ...) is left permissive (``True``) since
    this bias's only observed failure mode is stuffing missing slots into
    optional enum/opaque *component*/*object* properties, not array items.
    The lexer compiler exposes its active call through a grammar engine
    instead of schema frames, so its equivalent gate accepts only an
    explicitly tagged placeholder or a non-enum string property.
    """
    frames = list(getattr(state, "frames", ()))
    if not frames and getattr(state, "engine", None) is not None:
        from slm_training.dsl.grammar.fastpath.compiler_draft import _active_call
        from slm_training.dsl.lang_core import library_schema

        active = _active_call(state.engine)
        if active is None:
            return False
        component, index, _ = active
        definition = library_schema().get("$defs", {}).get(component) or {}
        properties = tuple((definition.get("properties") or {}).values())
        if not 0 <= index < len(properties):
            return False
        schema = properties[index]
        return bool(schema.get("x-openui-placeholder")) or (
            schema.get("type") == "string" and "enum" not in schema
        )
    if not frames:
        return True
    frame = frames[-1]
    kind = getattr(frame, "kind", None)
    if kind not in {"component", "object"}:
        return True
    schemas = tuple(getattr(frame, "schemas", ()))
    index = int(getattr(frame, "arg_index", -1))
    if not (0 <= index < len(schemas)):
        return False
    if kind == "component":
        return bool(schemas[index].get("x-openui-placeholder"))
    return schema_can_reach_visible_slot(dict(schemas[index]))


def choice_phase_evidence(state: Any) -> dict[str, object]:
    """Describe the bounded generated-state phase around a choice."""
    frames = list(getattr(state, "frames", ()))
    structural_list = bool(
        getattr(state, "mode", None) == "structural"
        and frames
        and frames[-1].kind == "variadic"
        and frames[-1].expr_type == "array"
    )
    if getattr(state, "current_marker", None) == "r=":
        aggregation_scope = "v05_root"
    elif structural_list:
        aggregation_scope = (
            "structural_root_list" if len(frames) == 1 else "structural_nested_list"
        )
    else:
        aggregation_scope = "other"
    return {
        "aggregation_scope": aggregation_scope,
        "frame_depth": len(frames),
        "frame_path_truncated": len(frames) > 8,
        "frame_path": [
            {
                "kind": str(frame.kind),
                "expr_type": str(frame.expr_type),
                "phase": str(frame.phase),
                "arg_index": int(frame.arg_index),
            }
            for frame in frames[-8:]
        ],
    }
