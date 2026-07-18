"""Canonicalize Lark-generated ASTs into StateAtoms."""

from __future__ import annotations

from typing import Any

from slm_training.dsl.analysis.arity.types import StateAtom
from slm_training.dsl.placeholders import is_placeholder


def _binding_order(node: Any) -> list[str]:
    """Return statement names in first-appearance order."""
    order: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            sid = value.get("statementId")
            if isinstance(sid, str) and sid not in order:
                order.append(sid)
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(node)
    return order


def _placeholder_index(value: str, placeholder_order: list[str]) -> int:
    """Return stable index for a placeholder value."""
    if value not in placeholder_order:
        placeholder_order.append(value)
    return placeholder_order.index(value)


def _canonicalize_value(
    value: Any,
    bindings: dict[str, int],
    placeholder_order: list[str],
) -> StateAtom:
    """Recursively canonicalize one AST value."""
    if value is None:
        return StateAtom.literal(None)
    if isinstance(value, bool):
        return StateAtom.literal(value)
    if isinstance(value, (int, float)):
        return StateAtom.literal(value)
    if isinstance(value, str):
        if is_placeholder(value):
            return StateAtom.placeholder(_placeholder_index(value, placeholder_order))
        return StateAtom.literal(value)
    if isinstance(value, list):
        return StateAtom.list(tuple(_canonicalize_value(item, bindings, placeholder_order) for item in value))
    if isinstance(value, dict):
        # Resolved reference to another binding.
        if value.get("type") == "ref":
            name = value.get("name")
            if isinstance(name, str) and name in bindings:
                return StateAtom.ref(bindings[name])
            return StateAtom.hole("ref")

        # Component / element node.
        type_name = value.get("typeName") or value.get("name")
        if isinstance(type_name, str):
            props: dict[str, Any] = value.get("props", {})
            # Stable key order: known positional keys first, then alphabetical.
            ordered_keys = sorted(props.keys())
            canonical_props = tuple(
                (key, _canonicalize_value(props[key], bindings, placeholder_order))
                for key in ordered_keys
            )
            return StateAtom.component(type_name, canonical_props)

        # Fallback: treat as anonymous record.
        items = tuple(
            (key, _canonicalize_value(val, bindings, placeholder_order))
            for key, val in sorted(value.items())
            if key != "statementId"
        )
        return StateAtom.component("record", items)

    return StateAtom.hole(f"unknown:{type(value).__name__}")


def canonicalize_ast(root: Any) -> tuple[StateAtom, ...]:
    """Convert a resolved typed AST root into a canonical atom tuple.

    The returned tuple is hashable and stable under:
    - renaming statement bindings (refs become De-Bruijn-like indices),
    - reordering component props (sorted canonically),
    - renaming placeholder surface strings (placeholder atoms use sorted indices).
    """
    bindings = {name: idx for idx, name in enumerate(_binding_order(root))}
    placeholder_order: list[str] = []
    atom = _canonicalize_value(root, bindings, placeholder_order)
    return (atom,)
