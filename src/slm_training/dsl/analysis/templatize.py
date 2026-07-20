"""Deterministic content-literal → placeholder templatizer.

The official parser's placeholder content policy rejects quoted literals in
user-facing content props (``CONTENT_PROPS``: text, label, title, …) — a
candidate carrying ``TextContent("Welcome back")`` cannot pass ``validate``
at all. This pass rewrites exactly those literals into ``:namespace.slot``
placeholder tokens so the program becomes policy-admissible, while the
original strings are preserved as meta-only provenance by the caller.

Naming is a pure function of AST position, never of content or run state:
the namespace is the enclosing statement's *canonical* binder (``root`` /
``v0, v1, …`` derived from the deterministic statement order, independent of
surface names), the slot is the prop name, and repeated ``(binder, prop)``
occurrences get ``_2, _3, …`` ordinals in traversal order. Alpha-equivalent
inputs therefore templatize to byte-identical canonical output.

Replacement scope (v1) is the dual of the two gates that reject literals:

- every scalar string literal in a ``CONTENT_PROPS`` position (the official
  parser's placeholder policy rejects these outright), and
- free-form scalar strings in other props — exactly the strings the quality
  gate hard-rejects as ``non_placeholder_string`` (``assess_record``); its
  enum-like ``[a-z0-9_-]+`` and numeric exemptions are mirrored here, so
  machine-ish tokens (``"submit"``, ``"email"``, ``"python"``) survive.

Enum-admissible values, style/structural tokens, and strings inside arrays
are counted but never rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from typing import Any

from slm_training.data.structure import STYLE_STRING_TOKENS
from slm_training.dsl.analysis.optimize import STRUCTURAL_LITERALS
from slm_training.dsl.lang_core import library_schema
from slm_training.dsl.placeholders import CONTENT_PROPS, extract_placeholders, is_placeholder
from slm_training.dsl.production_codec import (
    emit_statement_bindings,
    parse_statement_bindings,
    statement_binding_order,
)


# Mirror assess_record's non_placeholder_string exemptions (data/quality.py):
# lowercase enum/field-name tokens and numeric strings are not "content".
_ENUM_LIKE_RE = re.compile(r"[a-z0-9_\-]+")
_NUMERIC_RE = re.compile(r"[0-9]+(\.[0-9]+)?")


@dataclass(frozen=True)
class TemplatizeResult:
    source: str
    placeholders: tuple[str, ...]
    replacements: dict[str, str]
    skipped: dict[str, int]

    @property
    def changed(self) -> bool:
        return bool(self.replacements)


def _prop_enum(defs: dict[str, Any], component: str, prop: str) -> frozenset[str]:
    spec = ((defs.get(component) or {}).get("properties") or {}).get(prop) or {}
    values = spec.get("enum")
    if isinstance(values, list):
        return frozenset(str(value) for value in values)
    return frozenset()


def templatize(source: str, *, dsl: str | None = None) -> TemplatizeResult:
    """Replace content-prop string literals with positional placeholder tokens.

    Reads the binding AST without the official policy check (the literals we
    are repairing are exactly what that check rejects); the caller re-validates
    the emitted result through the official parser. The output is emitted via
    the production codec, so it is D2-canonical.
    """
    bindings = parse_statement_bindings(source, dsl=dsl, validate=False)
    order = statement_binding_order(bindings, dsl=dsl)
    canonical_names = {
        name: ("root" if name == "root" else f"v{index}")
        for index, name in enumerate(order[:-1])
    }
    canonical_names["root"] = "root"
    defs = dict(library_schema().get("$defs") or {})

    used_tokens = set(extract_placeholders(source))
    ordinals: dict[tuple[str, str], int] = {}
    replacements: dict[str, str] = {}
    skipped = {
        "enum_value": 0,
        "structural_literal": 0,
        "enum_like": 0,
        "array_string": 0,
        "extra_arg": 0,
    }

    def next_token(namespace: str, slot: str) -> str:
        ordinal = ordinals.get((namespace, slot), 0) + 1
        while True:
            token = (
                f":{namespace}.{slot}"
                if ordinal == 1
                else f":{namespace}.{slot}_{ordinal}"
            )
            if token not in used_tokens:
                ordinals[(namespace, slot)] = ordinal
                used_tokens.add(token)
                return token
            ordinal += 1

    def rewrite_value(
        value: Any, namespace: str, component: str, prop: str, in_array: bool
    ) -> Any:
        if isinstance(value, str):
            if is_placeholder(value):
                return value
            if in_array:
                skipped["array_string"] += 1
                return value
            if value in STYLE_STRING_TOKENS or value.lower() in STRUCTURAL_LITERALS:
                skipped["structural_literal"] += 1
                return value
            if value in _prop_enum(defs, component, prop):
                skipped["enum_value"] += 1
                return value
            if prop not in CONTENT_PROPS and (
                _ENUM_LIKE_RE.fullmatch(value) or _NUMERIC_RE.fullmatch(value)
            ):
                skipped["enum_like"] += 1
                return value
            token = next_token(namespace, prop)
            replacements[token] = value
            return token
        if isinstance(value, list):
            return [
                rewrite_value(item, namespace, component, prop, True)
                if isinstance(item, str)
                else rewrite_node(item, namespace)
                for item in value
            ]
        return rewrite_node(value, namespace)

    def rewrite_node(node: Any, namespace: str) -> Any:
        if isinstance(node, list):
            return [rewrite_node(item, namespace) for item in node]
        if not isinstance(node, dict):
            return node
        if node.get("type") == "element":
            type_name = str(node.get("typeName") or "")
            props = node.get("props") or {}
            new_props: dict[str, Any] = {}
            for key, value in props.items():
                if key == "_args":
                    counted = []
                    for item in value or []:
                        if isinstance(item, str) and not is_placeholder(item):
                            skipped["extra_arg"] += 1
                            counted.append(item)
                        else:
                            counted.append(rewrite_node(item, namespace))
                    new_props[key] = counted
                    continue
                new_props[key] = rewrite_value(
                    value, namespace, type_name, key, in_array=False
                )
            return {**node, "props": new_props}
        if node.get("type") == "call":
            args = [rewrite_node(item, namespace) for item in node.get("args") or []]
            return {**node, "args": args}
        return node

    # Traverse in canonical statement order so ordinal assignment (and any
    # collision bumping) is invariant across alpha-equivalent inputs.
    rewritten = {
        name: rewrite_node(bindings[name], canonical_names.get(name, "root"))
        for name in order
    }
    emitted = emit_statement_bindings(rewritten, dsl=dsl)
    return TemplatizeResult(
        source=emitted,
        placeholders=tuple(extract_placeholders(emitted)),
        replacements=replacements,
        skipped=skipped,
    )


__all__ = ["TemplatizeResult", "templatize"]
