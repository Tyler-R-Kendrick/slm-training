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

Only user-facing content properties are rewritten. Open identifiers, actions,
extra arguments, and other non-content strings remain visible so the caller's
symbol-only output check rejects them instead of disguising them as slots.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from typing import Any

from slm_training.dsl.language_contract import (
    STRUCTURAL_ID_ATOMS,
    grammar_string_literals,
)
from slm_training.dsl.lang_core import library_schema
from slm_training.dsl.placeholders import (
    STRUCTURAL_ID_PROPS,
    TEMPLATIZABLE_PROPS,
    extract_placeholders,
    is_placeholder,
)
from slm_training.dsl.production_codec import (
    emit_statement_bindings,
    parse_statement_bindings,
    statement_binding_order,
)

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
    used_structural_ids = {
        value for value in STRUCTURAL_ID_ATOMS if json.dumps(value) in source
    }
    structural_id_ordinal = 0
    ordinals: dict[tuple[str, str], int] = {}
    replacements: dict[str, str] = {}
    skipped = {
        "enum_value": 0,
        "structural_literal": 0,
        "enum_like": 0,
        "array_string": 0,
        "extra_arg": 0,
        "non_content_string": 0,
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

    def next_structural_id() -> str:
        nonlocal structural_id_ordinal
        while True:
            value = f"${structural_id_ordinal}"
            structural_id_ordinal += 1
            if value not in used_structural_ids:
                used_structural_ids.add(value)
                return value

    def rewrite_value(
        value: Any, namespace: str, component: str, prop: str, in_array: bool
    ) -> Any:
        if isinstance(value, str):
            if is_placeholder(value):
                if prop not in TEMPLATIZABLE_PROPS:
                    if prop in STRUCTURAL_ID_PROPS:
                        skipped["non_content_string"] += 1
                        return next_structural_id()
                    raise ValueError(
                        f"placeholder {value!r} is not allowed in non-content "
                        f"property {component}.{prop}"
                    )
                return value
            if value in _prop_enum(defs, component, prop):
                skipped["enum_value"] += 1
                return value
            if value in STRUCTURAL_ID_ATOMS and prop in STRUCTURAL_ID_PROPS:
                skipped["structural_literal"] += 1
                return value
            if prop not in TEMPLATIZABLE_PROPS:
                if prop in STRUCTURAL_ID_PROPS:
                    skipped["non_content_string"] += 1
                    return next_structural_id()
                raise ValueError(
                    f"open string {value!r} is not allowed in non-content "
                    f"property {component}.{prop}"
                )
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
                    for index, item in enumerate(value or []):
                        counted.append(
                            rewrite_value(
                                item, namespace, type_name, f"arg_{index + 1}", False
                            )
                        )
                    new_props[key] = counted
                    continue
                new_props[key] = rewrite_value(
                    value, namespace, type_name, key, in_array=False
                )
            return {**node, "props": new_props}
        if node.get("type") == "call":
            args = [
                rewrite_value(item, namespace, "Call", f"arg_{index + 1}", False)
                for index, item in enumerate(node.get("args") or [])
            ]
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


_QUOTED_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def templatize_fragment(
    source: str, *, output_kind: str | None = None
) -> TemplatizeResult:
    """Templatize free-form strings in a non-document output surface."""
    if output_kind == "document":
        return templatize(source)
    if output_kind in {"expression", "statement"}:
        prefix = ""
        expression = source.strip()
        if output_kind == "statement":
            match = re.fullmatch(r"\s*([a-z][A-Za-z0-9_]*)\s*=\s*(.+)\s*", source, re.S)
            if not match:
                raise ValueError("statement target must be one binding")
            prefix = f"{match.group(1)} = "
            expression = match.group(2)
        result = templatize(f"root = {expression}")
        root_line, *tail = result.source.splitlines()
        if not root_line.startswith("root = "):
            raise ValueError("templatized fragment lost root expression")
        output = prefix + root_line.removeprefix("root = ")
        if tail:
            output = "\n".join([output, *tail])
        return TemplatizeResult(
            source=output,
            placeholders=tuple(extract_placeholders(output)),
            replacements=result.replacements,
            skipped=result.skipped,
        )
    allowed = grammar_string_literals()
    used_tokens = set(extract_placeholders(source))
    replacements: dict[str, str] = {}
    ordinal = 0

    def replace_literal(match: re.Match[str]) -> str:
        nonlocal ordinal
        value = ast.literal_eval(match.group(0))
        if not isinstance(value, str) or is_placeholder(value) or value in allowed:
            return match.group(0)
        ordinal += 1
        token = f":fragment.string_{ordinal}"
        while token in used_tokens:
            ordinal += 1
            token = f":fragment.string_{ordinal}"
        used_tokens.add(token)
        replacements[token] = value
        return json.dumps(token)

    output = _QUOTED_LITERAL_RE.sub(replace_literal, source)
    return TemplatizeResult(
        source=output,
        placeholders=tuple(extract_placeholders(output)),
        replacements=replacements,
        skipped={
            "enum_value": 0,
            "structural_literal": 0,
            "enum_like": 0,
            "array_string": 0,
            "extra_arg": 0,
            "non_content_string": 0,
        },
    )


def role_contract_violations(
    source: str, *, output_kind: str = "document"
) -> tuple[str, ...]:
    """Return strings placed in a schema role they are not allowed to occupy."""
    if output_kind == "expression":
        source = f"root = {source}"
    elif output_kind == "statement":
        match = re.fullmatch(r"\s*[a-z][A-Za-z0-9_]*\s*=\s*(.+)\s*", source, re.S)
        if not match:
            return ("statement target must be one binding",)
        source = f"root = {match.group(1)}"
    elif output_kind != "document":
        return ()

    try:
        bindings = parse_statement_bindings(source, validate=False)
    except Exception as exc:  # noqa: BLE001 - caller reports contract failure
        return (f"role-contract parse failed: {exc}",)

    defs = dict(library_schema().get("$defs") or {})
    violations: list[str] = []

    def check_prop(component: str, prop: str, child: Any) -> None:
        if isinstance(child, list):
            for item in child:
                check_prop(component, prop, item)
            return
        if not isinstance(child, str):
            walk(child)
            return
        enum_values = _prop_enum(defs, component, prop)
        if is_placeholder(child):
            if prop not in TEMPLATIZABLE_PROPS:
                violations.append(
                    f"placeholder {child!r} in non-content property "
                    f"{component}.{prop}"
                )
        elif child in STRUCTURAL_ID_ATOMS:
            if prop not in STRUCTURAL_ID_PROPS:
                violations.append(
                    f"structural id {child!r} in content property "
                    f"{component}.{prop}"
                )
        elif child not in enum_values:
            violations.append(f"open string {child!r} in property {component}.{prop}")

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if not isinstance(value, dict):
            return
        if value.get("type") == "element":
            component = str(value.get("typeName") or "")
            for prop, child in (value.get("props") or {}).items():
                check_prop(component, prop, child)
        elif value.get("type") == "call":
            walk(value.get("args") or [])

    for binding in bindings.values():
        walk(binding)
    return tuple(dict.fromkeys(violations))


def assert_role_safe_output(source: str, *, output_kind: str = "document") -> None:
    violations = role_contract_violations(source, output_kind=output_kind)
    if violations:
        raise ValueError("; ".join(violations[:3]))


__all__ = [
    "TemplatizeResult",
    "assert_role_safe_output",
    "role_contract_violations",
    "templatize",
    "templatize_fragment",
]
