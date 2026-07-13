"""AST helpers shared across grammar backends."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from slm_training.dsl.placeholders import PLACEHOLDER_RE


def collect_placeholders_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in PLACEHOLDER_RE.finditer(text or ""):
        token = m.group(0)
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def collect_placeholders_from_ast(node: Any) -> list[str]:
    """Walk ElementNode-like trees and collect placeholder string values."""
    seen: set[str] = set()
    out: list[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            for m in PLACEHOLDER_RE.finditer(value):
                token = m.group(0)
                if token not in seen:
                    seen.add(token)
                    out.append(token)
            return
        if isinstance(value, dict):
            for v in value.values():
                visit(v)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return

    visit(node)
    return out


# Back-compat alias used by early drafts.
collect_placeholders = collect_placeholders_from_text


def _walk_type_names(node: Any, counts: Counter[str]) -> None:
    if node is None:
        return
    if isinstance(node, dict):
        name = node.get("typeName") or node.get("name")
        if isinstance(name, str) and name and name not in {"element", "ref", "token", "tree", "call"}:
            counts[name] += 1
        props = node.get("props")
        if isinstance(props, dict):
            for key, value in props.items():
                if key == "children":
                    continue
                _walk_type_names(value, counts)
            children = props.get("children")
            if isinstance(children, list):
                for child in children:
                    _walk_type_names(child, counts)
        elif isinstance(node.get("children"), list):
            for child in node["children"]:
                _walk_type_names(child, counts)
        root = node.get("root")
        if root is not None:
            _walk_type_names(root, counts)
        return
    if isinstance(node, list):
        for item in node:
            _walk_type_names(item, counts)


def component_multiset(ast: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    _walk_type_names(ast, counts)
    return dict(sorted(counts.items()))


def ast_fingerprint(ast: Any) -> str:
    payload = json.dumps(component_multiset(ast), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ast_summary(ast: Any) -> dict[str, Any]:
    multi = component_multiset(ast)
    return {
        "fingerprint": ast_fingerprint(ast),
        "components": multi,
        "n_nodes": sum(multi.values()),
        "root": next(iter(multi), None),
    }


def lark_tree_to_dict(tree: Any) -> Any:
    """Convert a Lark Tree/Token into a JSON-friendly nested dict."""
    from lark import Token, Tree

    if isinstance(tree, Token):
        return {"type": "token", "name": tree.type, "value": str(tree)}
    if isinstance(tree, Tree):
        return {
            "type": "tree",
            "name": tree.data,
            "children": [lark_tree_to_dict(c) for c in tree.children],
        }
    if isinstance(tree, list):
        return [lark_tree_to_dict(x) for x in tree]
    if isinstance(tree, dict):
        return {k: lark_tree_to_dict(v) for k, v in tree.items()}
    return tree


def _is_node_like(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") in {"element", "ref", "call"}


def map_positional_props(
    type_name: str,
    args: list[Any],
    prop_order: dict[str, list[str]] | None,
) -> dict[str, Any]:
    """Map positional call args onto named props using a schema prop-order table."""
    props: dict[str, Any] = {}
    remaining = list(args)
    order = list((prop_order or {}).get(type_name) or [])
    wants_children = (not order) or ("children" in order)

    # Explicit list children: Stack([a, b], "column")
    if remaining and isinstance(remaining[0], list) and wants_children:
        props["children"] = remaining.pop(0)
        order = [p for p in order if p != "children"]
    # Variadic children form: row(a, b) — used by toy-layout and similar DSLs.
    elif wants_children and remaining and _is_node_like(remaining[0]):
        children: list[Any] = []
        while remaining and _is_node_like(remaining[0]):
            children.append(remaining.pop(0))
        props["children"] = children
        order = [p for p in order if p != "children"]

    for prop in order:
        if not remaining:
            break
        if prop in props:
            continue
        props[prop] = remaining.pop(0)

    if remaining:
        props["_args"] = remaining
    return props
