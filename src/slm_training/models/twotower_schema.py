"""Library-schema traversal for the two-tower decoder.

One responsibility: answering structural questions about the OpenUI library
schema -- which families descend from a node, which are required, whether a
visible slot is reachable, and what enum values a slot admits. Pure schema
reasoning with no model state.

Extracted from ``TwoTowerModel``. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from typing import Any


def schema_descendant_families(families: set[str]) -> set[str]:
    """Return cycle-safe component refs nested below planned families."""
    from slm_training.dsl.lang_core import library_schema

    definitions = library_schema().get("$defs", {})
    descendants: set[str] = set()
    visited = set(families)
    pending = [definitions[family] for family in families if family in definitions]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
            continue
        if not isinstance(value, dict):
            continue
        reference = str(value.get("$ref") or "")
        if reference.startswith("#/$defs/"):
            family = reference.rsplit("/", 1)[-1]
            descendants.add(family)
            if family not in visited and family in definitions:
                visited.add(family)
                pending.append(definitions[family])
        pending.extend(value.values())
    return descendants


def schema_required_descendant_families(family: str) -> set[str]:
    """Return families reachable through required, non-alternative paths."""
    from slm_training.dsl.lang_core import library_schema

    definitions = library_schema().get("$defs", {})

    def visit(schema: object, seen: frozenset[str]) -> set[str]:
        if not isinstance(schema, dict):
            return set()
        reference = str(schema.get("$ref") or "")
        if reference.startswith("#/$defs/"):
            name = reference.rsplit("/", 1)[-1]
            if name in seen:
                return {name}
            return {name} | visit(definitions.get(name), seen | {name})
        alternatives = [
            option
            for key in ("anyOf", "oneOf")
            for option in schema.get(key, ())
            if isinstance(option, dict)
        ]
        if alternatives:
            common = visit(alternatives[0], seen)
            for option in alternatives[1:]:
                common.intersection_update(visit(option, seen))
            return common
        if schema.get("type") == "array":
            return visit(schema.get("items"), seen)
        required = set(schema.get("required", ()))
        return set().union(
            *(
                visit(child, seen)
                for name, child in schema.get("properties", {}).items()
                if name in required
            ),
            set(),
        )

    return visit(definitions.get(family), frozenset({family}))


def schema_has_opaque_required_collection(family: str) -> bool:
    """Whether a required collection intentionally accepts broad elements."""
    from slm_training.dsl.lang_core import library_schema

    definition = library_schema().get("$defs", {}).get(family, {})
    required = set(definition.get("required", ()))
    return any(
        isinstance(schema, dict)
        and schema.get("type") == "array"
        and not schema.get("items")
        for name, schema in definition.get("properties", {}).items()
        if name in required
    )


def schema_can_reach_visible_slot(
    schema: dict[str, Any], seen: frozenset[str] = frozenset()
) -> bool:
    reference = str(schema.get("$ref") or "")
    if reference.startswith("#/$defs/"):
        name = reference.rsplit("/", 1)[-1]
        if name in seen:
            return False
        from slm_training.dsl.lang_core import library_schema

        target = library_schema().get("$defs", {}).get(name)
        return isinstance(target, dict) and schema_can_reach_visible_slot(
            target, seen | {name}
        )
    if schema.get("x-openui-placeholder") or schema.get("type") == "string":
        return True
    return any(
        schema_can_reach_visible_slot(dict(child), seen)
        for child in (
            *schema.get("anyOf", ()),
            *schema.get("properties", {}).values(),
            *((schema["items"],) if isinstance(schema.get("items"), dict) else ()),
        )
        if isinstance(child, dict)
    )


def schema_contains_enum(schema: dict[str, Any]) -> bool:
    if schema.get("enum"):
        return True
    return any(
        schema_contains_enum(dict(option))
        for key in ("anyOf", "oneOf")
        for option in schema.get(key, ())
        if isinstance(option, dict)
    )


def schema_enum_values(schema: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = list(schema.get("enum", ()))
    for key in ("anyOf", "oneOf"):
        for option in schema.get(key, ()):
            if isinstance(option, dict):
                values.extend(schema_enum_values(dict(option)))
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)
