"""Semantic-role capacity and candidate selection.

One responsibility: given the library schema, which roles a family can still
absorb -- namespace groups, enum candidates, joint candidates across
placeholders, and whether a family (or one reachable from it) has capacity left.

Extracted from ``TwoTowerModel``. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import re
from typing import Any


def semantic_role_joint_candidates(
    placeholders: list[str], component_names: list[str]
) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Partition namespaces into jointly coverable direct-string role groups."""
    from itertools import combinations

    from slm_training.data.quality import semantic_role_properties
    from slm_training.dsl.lang_core import library_schema

    groups: dict[str, list[str]] = {}
    for placeholder in sorted(set(placeholders)):
        parts = placeholder.removeprefix(":").split(".")
        if len(parts) > 1:
            groups.setdefault(".".join(parts[:-1]), []).append(placeholder)
    definitions = library_schema().get("$defs", {})
    properties_by_slot = semantic_role_properties(placeholders)
    direct_string_properties = {
        name: {
            property_name
            for property_name, schema in (definition.get("properties") or {}).items()
            if isinstance(schema, dict) and schema.get("type") == "string"
        }
        for name in sorted(set(component_names))
        if isinstance((definition := definitions.get(name)), dict)
    }
    slot_candidate_counts = {
        slot: sum(
            bool(set(properties).intersection(component_properties))
            for component_properties in direct_string_properties.values()
        )
        for slot, properties in properties_by_slot.items()
    }

    def covers(slots: tuple[str, ...], properties: dict[str, Any]) -> bool:
        string_properties = {
            name
            for name, schema in properties.items()
            if isinstance(schema, dict) and schema.get("type") == "string"
        }

        def match(index: int, used: frozenset[str]) -> bool:
            if index == len(slots):
                return True
            return any(
                match(index + 1, used | {name})
                for name in properties_by_slot[slots[index]]
                if name in string_properties and name not in used
            )

        return match(0, frozenset())

    result: dict[tuple[str, ...], tuple[str, ...]] = {}
    for slots_list in groups.values():
        candidates: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for size in range(len(slots_list), 1, -1):
            for slots in combinations(slots_list, size):
                compatible = tuple(
                    name
                    for name in sorted(set(component_names))
                    if isinstance((definition := definitions.get(name)), dict)
                    and covers(slots, definition.get("properties") or {})
                )
                if compatible:
                    candidates.append((slots, compatible))
        used: set[str] = set()
        for slots, compatible in sorted(
            candidates,
            key=lambda item: (
                -len(item[0]),
                sum(slot_candidate_counts[slot] for slot in item[0]),
                len(item[1]),
                item[0],
            ),
        ):
            if used.intersection(slots):
                continue
            result[slots] = compatible
            used.update(slots)
    return result


def semantic_role_namespace_groups(
    slots: tuple[str, ...] | list[str],
) -> tuple[tuple[str, ...], ...]:
    """Group declared markers by their semantic owner namespace."""
    groups: dict[str, list[str]] = {}
    for slot in slots:
        namespace, separator, _property = slot.removeprefix(":").rpartition(".")
        groups.setdefault(namespace if separator else slot, []).append(slot)
    return tuple(tuple(group) for group in groups.values())


def semantic_role_enum_candidates(
    slot: str, candidates: tuple[str, ...]
) -> tuple[str, ...]:
    """Return candidate families whose public enum names the visible role."""
    from slm_training.dsl.lang_core import library_schema

    role = re.sub(r"\d+$", "", slot.removeprefix(":").split(".")[-1])
    definitions = library_schema().get("$defs", {})

    def enum_values(value: object) -> set[str]:
        if isinstance(value, dict):
            return {str(item).lower() for item in value.get("enum", ())} | set().union(
                *(enum_values(child) for child in value.values())
            )
        if isinstance(value, list):
            return set().union(*(enum_values(child) for child in value), set())
        return set()

    return tuple(
        family
        for family in candidates
        if role.lower() in enum_values(definitions.get(family, {}))
    )


def semantic_role_family_has_capacity(
    family: str, slots: tuple[str, ...], instances: int
) -> bool:
    """Return whether instances have distinct compatible public strings."""
    from slm_training.data.quality import semantic_role_properties
    from slm_training.dsl.lang_core import library_schema

    slots = tuple(dict.fromkeys(slots))
    definition = library_schema().get("$defs", {}).get(family, {})
    string_properties = {
        name
        for name, schema in (definition.get("properties") or {}).items()
        if isinstance(schema, dict) and schema.get("type") == "string"
    }
    if not string_properties:
        return len(semantic_role_namespace_groups(slots)) <= max(0, instances)
    properties_by_slot = semantic_role_properties(list(slots))
    if any(
        not string_properties.intersection(properties_by_slot.get(slot, ()))
        for slot in slots
    ):
        return True
    available = tuple(
        property_name
        for _instance in range(max(0, instances))
        for property_name in sorted(string_properties)
    )

    def match(index: int, used: frozenset[int]) -> bool:
        if index == len(slots):
            return True
        return any(
            match(index + 1, used | {position})
            for position, property_name in enumerate(available)
            if position not in used
            and property_name in properties_by_slot.get(slots[index], ())
        )

    return match(0, frozenset())


def semantic_role_reachable_family_has_capacity(
    family: str, slots: tuple[str, ...], instances: int
) -> bool:
    """Bound nested role namespaces to one structural owner per instance."""
    slots = tuple(dict.fromkeys(slots))
    return semantic_role_family_has_capacity(family, slots, instances) and len(
        semantic_role_namespace_groups(slots)
    ) <= max(0, instances)
