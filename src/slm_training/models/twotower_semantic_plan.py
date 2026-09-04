"""The semantic plan a decode must satisfy.

One responsibility: turning family counts and placeholders into the role
obligations a plan carries, and whether a seed is still active for it. Built on
``twotower_semantic_roles`` for the per-family capacity questions.

Extracted from ``TwoTowerModel``. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from slm_training.models.twotower_schema import (
    schema_descendant_families,
)
from slm_training.models.twotower_semantic_roles import (
    semantic_role_enum_candidates,
    semantic_role_family_has_capacity,
    semantic_role_joint_candidates,
    semantic_role_reachable_family_has_capacity,
)


def semantic_plan_role_obligations(
    family_counts: Counter[str],
    role_candidates: dict[str, tuple[str, ...]] | None,
    reachable_role_candidates: dict[str, tuple[str, ...]] | None = None,
) -> tuple[Counter[str], dict[str, tuple[str, ...]]]:
    """Pair uncovered visible roles with preferred compatible leaf families."""
    if not role_candidates:
        return family_counts, {}
    from slm_training.data.house_style.policy import DEFAULT_HOUSE_STYLE

    completed = family_counts.copy()
    planned = set(family_counts)
    schema_descendants = schema_descendant_families(planned)
    bindings: dict[str, list[str]] = defaultdict(list)
    coverage: dict[str, list[str]] = defaultdict(list)
    bound_slots: set[str] = set()
    component_names = sorted(
        {
            component
            for candidates in role_candidates.values()
            for component in candidates
        }
    )

    def covered_by_planned(slots: tuple[str, ...]) -> bool:
        trial = {family: list(assigned) for family, assigned in coverage.items()}

        def match(index: int) -> bool:
            if index == len(slots):
                return True
            slot = slots[index]
            compatible = set(role_candidates[slot]) | set(
                (reachable_role_candidates or {}).get(slot, ())
            )
            for family in sorted(planned.intersection(compatible)):
                assigned = trial.setdefault(family, [])
                assigned_slots = (*assigned, *bindings.get(family, ()), slot)
                has_capacity = (
                    semantic_role_family_has_capacity(
                        family, assigned_slots, completed[family]
                    )
                    if family in role_candidates[slot]
                    else semantic_role_reachable_family_has_capacity(
                        family, assigned_slots, completed[family]
                    )
                )
                if not has_capacity:
                    continue
                assigned.append(slot)
                if match(index + 1):
                    return True
                assigned.pop()
            return False

        if not match(0):
            return False
        coverage.update(trial)
        return True

    for slots, candidates in semantic_role_joint_candidates(
        list(role_candidates), component_names
    ).items():
        if covered_by_planned(slots):
            continue
        candidates = tuple(
            family
            for family in candidates
            if all(family in role_candidates[slot] for slot in slots)
        )
        if not candidates:
            continue
        family = next(
            (
                preferred
                for preferred in DEFAULT_HOUSE_STYLE.preferred_components
                if preferred in planned and preferred in candidates
            ),
            next(iter(sorted(planned.intersection(candidates))), None),
        )
        if family is None:
            family = next(iter(candidates))
            completed[family] += 1
            planned.add(family)
        bindings[family].extend(slots)
        bound_slots.update(slots)
    for slot, candidates in role_candidates.items():
        if slot in bound_slots:
            continue
        planned_family = next(
            (
                preferred
                for preferred in DEFAULT_HOUSE_STYLE.preferred_components
                if preferred in planned
                and preferred in candidates
                and semantic_role_family_has_capacity(
                    preferred,
                    (
                        *coverage.get(preferred, ()),
                        *bindings.get(preferred, ()),
                        slot,
                    ),
                    completed[preferred],
                )
            ),
            next(
                (
                    family
                    for family in sorted(planned.intersection(candidates))
                    if semantic_role_family_has_capacity(
                        family,
                        (
                            *coverage.get(family, ()),
                            *bindings.get(family, ()),
                            slot,
                        ),
                        completed[family],
                    )
                ),
                None,
            ),
        )
        if planned_family is not None:
            bindings[planned_family].append(slot)
            continue
        reachable_family = next(
            (
                family
                for family in sorted(
                    planned.intersection(
                        (reachable_role_candidates or {}).get(slot, ())
                    )
                )
                if semantic_role_reachable_family_has_capacity(
                    family,
                    (
                        *coverage.get(family, ()),
                        *bindings.get(family, ()),
                        slot,
                    ),
                    completed[family],
                )
            ),
            None,
        )
        if reachable_family is not None:
            family = next(
                (
                    preferred
                    for preferred in DEFAULT_HOUSE_STYLE.preferred_components
                    if preferred in candidates
                ),
                None,
            )
            if family is not None:
                added_direct = False
                exhausted_direct = any(
                    direct in planned
                    and not semantic_role_family_has_capacity(
                        direct,
                        (
                            *coverage.get(direct, ()),
                            *bindings.get(direct, ()),
                            slot,
                        ),
                        completed[direct],
                    )
                    for direct in candidates
                )
                if exhausted_direct and family not in schema_descendants:
                    completed[family] += 1
                    planned.add(family)
                    added_direct = True
                coverage[reachable_family].append(slot)
                if added_direct:
                    bindings[family].append(slot)
                continue
        nested_candidates = tuple(
            family for family in candidates if family in schema_descendants
        )
        if len(nested_candidates) == 1:
            family = nested_candidates[0]
            completed[family] += 1
            bindings[family].append(slot)
            continue
        family = next(
            (
                preferred
                for preferred in DEFAULT_HOUSE_STYLE.preferred_components
                if preferred in candidates
            ),
            None,
        )
        enum_candidates = semantic_role_enum_candidates(slot, candidates)
        if family is None and len(enum_candidates) == 1:
            family = enum_candidates[0]
        if family is None and len(candidates) == 1:
            family = candidates[0]
        if family is not None:
            completed[family] += 1
            bindings[family].append(slot)
    return completed, {family: tuple(slots) for family, slots in bindings.items()}


def semantic_plan_seed_active(
    state: Any | None,
    candidate_kinds: tuple[str, ...],
) -> bool:
    return bool(
        state is not None
        and not tuple(getattr(state, "section_types", ()))
        and any(
            kind in {"component_root", "component_root_or_bound"}
            for kind in candidate_kinds
        )
    )
