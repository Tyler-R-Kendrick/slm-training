"""Canonicalize SemanticPlanV1 for stable fingerprints and equivalence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from slm_training.data.progspec.semantic_plan import (
    PlanBinding,
    PlanSymbol,
    RoleSlot,
    SemanticPlanV1,
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_plan(plan: SemanticPlanV1) -> SemanticPlanV1:
    """Return a plan with normalized role/symbol IDs for equivalence testing.

    Normalization makes the plan invariant to alpha-renaming and sibling
    permutation where semantics are unchanged. The exact fingerprint of the
    returned plan is the canonical fingerprint.
    """
    if plan.compile_to_baseline():
        return plan

    # Build a content-based renaming for symbols and roles.
    symbol_renames = _symbol_renames(plan)
    role_renames = _role_renames(plan)

    def rename_role(role_id: str) -> str:
        return role_renames.get(role_id, role_id)

    def rename_symbol(symbol_id: str) -> str:
        return symbol_renames.get(symbol_id, symbol_id)

    new_role_slots = tuple(
        RoleSlot(
            role_id=rename_role(slot.role_id),
            component_family=slot.component_family,
            candidate_distribution=slot.candidate_distribution,
            min_cardinality=slot.min_cardinality,
            max_cardinality=slot.max_cardinality,
            required=slot.required,
            evidence_spans=slot.evidence_spans,
        )
        for slot in plan.role_slots
    )

    new_symbols = tuple(
        PlanSymbol(
            symbol_id=rename_symbol(sym.symbol_id),
            semantic_role=sym.semantic_role,
            allowed_pointer_targets=sym.allowed_pointer_targets,
        )
        for sym in plan.symbols
    )

    new_bindings: list[PlanBinding] = []
    for binding in plan.bindings:
        new_bindings.append(
            PlanBinding(
                role_slot_id=rename_role(binding.role_slot_id),
                candidate_symbols=tuple(
                    rename_symbol(s) for s in (binding.candidate_symbols or ())
                )
                or None,
                placeholder_fallback=binding.placeholder_fallback,
            )
        )

    topology = plan.topology
    new_edges: tuple[dict[str, Any], ...] | None = None
    if topology.parent_relation_candidates:
        edges = []
        for edge in topology.parent_relation_candidates:
            edges.append(
                {
                    "parent_role_id": rename_role(str(edge.get("parent_role_id") or "")),
                    "child_role_id": rename_role(str(edge.get("child_role_id") or "")),
                    "relation": str(edge.get("relation") or "contains"),
                }
            )
        # Sort to make order-independent.
        new_edges = tuple(sorted(edges, key=_stable_json))

    new_siblings: tuple[tuple[str, ...], ...] | None = None
    if topology.sibling_order_groups:
        groups = []
        for group in topology.sibling_order_groups:
            groups.append(tuple(sorted(rename_role(r) for r in group)))
        new_siblings = tuple(sorted(groups))

    new_topology = plan.topology.model_copy(
        update={
            "parent_relation_candidates": new_edges,
            "sibling_order_groups": new_siblings,
        }
    )

    return plan.model_copy(
        update={
            "role_slots": new_role_slots,
            "symbols": new_symbols,
            "bindings": tuple(new_bindings),
            "topology": new_topology,
        }
    )


def plan_factor_fingerprints(plan: SemanticPlanV1) -> dict[str, str]:
    """SHA-256 fingerprints for the whole plan and each factor."""
    canonical = canonicalize_plan(plan)
    factors = {
        "exact": canonical.model_dump(mode="json"),
        "archetype": canonical.archetype.model_dump(mode="json"),
        "role_set": tuple(sorted(slot.role_id for slot in canonical.role_slots)),
        "topology": canonical.topology.model_dump(mode="json"),
        "bindings": tuple(
            _stable_json(
                {
                    "role_slot_id": binding.role_slot_id,
                    "candidate_symbols": binding.candidate_symbols,
                }
            )
            for binding in canonical.bindings
        ),
    }
    return {name: _sha256(_stable_json(value)) for name, value in factors.items()}


def _symbol_renames(plan: SemanticPlanV1) -> dict[str, str]:
    """Map each symbol to a canonical index ordered by semantic content."""
    symbols = sorted(
        plan.symbols,
        key=lambda sym: (sym.semantic_role or "", sym.symbol_id),
    )
    return {sym.symbol_id: f"sym_{index:04d}" for index, sym in enumerate(symbols)}


def _role_renames(plan: SemanticPlanV1) -> dict[str, str]:
    """Map each role to a canonical index ordered by (component_family, path-like id).

    This is intentionally simple: two plans with the same component structure but
    different statementIds receive the same normalized role IDs.
    """
    slots = sorted(
        plan.role_slots,
        key=lambda slot: (slot.component_family or "", slot.role_id),
    )
    return {slot.role_id: f"role_{index:04d}" for index, slot in enumerate(slots)}
