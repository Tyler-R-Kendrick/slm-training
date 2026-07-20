"""Plan-level semantic corruption transforms for SPV2-01.

Every transform mutates a pack-neutral :class:`SemanticPlanV1` and is compiled
back through :class:`PlanSeedBuilder` so that the surface OpenUI stays parser /
schema valid.  The corruption is semantic: the program is well-formed but no
longer matches the prompt contract encoded by the positive record.
"""

from __future__ import annotations

import copy
from typing import Any

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_contrast.schema import ContrastFamily, ContrastSeverity


class TransformCandidate:
    """One corruption hypothesis produced from a source plan."""

    def __init__(
        self,
        transform_id: str,
        family: ContrastFamily,
        severity: ContrastSeverity,
        description: str,
        plan: SemanticPlanV1,
    ) -> None:
        self.transform_id = transform_id
        self.family = family
        self.severity = severity
        self.description = description
        self.plan = plan

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "family": self.family.value,
            "severity": self.severity.value,
            "description": self.description,
            "plan": self.plan.to_dict(),
        }


def _clone_plan(plan: SemanticPlanV1) -> dict[str, Any]:
    return copy.deepcopy(plan.to_dict())


def _rebuild_plan(payload: dict[str, Any]) -> SemanticPlanV1:
    return SemanticPlanV1.from_dict(payload)


def _role_slots(payload: dict[str, Any]) -> list[dict[str, Any]]:
    slots = payload.setdefault("role_slots", [])
    if not isinstance(slots, list):
        payload["role_slots"] = list(slots)
    return payload["role_slots"]


def _symbols(payload: dict[str, Any]) -> list[dict[str, Any]]:
    symbols = payload.setdefault("symbols", [])
    if not isinstance(symbols, list):
        payload["symbols"] = list(symbols)
    return payload["symbols"]


def _bindings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    bindings = payload.setdefault("bindings", [])
    if not isinstance(bindings, list):
        payload["bindings"] = list(bindings)
    return payload["bindings"]


def _topology(payload: dict[str, Any]) -> dict[str, Any]:
    topo = payload.setdefault("topology", {})
    if not isinstance(topo, dict):
        payload["topology"] = {}
    return payload["topology"]


def _edges(payload: dict[str, Any]) -> list[dict[str, Any]]:
    topo = _topology(payload)
    edges = topo.setdefault("parent_relation_candidates", [])
    if not isinstance(edges, list):
        topo["parent_relation_candidates"] = list(edges)
    return topo["parent_relation_candidates"]


def _content_families() -> tuple[str, ...]:
    return ("TextContent", "Button", "Label", "Title")


def _content_component_roles(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        slot
        for slot in slots
        if str(slot.get("component_family") or "") in _content_families()
    ]


def _new_symbol_id(symbols: list[dict[str, Any]]) -> str:
    idx = len(symbols)
    return f"sym_{idx:04d}"


def _transform_positive_control(plan: SemanticPlanV1) -> TransformCandidate:
    return TransformCandidate(
        transform_id="positive_control_identity",
        family=ContrastFamily.POSITIVE,
        severity=ContrastSeverity.BENIGN,
        description="Original plan compiled unchanged as a positive control.",
        plan=plan,
    )


def _transform_content_swap_family(plan: SemanticPlanV1) -> TransformCandidate | None:
    payload = _clone_plan(plan)
    slots = _role_slots(payload)
    candidates = _content_component_roles(slots)
    if not candidates:
        return None
    target = candidates[0]
    old_family = str(target.get("component_family") or "")
    alternatives = [f for f in _content_families() if f != old_family]
    if not alternatives:
        return None
    target["component_family"] = alternatives[0]
    return TransformCandidate(
        transform_id="content_swap_family",
        family=ContrastFamily.CONTENT,
        severity=ContrastSeverity.MODERATE,
        description=(
            f"Changed role slot component family from {old_family!r} to "
            f"{alternatives[0]!r} while preserving the original content binding."
        ),
        plan=_rebuild_plan(payload),
    )


def _transform_content_invert_role(plan: SemanticPlanV1) -> TransformCandidate | None:
    payload = _clone_plan(plan)
    symbols = _symbols(payload)
    if not symbols:
        return None
    target = symbols[0]
    old_role = str(target.get("semantic_role") or "text")
    inverted = "icon" if old_role != "icon" else "value"
    target["semantic_role"] = inverted
    return TransformCandidate(
        transform_id="content_invert_role",
        family=ContrastFamily.CONTENT,
        severity=ContrastSeverity.MODERATE,
        description=(
            f"Changed symbol semantic role from {old_role!r} to {inverted!r} "
            "so the compiled content prop no longer matches the component schema."
        ),
        plan=_rebuild_plan(payload),
    )


def _transform_topology_delete_leaf(plan: SemanticPlanV1) -> TransformCandidate | None:
    """Remove a leaf content role and its edge, changing the visible topology."""
    payload = _clone_plan(plan)
    slots = _role_slots(payload)
    edges = _edges(payload)
    bindings = _bindings(payload)
    symbols = _symbols(payload)
    if len(slots) < 3 or len(edges) < 2:
        return None
    parent_ids = {str(e.get("parent_role_id") or "") for e in edges}
    leaves = [
        slot
        for slot in slots
        if str(slot.get("role_id") or "") not in parent_ids
        and str(slot.get("component_family") or "") in _content_families()
    ]
    if not leaves:
        return None
    target = leaves[0]
    target_role_id = str(target["role_id"])
    # Remove the slot, its parent edge, and any binding unique to it.
    slots[:] = [s for s in slots if str(s.get("role_id") or "") != target_role_id]
    edges[:] = [e for e in edges if str(e.get("child_role_id") or "") != target_role_id]
    dropped_binding = next(
        (b for b in bindings if str(b.get("role_slot_id") or "") == target_role_id), None
    )
    if dropped_binding is not None:
        bindings.remove(dropped_binding)
        dropped_syms = set(dropped_binding.get("candidate_symbols") or ())
        # Drop symbols that are no longer referenced by any binding.
        used_symbols = {
            sym
            for b in bindings
            for sym in (b.get("candidate_symbols") or ())
        }
        symbols[:] = [s for s in symbols if str(s.get("symbol_id") or "") not in (dropped_syms - used_symbols)]
    return TransformCandidate(
        transform_id="topology_delete_leaf",
        family=ContrastFamily.TOPOLOGY,
        severity=ContrastSeverity.SEVERE,
        description=(
            f"Deleted leaf role {target_role_id!r} and its binding, "
            "removing its component and placeholder from the rendered program."
        ),
        plan=_rebuild_plan(payload),
    )


def _transform_topology_reparent(plan: SemanticPlanV1) -> TransformCandidate | None:
    payload = _clone_plan(plan)
    edges = _edges(payload)
    slots = _role_slots(payload)
    if len(edges) < 2 or len(slots) < 3:
        return None
    first = edges[0]
    second = edges[1]
    child = str(first.get("child_role_id") or "")
    new_parent = str(second.get("parent_role_id") or "")
    old_parent = str(first.get("parent_role_id") or "")
    if child == new_parent or old_parent == new_parent:
        return None
    descendants: set[str] = set()

    def collect(parent: str) -> None:
        for edge in edges:
            if str(edge.get("parent_role_id") or "") == parent:
                c = str(edge.get("child_role_id") or "")
                if c not in descendants:
                    descendants.add(c)
                    collect(c)

    collect(child)
    if new_parent in descendants:
        return None
    first["parent_role_id"] = new_parent
    return TransformCandidate(
        transform_id="topology_reparent",
        family=ContrastFamily.TOPOLOGY,
        severity=ContrastSeverity.SEVERE,
        description=(
            f"Re-parented role {child!r} from {old_parent!r} to {new_parent!r}, "
            "changing the semantic nesting of the program."
        ),
        plan=_rebuild_plan(payload),
    )


def _transform_binding_swap_symbol(plan: SemanticPlanV1) -> TransformCandidate | None:
    payload = _clone_plan(plan)
    bindings = _bindings(payload)
    symbols = _symbols(payload)
    if len(bindings) < 1 or len(symbols) < 2:
        return None
    binding = bindings[0]
    candidates = [str(s.get("symbol_id") or "") for s in symbols]
    current = list(binding.get("candidate_symbols") or ())
    swapped = candidates[1] if not current or current[0] != candidates[1] else candidates[0]
    binding["candidate_symbols"] = (swapped,)
    return TransformCandidate(
        transform_id="binding_swap_symbol",
        family=ContrastFamily.BINDING,
        severity=ContrastSeverity.SEVERE,
        description=(
            f"Replaced binding candidate symbols with {swapped!r}, "
            "binding the role to semantically mismatched content."
        ),
        plan=_rebuild_plan(payload),
    )


def _transform_binding_introduce_incompatible_symbol(
    plan: SemanticPlanV1,
) -> TransformCandidate | None:
    """Bind a role to a freshly introduced placeholder of an incompatible role."""
    payload = _clone_plan(plan)
    bindings = _bindings(payload)
    symbols = _symbols(payload)
    if not bindings:
        return None
    binding = bindings[0]
    new_sym_id = _new_symbol_id(symbols)
    symbols.append(
        {
            "symbol_id": new_sym_id,
            "semantic_role": "icon",
            "allowed_pointer_targets": None,
        }
    )
    binding["candidate_symbols"] = (new_sym_id,)
    return TransformCandidate(
        transform_id="binding_introduce_incompatible_symbol",
        family=ContrastFamily.BINDING,
        severity=ContrastSeverity.SEVERE,
        description=(
            f"Introduced placeholder {new_sym_id!r} with semantic role 'icon' "
            "and rebound a content role to it, breaking the prompt slot contract."
        ),
        plan=_rebuild_plan(payload),
    )


def _transform_contract_unresolve(plan: SemanticPlanV1) -> TransformCandidate | None:
    """Mark a requirement unresolved and remove the role that satisfied it."""
    payload = _clone_plan(plan)
    coverage = payload.setdefault("coverage", {})
    if not isinstance(coverage, dict):
        payload["coverage"] = {}
        coverage = payload["coverage"]
    accounted = list(coverage.get("named_requirements_accounted_for") or ())
    if not accounted:
        accounted = ["semantic_content"]
    target_req = accounted[0]
    coverage["named_requirements_accounted_for"] = accounted[1:] if len(accounted) > 1 else []
    coverage["unresolved_requirements"] = [target_req]

    slots = _role_slots(payload)
    bindings = _bindings(payload)
    symbols = _symbols(payload)
    edges = _edges(payload)
    # Find a symbol whose semantic role matches the unresolved requirement.
    target_sym = next(
        (s for s in symbols if str(s.get("semantic_role") or "") == target_req), None
    )
    if target_sym is None:
        return TransformCandidate(
            transform_id="contract_unresolve",
            family=ContrastFamily.CONTRACT,
            severity=ContrastSeverity.MODERATE,
            description=(
                f"Marked requirement {target_req!r} as unresolved, "
                "even though the surface program still contains its content."
            ),
            plan=_rebuild_plan(payload),
        )
    target_sym_id = str(target_sym["symbol_id"])
    # Remove the binding that uses this symbol and its role.
    target_binding = next(
        (b for b in bindings if target_sym_id in (b.get("candidate_symbols") or ())), None
    )
    if target_binding is not None:
        target_role_id = str(target_binding["role_slot_id"])
        slots[:] = [s for s in slots if str(s.get("role_id") or "") != target_role_id]
        edges[:] = [e for e in edges if str(e.get("child_role_id") or "") != target_role_id]
        bindings.remove(target_binding)
        used_symbols = {
            sym for b in bindings for sym in (b.get("candidate_symbols") or ())
        }
        symbols[:] = [s for s in symbols if str(s.get("symbol_id") or "") in used_symbols]
    return TransformCandidate(
        transform_id="contract_unresolve",
        family=ContrastFamily.CONTRACT,
        severity=ContrastSeverity.SEVERE,
        description=(
            f"Marked requirement {target_req!r} as unresolved and removed the "
            "role that satisfied it, so the program no longer covers the contract."
        ),
        plan=_rebuild_plan(payload),
    )


def _transform_contract_archetype_mismatch(
    plan: SemanticPlanV1,
) -> TransformCandidate | None:
    payload = _clone_plan(plan)
    archetype = payload.setdefault("archetype", {})
    if not isinstance(archetype, dict):
        payload["archetype"] = {}
        archetype = payload["archetype"]
    old_id = str(archetype.get("id") or "unknown")
    archetype["id"] = "mismatched_archetype"
    return TransformCandidate(
        transform_id="contract_archetype_mismatch",
        family=ContrastFamily.CONTRACT,
        severity=ContrastSeverity.BENIGN,
        description=(
            f"Changed archetype id from {old_id!r} to 'mismatched_archetype', "
            "contradicting the prompt's stated layout archetype."
        ),
        plan=_rebuild_plan(payload),
    )


TRANSFORM_ORDER = (
    _transform_positive_control,
    _transform_content_swap_family,
    _transform_content_invert_role,
    _transform_topology_delete_leaf,
    _transform_topology_reparent,
    _transform_binding_swap_symbol,
    _transform_binding_introduce_incompatible_symbol,
    _transform_contract_unresolve,
    _transform_contract_archetype_mismatch,
)


def generate_transforms(plan: SemanticPlanV1) -> tuple[TransformCandidate, ...]:
    """Return all corruption candidates that can be applied to *plan*."""
    results: list[TransformCandidate] = []
    for factory in TRANSFORM_ORDER:
        try:
            candidate = factory(plan)
        except Exception:  # noqa: BLE001 - corruption hypotheses are best-effort
            continue
        if candidate is not None:
            results.append(candidate)
    return tuple(results)


__all__ = ["TransformCandidate", "generate_transforms"]
