"""Construct valid OpenUI seeds from SemanticPlanV1 using pack-owned constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.dsl.pack import DslPack
from slm_training.dsl.parser import validate


@dataclass(frozen=True)
class SeedResult:
    seed: str | None
    ok: bool
    reason: str | None


class PlanSeedBuilder:
    """Build a valid OpenUI seed from a semantic plan.

    Only certified/authored plan facts may hard-restrict candidate membership.
    Predicted/gold semantic preferences remain soft: the builder uses them as
    hints but falls back to the pack canonicalizer for validity.
    """

    def __init__(self, pack: DslPack) -> None:
        self.pack = pack

    def build(self, plan: SemanticPlanV1) -> SeedResult:
        has_actionable = (
            plan.role_slots
            or plan.topology.parent_relation_candidates is not None
            or plan.symbols
        )
        if not has_actionable:
            return SeedResult(seed=None, ok=True, reason="baseline: no actionable plan")

        roles: dict[str, dict[str, Any]] = {
            slot.role_id: {
                "component_family": slot.component_family or "Stack",
                "content": None,
            }
            for slot in plan.role_slots
        }

        # Surface syntax uses request-local ordinals only.  Semantic roles remain
        # typed authority for choosing the component property; neither the caller's
        # symbol spelling nor the role name becomes a template-marker identity.
        symbol_slot = {
            sym.symbol_id: f"slot_{index}" for index, sym in enumerate(plan.symbols)
        }

        # Apply content symbols to matching role slots.
        for binding in plan.bindings:
            candidates = binding.candidate_symbols or ()
            if candidates and binding.role_slot_id in roles:
                if candidates[0] not in symbol_slot:
                    return SeedResult(
                        seed=None,
                        ok=False,
                        reason=f"unknown content symbol {candidates[0]}",
                    )
                roles[binding.role_slot_id]["content"] = candidates[0]

        # Identify root role.
        child_ids = set()
        if plan.topology.parent_relation_candidates:
            for edge in plan.topology.parent_relation_candidates:
                child = str(edge.get("child_role_id") or "")
                if child:
                    child_ids.add(child)
        root_roles = [rid for rid in roles if rid not in child_ids]
        if len(root_roles) != 1:
            return SeedResult(
                seed=None,
                ok=False,
                reason=f"expected exactly one root role, got {len(root_roles)}",
            )
        root_role_id = root_roles[0]

        children_map: dict[str, list[str]] = {}
        if plan.topology.parent_relation_candidates:
            for edge in plan.topology.parent_relation_candidates:
                parent = str(edge.get("parent_role_id") or "")
                child = str(edge.get("child_role_id") or "")
                if parent and child:
                    children_map.setdefault(parent, []).append(child)

        # Render each role to a statement id and an expression.
        rendered_expr: dict[str, str] = {}
        statement_id_for: dict[str, str] = {}

        def render(role_id: str) -> str:
            if role_id in rendered_expr:
                return rendered_expr[role_id]
            info = roles.get(role_id)
            if info is None:
                raise ValueError(f"unknown role {role_id}")
            family = info["component_family"]
            children = children_map.get(role_id, [])

            args: list[str] = []
            if children:
                for child_role in children:
                    child_stmt = statement_id_for.get(child_role)
                    if child_stmt is None:
                        render(child_role)
                        child_stmt = statement_id_for[child_role]
                    args.append(child_stmt)
            content_symbol = info["content"]
            if content_symbol is not None:
                slot = symbol_slot.get(content_symbol)
                if slot is None:
                    raise ValueError(f"unknown content symbol {content_symbol}")
                args.append(f'":{slot}"')

            if args:
                expr = (
                    f"{family}([{', '.join(args)}])"
                    if children
                    else f"{family}({', '.join(args)})"
                )
            else:
                expr = f"{family}()"
            rendered_expr[role_id] = expr
            return expr

        # Topological order: children before parents.
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(role_id: str) -> None:
            if role_id in visited:
                return
            visited.add(role_id)
            for child in children_map.get(role_id, []):
                visit(child)
            stmt_id = f"node_{len(ordered)}"
            statement_id_for[role_id] = stmt_id
            render(role_id)
            ordered.append(role_id)

        visit(root_role_id)

        statements: list[str] = []
        for role_id in ordered:
            stmt_id = statement_id_for[role_id]
            expr = rendered_expr[role_id]
            statements.append(f"{stmt_id} = {expr}")

        # Rename the final statement to root.
        if statements:
            last_eq = statements[-1].index(" = ")
            statements[-1] = "root" + statements[-1][last_eq:]

        seed_text = "\n".join(statements)

        try:
            validate(seed_text)
        except Exception as exc:  # noqa: BLE001
            return SeedResult(seed=None, ok=False, reason=f"invalid seed: {exc}")

        return SeedResult(seed=seed_text, ok=True, reason=None)
