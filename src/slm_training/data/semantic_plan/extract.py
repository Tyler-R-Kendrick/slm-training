"""Extract SemanticPlanV1 from ProgramSpec/AST."""

from __future__ import annotations

from typing import Any, Protocol

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanConfidenceCalibration,
    PlanCoverage,
    PlanIdentity,
    PlanSymbol,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.dsl.pack import DslPack


class SemanticPlanExtractor(Protocol):
    """Pack-facing protocol for deriving a canonical semantic plan."""

    def extract(self, program_spec: ProgramSpec, pack: DslPack) -> SemanticPlanV1: ...


class OpenUISemanticPlanExtractor:
    """Deterministic gold plan extraction from an OpenUI typed AST."""

    # Components whose primary prop is a content placeholder.
    _CONTENT_PROPS = {"text", "label", "title", "hint", "placeholder"}
    # Container components whose children carry semantic meaning.
    _CONTAINER_TYPES = {"Stack", "Card", "List", "Grid", "Form", "Page"}

    def extract(self, program_spec: ProgramSpec, pack: DslPack) -> SemanticPlanV1:
        ast = program_spec.ast
        if not isinstance(ast, dict):
            raise ValueError("OpenUI ProgramSpec ast must be a dict")
        # ProgramSpec.ast is the root element dict for OpenUI.
        if ast.get("type") == "element" or ast.get("typeName"):
            root = ast
        else:
            root = ast.get("root")
        if not isinstance(root, dict):
            raise ValueError("OpenUI AST root must be an element dict")

        root_type = str(root.get("typeName") or "")
        root_props = dict(root.get("props") or {})
        direction = str(root_props.get("direction") or "") if "direction" in root_props else ""

        identity = PlanIdentity(
            pack_id=pack.pack_id,
            contract_hash=program_spec.contract_id,
            source_program_fingerprint=None,
            prompt_context_hash=None,
            provenance="gold",
        )

        archetype = PlanArchetype(
            id=self._archetype_id(root_type, direction),
            confidence=1.0,
        )

        role_slots: list[RoleSlot] = []
        symbols: list[PlanSymbol] = []
        bindings: list[PlanBinding] = []
        topology_edges: list[dict[str, Any]] = []
        sibling_groups: list[tuple[str, ...]] = []

        symbol_index: dict[str, int] = {}
        role_index: dict[str, int] = {}

        def get_symbol_id(raw: str) -> str:
            idx = symbol_index.setdefault(raw, len(symbol_index))
            return f"sym_{idx:04d}"

        def get_role_id(raw: str) -> str:
            idx = role_index.setdefault(raw, len(role_index))
            return f"role_{idx:04d}"

        def walk(node: dict[str, Any], parent_role_id: str | None, path: tuple[int, ...]) -> str:
            if not isinstance(node, dict):
                raise ValueError("AST node must be an element dict")
            type_name = str(node.get("typeName") or "")
            statement_id = str(node.get("statementId") or "")
            props = dict(node.get("props") or {})

            role_name = self._derive_role(type_name, props, path, parent_role_id)
            role_id = get_role_id(f"{statement_id or path}_{role_name}")
            role_slots.append(
                RoleSlot(
                    role_id=role_id,
                    component_family=type_name,
                    required=self._is_required(props, type_name),
                )
            )

            if parent_role_id is not None:
                topology_edges.append(
                    {
                        "parent_role_id": parent_role_id,
                        "child_role_id": role_id,
                        "relation": "contains",
                    }
                )

            # Extract placeholder/content references from content props.
            candidate_symbols: list[str] = []
            for prop_name, value in props.items():
                if prop_name == "children":
                    continue
                raw = str(value) if value is not None else ""
                if raw.startswith(":"):
                    symbol_id = get_symbol_id(raw)
                    symbols.append(
                        PlanSymbol(
                            symbol_id=symbol_id,
                            semantic_role=prop_name,
                            allowed_pointer_targets=None,
                        )
                    )
                    candidate_symbols.append(symbol_id)
            if candidate_symbols:
                bindings.append(
                    PlanBinding(
                        role_slot_id=role_id,
                        candidate_symbols=tuple(candidate_symbols),
                        placeholder_fallback=True,
                    )
                )

            children = props.get("children")
            if isinstance(children, list):
                child_roles: list[str] = []
                for index, child in enumerate(children):
                    if isinstance(child, dict):
                        child_role_id = walk(child, role_id, (*path, index))
                        child_roles.append(child_role_id)
                if len(child_roles) > 1:
                    sibling_groups.append(tuple(sorted(child_roles)))
            return role_id

        walk(root, None, (0,))

        topology = PlanTopology(
            parent_relation_candidates=tuple(topology_edges) or None,
            sibling_order_groups=tuple(sibling_groups) or None,
        )

        coverage = PlanCoverage(
            named_requirements_accounted_for=tuple(
                sorted({sym.semantic_role for sym in symbols if sym.semantic_role})
            )
            or None,
            unresolved_requirements=None,
        )

        return SemanticPlanV1(
            identity=identity,
            archetype=archetype,
            role_slots=tuple(role_slots),
            topology=topology,
            symbols=tuple(symbols),
            bindings=tuple(bindings),
            coverage=coverage,
            confidence_calibration=PlanConfidenceCalibration(
                per_factor_confidence={
                    "archetype": 1.0,
                    "role_slots": 1.0,
                    "topology": 1.0,
                    "symbols": 1.0,
                    "bindings": 1.0,
                }
            ),
        )

    @classmethod
    def _archetype_id(cls, root_type: str, direction: str) -> str:
        parts = [root_type.lower()]
        if direction:
            parts.append(direction)
        return "_".join(parts)

    @classmethod
    def _derive_role(
        cls,
        type_name: str,
        props: dict[str, Any],
        path: tuple[int, ...],
        parent_role_id: str | None,
    ) -> str:
        # Prefer a content prop name as role when available.
        for prop_name in props:
            if prop_name in cls._CONTENT_PROPS:
                return f"{type_name.lower()}_{prop_name}"
        # Use position-based role for containers.
        if type_name in cls._CONTAINER_TYPES:
            if path == (0,):
                return "root_container"
            return f"container_{'_'.join(str(p) for p in path)}"
        # Fallback: component type + position.
        return f"{type_name.lower()}_{'_'.join(str(p) for p in path)}"

    @classmethod
    def _is_required(cls, props: dict[str, Any], type_name: str) -> bool | None:
        # Conservative default: unknown.
        return None
