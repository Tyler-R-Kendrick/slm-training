"""Factor-wise oracle substitution for SemanticPlanV1."""

from __future__ import annotations

from typing import Any, Literal

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)

PlanSource = Literal["none", "predicted", "gold"]
OracleFactor = Literal["archetype", "roles", "topology", "bindings"]
UseMode = Literal["seed", "features", "soft_bias", "certified_restrictions"]

_ALLOWED_FACTORS: set[str] = {"archetype", "roles", "topology", "bindings"}


class PlanOracleSubstitutor:
    """Fail-closed factor-wise oracle substitution for ceiling experiments.

    Gold plans are accepted only when ``honesty_mode == "oracle_diagnostic"``.
    Oracle arms must never enter a production/ship manifest.
    """

    def __init__(
        self,
        *,
        plan_source: PlanSource = "none",
        oracle_factors: tuple[OracleFactor, ...] | None = None,
        use_mode: UseMode = "seed",
        honesty_mode: str = "production",
    ) -> None:
        if plan_source not in {"none", "predicted", "gold"}:
            raise ValueError(f"invalid plan_source: {plan_source!r}")
        if use_mode not in {"seed", "features", "soft_bias", "certified_restrictions"}:
            raise ValueError(f"invalid use_mode: {use_mode!r}")
        if honesty_mode not in {"production", "oracle_diagnostic"}:
            raise ValueError(f"invalid honesty_mode: {honesty_mode!r}")
        if oracle_factors is not None:
            bad = set(oracle_factors) - _ALLOWED_FACTORS
            if bad:
                raise ValueError(f"invalid oracle factors: {sorted(bad)}")
        self.plan_source = plan_source
        self.oracle_factors = tuple(oracle_factors) if oracle_factors else ()
        self.use_mode = use_mode
        self.honesty_mode = honesty_mode

    def apply(self, baseline: SemanticPlanV1, oracle: SemanticPlanV1) -> SemanticPlanV1:
        """Return a plan with selected oracle factors substituted into baseline.

        Unknown factors preserve baseline behavior. Gold source is rejected
        outside oracle_diagnostic mode.
        """
        if self.plan_source == "none":
            return baseline
        if self.plan_source == "gold" and self.honesty_mode != "oracle_diagnostic":
            raise ValueError(
                "gold plan source requires honesty_mode=oracle_diagnostic"
            )
        if self.plan_source == "predicted" and oracle.identity.provenance == "gold":
            raise ValueError(
                "predicted plan source cannot consume a gold oracle plan"
            )

        updates: dict[str, Any] = {}
        for factor in self.oracle_factors:
            if factor == "archetype":
                updates["archetype"] = PlanArchetype(
                    id=oracle.archetype.id,
                    distribution=oracle.archetype.distribution,
                    confidence=oracle.archetype.confidence,
                )
            elif factor == "roles":
                updates["role_slots"] = tuple(
                    RoleSlot(
                        role_id=slot.role_id,
                        component_family=slot.component_family,
                        candidate_distribution=slot.candidate_distribution,
                        min_cardinality=slot.min_cardinality,
                        max_cardinality=slot.max_cardinality,
                        required=slot.required,
                        evidence_spans=slot.evidence_spans,
                    )
                    for slot in oracle.role_slots
                )
            elif factor == "topology":
                updates["topology"] = PlanTopology(
                    parent_relation_candidates=oracle.topology.parent_relation_candidates,
                    sibling_order_groups=oracle.topology.sibling_order_groups,
                    depth_bounds=oracle.topology.depth_bounds,
                    cardinality_bounds=oracle.topology.cardinality_bounds,
                    partial_order_constraints=oracle.topology.partial_order_constraints,
                )
            elif factor == "bindings":
                updates["bindings"] = tuple(
                    PlanBinding(
                        role_slot_id=binding.role_slot_id,
                        candidate_symbols=binding.candidate_symbols,
                        placeholder_fallback=binding.placeholder_fallback,
                    )
                    for binding in oracle.bindings
                )
        if not updates:
            return baseline
        return baseline.model_copy(update=updates)

    def contamination_banner(self) -> str | None:
        """Banner to attach to any artifact produced by an oracle arm."""
        if self.plan_source in {"gold", "oracle_override"}:
            return (
                f"ORACLE_DIAGNOSTIC: plan_source={self.plan_source} "
                f"factors={list(self.oracle_factors)} use_mode={self.use_mode}"
            )
        return None
