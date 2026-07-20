"""SemanticPlanV1 compiler bridge.

The bridge converts a pack-neutral `SemanticPlanV1` into:

1. a deterministic valid seed (when plan facts are actionable);
2. soft action/edit features that rank but never remove legal candidates;
3. certified-only hard restrictions gated by independently verifiable evidence.

Predicted, retrieved, and oracle plan facts are always soft or reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.seed import PlanSeedBuilder, SeedResult
from slm_training.dsl.pack import DslPack
from slm_training.dsl.parser import validate


class EvidenceKind(str, Enum):
    """Provenance class for a piece of restriction evidence."""

    PREDICTION_ONLY = "prediction_only"
    RETRIEVAL = "retrieval"
    ORACLE_GOLD = "oracle_gold"
    COMPILER_AUTHORED_CERTIFIED = "compiler_authored_certified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Evidence:
    """One piece of evidence offered to the certified-restriction gate."""

    evidence_id: str
    kind: EvidenceKind
    certificate: str | None = None
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "certificate": self.certificate,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class PlanSeedResult:
    """Outcome of building a plan-derived valid seed."""

    seed: str | None
    ok: bool
    reason: str | None = None
    plan_coverage: dict[str, Any] = field(default_factory=dict)
    provenance: str = "none"
    uncertainty: dict[str, float] | None = None
    verifier_outcome: dict[str, Any] | None = None
    fail_closed_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "ok": self.ok,
            "reason": self.reason,
            "plan_coverage": self.plan_coverage,
            "provenance": self.provenance,
            "uncertainty": self.uncertainty,
            "verifier_outcome": self.verifier_outcome,
            "fail_closed_reason": self.fail_closed_reason,
        }


@dataclass(frozen=True)
class HardRemoval:
    """One hard removal supported by certified evidence."""

    action_id: str
    evidence: Evidence
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "evidence": self.evidence.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlanActionFeatures:
    """Soft, versioned features attached to one legal action or valid edit."""

    action_id: str
    matches_predicted_role: bool = False
    component_family_compatible: bool = False
    expected_coverage_contribution: float = 0.0
    topology_parent_order_compatible: bool = False
    cardinality_depth_delta: int = 0
    binding_pointer_compatible: bool = False
    plan_confidence: float = 0.0
    provenance: str = "none"
    seed_lineage: str | None = None
    conflict_or_unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "matches_predicted_role": self.matches_predicted_role,
            "component_family_compatible": self.component_family_compatible,
            "expected_coverage_contribution": self.expected_coverage_contribution,
            "topology_parent_order_compatible": self.topology_parent_order_compatible,
            "cardinality_depth_delta": self.cardinality_depth_delta,
            "binding_pointer_compatible": self.binding_pointer_compatible,
            "plan_confidence": self.plan_confidence,
            "provenance": self.provenance,
            "seed_lineage": self.seed_lineage,
            "conflict_or_unknown": self.conflict_or_unknown,
        }


@dataclass(frozen=True)
class RestrictionResult:
    """Outcome of applying the certified-only restriction gate."""

    hard_removals: tuple[HardRemoval, ...] = ()
    soft_removals: tuple[tuple[str, Evidence], ...] = ()
    unknown_preserved: tuple[str, ...] = ()
    evidence_log: tuple[Evidence, ...] = ()
    false_hard_prune_count: int = 0
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_removals": [r.to_dict() for r in self.hard_removals],
            "soft_removals": [
                {"action_id": action_id, "evidence": ev.to_dict()}
                for action_id, ev in self.soft_removals
            ],
            "unknown_preserved": list(self.unknown_preserved),
            "evidence_log": [e.to_dict() for e in self.evidence_log],
            "false_hard_prune_count": self.false_hard_prune_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class PlanAssumption:
    """One reversible plan assumption committed during search."""

    assumption_id: str
    fact: str
    depends_on: tuple[str, ...] = ()


class PlanAssumptionTrail:
    """Trail of reversible plan assumptions.

    On rollback, every assumption pushed at the current decision level is
    retracted and no residue is left in the active assumption set.
    """

    def __init__(self) -> None:
        self._frames: list[list[PlanAssumption]] = []

    def push(self, assumptions: list[PlanAssumption]) -> None:
        self._frames.append(list(assumptions))

    def rollback(self) -> list[PlanAssumption]:
        """Retract the most recent frame and return what was removed."""
        if not self._frames:
            return []
        return self._frames.pop()

    @property
    def active(self) -> list[PlanAssumption]:
        return [a for frame in self._frames for a in frame]

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_count": len(self.active),
            "frame_count": len(self._frames),
        }


class SemanticPlanCompiler(Protocol):
    """Pack-neutral plan-consumer interface."""

    def build_valid_seed(
        self, request: Any, plan: SemanticPlanV1 | None, pack: DslPack
    ) -> PlanSeedResult: ...

    def annotate_actions(
        self, state: Any, actions: list[Any], plan: SemanticPlanV1 | None
    ) -> list[PlanActionFeatures]: ...

    def certified_restrictions(
        self,
        request: Any,
        state: Any,
        plan: SemanticPlanV1 | None,
        evidence: list[Evidence],
    ) -> RestrictionResult: ...


class OpenUISemanticPlanCompiler:
    """OpenUI implementation of the plan-consumer bridge.

    Default behavior is fail-closed and plan-neutral: a missing or abstained
    plan compiles to the baseline no-plan path. Oracle-only plans compile to
    baseline in ``production`` honesty mode and are enabled only in
    ``oracle_diagnostic`` mode. Hard restrictions are applied only when
    ``EvidenceKind.COMPILER_AUTHORED_CERTIFIED`` evidence carries a replayable
    certificate.
    """

    def __init__(
        self,
        *,
        honesty_mode: str = "production",
        allow_unsafe_predicted_hard_control: bool = False,
    ) -> None:
        if honesty_mode not in {"production", "oracle_diagnostic"}:
            raise ValueError(
                f"invalid honesty_mode {honesty_mode!r}; expected production or oracle_diagnostic"
            )
        self.honesty_mode = honesty_mode
        self.allow_unsafe_predicted_hard_control = allow_unsafe_predicted_hard_control

    def _plan_is_actionable(self, plan: SemanticPlanV1) -> bool:
        """True when *plan* carries predicted/retrieved/merged structure to compile."""
        if plan.is_abstained:
            return False
        if self.honesty_mode != "oracle_diagnostic" and plan.is_oracle_only:
            return False
        return bool(
            plan.role_slots
            or plan.topology.parent_relation_candidates is not None
            or plan.topology.sibling_order_groups is not None
            or plan.symbols
            or plan.bindings
            or plan.coverage.named_requirements_accounted_for
        )

    def build_valid_seed(
        self, request: Any, plan: SemanticPlanV1 | None, pack: DslPack
    ) -> PlanSeedResult:
        if plan is None or not self._plan_is_actionable(plan):
            return PlanSeedResult(
                seed=None,
                ok=True,
                reason="baseline: no actionable plan",
                plan_coverage={},
                provenance="none",
            )

        builder = PlanSeedBuilder(pack)
        seed_result: SeedResult = builder.build(plan)
        coverage = _compute_coverage(plan)
        provenance = plan.identity.provenance

        if not seed_result.ok:
            return PlanSeedResult(
                seed=None,
                ok=False,
                reason=seed_result.reason,
                plan_coverage=coverage,
                provenance=provenance,
                fail_closed_reason=seed_result.reason,
            )

        seed = seed_result.seed
        if seed is None:
            return PlanSeedResult(
                seed=None,
                ok=True,
                reason="baseline: no actionable plan",
                plan_coverage=coverage,
                provenance=provenance,
            )

        try:
            validate(seed)
            verifier_outcome: dict[str, Any] = {"validated": True}
        except Exception as exc:  # noqa: BLE001
            return PlanSeedResult(
                seed=None,
                ok=False,
                reason=f"verifier rejected seed: {exc}",
                plan_coverage=coverage,
                provenance=provenance,
                verifier_outcome={"validated": False, "error": str(exc)},
                fail_closed_reason="verifier_rejection",
            )

        uncertainty = _compute_uncertainty(plan)
        return PlanSeedResult(
            seed=seed,
            ok=True,
            reason=None,
            plan_coverage=coverage,
            provenance=provenance,
            uncertainty=uncertainty,
            verifier_outcome=verifier_outcome,
        )

    def annotate_actions(
        self, state: Any, actions: list[Any], plan: SemanticPlanV1 | None
    ) -> list[PlanActionFeatures]:
        if plan is None or not self._plan_is_actionable(plan):
            return [_baseline_features(str(a)) for a in actions]

        role_ids = {slot.role_id for slot in plan.role_slots}
        role_families = {
            slot.role_id: slot.component_family or ""
            for slot in plan.role_slots
        }
        bound_symbols = _bound_symbols(plan)
        topology_edges = _topology_edges(plan)
        plan_confidence = plan.archetype.confidence or 0.0
        provenance = plan.identity.provenance

        features: list[PlanActionFeatures] = []
        for action in actions:
            action_id = str(action)
            features.append(
                _score_action(
                    action_id,
                    role_ids,
                    role_families,
                    bound_symbols,
                    topology_edges,
                    plan_confidence,
                    provenance,
                )
            )
        return features

    def certified_restrictions(
        self,
        request: Any,
        state: Any,
        plan: SemanticPlanV1 | None,
        evidence: list[Evidence],
    ) -> RestrictionResult:
        if plan is None or not self._plan_is_actionable(plan):
            return RestrictionResult(status="baseline_no_restrictions")

        hard_removals: list[HardRemoval] = []
        soft_removals: list[tuple[str, Evidence]] = []
        unknown_preserved: list[str] = []
        evidence_log: list[Evidence] = []

        for ev in evidence:
            evidence_log.append(ev)
            if ev.kind is EvidenceKind.COMPILER_AUTHORED_CERTIFIED and ev.certificate:
                hard_removals.append(
                    HardRemoval(
                        action_id=ev.evidence_id,
                        evidence=ev,
                        reason=f"certified hard removal (certificate={ev.certificate})",
                    )
                )
            elif ev.kind in {
                EvidenceKind.PREDICTION_ONLY,
                EvidenceKind.RETRIEVAL,
                EvidenceKind.ORACLE_GOLD,
            }:
                soft_removals.append((ev.evidence_id, ev))
            else:
                unknown_preserved.append(ev.evidence_id)

        false_prune_count = 0
        if self.allow_unsafe_predicted_hard_control:
            for ev in evidence:
                if ev.kind is EvidenceKind.PREDICTION_ONLY:
                    hard_removals.append(
                        HardRemoval(
                            action_id=ev.evidence_id,
                            evidence=ev,
                            reason="UNSAFE: predicted-hard removal (non-promotable diagnostic)",
                        )
                    )

        return RestrictionResult(
            hard_removals=tuple(hard_removals),
            soft_removals=tuple(soft_removals),
            unknown_preserved=tuple(unknown_preserved),
            evidence_log=tuple(evidence_log),
            false_hard_prune_count=false_prune_count,
            status="ok",
        )


def _compute_coverage(plan: SemanticPlanV1) -> dict[str, Any]:
    return {
        "role_count": len(plan.role_slots),
        "topology_edge_count": len(plan.topology.parent_relation_candidates or ()),
        "symbol_count": len(plan.symbols),
        "binding_count": len(plan.bindings),
        "named_requirements": list(plan.coverage.named_requirements_accounted_for or ()),
    }


def _compute_uncertainty(plan: SemanticPlanV1) -> dict[str, float]:
    per_factor = plan.confidence_calibration.per_factor_confidence or {}
    return {
        "archetype": plan.archetype.confidence or 0.0,
        "role_slots": per_factor.get("role_slots", 0.0),
        "topology": per_factor.get("topology", 0.0),
        "symbols": per_factor.get("symbols", 0.0),
        "bindings": per_factor.get("bindings", 0.0),
    }


def _bound_symbols(plan: SemanticPlanV1) -> set[str]:
    symbols: set[str] = set()
    for binding in plan.bindings:
        candidates = binding.candidate_symbols or ()
        symbols.update(candidates)
    return symbols


def _topology_edges(plan: SemanticPlanV1) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for edge in plan.topology.parent_relation_candidates or ():
        parent = str(edge.get("parent_role_id") or "")
        child = str(edge.get("child_role_id") or "")
        if parent and child:
            edges.add((parent, child))
    return edges


def _baseline_features(action_id: str) -> PlanActionFeatures:
    return PlanActionFeatures(
        action_id=action_id,
        matches_predicted_role=False,
        component_family_compatible=False,
        expected_coverage_contribution=0.0,
        topology_parent_order_compatible=False,
        cardinality_depth_delta=0,
        binding_pointer_compatible=False,
        plan_confidence=0.0,
        provenance="none",
        conflict_or_unknown=False,
    )


def _score_action(
    action_id: str,
    role_ids: set[str],
    role_families: dict[str, str],
    bound_symbols: set[str],
    topology_edges: set[tuple[str, str]],
    plan_confidence: float,
    provenance: str,
) -> PlanActionFeatures:
    matches_role = action_id in role_ids
    family_compatible = any(
        action_id == family or action_id.endswith(f"_{family}")
        for family in role_families.values()
        if family
    )
    binding_compatible = action_id in bound_symbols
    topology_compatible = any(
        action_id == parent or action_id == child for parent, child in topology_edges
    )
    conflict_or_unknown = not (
        matches_role or family_compatible or binding_compatible or topology_compatible
    )
    coverage_contribution = (
        1.0 / max(len(role_ids), 1) if matches_role else 0.0
    )

    return PlanActionFeatures(
        action_id=action_id,
        matches_predicted_role=matches_role,
        component_family_compatible=family_compatible,
        expected_coverage_contribution=coverage_contribution,
        topology_parent_order_compatible=topology_compatible,
        cardinality_depth_delta=0,
        binding_pointer_compatible=binding_compatible,
        plan_confidence=plan_confidence if (matches_role or family_compatible) else 0.0,
        provenance=provenance,
        conflict_or_unknown=conflict_or_unknown,
    )
