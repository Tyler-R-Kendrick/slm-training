"""Integrate prompt requirements with SemanticPlanV1 consumers (SGS-006).

Advisory-only adapters: score candidate plans against
``PromptSemanticRequirementsV1`` facts, annotate soft action features, and
never hard-prune the legal domain. Missing/empty/abstained requirements
compile to baseline (no-op) behavior.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Literal

from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.compiler import (
    EvidenceKind,
    PlanActionFeatures,
)
from slm_training.data.semantic_plan.requirements_project import (
    requirements_may_hard_prune,
)

__all__ = [
    "FactSatisfaction",
    "RequirementSatisfactionReport",
    "score_plan_against_requirements",
    "annotate_actions_with_requirements",
    "requirements_hard_prune_allowed",
]

SatisfactionStatus = Literal["satisfied", "violated", "unknown"]

_REQUIRED_COMPONENT_RE = re.compile(
    r"^prompt requires component ([A-Za-z][A-Za-z0-9]*)(?: count>=(\d+))?$"
)
_FORBIDDEN_COMPONENT_RE = re.compile(
    r"^prompt forbids component ([A-Za-z][A-Za-z0-9]*)$"
)
_DECLARED_MARKER_RE = re.compile(r"^declared template marker (:slot_\d+)$")
_DECLARED_SYMBOL_RE = re.compile(
    r"^declared runtime symbol (:slot_\d+|[^ ]+) role=([A-Za-z_]+)$"
)


class FactSatisfaction:
    __slots__ = ("fact_id", "status", "detail", "source_span")

    def __init__(
        self,
        *,
        fact_id: str,
        status: SatisfactionStatus,
        detail: str,
        source_span: str | None = None,
    ) -> None:
        self.fact_id = fact_id
        self.status = status
        self.detail = detail
        self.source_span = source_span

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "status": self.status,
            "detail": self.detail,
            "source_span": self.source_span,
        }


class RequirementSatisfactionReport:
    __slots__ = ("baseline", "facts", "requirements_version", "prompt_context_hash")

    def __init__(
        self,
        *,
        baseline: bool,
        facts: tuple[FactSatisfaction, ...],
        requirements_version: str | None = None,
        prompt_context_hash: str | None = None,
    ) -> None:
        self.baseline = baseline
        self.facts = facts
        self.requirements_version = requirements_version
        self.prompt_context_hash = prompt_context_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "requirements_version": self.requirements_version,
            "prompt_context_hash": self.prompt_context_hash,
            "facts": [fact.to_dict() for fact in self.facts],
            "satisfied": [f.fact_id for f in self.facts if f.status == "satisfied"],
            "violated": [f.fact_id for f in self.facts if f.status == "violated"],
            "unknown": [f.fact_id for f in self.facts if f.status == "unknown"],
        }


def requirements_hard_prune_allowed(
    requirements: PromptSemanticRequirementsV1 | None,
    *,
    evidence_kinds: tuple[EvidenceKind, ...] = (),
) -> bool:
    """Hard prune only when separately certified — never from requirements alone."""
    if requirements is None or not requirements.facts:
        return False
    return requirements_may_hard_prune(requirements, evidence_kinds=evidence_kinds)


def _component_counts(plan: SemanticPlanV1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in plan.role_slots:
        family = slot.component_family
        if not family:
            continue
        weight = slot.min_cardinality if slot.min_cardinality is not None else 1
        counts[family] = counts.get(family, 0) + max(1, int(weight))
    return counts


def _accounted_names(plan: SemanticPlanV1) -> set[str]:
    names = set(plan.coverage.named_requirements_accounted_for or ())
    names.update(symbol.symbol_id for symbol in plan.symbols)
    names.update(binding.role_slot_id for binding in plan.bindings)
    return names


def _score_fact(fact: RequirementFact, plan: SemanticPlanV1) -> FactSatisfaction:
    counts = _component_counts(plan)
    accounted = _accounted_names(plan)

    match = _REQUIRED_COMPONENT_RE.match(fact.statement)
    if match and fact.disposition == "required":
        name, raw_count = match.group(1), match.group(2)
        need = int(raw_count or "1")
        have = counts.get(name, 0)
        if have >= need:
            return FactSatisfaction(
                fact_id=fact.fact_id,
                status="satisfied",
                detail=f"{name} count {have} >= {need}",
                source_span=fact.source_span,
            )
        if not counts:
            return FactSatisfaction(
                fact_id=fact.fact_id,
                status="unknown",
                detail=f"no component_family inventory on plan for {name}",
                source_span=fact.source_span,
            )
        return FactSatisfaction(
            fact_id=fact.fact_id,
            status="violated",
            detail=f"{name} count {have} < {need}",
            source_span=fact.source_span,
        )

    match = _FORBIDDEN_COMPONENT_RE.match(fact.statement)
    if match and fact.disposition == "forbidden":
        name = match.group(1)
        have = counts.get(name, 0)
        if not counts and not plan.role_slots:
            return FactSatisfaction(
                fact_id=fact.fact_id,
                status="unknown",
                detail=f"no inventory to evaluate forbidden {name}",
                source_span=fact.source_span,
            )
        if have == 0:
            return FactSatisfaction(
                fact_id=fact.fact_id,
                status="satisfied",
                detail=f"{name} absent",
                source_span=fact.source_span,
            )
        return FactSatisfaction(
            fact_id=fact.fact_id,
            status="violated",
            detail=f"{name} present count={have}",
            source_span=fact.source_span,
        )

    match = _DECLARED_MARKER_RE.match(fact.statement) or _DECLARED_SYMBOL_RE.match(
        fact.statement
    )
    if match:
        token = match.group(1)
        if token in accounted:
            return FactSatisfaction(
                fact_id=fact.fact_id,
                status="satisfied",
                detail=f"{token} accounted on plan",
                source_span=fact.source_span,
            )
        if not accounted:
            return FactSatisfaction(
                fact_id=fact.fact_id,
                status="unknown",
                detail=f"plan has no accounted names to check {token}",
                source_span=fact.source_span,
            )
        return FactSatisfaction(
            fact_id=fact.fact_id,
            status="violated",
            detail=f"{token} missing from plan accounting",
            source_span=fact.source_span,
        )

    return FactSatisfaction(
        fact_id=fact.fact_id,
        status="unknown",
        detail="no deterministic plan projection for this fact",
        source_span=fact.source_span,
    )


def score_plan_against_requirements(
    plan: SemanticPlanV1 | None,
    requirements: PromptSemanticRequirementsV1 | None,
) -> RequirementSatisfactionReport:
    """Score a candidate plan against prompt requirements (advisory)."""
    if requirements is None or not requirements.facts:
        return RequirementSatisfactionReport(baseline=True, facts=())
    if plan is None:
        return RequirementSatisfactionReport(
            baseline=False,
            facts=tuple(
                FactSatisfaction(
                    fact_id=fact.fact_id,
                    status="unknown",
                    detail="no candidate plan",
                    source_span=fact.source_span,
                )
                for fact in requirements.facts
            ),
            requirements_version=requirements.requirements_version,
            prompt_context_hash=requirements.prompt_context_hash,
        )
    scored = tuple(_score_fact(fact, plan) for fact in requirements.facts)
    return RequirementSatisfactionReport(
        baseline=False,
        facts=scored,
        requirements_version=requirements.requirements_version,
        prompt_context_hash=requirements.prompt_context_hash,
    )


def annotate_actions_with_requirements(
    actions: list[Any],
    requirements: PromptSemanticRequirementsV1 | None,
    *,
    base_features: list[PlanActionFeatures] | None = None,
) -> list[PlanActionFeatures]:
    """Soft-annotate actions; never removes actions from ``actions``."""
    if base_features is not None and len(base_features) != len(actions):
        raise ValueError("base_features length must match actions")

    if requirements is None or not requirements.facts:
        if base_features is not None:
            return list(base_features)
        return [
            PlanActionFeatures(action_id=str(action), provenance="none")
            for action in actions
        ]

    required_names = set()
    for fact in requirements.facts:
        match = _REQUIRED_COMPONENT_RE.match(fact.statement)
        if match and fact.disposition == "required":
            required_names.add(match.group(1))

    out: list[PlanActionFeatures] = []
    for index, action in enumerate(actions):
        action_id = str(action)
        base = (
            base_features[index]
            if base_features is not None
            else PlanActionFeatures(action_id=action_id, provenance="none")
        )
        compatible = any(name in action_id for name in required_names)
        out.append(
            replace(
                base,
                component_family_compatible=base.component_family_compatible
                or compatible,
                provenance=(
                    base.provenance
                    if base.provenance != "none"
                    else "prompt_requirements"
                ),
            )
        )
    assert len(out) == len(actions)
    return out
