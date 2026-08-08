"""Authority projections for prompt-side requirements (SGS-005 / SLM-443).

Pure, deterministic projections that make authority transitions explicit when
``PromptSemanticRequirementsV1`` (and, by reuse, ``SemanticPlanV1`` production
manifests) are consumed by production, oracle diagnostics, or evaluators.

These projections never change the exact candidate domain by themselves: they
only rewrite/reject requirement manifests. Hard pruning remains owned by
``OpenUISemanticPlanCompiler.certified_restrictions`` and requires
``EvidenceKind.COMPILER_AUTHORED_CERTIFIED`` evidence — never a prompt-side
requirement alone.
"""

from __future__ import annotations

from typing import Any, Literal

from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementAuthority,
    RequirementFact,
)
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.compiler import EvidenceKind
from slm_training.data.semantic_plan.requirements_canonicalize import (
    canonicalize_requirements,
)

__all__ = [
    "ProjectionMode",
    "ProjectionResult",
    "UnsafeAuthorityError",
    "project_prompt_requirements",
    "project_semantic_plan",
    "requirements_may_hard_prune",
]

ProjectionMode = Literal["production", "oracle_diagnostic", "evaluation_only"]

_MODE_TARGET_AUTHORITY: dict[str, RequirementAuthority] = {
    "production": "advisory-learned",
    "oracle_diagnostic": "oracle-diagnostic",
    "evaluation_only": "evaluation-only",
}


class UnsafeAuthorityError(ValueError):
    """Raised when a projection would smuggle oracle/gold into a production path."""


class ProjectionResult:
    """Projected requirements plus an explicit downgrade audit trail."""

    __slots__ = ("mode", "requirements", "downgrades", "rejected_fact_ids")

    def __init__(
        self,
        *,
        mode: ProjectionMode,
        requirements: PromptSemanticRequirementsV1,
        downgrades: tuple[dict[str, str], ...] = (),
        rejected_fact_ids: tuple[str, ...] = (),
    ) -> None:
        self.mode = mode
        self.requirements = requirements
        self.downgrades = downgrades
        self.rejected_fact_ids = rejected_fact_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requirements": self.requirements.to_dict(),
            "downgrades": list(self.downgrades),
            "rejected_fact_ids": list(self.rejected_fact_ids),
        }


def requirements_may_hard_prune(
    requirements: PromptSemanticRequirementsV1,
    *,
    evidence_kinds: tuple[EvidenceKind, ...] = (),
) -> bool:
    """Prompt requirements alone never hard-prune the legal domain.

    Returns True only when independently certified compiler evidence is also
    supplied — mirroring ``OpenUISemanticPlanCompiler.certified_restrictions``.
    """
    del requirements  # advisory surface; never sufficient alone
    return any(kind is EvidenceKind.COMPILER_AUTHORED_CERTIFIED for kind in evidence_kinds)


def _authority_for_mode(authority: str, mode: ProjectionMode) -> str:
    """Map a fact's authority into the consumer mode.

    ``combine_authority`` picks the *weaker* tier (evaluation-only <
    oracle-diagnostic < advisory-learned). Production is different: it must
    strip diagnostic privilege, so ``oracle-diagnostic`` is rewritten to
    ``advisory-learned`` (losing diagnostic-only rights) rather than kept as
    the weaker combine result.
    """
    if mode == "production":
        if authority == "oracle-diagnostic":
            return "advisory-learned"
        return authority  # advisory-learned or evaluation-only
    if mode == "oracle_diagnostic":
        return authority
    return "evaluation-only"


def _project_fact(
    fact: RequirementFact,
    *,
    mode: ProjectionMode,
) -> tuple[RequirementFact | None, dict[str, str] | None]:
    if mode == "production" and fact.provenance == "oracle_override":
        raise UnsafeAuthorityError(
            f"oracle_override fact {fact.fact_id!r} cannot enter a production projection"
        )

    reason = {
        "production": "production_strips_oracle_diagnostic_privilege",
        "oracle_diagnostic": "oracle_diagnostic_passthrough",
        "evaluation_only": "evaluation_only_forces_weakest_authority",
    }[mode]
    new_authority = _authority_for_mode(fact.authority, mode)
    if new_authority != fact.authority:
        return fact.model_copy(update={"authority": new_authority}), {
            "fact_id": fact.fact_id,
            "from_authority": fact.authority,
            "to_authority": new_authority,
            "reason": reason,
        }
    return fact, None


def project_prompt_requirements(
    requirements: PromptSemanticRequirementsV1,
    *,
    mode: ProjectionMode = "production",
    canonicalize: bool = True,
) -> ProjectionResult:
    """Project requirements into a consumer-safe authority view.

    Idempotent: projecting twice with the same mode yields an identical
    requirements fingerprint. Never escalates authority. Never invents
    compiler-hard / verifier-hard claims (those Literals are absent from the
    requirements contract).
    """
    if mode not in _MODE_TARGET_AUTHORITY:
        raise ValueError(f"unknown projection mode {mode!r}")

    projected: list[RequirementFact] = []
    downgrades: list[dict[str, str]] = []
    for fact in requirements.facts:
        new_fact, note = _project_fact(fact, mode=mode)
        if new_fact is None:
            continue
        projected.append(new_fact)
        if note is not None:
            downgrades.append(note)

    result = PromptSemanticRequirementsV1(
        requirements_version=requirements.requirements_version,
        prompt_context_hash=requirements.prompt_context_hash,
        facts=tuple(projected),
        ambiguity_groups=requirements.ambiguity_groups,
    )
    if canonicalize:
        result = canonicalize_requirements(result)
    return ProjectionResult(
        mode=mode,
        requirements=result,
        downgrades=tuple(downgrades),
        rejected_fact_ids=(),
    )


def project_semantic_plan(
    plan: SemanticPlanV1,
    *,
    mode: ProjectionMode = "production",
) -> dict[str, Any]:
    """Reuse ``SemanticPlanV1.to_production_dict`` honesty modes.

    ``evaluation_only`` maps to production stripping (no oracle) but tags the
    payload with an explicit non-promotable marker — plans have no separate
    evaluation-only honesty mode today.
    """
    if mode == "oracle_diagnostic":
        payload = plan.to_production_dict(honesty_mode="oracle_diagnostic")
        payload["projection_mode"] = mode
        return payload
    if mode in {"production", "evaluation_only"}:
        payload = plan.to_production_dict(honesty_mode="production")
        payload["projection_mode"] = mode
        if mode == "evaluation_only":
            payload["promotable"] = False
            payload["projection_note"] = "evaluation_only_non_promotable"
        return payload
    raise ValueError(f"unknown projection mode {mode!r}")
