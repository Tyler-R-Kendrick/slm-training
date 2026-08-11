"""Deterministic compilation into ``CompiledGoalConstraintSetV1`` (PGS-A02 / SLM-495).

Structured ``GenerationRequest`` fields, pack contract surfaces, and verifier
requirements compile as exact hard constraints. ``RequirementFact`` values
from ``VerifiedSynthesisProblemV1.requirements`` compile only through a closed,
versioned rule table into advisory, oracle-diagnostic, or evaluation-only
predicates — never into production pruning.

Gold/target AST material must not enter this entry point. Unknown requirement
statements are recorded in ``uncompiled_source_ids`` rather than guessed or
dropped.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintAmbiguityGroup,
    GoalConstraintV1,
    canonical_json_bytes,
)
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.synthesis_problem import (
    EvidenceProvenanceV1,
    PackIdentityV1,
    VerificationRequirementV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.dsl.pack import DslPack

__all__ = [
    "COMPILER_VERSION",
    "COMPILE_RULES",
    "CompileRule",
    "GoldTargetAccessError",
    "IdentityMismatchError",
    "compile_goal_constraints",
    "compile_rule_table",
    "digest_recipe",
    "pack_identity_digest",
    "request_production_digest",
    "source_digest",
    "source_to_authority_matrix",
]

COMPILER_VERSION = "pgs-a02.requirements_compile/v1"

_GOLD_LEAK_KEYS = frozenset(
    {
        "openui",
        "gold",
        "gold_ast",
        "gold_openui",
        "target",
        "target_ast",
        "reference",
        "reference_ast",
        "program_spec",
        "ast",
        "example_record",
        "record",
    }
)

_STRUCTURED_EXTRACTOR_PREFIXES = frozenset(
    {
        "sgs004.slot_contract.v1:",
        "sgs004.runtime_symbol.v1:",
        "sgs004.output_kind.v1:",
    }
)

_COMPONENT_REQUIRED_RE = re.compile(
    r"^prompt requires component ([A-Za-z][A-Za-z0-9]*)(?: count>=([0-9]+))?$"
)
_COMPONENT_FORBIDDEN_RE = re.compile(r"^prompt forbids component ([A-Za-z][A-Za-z0-9]*)$")
_EITHER_OR_RE = re.compile(r"^prompt offers component alternative ([A-Za-z][A-Za-z0-9]*)$")


class GoldTargetAccessError(ValueError):
    """Raised when gold/target material is offered to the production compiler."""


class IdentityMismatchError(ValueError):
    """Raised when problem/request/pack identities disagree."""


@dataclass(frozen=True)
class CompileRule:
    """Stable compile-rule identity for advisory/evaluation translation."""

    rule_id: str
    version: str
    description: str
    source_kind: str

    @property
    def qualified_id(self) -> str:
        return f"{self.rule_id}.{self.version}"


COMPILE_RULES: Mapping[str, CompileRule] = {
    "prompt_component_required": CompileRule(
        "pgs-a02.prompt_component_required",
        "v1",
        "Positive component inventory from authored RequirementFact statements.",
        "prompt_requirement",
    ),
    "prompt_component_forbidden": CompileRule(
        "pgs-a02.prompt_component_forbidden",
        "v1",
        "Explicit forbidden component mentions from RequirementFact statements.",
        "prompt_requirement",
    ),
    "prompt_either_or": CompileRule(
        "pgs-a02.prompt_either_or",
        "v1",
        "Explicit either/or alternatives; never resolved at compile time.",
        "prompt_requirement",
    ),
    "evaluation_fixture": CompileRule(
        "pgs-a02.evaluation_fixture",
        "v1",
        "Held-out/evaluation-only facts passed through the problem envelope.",
        "evaluation_fixture",
    ),
    "oracle_diagnostic": CompileRule(
        "pgs-a02.oracle_diagnostic",
        "v1",
        "Oracle-diagnostic annotations; scoring only, never production prune.",
        "oracle_diagnostic",
    ),
    "pack_unsupported_production": CompileRule(
        "pgs-a02.pack_unsupported_production",
        "v1",
        "Pack-declared unsupported grammar productions.",
        "pack_contract",
    ),
}


def digest_recipe() -> dict[str, Any]:
    """Machine-readable digest recipes for downstream consumers."""
    return {
        "request_production_digest": {
            "algorithm": "sha256",
            "payload": "canonical_json(GenerationRequest.to_dict())",
            "notes": "Sorted-key JSON with compact separators; default output_kind omitted.",
        },
        "pack_identity_digest": {
            "algorithm": "sha256",
            "payload": "canonical_json({pack_id, **PackIdentityV1.model_dump()})",
            "notes": "Live pack.pack_id is authoritative; problem.pack_identity fields are included.",
        },
        "source_digest": {
            "algorithm": "sha256",
            "payload": "canonical_json(rule-local source payload)",
        },
        "compiled_set_digest": {
            "algorithm": "sha256",
            "payload": "CompiledGoalConstraintSetV1._canonical_payload() excluding digest",
            "notes": "Owned by goal_constraints.py; constraints sorted by constraint_id.",
        },
    }


def compile_rule_table() -> dict[str, Any]:
    """Closed compilation-rule table (stable ids/versions, no generic substring rules)."""
    return {
        "compiler_version": COMPILER_VERSION,
        "rules": {
            key: {
                "rule_id": rule.rule_id,
                "version": rule.version,
                "qualified_id": rule.qualified_id,
                "description": rule.description,
                "default_source_kind": rule.source_kind,
            }
            for key, rule in COMPILE_RULES.items()
        },
        "structured_hard_sources": [
            "generation_request.slot_contract",
            "generation_request.runtime_symbols",
            "generation_request.output_kind",
            "generation_request.output_category",
            "pack.grammar_capability_authority.unsupported_alternatives",
            "problem.verification_requirements",
        ],
        "structured_extractor_skip_prefixes": sorted(_STRUCTURED_EXTRACTOR_PREFIXES),
    }


def source_to_authority_matrix() -> dict[str, Any]:
    """Source kind → default authority/completeness/prune posture."""
    return {
        "generation_request": {
            "authority_tier": "compiler-hard",
            "completeness": "EXACT",
            "may_prune": False,
            "partition": "hard",
        },
        "pack_contract": {
            "authority_tier": "compiler-hard",
            "completeness": "EXACT",
            "may_prune": True,
            "partition": "hard",
        },
        "verification_requirement": {
            "authority_tier": "verifier-hard",
            "completeness": "EXACT",
            "may_prune": True,
            "partition": "hard",
        },
        "prompt_requirement": {
            "authority_tier": "advisory-learned",
            "completeness": "HEURISTIC",
            "may_prune": False,
            "partition": "advisory",
        },
        "oracle_diagnostic": {
            "authority_tier": "oracle-diagnostic",
            "completeness": "HEURISTIC",
            "may_prune": False,
            "partition": "advisory",
        },
        "evaluation_fixture": {
            "authority_tier": "evaluation-only",
            "completeness": "EXACT",
            "may_prune": False,
            "partition": "evaluation",
        },
    }


def source_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def request_production_digest(request: GenerationRequest) -> str:
    return source_digest(request.to_dict())


def pack_identity_digest(identity: PackIdentityV1, pack: DslPack) -> str:
    payload = {"pack_id": pack.pack_id, **identity.model_dump(mode="json")}
    return source_digest(payload)


def _reject_gold_kwargs(kwargs: Mapping[str, Any]) -> None:
    leaked = sorted(_GOLD_LEAK_KEYS.intersection(kwargs))
    if leaked:
        raise GoldTargetAccessError(
            f"goal-constraint compilation refuses gold/target inputs: {leaked}"
        )


def _reject_gold_provenance(problem: VerifiedSynthesisProblemV1) -> None:
    leaked = sorted(_GOLD_LEAK_KEYS.intersection(problem.provenance))
    if leaked:
        raise GoldTargetAccessError(
            f"goal-constraint compilation refuses gold/target provenance keys: {leaked}"
        )


def _validate_identities(
    *,
    problem: VerifiedSynthesisProblemV1,
    request: GenerationRequest,
    pack: DslPack,
) -> None:
    if problem.pack_identity.pack_id != pack.pack_id:
        raise IdentityMismatchError(
            "problem.pack_identity.pack_id does not match live pack.pack_id "
            f"({problem.pack_identity.pack_id!r} != {pack.pack_id!r})"
        )

    effective = {
        (symbol.surface, symbol.role): symbol
        for symbol in request.effective_runtime_symbols()
    }
    for declared in problem.runtime_symbols:
        key = (declared.surface, declared.role)
        if key not in effective:
            raise IdentityMismatchError(
                "problem.runtime_symbols contains a declaration absent from the "
                f"GenerationRequest production inputs: {declared.surface!r}"
            )

    requirements = problem.requirements
    if requirements.prompt_context_hash is not None:
        expected = request_production_digest(request)
        if requirements.prompt_context_hash != expected:
            raise IdentityMismatchError(
                "problem.requirements.prompt_context_hash does not match the "
                "GenerationRequest production-input digest"
            )


def _hard_slot_inventory(request: GenerationRequest) -> GoalConstraintV1 | None:
    slots = tuple(sorted(request.slot_contract))
    if not slots:
        return None
    payload = {"slots": list(slots)}
    return GoalConstraintV1(
        constraint_id="hard.slot_inventory_exact",
        kind="slot_inventory_exact",
        parameters={"slots": list(slots)},
        source_kind="generation_request",
        source_id="generation_request.slot_contract",
        source_digest=source_digest(payload),
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=False,
    )


def _hard_runtime_symbols(request: GenerationRequest) -> tuple[GoalConstraintV1, ...]:
    constraints: list[GoalConstraintV1] = []
    for symbol in sorted(request.runtime_symbols, key=lambda item: item.surface):
        payload = {"surface": symbol.surface, "role": symbol.role}
        constraints.append(
            GoalConstraintV1(
                constraint_id=f"hard.runtime_symbol.{symbol.surface}",
                kind="runtime_symbol_accounted",
                parameters=payload,
                source_kind="generation_request",
                source_id=f"generation_request.runtime_symbols:{symbol.surface}",
                source_digest=source_digest(payload),
                authority_tier="compiler-hard",
                completeness="EXACT",
                may_prune=False,
            )
        )
    return tuple(constraints)


def _hard_output_kind(request: GenerationRequest) -> GoalConstraintV1:
    payload = {"output_kind": request.output_kind}
    return GoalConstraintV1(
        constraint_id="hard.output_kind_equals",
        kind="output_kind_equals",
        parameters=payload,
        source_kind="generation_request",
        source_id="generation_request.output_kind",
        source_digest=source_digest(payload),
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=False,
    )


def _hard_output_category(request: GenerationRequest) -> GoalConstraintV1 | None:
    if request.output_category is None:
        return None
    payload = {"output_category": request.output_category}
    return GoalConstraintV1(
        constraint_id="hard.output_category_equals",
        kind="output_category_equals",
        parameters=payload,
        source_kind="generation_request",
        source_id="generation_request.output_category",
        source_digest=source_digest(payload),
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=False,
    )


def _hard_verification_requirements(
    requirements: Iterable[VerificationRequirementV1],
) -> tuple[GoalConstraintV1, ...]:
    constraints: list[GoalConstraintV1] = []
    for requirement in sorted(requirements, key=lambda item: item.requirement_id):
        if requirement.kind == "gate":
            payload = {
                "gate": requirement.gate,
                "mandatory": requirement.mandatory,
            }
            parameters = {"gate": requirement.gate}
        else:
            payload = {
                "evaluator_version": requirement.evaluator_version,
                "evaluator_hash": requirement.evaluator_hash,
                "mandatory": requirement.mandatory,
            }
            parameters = {
                key: value
                for key, value in payload.items()
                if key != "mandatory" and value is not None
            }
        constraints.append(
            GoalConstraintV1(
                constraint_id=f"hard.verification.{requirement.requirement_id}",
                kind="required_gate_passes",
                parameters=parameters,
                source_kind="verification_requirement",
                source_id=f"verification_requirements:{requirement.requirement_id}",
                source_digest=source_digest(payload),
                authority_tier="verifier-hard",
                completeness="EXACT",
                may_prune=True,
            )
        )
    return tuple(constraints)


def _hard_pack_restrictions(pack: DslPack) -> tuple[GoalConstraintV1, ...]:
    authority = pack.grammar_capability_authority
    if authority is None or not authority.unsupported_alternatives:
        return ()
    rule = COMPILE_RULES["pack_unsupported_production"]
    constraints: list[GoalConstraintV1] = []
    for production_id in sorted(authority.unsupported_alternatives):
        reason = authority.unsupported_alternatives[production_id]
        payload = {"production_id": production_id, "reason": reason}
        constraints.append(
            GoalConstraintV1(
                constraint_id=f"hard.pack.unsupported.{production_id}",
                kind="binding_resolved",
                parameters={"production_id": production_id, "forbidden": True, "reason": reason},
                source_kind="pack_contract",
                source_id=f"{rule.qualified_id}:{production_id}",
                source_digest=source_digest(payload),
                authority_tier="compiler-hard",
                completeness="EXACT",
                may_prune=True,
            )
        )
    return tuple(constraints)


def _extractor_rule_key(fact: RequirementFact) -> str | None:
    for prefix in (
        "sgs004.prompt_component_required.v1:",
        "sgs004.prompt_component_forbidden.v1:",
        "sgs004.prompt_either_or.v1:",
    ):
        if fact.fact_id.startswith(prefix):
            return prefix.split(".")[1]  # prompt_component_required
    return None


def _compile_requirement_fact(
    fact: RequirementFact,
) -> GoalConstraintV1 | None:
    if any(fact.fact_id.startswith(prefix) for prefix in _STRUCTURED_EXTRACTOR_PREFIXES):
        return None

    if fact.authority == "evaluation-only":
        rule = COMPILE_RULES["evaluation_fixture"]
        authority = "evaluation-only"
        completeness = "EXACT"
        source_kind = "evaluation_fixture"
    elif fact.authority == "oracle-diagnostic" or fact.provenance == "oracle_override":
        rule = COMPILE_RULES["oracle_diagnostic"]
        authority = "oracle-diagnostic"
        completeness = "HEURISTIC"
        source_kind = "oracle_diagnostic"
    else:
        extractor_key = _extractor_rule_key(fact)
        if extractor_key == "prompt_component_required":
            rule = COMPILE_RULES["prompt_component_required"]
        elif extractor_key == "prompt_component_forbidden":
            rule = COMPILE_RULES["prompt_component_forbidden"]
        elif extractor_key == "prompt_either_or":
            rule = COMPILE_RULES["prompt_either_or"]
        else:
            return None
        authority = "advisory-learned"
        completeness = "HEURISTIC"
        source_kind = rule.source_kind

    parameters = _parameters_for_fact(fact, rule)
    if parameters is None:
        return None
    kind = parameters["_kind"]
    constraint_parameters = {
        key: value for key, value in parameters.items() if key != "_kind"
    }

    payload = {
        "fact_id": fact.fact_id,
        "statement": fact.statement,
        "parameters": constraint_parameters,
        "provenance": fact.provenance,
        "confidence": fact.confidence,
    }
    return GoalConstraintV1(
        constraint_id=f"compiled.{fact.fact_id}",
        kind=kind,  # type: ignore[arg-type]
        parameters=constraint_parameters,
        source_kind=source_kind,  # type: ignore[arg-type]
        source_id=f"{rule.qualified_id}:{fact.fact_id}",
        source_digest=source_digest(payload),
        authority_tier=authority,  # type: ignore[arg-type]
        completeness=completeness,  # type: ignore[arg-type]
        may_prune=False,
    )


def _parameters_for_fact(
    fact: RequirementFact, rule: CompileRule
) -> dict[str, Any] | None:
    if rule.rule_id == COMPILE_RULES["evaluation_fixture"].rule_id:
        return {
            "_kind": "component_count_at_least",
            "component": fact.factor_family,
            "min_count": 1 if fact.disposition == "required" else 0,
            "statement": fact.statement,
            "provenance": fact.provenance,
            "confidence": fact.confidence,
        }

    if rule.rule_id == COMPILE_RULES["oracle_diagnostic"].rule_id:
        return {
            "_kind": "component_count_at_least",
            "component": fact.factor_family,
            "min_count": 1 if fact.disposition == "required" else 0,
            "statement": fact.statement,
            "provenance": fact.provenance,
            "confidence": fact.confidence,
        }

    if rule.rule_id == COMPILE_RULES["prompt_component_required"].rule_id:
        match = _COMPONENT_REQUIRED_RE.fullmatch(fact.statement)
        if match is None:
            return None
        component, count_text = match.group(1), match.group(2)
        min_count = int(count_text) if count_text is not None else 1
        return {
            "_kind": "component_count_at_least",
            "component": component,
            "min_count": min_count,
            "statement": fact.statement,
            "provenance": fact.provenance,
            "confidence": fact.confidence,
        }

    if rule.rule_id == COMPILE_RULES["prompt_component_forbidden"].rule_id:
        match = _COMPONENT_FORBIDDEN_RE.fullmatch(fact.statement)
        if match is None:
            return None
        component = match.group(1)
        return {
            "_kind": "component_absent",
            "component": component,
            "statement": fact.statement,
            "provenance": fact.provenance,
            "confidence": fact.confidence,
        }

    if rule.rule_id == COMPILE_RULES["prompt_either_or"].rule_id:
        match = _EITHER_OR_RE.fullmatch(fact.statement)
        if match is None:
            return None
        component = match.group(1)
        return {
            "_kind": "component_count_at_least",
            "component": component,
            "min_count": 0,
            "alternative": True,
            "statement": fact.statement,
            "provenance": fact.provenance,
            "confidence": fact.confidence,
        }

    return None


def _compile_evidence_provenance(
    evidence: EvidenceProvenanceV1,
) -> GoalConstraintV1:
    if evidence.authority == "evaluation-only":
        rule = COMPILE_RULES["evaluation_fixture"]
        authority = "evaluation-only"
        source_kind = "evaluation_fixture"
        completeness = "EXACT"
    else:
        rule = COMPILE_RULES["oracle_diagnostic"]
        authority = "oracle-diagnostic"
        source_kind = "oracle_diagnostic"
        completeness = "HEURISTIC"

    payload = {
        "source": evidence.source,
        "identity": evidence.identity,
        "authority": evidence.authority,
    }
    safe_source = re.sub(r"[^A-Za-z0-9._-]+", "_", evidence.source)
    constraint_id = f"evidence.{safe_source}.{evidence.authority}"
    return GoalConstraintV1(
        constraint_id=constraint_id,
        kind="required_gate_passes",
        parameters={
            "evidence_source": evidence.source,
            "identity": evidence.identity,
        },
        source_kind=source_kind,  # type: ignore[arg-type]
        source_id=f"{rule.qualified_id}:{evidence.source}",
        source_digest=source_digest(payload),
        authority_tier=authority,  # type: ignore[arg-type]
        completeness=completeness,  # type: ignore[arg-type]
        may_prune=False,
    )


def _ambiguity_groups(
    requirements: PromptSemanticRequirementsV1,
    compiled_ids_by_fact: Mapping[str, str],
) -> tuple[GoalConstraintAmbiguityGroup, ...]:
    groups: list[GoalConstraintAmbiguityGroup] = []
    for index, group in enumerate(
        sorted(requirements.ambiguity_groups, key=lambda item: item.group_id)
    ):
        member_ids = tuple(
            sorted(
                compiled_ids_by_fact[fact_id]
                for fact_id in group.member_fact_ids
                if fact_id in compiled_ids_by_fact
            )
        )
        if len(member_ids) < 2:
            continue
        groups.append(
            GoalConstraintAmbiguityGroup(
                group_id=f"ambiguity.{index:04d}",
                member_constraint_ids=member_ids,
                note=group.note,
            )
        )
    return tuple(groups)


def _partition_constraints(
    constraints: Iterable[GoalConstraintV1],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    hard: list[str] = []
    advisory: list[str] = []
    evaluation: list[str] = []
    for constraint in constraints:
        if constraint.authority_tier in {"compiler-hard", "verifier-hard"}:
            hard.append(constraint.constraint_id)
        elif constraint.authority_tier == "evaluation-only":
            evaluation.append(constraint.constraint_id)
        else:
            advisory.append(constraint.constraint_id)
    return tuple(sorted(hard)), tuple(sorted(advisory)), tuple(sorted(evaluation))


def compile_goal_constraints(
    problem: VerifiedSynthesisProblemV1,
    request: GenerationRequest,
    pack: DslPack,
    **kwargs: Any,
) -> CompiledGoalConstraintSetV1:
    """Compile structured request/problem/pack inputs into goal constraints."""
    _reject_gold_kwargs(kwargs)
    _reject_gold_provenance(problem)
    _validate_identities(problem=problem, request=request, pack=pack)

    constraints: list[GoalConstraintV1] = []

    slot_inventory = _hard_slot_inventory(request)
    if slot_inventory is not None:
        constraints.append(slot_inventory)
    constraints.extend(_hard_runtime_symbols(request))
    constraints.append(_hard_output_kind(request))
    output_category = _hard_output_category(request)
    if output_category is not None:
        constraints.append(output_category)
    constraints.extend(_hard_verification_requirements(problem.verification_requirements))
    constraints.extend(_hard_pack_restrictions(pack))

    uncompiled: list[str] = []
    compiled_ids_by_fact: dict[str, str] = {}
    for fact in sorted(problem.requirements.facts, key=lambda item: item.fact_id):
        compiled = _compile_requirement_fact(fact)
        if compiled is None:
            if not any(
                fact.fact_id.startswith(prefix) for prefix in _STRUCTURED_EXTRACTOR_PREFIXES
            ):
                uncompiled.append(fact.fact_id)
            continue
        constraints.append(compiled)
        compiled_ids_by_fact[fact.fact_id] = compiled.constraint_id

    for evidence in sorted(problem.evidence_provenance, key=lambda item: item.source):
        constraints.append(_compile_evidence_provenance(evidence))

    hard_ids, advisory_ids, evaluation_ids = _partition_constraints(constraints)
    ambiguity_groups = _ambiguity_groups(problem.requirements, compiled_ids_by_fact)

    compiled = CompiledGoalConstraintSetV1(
        problem_digest=problem.digest,
        request_digest=request_production_digest(request),
        pack_identity_digest=pack_identity_digest(problem.pack_identity, pack),
        compiler_version=COMPILER_VERSION,
        constraints=tuple(constraints),
        hard_constraint_ids=hard_ids,
        advisory_constraint_ids=advisory_ids,
        evaluation_constraint_ids=evaluation_ids,
        uncompiled_source_ids=tuple(sorted(set(uncompiled))),
        ambiguity_groups=ambiguity_groups,
    )
    return CompiledGoalConstraintSetV1.from_dict(
        compiled.model_copy(update={"digest": compiled.compute_digest()}).to_dict()
    )
