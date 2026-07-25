"""Semantic-first preference costs for equivalent operator outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from itertools import combinations
from typing import Any, Iterable

from slm_training.dsl.operators.contracts import (
    AstOperatorV1,
    OperatorApplicationV1,
    _fingerprint,
    _require_digest,
    _require_identifier,
)

_COST_SCALE = Decimal(1_000_000)


class PreferenceIntent(str, Enum):
    SIMPLIFY = "simplify"
    EXPAND = "expand"
    PRESERVE = "preserve"


class SequenceDefectPolicy(str, Enum):
    REJECT = "reject"
    PENALIZE = "penalize"


class PreferenceScopeError(ValueError):
    """Candidates do not share one verified semantic comparison scope."""


@dataclass(frozen=True)
class CanonicalAstCostV1:
    node_count: int
    production_count: int
    optional_node_count: int
    marker_count: int
    schema: str = "canonical_ast_cost/v1"

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.node_count,
                self.production_count,
                self.optional_node_count,
                self.marker_count,
            )
        ):
            raise ValueError("canonical AST cost counts must be non-negative")

    def directional(self, intent: PreferenceIntent) -> tuple[int, int, int, int]:
        values = (
            self.node_count,
            self.production_count,
            self.optional_node_count,
            self.marker_count,
        )
        if intent is PreferenceIntent.SIMPLIFY:
            return values
        if intent is PreferenceIntent.EXPAND:
            return tuple(-value for value in values)
        return (0, 0, 0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "node_count": self.node_count,
            "production_count": self.production_count,
            "optional_node_count": self.optional_node_count,
            "marker_count": self.marker_count,
        }


@dataclass(frozen=True)
class OperatorPreferenceStepV1:
    operator_id: str
    operator_fingerprint: str
    semantic_action_id: str
    application_id: str
    before_state_fingerprint: str
    after_state_fingerprint: str
    before_ast_fingerprint: str
    after_ast_fingerprint: str
    locality_violations: int
    operator_cost_micros: int
    proven_redundant: bool = False
    schema: str = "operator_preference_step/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.operator_id, "operator_id")
        for field, value in (
            ("operator_fingerprint", self.operator_fingerprint),
            ("semantic_action_id", self.semantic_action_id),
            ("application_id", self.application_id),
            ("before_state_fingerprint", self.before_state_fingerprint),
            ("after_state_fingerprint", self.after_state_fingerprint),
            ("before_ast_fingerprint", self.before_ast_fingerprint),
            ("after_ast_fingerprint", self.after_ast_fingerprint),
        ):
            _require_digest(value, field)
        if self.locality_violations < 0 or self.operator_cost_micros < 0:
            raise ValueError("operator step costs must be non-negative")
        if not isinstance(self.proven_redundant, bool):
            raise TypeError("proven_redundant must be boolean")

    @classmethod
    def from_application(
        cls,
        *,
        declaration: AstOperatorV1,
        semantic_action_id: str,
        application: OperatorApplicationV1,
        locality_violations: int = 0,
    ) -> OperatorPreferenceStepV1:
        if not application.succeeded:
            raise ValueError("preference sequences require successful applications")
        if application.operator_fingerprint != declaration.fingerprint:
            raise ValueError("application does not match the operator declaration")
        assert application.after_state_digest is not None
        assert application.after_ast_digest is not None
        scaled = Decimal(str(declaration.cost)) * _COST_SCALE
        if (
            not math.isfinite(declaration.cost)
            or declaration.cost < 0
            or scaled != scaled.to_integral_value()
        ):
            raise ValueError(
                "operator_cost must be finite, non-negative, and exact to six decimals"
            )
        return cls(
            operator_id=declaration.operator_id,
            operator_fingerprint=application.operator_fingerprint,
            semantic_action_id=semantic_action_id,
            application_id=application.application_id,
            before_state_fingerprint=application.before_state_digest,
            after_state_fingerprint=application.after_state_digest,
            before_ast_fingerprint=application.before_ast_digest,
            after_ast_fingerprint=application.after_ast_digest,
            locality_violations=locality_violations,
            operator_cost_micros=int(scaled),
        )

    @property
    def is_no_op(self) -> bool:
        return self.before_state_fingerprint == self.after_state_fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "operator_fingerprint": self.operator_fingerprint,
            "semantic_action_id": self.semantic_action_id,
            "application_id": self.application_id,
            "before_state_fingerprint": self.before_state_fingerprint,
            "after_state_fingerprint": self.after_state_fingerprint,
            "before_ast_fingerprint": self.before_ast_fingerprint,
            "after_ast_fingerprint": self.after_ast_fingerprint,
            "locality_violations": self.locality_violations,
            "operator_cost_micros": self.operator_cost_micros,
            "proven_redundant": self.proven_redundant,
        }


@dataclass(frozen=True)
class OperatorSequenceDiagnosticsV1:
    no_op_step_indices: tuple[int, ...]
    cycle_step_indices: tuple[int, ...]
    redundant_step_indices: tuple[int, ...]
    schema: str = "operator_sequence_diagnostics/v1"

    @property
    def defect_count(self) -> int:
        return len(
            set(self.no_op_step_indices)
            | set(self.cycle_step_indices)
            | set(self.redundant_step_indices)
        )

    @property
    def clean(self) -> bool:
        return self.defect_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "no_op_step_indices": list(self.no_op_step_indices),
            "cycle_step_indices": list(self.cycle_step_indices),
            "redundant_step_indices": list(self.redundant_step_indices),
            "defect_count": self.defect_count,
        }


@dataclass(frozen=True)
class OperatorPreferenceSequenceV1:
    initial_state_fingerprint: str
    initial_ast_fingerprint: str
    steps: tuple[OperatorPreferenceStepV1, ...]
    schema: str = "operator_preference_sequence/v1"

    def __post_init__(self) -> None:
        _require_digest(self.initial_state_fingerprint, "initial_state_fingerprint")
        _require_digest(self.initial_ast_fingerprint, "initial_ast_fingerprint")
        expected_state = self.initial_state_fingerprint
        expected_ast = self.initial_ast_fingerprint
        for step in self.steps:
            if (
                step.before_state_fingerprint != expected_state
                or step.before_ast_fingerprint != expected_ast
            ):
                raise ValueError("operator preference sequence is not state-contiguous")
            expected_state = step.after_state_fingerprint
            expected_ast = step.after_ast_fingerprint

    @property
    def final_state_fingerprint(self) -> str:
        if not self.steps:
            return self.initial_state_fingerprint
        return self.steps[-1].after_state_fingerprint

    @property
    def final_ast_fingerprint(self) -> str:
        if not self.steps:
            return self.initial_ast_fingerprint
        return self.steps[-1].after_ast_fingerprint

    @property
    def semantic_fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": "operator_semantic_sequence/v1",
                "semantic_action_ids": [
                    step.semantic_action_id for step in self.steps
                ],
            }
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @property
    def diagnostics(self) -> OperatorSequenceDiagnosticsV1:
        seen_states = {self.initial_state_fingerprint}
        seen_applications: set[str] = set()
        no_ops: list[int] = []
        cycles: list[int] = []
        redundant: list[int] = []
        for index, step in enumerate(self.steps):
            if step.is_no_op:
                no_ops.append(index)
            if step.after_state_fingerprint in seen_states:
                cycles.append(index)
            if step.proven_redundant or step.application_id in seen_applications:
                redundant.append(index)
            seen_states.add(step.after_state_fingerprint)
            seen_applications.add(step.application_id)
        return OperatorSequenceDiagnosticsV1(
            no_op_step_indices=tuple(no_ops),
            cycle_step_indices=tuple(cycles),
            redundant_step_indices=tuple(redundant),
        )

    @property
    def locality_violations(self) -> int:
        return sum(step.locality_violations for step in self.steps)

    @property
    def operator_cost_micros(self) -> int:
        return sum(step.operator_cost_micros for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "initial_ast_fingerprint": self.initial_ast_fingerprint,
            "steps": [step.to_dict() for step in self.steps],
            "final_state_fingerprint": self.final_state_fingerprint,
            "final_ast_fingerprint": self.final_ast_fingerprint,
            "semantic_fingerprint": self.semantic_fingerprint,
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class SemanticPreferenceCandidateV1:
    candidate_id: str
    semantic_frame_fingerprint: str
    equivalence_class_fingerprint: str
    final_ast_fingerprint: str
    required_obligations: tuple[str, ...]
    satisfied_obligations: tuple[str, ...]
    verifier_valid: bool
    ast_cost: CanonicalAstCostV1
    sequence: OperatorPreferenceSequenceV1
    schema: str = "semantic_preference_candidate/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        for field, value in (
            ("semantic_frame_fingerprint", self.semantic_frame_fingerprint),
            ("equivalence_class_fingerprint", self.equivalence_class_fingerprint),
            ("final_ast_fingerprint", self.final_ast_fingerprint),
        ):
            _require_digest(value, field)
        for obligation in (*self.required_obligations, *self.satisfied_obligations):
            _require_identifier(obligation, "semantic obligation")
        if len(set(self.required_obligations)) != len(self.required_obligations):
            raise ValueError("required semantic obligations must be unique")
        if len(set(self.satisfied_obligations)) != len(self.satisfied_obligations):
            raise ValueError("satisfied semantic obligations must be unique")
        if not isinstance(self.verifier_valid, bool):
            raise TypeError("verifier_valid must be boolean")
        if self.sequence.final_ast_fingerprint != self.final_ast_fingerprint:
            raise ValueError("final AST and operator-sequence state disagree")

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self.required_obligations) - set(self.satisfied_obligations))
        )

    @property
    def semantic_complete(self) -> bool:
        return not self.missing_obligations

    @property
    def eligible(self) -> bool:
        return self.semantic_complete and self.verifier_valid

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "semantic_frame_fingerprint": self.semantic_frame_fingerprint,
            "equivalence_class_fingerprint": self.equivalence_class_fingerprint,
            "final_ast_fingerprint": self.final_ast_fingerprint,
            "required_obligations": sorted(self.required_obligations),
            "satisfied_obligations": sorted(self.satisfied_obligations),
            "missing_obligations": list(self.missing_obligations),
            "semantic_complete": self.semantic_complete,
            "verifier_valid": self.verifier_valid,
            "ast_cost": self.ast_cost.to_dict(),
            "sequence": self.sequence.to_dict(),
        }


@dataclass(frozen=True)
class SemanticPreferenceCostV1:
    eligible_penalty: int
    missing_semantic_obligations: int
    verifier_invalid: int
    sequence_defects: int
    ast_nodes: int
    ast_productions: int
    ast_optional_nodes: int
    ast_markers: int
    locality_violations: int
    sequence_length: int
    operator_cost_micros: int
    schema: str = "semantic_preference_cost/v1"

    @property
    def ordering_key(self) -> tuple[int, ...]:
        return (
            self.eligible_penalty,
            self.missing_semantic_obligations,
            self.verifier_invalid,
            self.sequence_defects,
            self.ast_nodes,
            self.ast_productions,
            self.ast_optional_nodes,
            self.ast_markers,
            self.locality_violations,
            self.sequence_length,
            self.operator_cost_micros,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "eligible_penalty": self.eligible_penalty,
            "missing_semantic_obligations": self.missing_semantic_obligations,
            "verifier_invalid": self.verifier_invalid,
            "sequence_defects": self.sequence_defects,
            "ast_nodes": self.ast_nodes,
            "ast_productions": self.ast_productions,
            "ast_optional_nodes": self.ast_optional_nodes,
            "ast_markers": self.ast_markers,
            "locality_violations": self.locality_violations,
            "sequence_length": self.sequence_length,
            "operator_cost_micros": self.operator_cost_micros,
        }


@dataclass(frozen=True)
class PreferenceTieGroupV1:
    rank: int
    candidate_ids: tuple[str, ...]
    cost: SemanticPreferenceCostV1
    schema: str = "preference_tie_group/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rank": self.rank,
            "candidate_ids": list(self.candidate_ids),
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class PreferenceEquivalenceGroupV1:
    fingerprint: str
    candidate_ids: tuple[str, ...]
    schema: str = "preference_equivalence_group/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fingerprint": self.fingerprint,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True)
class OperatorPreferencePairV1:
    chosen_candidate_id: str
    rejected_candidate_id: str
    first_differing_cost_axis: str
    schema: str = "operator_preference_pair/v1"

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "chosen_candidate_id": self.chosen_candidate_id,
            "rejected_candidate_id": self.rejected_candidate_id,
            "first_differing_cost_axis": self.first_differing_cost_axis,
        }


@dataclass(frozen=True)
class OperatorPreferenceGroupV1:
    semantic_frame_fingerprint: str
    equivalence_class_fingerprint: str
    intent: PreferenceIntent
    sequence_defect_policy: SequenceDefectPolicy
    ranked_tiers: tuple[PreferenceTieGroupV1, ...]
    rejected_candidate_ids: tuple[str, ...]
    final_ast_groups: tuple[PreferenceEquivalenceGroupV1, ...]
    semantic_sequence_groups: tuple[PreferenceEquivalenceGroupV1, ...]
    preference_pairs: tuple[OperatorPreferencePairV1, ...]
    schema: str = "operator_preference_group/v1"

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "semantic_frame_fingerprint": self.semantic_frame_fingerprint,
            "equivalence_class_fingerprint": self.equivalence_class_fingerprint,
            "intent": self.intent.value,
            "sequence_defect_policy": self.sequence_defect_policy.value,
            "ranked_tiers": [tier.to_dict() for tier in self.ranked_tiers],
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "final_ast_groups": [group.to_dict() for group in self.final_ast_groups],
            "semantic_sequence_groups": [
                group.to_dict() for group in self.semantic_sequence_groups
            ],
            "preference_pairs": [pair.to_dict() for pair in self.preference_pairs],
        }


def preference_cost(
    candidate: SemanticPreferenceCandidateV1,
    *,
    intent: PreferenceIntent,
    sequence_defect_policy: SequenceDefectPolicy,
) -> SemanticPreferenceCostV1:
    if not isinstance(intent, PreferenceIntent):
        raise TypeError("intent must be PreferenceIntent")
    if not isinstance(sequence_defect_policy, SequenceDefectPolicy):
        raise TypeError("sequence_defect_policy must be SequenceDefectPolicy")
    diagnostics = candidate.sequence.diagnostics
    if (
        sequence_defect_policy is SequenceDefectPolicy.REJECT
        and not diagnostics.clean
    ):
        raise ValueError("defective operator sequence is rejected by policy")
    ast = candidate.ast_cost.directional(intent)
    return SemanticPreferenceCostV1(
        eligible_penalty=int(not candidate.eligible),
        missing_semantic_obligations=len(candidate.missing_obligations),
        verifier_invalid=int(not candidate.verifier_valid),
        sequence_defects=(
            diagnostics.defect_count
            if sequence_defect_policy is SequenceDefectPolicy.PENALIZE
            else 0
        ),
        ast_nodes=ast[0],
        ast_productions=ast[1],
        ast_optional_nodes=ast[2],
        ast_markers=ast[3],
        locality_violations=candidate.sequence.locality_violations,
        sequence_length=len(candidate.sequence.steps),
        operator_cost_micros=candidate.sequence.operator_cost_micros,
    )


def _equivalence_groups(
    candidates: Iterable[SemanticPreferenceCandidateV1],
    *,
    key,
) -> tuple[PreferenceEquivalenceGroupV1, ...]:
    grouped: dict[str, list[str]] = {}
    for candidate in candidates:
        grouped.setdefault(key(candidate), []).append(candidate.candidate_id)
    return tuple(
        PreferenceEquivalenceGroupV1(
            fingerprint=fingerprint,
            candidate_ids=tuple(sorted(candidate_ids)),
        )
        for fingerprint, candidate_ids in sorted(grouped.items())
    )


_COST_AXES = (
    "eligibility",
    "semantic_completeness",
    "verifier_validity",
    "sequence_defects",
    "ast_nodes",
    "ast_productions",
    "ast_optional_nodes",
    "ast_markers",
    "locality_violations",
    "sequence_length",
    "operator_specific_cost",
)


def _first_differing_axis(
    chosen: SemanticPreferenceCostV1, rejected: SemanticPreferenceCostV1
) -> str:
    return next(
        axis
        for axis, left, right in zip(
            _COST_AXES,
            chosen.ordering_key,
            rejected.ordering_key,
            strict=True,
        )
        if left != right
    )


def build_operator_preference_group(
    candidates: Iterable[SemanticPreferenceCandidateV1],
    *,
    intent: PreferenceIntent,
    sequence_defect_policy: SequenceDefectPolicy = SequenceDefectPolicy.REJECT,
    max_pairs: int = 10_000,
) -> OperatorPreferenceGroupV1:
    """Rank only one semantic scope; semantic validity cannot trade for brevity."""
    values = tuple(candidates)
    if not values:
        raise ValueError("preference group requires at least one candidate")
    if not isinstance(intent, PreferenceIntent):
        raise TypeError("intent must be PreferenceIntent")
    if not isinstance(sequence_defect_policy, SequenceDefectPolicy):
        raise TypeError("sequence_defect_policy must be SequenceDefectPolicy")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if len({candidate.candidate_id for candidate in values}) != len(values):
        raise ValueError("preference candidate IDs must be unique")
    frames = {candidate.semantic_frame_fingerprint for candidate in values}
    classes = {candidate.equivalence_class_fingerprint for candidate in values}
    if len(frames) != 1 or len(classes) != 1:
        raise PreferenceScopeError(
            "preference candidates must share one SemanticFrame/equivalence class"
        )

    rejected = tuple(
        sorted(
            candidate.candidate_id
            for candidate in values
            if sequence_defect_policy is SequenceDefectPolicy.REJECT
            and not candidate.sequence.diagnostics.clean
        )
    )
    rejected_set = set(rejected)
    ranked = [candidate for candidate in values if candidate.candidate_id not in rejected_set]
    by_cost: dict[tuple[int, ...], list[SemanticPreferenceCandidateV1]] = {}
    costs: dict[str, SemanticPreferenceCostV1] = {}
    for candidate in ranked:
        cost = preference_cost(
            candidate,
            intent=intent,
            sequence_defect_policy=sequence_defect_policy,
        )
        costs[candidate.candidate_id] = cost
        by_cost.setdefault(cost.ordering_key, []).append(candidate)

    tiers = tuple(
        PreferenceTieGroupV1(
            rank=rank,
            candidate_ids=tuple(
                sorted(candidate.candidate_id for candidate in by_cost[key])
            ),
            cost=costs[by_cost[key][0].candidate_id],
        )
        for rank, key in enumerate(sorted(by_cost), start=1)
    )
    pair_count = sum(
        len(chosen.candidate_ids) * len(lower.candidate_ids)
        for chosen, lower in combinations(tiers, 2)
    )
    if pair_count > max_pairs:
        raise ValueError(
            f"preference pair bound exceeded: {pair_count} > {max_pairs}"
        )
    pairs = tuple(
        OperatorPreferencePairV1(
            chosen_candidate_id=chosen_id,
            rejected_candidate_id=rejected_id,
            first_differing_cost_axis=_first_differing_axis(
                chosen.cost, lower.cost
            ),
        )
        for chosen, lower in combinations(tiers, 2)
        for chosen_id in chosen.candidate_ids
        for rejected_id in lower.candidate_ids
    )
    ordered_values = tuple(sorted(values, key=lambda candidate: candidate.candidate_id))
    return OperatorPreferenceGroupV1(
        semantic_frame_fingerprint=next(iter(frames)),
        equivalence_class_fingerprint=next(iter(classes)),
        intent=intent,
        sequence_defect_policy=sequence_defect_policy,
        ranked_tiers=tiers,
        rejected_candidate_ids=rejected,
        final_ast_groups=_equivalence_groups(
            ordered_values, key=lambda candidate: candidate.final_ast_fingerprint
        ),
        semantic_sequence_groups=_equivalence_groups(
            ordered_values,
            key=lambda candidate: candidate.sequence.semantic_fingerprint,
        ),
        preference_pairs=pairs,
    )


__all__ = [
    "CanonicalAstCostV1",
    "OperatorPreferenceGroupV1",
    "OperatorPreferencePairV1",
    "OperatorPreferenceSequenceV1",
    "OperatorPreferenceStepV1",
    "OperatorSequenceDiagnosticsV1",
    "PreferenceEquivalenceGroupV1",
    "PreferenceIntent",
    "PreferenceScopeError",
    "PreferenceTieGroupV1",
    "SemanticPreferenceCandidateV1",
    "SemanticPreferenceCostV1",
    "SequenceDefectPolicy",
    "build_operator_preference_group",
    "preference_cost",
]
