"""Compiler-masked hierarchical operator action head (DSH3-15, wiring-only).

Extends the SLM-168 dynamic pointer/choice seam (``DynamicPointerScorer``)
with a two-stage action-kind/operator -> typed-argument decision over a live
``OperatorLegalSetV1``, instead of a flat single-stage scorer. This module
does not create a parallel scorer, a parallel legal-set representation, or a
parallel serialization: every candidate is drawn from
``OperatorLegalSetV1.operator_actions`` (the exact, compiler-verified legal
set) and both stages reuse ``DynamicPointerScorer``.

Gate status (read before wiring this into training): SLM-383's own decision
text authorizes testing this head "only after the discrete token baseline
proves causal value." That baseline is DSH3-14 / SLM-382 (E803,
``docs/design/e803-reserved-operator-baseline-20260723/``), which was
*rejected* -- every arm tied with no causal improvement. The same-day
disposition DSH3-17 / SLM-385 (``docs/design/dsh3-17-cap2-disposition-20260723/``,
``slm_training.evals.cap2_disposition``) already records
``HIERARCHICAL_HEAD`` as ``UNRUN_CONDITIONAL`` for exactly that reason.

AGENTS.md IV.11 notes E803 rejected only the *decoder-visible* token
hypothesis and says nothing about an encoder-side/masked scoring mechanism,
which is what this module implements: a ranking-only lever over an
already-computed legal domain, never a legality or singleton-bypass change.
That leaves the hierarchical-head *hypothesis* open, but this module does
not, by itself, satisfy DSH3-15's acceptance criteria -- it ships wiring and
fixture-evidence tests only (feature-off parity, legal-set-membership
closure, permutation invariance). The multi-seed causal-attribution
experiment (token baseline vs. weight-zero vs. enabled, matched capacity,
``causal_change_rate`` and ``zero_false_legal_admissions`` gates as in
``reserved_operator_baseline.py``) is a follow-up harness run. Do not treat
anything produced here as ship or promotion evidence, and do not flip
``cap2_disposition.py``'s ``HIERARCHICAL_HEAD`` verdict from this module
alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import torch

from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    EffectDeltaKind,
)
from slm_training.dsl.operators.legal_set import OperatorLegalSetV1
from slm_training.dsl.operators.registry import OperatorLibraryV1, OperatorStateV1
from slm_training.dsl.pack import DslPack
from slm_training.models.dynamic_pointer_scorer import (
    DynamicPointerScorer,
    DynamicPointerScorerConfig,
    PointerCandidate,
    PointerCandidateSet,
)

__all__ = [
    "HIERARCHICAL_OPERATOR_HEAD_SCHEMA_VERSION",
    "HierarchicalOperatorHeadConfigV1",
    "HierarchicalOperatorHeadMode",
    "HierarchicalDecisionKind",
    "OperatorActionFeatureViewV1",
    "OperatorGroupFeatureViewV1",
    "HierarchicalOperatorDecisionV1",
    "HierarchicalOperatorActionHead",
    "build_operator_action_features",
    "group_operator_features",
]

HIERARCHICAL_OPERATOR_HEAD_SCHEMA_VERSION = "hierarchical_operator_head/v1"

HierarchicalOperatorHeadMode = Literal["off", "dynamic_head"]

_EFFECT_KIND_ORDER: tuple[EffectDeltaKind, ...] = (
    EffectDeltaKind.CARDINALITY,
    EffectDeltaKind.PROPERTY,
    EffectDeltaKind.SCOPE,
    EffectDeltaKind.TOPOLOGY,
)


class HierarchicalDecisionKind(str, Enum):
    """Compiler-masked disposition for one hierarchical head decision."""

    APPLY = "apply"
    DEFER = "defer"


@dataclass(frozen=True)
class HierarchicalOperatorHeadConfigV1:
    """Pinned, default-off configuration for the hierarchical operator head.

    ``mode="off"`` (the default) makes both stages true no-ops: the module
    always defers to the ordinary decoder/search path and builds zero
    trainable parameters, mirroring ``DynamicPointerScorerConfig``'s
    ``pointer_mode="legacy_tokens"`` contract.
    """

    mode: HierarchicalOperatorHeadMode = "off"
    d_model: int = 128
    hidden_dim: int = 256
    heads: int = 4
    temperature: float = 1.0
    dropout: float = 0.0
    schema: str = HIERARCHICAL_OPERATOR_HEAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mode not in ("off", "dynamic_head"):
            raise ValueError(f"unknown hierarchical operator head mode: {self.mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "temperature": self.temperature,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalOperatorHeadConfigV1":
        return cls(
            mode=str(data.get("mode", "off")),  # type: ignore[arg-type]
            d_model=int(data.get("d_model", 128)),
            hidden_dim=int(data.get("hidden_dim", 256)),
            heads=int(data.get("heads", 4)),
            temperature=float(data.get("temperature", 1.0)),
            dropout=float(data.get("dropout", 0.0)),
        )


@dataclass(frozen=True)
class OperatorActionFeatureViewV1:
    """Typed, display-free features for one live ``LegalOperatorActionV1``.

    Every field is derived from ``ActionEffectV1`` counts and compiler-owned
    coverage/cost -- never from an opaque id, a request id, or a display
    name. ``semantic_id`` (not ``application_id``) is the permutation-
    invariant identity: it is stable under reference-table re-permutation
    (``ReferenceTableV1.permuted``) because it is derived from each
    argument's semantic fingerprint, not its opaque id.
    """

    operator_id: str
    semantic_id: str
    application_id: str
    scope_delta_count: int
    cardinality_delta_count: int
    property_delta_count: int
    topology_delta_count: int
    consumed_role_count: int
    produced_role_count: int
    consumed_binder_count: int
    produced_binder_count: int
    compiler_coverage: str
    estimated_completion_cost: float
    schema: str = "operator_action_feature_view/v1"

    @property
    def dominant_effect_kind(self) -> str:
        counts = {
            EffectDeltaKind.CARDINALITY: self.cardinality_delta_count,
            EffectDeltaKind.PROPERTY: self.property_delta_count,
            EffectDeltaKind.SCOPE: self.scope_delta_count,
            EffectDeltaKind.TOPOLOGY: self.topology_delta_count,
        }
        return max(_EFFECT_KIND_ORDER, key=lambda kind: counts[kind]).value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "semantic_id": self.semantic_id,
            "application_id": self.application_id,
            "scope_delta_count": self.scope_delta_count,
            "cardinality_delta_count": self.cardinality_delta_count,
            "property_delta_count": self.property_delta_count,
            "topology_delta_count": self.topology_delta_count,
            "consumed_role_count": self.consumed_role_count,
            "produced_role_count": self.produced_role_count,
            "consumed_binder_count": self.consumed_binder_count,
            "produced_binder_count": self.produced_binder_count,
            "compiler_coverage": self.compiler_coverage,
            "estimated_completion_cost": self.estimated_completion_cost,
        }


@dataclass(frozen=True)
class OperatorGroupFeatureViewV1:
    """Per-operator rollup of ``OperatorActionFeatureViewV1`` members."""

    operator_id: str
    candidate_count: int
    dominant_effect_kind: str
    total_scope_deltas: int
    total_cardinality_deltas: int
    total_property_deltas: int
    total_topology_deltas: int
    schema: str = "operator_group_feature_view/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "candidate_count": self.candidate_count,
            "dominant_effect_kind": self.dominant_effect_kind,
            "total_scope_deltas": self.total_scope_deltas,
            "total_cardinality_deltas": self.total_cardinality_deltas,
            "total_property_deltas": self.total_property_deltas,
            "total_topology_deltas": self.total_topology_deltas,
        }


@dataclass(frozen=True)
class HierarchicalOperatorDecisionV1:
    """One two-stage compiler-masked decision over a live legal set."""

    legal_set_fingerprint: str
    decision: HierarchicalDecisionKind
    mode: str
    operator_id: str | None
    application_id: str | None
    operator_candidate_ids: tuple[str, ...]
    operator_scores: tuple[float, ...]
    argument_candidate_ids: tuple[str, ...]
    argument_scores: tuple[float, ...]
    schema: str = "hierarchical_operator_decision/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "legal_set_fingerprint": self.legal_set_fingerprint,
            "decision": self.decision.value,
            "mode": self.mode,
            "operator_id": self.operator_id,
            "application_id": self.application_id,
            "operator_candidate_ids": list(self.operator_candidate_ids),
            "operator_scores": list(self.operator_scores),
            "argument_candidate_ids": list(self.argument_candidate_ids),
            "argument_scores": list(self.argument_scores),
        }


def build_operator_action_features(
    *,
    legal_set: OperatorLegalSetV1,
    library: OperatorLibraryV1,
    pack: DslPack,
    state: OperatorStateV1,
    provenance: ApplicationProvenanceV1,
) -> tuple[OperatorActionFeatureViewV1, ...]:
    """Derive typed features for every live legal operator action.

    Candidates are constructed solely from ``legal_set.operator_actions`` --
    this function never invents an action and never drops one. Re-running
    ``library.dry_run`` is pure (no mutation of ``state``) and must replay
    the exact ``application_id`` already recorded on the legal action, or
    the legal set is stale relative to ``library``/``pack``/``state`` and
    construction fails closed rather than silently admitting drift.
    """
    features: list[OperatorActionFeatureViewV1] = []
    for action in legal_set.operator_actions:
        application = library.dry_run(
            pack, state, action.operator_id, action.arguments, provenance
        )
        if not application.succeeded:
            raise ValueError(
                f"legal action {action.application_id} failed to replay as legal"
            )
        if application.application_id != action.application_id:
            raise ValueError(
                f"legal action {action.application_id} did not replay identically"
            )
        effect = application.effect
        assert effect is not None
        features.append(
            OperatorActionFeatureViewV1(
                operator_id=action.operator_id,
                semantic_id=action.semantic_id,
                application_id=action.application_id,
                scope_delta_count=len(effect.scope_deltas),
                cardinality_delta_count=len(effect.cardinality_deltas),
                property_delta_count=len(effect.property_deltas),
                topology_delta_count=len(effect.topology_deltas),
                consumed_role_count=len(effect.consumed_roles),
                produced_role_count=len(effect.produced_roles),
                consumed_binder_count=len(effect.consumed_binders),
                produced_binder_count=len(effect.produced_binders),
                compiler_coverage=effect.compiler_coverage.value,
                estimated_completion_cost=effect.estimated_completion_cost,
            )
        )
    return tuple(features)


def group_operator_features(
    features: tuple[OperatorActionFeatureViewV1, ...],
) -> tuple[OperatorGroupFeatureViewV1, ...]:
    """Roll typed action features up to one summary per ``operator_id``."""
    by_operator: dict[str, list[OperatorActionFeatureViewV1]] = {}
    for feature in features:
        by_operator.setdefault(feature.operator_id, []).append(feature)
    groups: list[OperatorGroupFeatureViewV1] = []
    for operator_id in sorted(by_operator):
        members = by_operator[operator_id]
        counts = {
            EffectDeltaKind.CARDINALITY: sum(m.cardinality_delta_count for m in members),
            EffectDeltaKind.PROPERTY: sum(m.property_delta_count for m in members),
            EffectDeltaKind.SCOPE: sum(m.scope_delta_count for m in members),
            EffectDeltaKind.TOPOLOGY: sum(m.topology_delta_count for m in members),
        }
        dominant = max(_EFFECT_KIND_ORDER, key=lambda kind: counts[kind])
        groups.append(
            OperatorGroupFeatureViewV1(
                operator_id=operator_id,
                candidate_count=len(members),
                dominant_effect_kind=dominant.value,
                total_scope_deltas=counts[EffectDeltaKind.SCOPE],
                total_cardinality_deltas=counts[EffectDeltaKind.CARDINALITY],
                total_property_deltas=counts[EffectDeltaKind.PROPERTY],
                total_topology_deltas=counts[EffectDeltaKind.TOPOLOGY],
            )
        )
    return tuple(groups)


def _action_display_text(feature: OperatorActionFeatureViewV1) -> str:
    """Typed, opaque-free signal derived only from ``ActionEffectV1`` counts."""
    return (
        f"scope={feature.scope_delta_count}"
        f";cardinality={feature.cardinality_delta_count}"
        f";property={feature.property_delta_count}"
        f";topology={feature.topology_delta_count}"
        f";consumed_roles={feature.consumed_role_count}"
        f";produced_roles={feature.produced_role_count}"
        f";consumed_binders={feature.consumed_binder_count}"
        f";produced_binders={feature.produced_binder_count}"
        f";coverage={feature.compiler_coverage}"
    )


def _group_display_text(group: OperatorGroupFeatureViewV1) -> str:
    return (
        f"candidates={group.candidate_count}"
        f";scope={group.total_scope_deltas}"
        f";cardinality={group.total_cardinality_deltas}"
        f";property={group.total_property_deltas}"
        f";topology={group.total_topology_deltas}"
    )


def _manifest_hash(candidates: tuple[PointerCandidate, ...]) -> str:
    payload = json.dumps([c.to_dict() for c in candidates], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _build_stage_one_candidates(
    groups: tuple[OperatorGroupFeatureViewV1, ...],
) -> PointerCandidateSet:
    """Stage 1 (action-kind/operator) candidates, keyed by ``operator_id``."""
    candidates = tuple(
        PointerCandidate(
            stable_id=group.operator_id,
            display_text=_group_display_text(group),
            kind="schema_entity",
            type_name=group.dominant_effect_kind,
            provenance="compiler_scope",
        )
        for group in sorted(groups, key=lambda g: g.operator_id)
    )
    return PointerCandidateSet(
        candidates=candidates,
        permitted_sources=("structured_contract",),
        manifest_hash=_manifest_hash(candidates),
    )


def _build_stage_two_candidates(
    features: tuple[OperatorActionFeatureViewV1, ...],
) -> PointerCandidateSet:
    """Stage 2 (typed-argument) candidates, keyed by permutation-invariant semantic id."""
    candidates = tuple(
        PointerCandidate(
            stable_id=feature.semantic_id,
            display_text=_action_display_text(feature),
            kind="schema_entity",
            type_name=feature.operator_id,
            provenance="compiler_scope",
        )
        for feature in sorted(features, key=lambda f: f.semantic_id)
    )
    return PointerCandidateSet(
        candidates=candidates,
        permitted_sources=("structured_contract",),
        manifest_hash=_manifest_hash(candidates),
    )


class HierarchicalOperatorActionHead:
    """Two-stage compiler-masked operator action head over a live legal set.

    Reuses ``DynamicPointerScorer`` for both stages instead of introducing a
    parallel scorer. In ``mode="off"`` (default), the head still builds both
    stages' candidate sets and shadow-scores them (zero logits, since the
    underlying scorer is a no-op in that mode) but the decision is always
    ``DEFER`` -- legal-set membership, exact-singleton force emission, and
    ordinary decode/search behavior are completely untouched.
    """

    def __init__(
        self,
        config: HierarchicalOperatorHeadConfigV1 | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or HierarchicalOperatorHeadConfigV1()
        self.device = str(device)
        self._scorer = DynamicPointerScorer(
            DynamicPointerScorerConfig(
                pointer_mode=(
                    "dynamic_head" if self.config.mode == "dynamic_head" else "legacy_tokens"
                ),
                pointer_candidate_source="structured_contract",
                d_model=self.config.d_model,
                pointer_hidden_dim=self.config.hidden_dim,
                pointer_heads=self.config.heads,
                pointer_temperature=self.config.temperature,
                dropout=self.config.dropout,
            ),
            device=device,
        )

    @property
    def enabled(self) -> bool:
        return self.config.mode == "dynamic_head"

    def decide(
        self,
        state_vector: torch.Tensor,
        legal_set: OperatorLegalSetV1,
        features: tuple[OperatorActionFeatureViewV1, ...],
    ) -> HierarchicalOperatorDecisionV1:
        """Return a hierarchical decision; always DEFER when ``mode="off"``."""
        if any(
            feature.application_id
            not in {action.application_id for action in legal_set.operator_actions}
            for feature in features
        ):
            raise ValueError("candidate features are not members of the live legal set")

        if not features:
            return HierarchicalOperatorDecisionV1(
                legal_set_fingerprint=legal_set.fingerprint,
                decision=HierarchicalDecisionKind.DEFER,
                mode=self.config.mode,
                operator_id=None,
                application_id=None,
                operator_candidate_ids=(),
                operator_scores=(),
                argument_candidate_ids=(),
                argument_scores=(),
            )

        groups = group_operator_features(features)
        stage_one = _build_stage_one_candidates(groups)
        stage_one_logits = self._scorer.forward(state_vector, stage_one)

        if not self.enabled:
            stage_two = _build_stage_two_candidates(features)
            stage_two_logits = self._scorer.forward(state_vector, stage_two)
            return HierarchicalOperatorDecisionV1(
                legal_set_fingerprint=legal_set.fingerprint,
                decision=HierarchicalDecisionKind.DEFER,
                mode=self.config.mode,
                operator_id=None,
                application_id=None,
                operator_candidate_ids=tuple(c.stable_id for c in stage_one.candidates),
                operator_scores=tuple(float(x) for x in stage_one_logits.tolist()),
                argument_candidate_ids=tuple(c.stable_id for c in stage_two.candidates),
                argument_scores=tuple(float(x) for x in stage_two_logits.tolist()),
            )

        top_group = int(torch.argmax(stage_one_logits).item())
        chosen_operator_id = stage_one.candidates[top_group].stable_id
        stage_two_features = tuple(
            f for f in features if f.operator_id == chosen_operator_id
        )
        stage_two = _build_stage_two_candidates(stage_two_features)
        stage_two_logits = self._scorer.forward(state_vector, stage_two)
        top_argument = int(torch.argmax(stage_two_logits).item())
        chosen = stage_two.candidates[top_argument]
        chosen_application_id = next(
            f.application_id
            for f in stage_two_features
            if f.semantic_id == chosen.stable_id
        )
        return HierarchicalOperatorDecisionV1(
            legal_set_fingerprint=legal_set.fingerprint,
            decision=HierarchicalDecisionKind.APPLY,
            mode=self.config.mode,
            operator_id=chosen_operator_id,
            application_id=chosen_application_id,
            operator_candidate_ids=tuple(c.stable_id for c in stage_one.candidates),
            operator_scores=tuple(float(x) for x in stage_one_logits.tolist()),
            argument_candidate_ids=tuple(c.stable_id for c in stage_two.candidates),
            argument_scores=tuple(float(x) for x in stage_two_logits.tolist()),
        )
