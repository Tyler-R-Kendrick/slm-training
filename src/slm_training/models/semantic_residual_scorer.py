"""Default-off advisory residual scores over compiler-legal candidates.

Authority rule: keys(Δ) ⊆ A_compiler(s). Complete singletons never invoke
factor projection, propagation, ranking, or neural work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from slm_training.data.progspec.semantic_evidence import (
    FactorRepresentation,
    SemanticEvidenceSnapshotV1,
    build_factor_representation,
    lossy_pairwise_edges,
)
from slm_training.models.semantic_factor_propagation import (
    PropagationConfig,
    build_incidence_matrix,
    propagate_restart,
    split_polarity_channels,
)

__all__ = [
    "AdvisoryScoreResult",
    "SemanticResidualScorerConfig",
    "SemanticResidualScorerV1",
    "assert_score_keys_legal",
]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Map scores to [0,1] by peak so residual can overcome baseline margins."""

    if not scores:
        return scores
    peak = max(abs(v) for v in scores.values())
    if peak <= 0:
        return scores
    return {k: float(v) / peak for k, v in scores.items()}


@dataclass(frozen=True)
class SemanticResidualScorerConfig:
    representation: FactorRepresentation = FactorRepresentation.ROLE_PORTED_FACTOR
    alpha: float = 1.0  # 0.0 => train-on/decode-off
    lambda_restart: float = 0.15
    max_steps: int = 16
    seed: int = 0
    decode_apply: bool = False  # production default-off
    # External scorer_params.defaults (+ optional arm overrides). Empty => built-in
    # fallbacks for unit tests that construct configs directly.
    params: Mapping[str, Any] | None = None

    @classmethod
    def from_external(
        cls,
        *,
        arm: Mapping[str, Any],
        defaults: Mapping[str, Any],
        seed: int = 0,
    ) -> SemanticResidualScorerConfig:
        """Build config from scorer_params.v1.json arm + defaults blocks."""

        return cls(
            representation=FactorRepresentation(str(arm["representation"])),
            alpha=float(arm["alpha"]),
            decode_apply=bool(arm["decode_apply"]),
            lambda_restart=float(
                arm.get("lambda_restart", defaults.get("lambda_restart", 0.15))
            ),
            max_steps=int(arm.get("max_steps", defaults.get("max_steps", 16))),
            seed=int(seed),
            params=dict(defaults),
        )


@dataclass
class AdvisoryScoreResult:
    deltas: dict[str, float]
    applications: int
    choice_changed: bool
    abstained: bool
    singleton_skip: bool
    incomplete_skip: bool
    stale_digest_rejection: bool
    illegal_score_rejection: bool
    unknown_preserved: bool
    notes: list[str] = field(default_factory=list)
    telemetry: dict[str, int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deltas": dict(self.deltas),
            "applications": self.applications,
            "choice_changed": self.choice_changed,
            "abstained": self.abstained,
            "singleton_skip": self.singleton_skip,
            "incomplete_skip": self.incomplete_skip,
            "stale_digest_rejection": self.stale_digest_rejection,
            "illegal_score_rejection": self.illegal_score_rejection,
            "unknown_preserved": self.unknown_preserved,
            "notes": list(self.notes),
            "telemetry": dict(self.telemetry),
        }


def assert_score_keys_legal(
    deltas: Mapping[str, float], legal: Sequence[str]
) -> None:
    legal_set = set(legal)
    illegal = [k for k in deltas if k not in legal_set]
    if illegal:
        raise ValueError(f"score keys outside legal set: {sorted(illegal)}")
    for key, value in deltas.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite score for {key}")


class SemanticResidualScorerV1:
    """Deterministic/zero-parameter residual over factor evidence (first baseline)."""

    def __init__(self, config: SemanticResidualScorerConfig | None = None) -> None:
        self.config = config or SemanticResidualScorerConfig()
        self._params: dict[str, Any] = dict(self.config.params or {})

    @property
    def trainable_parameters(self) -> int:
        return 0

    def _p(self, key: str, default: Any) -> Any:
        return self._params.get(key, default)

    def score(
        self,
        *,
        legal_candidate_ids: Sequence[str],
        coverage: str,
        snapshot: SemanticEvidenceSnapshotV1 | None,
        expected_source_digest: str | None = None,
        baseline_scores: Mapping[str, float] | None = None,
    ) -> AdvisoryScoreResult:
        legal = tuple(str(x) for x in legal_candidate_ids)
        tele: dict[str, int | float] = {
            "semantic_factor_propagation_calls": 0,
            "semantic_factor_ranker_applications": 0,
            "semantic_factor_choice_changes": 0,
            "semantic_factor_abstentions": 0,
            "semantic_factor_singleton_skips": 0,
            "semantic_factor_incomplete_coverage_skips": 0,
            "semantic_factor_stale_digest_rejections": 0,
            "semantic_factor_illegal_score_rejections": 0,
            "semantic_factor_unknown_preserved": 0,
            "neural_forwards": 0,
        }
        # Complete singleton: hard refusal — zero factor/propagation/ranker work.
        if coverage == "complete" and len(legal) == 1:
            tele["semantic_factor_singleton_skips"] = 1
            return AdvisoryScoreResult(
                deltas={},
                applications=0,
                choice_changed=False,
                abstained=True,
                singleton_skip=True,
                incomplete_skip=False,
                stale_digest_rejection=False,
                illegal_score_rejection=False,
                unknown_preserved=False,
                notes=("singleton_zero_work",),
                telemetry=tele,
            )
        if coverage != "complete":
            tele["semantic_factor_incomplete_coverage_skips"] = 1
            tele["semantic_factor_unknown_preserved"] = 1
            return AdvisoryScoreResult(
                deltas={},
                applications=0,
                choice_changed=False,
                abstained=True,
                singleton_skip=False,
                incomplete_skip=True,
                stale_digest_rejection=False,
                illegal_score_rejection=False,
                unknown_preserved=True,
                notes=("unknown_preserved",),
                telemetry=tele,
            )
        if snapshot is None or self.config.representation is FactorRepresentation.NONE:
            tele["semantic_factor_abstentions"] = 1
            return AdvisoryScoreResult(
                deltas={cid: 0.0 for cid in legal},
                applications=0,
                choice_changed=False,
                abstained=True,
                singleton_skip=False,
                incomplete_skip=False,
                stale_digest_rejection=False,
                illegal_score_rejection=False,
                unknown_preserved=False,
                notes=("no_evidence_or_none_arm",),
                telemetry=tele,
            )
        if expected_source_digest is not None and snapshot.source_digest != expected_source_digest:
            tele["semantic_factor_stale_digest_rejections"] = 1
            return AdvisoryScoreResult(
                deltas={},
                applications=0,
                choice_changed=False,
                abstained=True,
                singleton_skip=False,
                incomplete_skip=False,
                stale_digest_rejection=True,
                illegal_score_rejection=False,
                unknown_preserved=False,
                notes=("stale_digest",),
                telemetry=tele,
            )
        if set(snapshot.legal_candidate_ids) != set(legal):
            tele["semantic_factor_illegal_score_rejections"] = 1
            return AdvisoryScoreResult(
                deltas={},
                applications=0,
                choice_changed=False,
                abstained=True,
                singleton_skip=False,
                incomplete_skip=False,
                stale_digest_rejection=False,
                illegal_score_rejection=True,
                unknown_preserved=False,
                notes=("legal_set_mismatch",),
                telemetry=tele,
            )

        alpha = float(self.config.alpha if self.config.decode_apply else 0.0)
        deltas = self._compute_deltas(snapshot, legal)
        try:
            assert_score_keys_legal(deltas, legal)
        except ValueError as exc:
            tele["semantic_factor_illegal_score_rejections"] = 1
            return AdvisoryScoreResult(
                deltas={},
                applications=0,
                choice_changed=False,
                abstained=True,
                singleton_skip=False,
                incomplete_skip=False,
                stale_digest_rejection=False,
                illegal_score_rejection=True,
                unknown_preserved=False,
                notes=(f"illegal:{exc}",),
                telemetry=tele,
            )

        applied = {cid: alpha * float(deltas.get(cid, 0.0)) for cid in legal}
        tele["semantic_factor_ranker_applications"] = 1 if alpha != 0.0 else 0
        applications = 1 if alpha != 0.0 else 0

        baseline = {cid: float((baseline_scores or {}).get(cid, 0.0)) for cid in legal}
        combined = {cid: baseline[cid] + applied[cid] for cid in legal}
        base_choice = max(legal, key=lambda c: (baseline[c], c))
        new_choice = max(legal, key=lambda c: (combined[c], c))
        changed = applications > 0 and base_choice != new_choice
        if changed:
            tele["semantic_factor_choice_changes"] = 1
        if applications == 0:
            tele["semantic_factor_abstentions"] = 1
        return AdvisoryScoreResult(
            deltas=applied,
            applications=applications,
            choice_changed=changed,
            abstained=applications == 0,
            singleton_skip=False,
            incomplete_skip=False,
            stale_digest_rejection=False,
            illegal_score_rejection=False,
            unknown_preserved=False,
            notes=("scored",),
            telemetry=tele,
        )

    def _compute_deltas(
        self,
        snapshot: SemanticEvidenceSnapshotV1,
        legal: Sequence[str],
    ) -> dict[str, float]:
        rep = self.config.representation
        member_roles = set(self._p("member_roles", ["MEMBER", "COMPONENT"]))
        normalize = bool(self._p("normalize_scores", True))

        if rep is FactorRepresentation.EXACT_TYPED_ZERO_PARAMETER:
            mults = dict(
                self._p(
                    "exact_typed_role_multipliers",
                    {"USE": 3.0, "TARGET": 3.0, "DEFINITION": 0.5, "SOURCE": 0.5, "default": 1.0},
                )
            )
            scores = {cid: 0.0 for cid in legal}
            for factor in snapshot.factors:
                for port in factor.ports:
                    if port.node_id not in scores:
                        continue
                    w = float(factor.confidence)
                    w *= float(mults.get(port.role, mults.get("default", 1.0)))
                    scores[port.node_id] += w
            return _normalize_scores(scores) if normalize else scores
        if rep is FactorRepresentation.DIRECT_FACTORS:
            mults = dict(
                self._p(
                    "direct_role_multipliers",
                    {
                        "USE": 4.0,
                        "TARGET": 4.0,
                        "default_non_member": 2.0,
                        "MEMBER": 1.0,
                        "COMPONENT": 1.0,
                    },
                )
            )
            scores = {cid: 0.0 for cid in legal}
            for factor in snapshot.factors:
                weight = float(factor.confidence)
                if factor.polarity.value == "contradicts":
                    weight = -weight
                for port in factor.ports:
                    if port.node_id not in scores:
                        continue
                    if port.role in mults:
                        role_mul = float(mults[port.role])
                    elif port.role not in member_roles:
                        role_mul = float(mults.get("default_non_member", 2.0))
                    else:
                        role_mul = 1.0
                    scores[port.node_id] += weight * role_mul
            return _normalize_scores(scores) if normalize else scores
        if rep is FactorRepresentation.LOSSY_PAIRWISE:
            edge_w = float(self._p("lossy_pairwise_edge_weight", 0.5))
            scores = {cid: 0.0 for cid in legal}
            for a, b in lossy_pairwise_edges(snapshot.factors):
                if a in scores:
                    scores[a] += edge_w
                if b in scores:
                    scores[b] += edge_w
            return _normalize_scores(scores) if normalize else scores
        if rep is FactorRepresentation.ORACLE_DIAGNOSTIC:
            scores = {cid: 0.0 for cid in legal}
            for factor in snapshot.factors:
                for cid in legal:
                    if cid in factor.source_ref or any(
                        cid == p.node_id for p in factor.ports
                    ):
                        scores[cid] += 1.0
            return _normalize_scores(scores) if normalize else scores

        view = build_factor_representation(snapshot, rep, seed=self.config.seed)
        if view is None or not view.vertex_ids:
            return {cid: 0.0 for cid in legal}
        B = build_incidence_matrix(view)
        cfg = PropagationConfig(
            lambda_restart=float(self.config.lambda_restart),
            max_steps=int(self.config.max_steps),
        )
        restart_role_bonus = dict(
            self._p(
                "restart_role_bonus",
                {"USE": 2.0, "TARGET": 2.0, "DEFINITION": 2.0, "SOURCE": 2.0},
            )
        )
        legal_bonus = float(self._p("restart_legal_bonus", 1.5))
        non_member_bonus = float(self._p("restart_non_member_bonus", 1.0))
        contradict_scale = float(self._p("restart_contradict_scale", 0.25))
        r = np.zeros(len(view.vertex_ids), dtype=np.float64)
        legal_set = set(legal)
        for (v_idx, f_idx), role in zip(view.incidence, view.port_roles, strict=False):
            vid = view.vertex_ids[v_idx]
            weight = 1.0
            weight += float(restart_role_bonus.get(role, 0.0))
            if role not in member_roles:
                weight += non_member_bonus
            if vid in legal_set:
                weight += legal_bonus
            if f_idx < len(view.polarity) and view.polarity[f_idx] == "contradicts":
                weight *= contradict_scale
            r[v_idx] += weight
        if float(r.sum()) <= 0:
            r = np.ones(len(view.vertex_ids), dtype=np.float64)

        if any(p == "contradicts" for p in view.polarity):
            _pos, _neg, delta = split_polarity_channels(view, config=cfg)
            conf = delta
        else:
            result = propagate_restart(B, r, config=cfg)
            conf = result.confidence
        v_score = {vid: float(conf[i]) for i, vid in enumerate(view.vertex_ids)}
        role_boost_cfg = dict(
            self._p("role_boost", {"USE": 0.5, "TARGET": 0.5, "default_non_member": 0.25})
        )
        role_boost = {vid: 0.0 for vid in view.vertex_ids}
        for (v_idx, _f_idx), role in zip(view.incidence, view.port_roles, strict=False):
            if role in role_boost_cfg:
                role_boost[view.vertex_ids[v_idx]] += float(role_boost_cfg[role])
            elif role not in member_roles:
                role_boost[view.vertex_ids[v_idx]] += float(
                    role_boost_cfg.get("default_non_member", 0.25)
                )
        scores = {cid: 0.0 for cid in legal}
        for cid in legal:
            scores[cid] = v_score.get(cid, 0.0) + role_boost.get(cid, 0.0)
        return _normalize_scores(scores) if normalize else scores
