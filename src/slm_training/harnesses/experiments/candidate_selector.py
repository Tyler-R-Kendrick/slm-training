"""SLM-127 EFS3-04 contract-grounded candidate selector with calibrated abstention.

Wiring/fixture harness only. The learned head is a tiny MLP over per-candidate
features; real generalization requires a trained OpenUI checkpoint and a labeled
candidate corpus. No ship claim, no real checkpoint training, no GPU run.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from slm_training.lineage.records import content_sha

# Reuse the existing calibration helper from the judge-independence audit.
from slm_training.evals.judge_independence import calibration_error

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be absent in minimal environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

__all__ = [
    "CANDIDATE_SELECTOR_SCHEMA",
    "CandidateSelectionGroupV1",
    "CandidateSelector",
    "ContractSelectorScorer",
    "EnergyScoreSelector",
    "HardThenSimpleSelector",
    "LearnedCandidateSelector",
    "ModelScoreSelector",
    "SelectionCandidate",
    "SelectionDecision",
    "ThresholdManifestV1",
    "ValueScoreSelector",
    "brier_score",
    "evaluate_selector",
    "expected_calibration_error",
    "load_selection_groups",
    "make_fixture_groups",
    "risk_coverage_curve",
    "select_threshold_on_validation",
    "selection_group_from_dict",
    "selection_group_to_dict",
    "train_selector_fixture",
    "write_selection_groups",
]

CANDIDATE_SELECTOR_SCHEMA = "CandidateSelectionGroupV1"


@dataclass(frozen=True)
class SelectionCandidate:
    """One generated candidate with contract-relevant scores and features."""

    candidate_id: str
    canonical_program: str
    ast_fingerprint: str
    generator_id: str
    generator_score: float | None
    value_score: float | None
    energy_score: float | None
    semantic_success: bool | None
    acceptable_set: bool
    available_features: Mapping[str, float | int | bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "canonical_program": self.canonical_program,
            "ast_fingerprint": self.ast_fingerprint,
            "generator_id": self.generator_id,
            "generator_score": self.generator_score,
            "value_score": self.value_score,
            "energy_score": self.energy_score,
            "semantic_success": self.semantic_success,
            "acceptable_set": self.acceptable_set,
            "available_features": dict(self.available_features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SelectionCandidate:
        return cls(
            candidate_id=str(value["candidate_id"]),
            canonical_program=str(value["canonical_program"]),
            ast_fingerprint=str(value["ast_fingerprint"]),
            generator_id=str(value["generator_id"]),
            generator_score=_optional_float(value.get("generator_score")),
            value_score=_optional_float(value.get("value_score")),
            energy_score=_optional_float(value.get("energy_score")),
            semantic_success=_optional_bool(value.get("semantic_success")),
            acceptable_set=bool(value.get("acceptable_set", False)),
            available_features=dict(value.get("available_features") or {}),
        )


@dataclass(frozen=True)
class SelectionDecision:
    """Outcome of applying a selector to one candidate set."""

    selected_candidate_id: str | None
    abstained: bool
    fallback_policy: str
    predicted_success: float | None
    utility_scores: tuple[float, ...]
    selector_id: str
    threshold_id: str
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "abstained": self.abstained,
            "fallback_policy": self.fallback_policy,
            "predicted_success": self.predicted_success,
            "utility_scores": list(self.utility_scores),
            "selector_id": self.selector_id,
            "threshold_id": self.threshold_id,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SelectionDecision:
        return cls(
            selected_candidate_id=_optional_str(value.get("selected_candidate_id")),
            abstained=bool(value["abstained"]),
            fallback_policy=str(value["fallback_policy"]),
            predicted_success=_optional_float(value.get("predicted_success")),
            utility_scores=tuple(float(x) for x in value.get("utility_scores") or ()),
            selector_id=str(value["selector_id"]),
            threshold_id=str(value["threshold_id"]),
            reason_code=str(value["reason_code"]),
        )


@dataclass(frozen=True)
class CandidateSelectionGroupV1:
    """A prompt/contract group with a bounded candidate set and oracle labels."""

    group_id: str
    prompt_hash: str
    contract_hash: str
    generator_id: str
    checkpoint_sha: str
    seed: int
    k: int
    candidates: tuple[SelectionCandidate, ...]
    acceptable_set: tuple[str, ...]
    oracle_best_id: str | None
    split: str
    unknown_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be train/validation/test, got {self.split!r}")
        if self.k != len(self.candidates):
            raise ValueError(f"k={self.k} does not match candidate count {len(self.candidates)}")
        ids = {c.candidate_id for c in self.candidates}
        if len(ids) != len(self.candidates):
            raise ValueError("candidate_id values must be unique within a group")
        if any(cid not in ids for cid in self.acceptable_set):
            raise ValueError("acceptable_set must contain candidate_ids present in candidates")
        if self.oracle_best_id is not None and self.oracle_best_id not in ids:
            raise ValueError("oracle_best_id must be a candidate_id in candidates")

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["candidates"] = [c.to_dict() for c in self.candidates]
        data["acceptable_set"] = list(self.acceptable_set)
        data["unknown_fields"] = list(self.unknown_fields)
        data["schema"] = CANDIDATE_SELECTOR_SCHEMA
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateSelectionGroupV1:
        if value.get("schema") != CANDIDATE_SELECTOR_SCHEMA:
            raise ValueError(f"unsupported group schema: {value.get('schema')!r}")
        fields = dict(value)
        fields.pop("schema", None)
        fields["candidates"] = tuple(
            SelectionCandidate.from_dict(c) for c in fields.get("candidates", ())
        )
        fields["acceptable_set"] = tuple(fields.get("acceptable_set") or ())
        fields["unknown_fields"] = tuple(fields.get("unknown_fields") or ())
        return cls(**fields)


@dataclass(frozen=True)
class ThresholdManifestV1:
    """Calibrated abstention threshold from a validation sweep."""

    threshold_id: str
    selector_id: str
    calibration_set_hash: str
    metric_label_version: str
    threshold: float
    validation_coverage: float
    validation_risk: float
    policy: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_id": self.threshold_id,
            "selector_id": self.selector_id,
            "calibration_set_hash": self.calibration_set_hash,
            "metric_label_version": self.metric_label_version,
            "threshold": self.threshold,
            "validation_coverage": self.validation_coverage,
            "validation_risk": self.validation_risk,
            "policy": self.policy,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ThresholdManifestV1:
        return cls(
            threshold_id=str(value["threshold_id"]),
            selector_id=str(value["selector_id"]),
            calibration_set_hash=str(value["calibration_set_hash"]),
            metric_label_version=str(value["metric_label_version"]),
            threshold=float(value["threshold"]),
            validation_coverage=float(value["validation_coverage"]),
            validation_risk=float(value["validation_risk"]),
            policy=str(value["policy"]),
            timestamp=str(value["timestamp"]),
        )


@runtime_checkable
class CandidateSelector(Protocol):
    """Protocol for a selector that scores and picks one candidate per group."""

    @property
    def selector_id(self) -> str:
        ...

    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision:
        ...


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _safe_number(value: float | None, *, default: float = 0.0) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return float(value)


class _BaseScoreSelector:
    """Shared score fallback and deterministic tie-breaking."""

    def _pick(
        self,
        candidates: Sequence[SelectionCandidate],
        scores: Sequence[float | None],
        reason_code: str,
    ) -> tuple[SelectionCandidate | None, str, str]:
        if not candidates:
            return None, "empty_candidate_set", "empty_candidate_set"
        safe = [
            (score, index)
            for index, score in enumerate(scores)
            if score is not None and not math.isnan(score)
        ]
        if not safe:
            winner = sorted(candidates, key=lambda c: c.candidate_id)[0]
            return winner, "missing_score_fallback", "missing_score_fallback"
        # Highest score, tie-break by candidate_id for determinism.
        best_index = max(
            safe,
            key=lambda item: (item[0], candidates[item[1]].candidate_id),
        )[1]
        return candidates[best_index], reason_code, reason_code


class ModelScoreSelector(_BaseScoreSelector):
    """Select the candidate with the highest generator_score."""

    selector_id = "model_score"

    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision:
        winner, reason, fallback = self._pick(
            candidates,
            [c.generator_score for c in candidates],
            "selected_model_score",
        )
        if winner is None:
            return SelectionDecision(
                selected_candidate_id=None,
                abstained=True,
                fallback_policy=fallback,
                predicted_success=None,
                utility_scores=(),
                selector_id=self.selector_id,
                threshold_id="",
                reason_code=reason,
            )
        return SelectionDecision(
            selected_candidate_id=winner.candidate_id,
            abstained=False,
            fallback_policy=fallback,
            predicted_success=None,
            utility_scores=(),
            selector_id=self.selector_id,
            threshold_id="",
            reason_code=reason,
        )


class ValueScoreSelector(_BaseScoreSelector):
    """Select the candidate with the highest value_score."""

    selector_id = "value_score"

    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision:
        winner, reason, fallback = self._pick(
            candidates,
            [c.value_score for c in candidates],
            "selected_value_score",
        )
        if winner is None:
            return SelectionDecision(
                selected_candidate_id=None,
                abstained=True,
                fallback_policy=fallback,
                predicted_success=None,
                utility_scores=(),
                selector_id=self.selector_id,
                threshold_id="",
                reason_code=reason,
            )
        return SelectionDecision(
            selected_candidate_id=winner.candidate_id,
            abstained=False,
            fallback_policy=fallback,
            predicted_success=None,
            utility_scores=(),
            selector_id=self.selector_id,
            threshold_id="",
            reason_code=reason,
        )


class EnergyScoreSelector(_BaseScoreSelector):
    """Select the candidate with the highest energy_score."""

    selector_id = "energy_score"

    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision:
        winner, reason, fallback = self._pick(
            candidates,
            [c.energy_score for c in candidates],
            "selected_energy_score",
        )
        if winner is None:
            return SelectionDecision(
                selected_candidate_id=None,
                abstained=True,
                fallback_policy=fallback,
                predicted_success=None,
                utility_scores=(),
                selector_id=self.selector_id,
                threshold_id="",
                reason_code=reason,
            )
        return SelectionDecision(
            selected_candidate_id=winner.candidate_id,
            abstained=False,
            fallback_policy=fallback,
            predicted_success=None,
            utility_scores=(),
            selector_id=self.selector_id,
            threshold_id="",
            reason_code=reason,
        )


class HardThenSimpleSelector:
    """Prefer hard-verified successes, then generator score, then candidate_id."""

    selector_id = "hard_then_simple"

    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision:
        if not candidates:
            return SelectionDecision(
                selected_candidate_id=None,
                abstained=True,
                fallback_policy="empty_candidate_set",
                predicted_success=None,
                utility_scores=(),
                selector_id=self.selector_id,
                threshold_id="",
                reason_code="empty_candidate_set",
            )
        hard = [c for c in candidates if c.semantic_success is True]
        pool = hard if hard else candidates
        fallback = "hard_then_simple" if hard else "hard_then_simple_fallback"
        winner = sorted(
            pool,
            key=lambda c: (
                _safe_number(c.generator_score, default=float("-inf")),
                c.candidate_id,
            ),
            reverse=True,
        )[0]
        return SelectionDecision(
            selected_candidate_id=winner.candidate_id,
            abstained=False,
            fallback_policy=fallback,
            predicted_success=None,
            utility_scores=(),
            selector_id=self.selector_id,
            threshold_id="",
            reason_code="selected_hard_then_simple",
        )


# Define the torch-backed scorer only when torch is available. In torch-free
# environments the class still exists but raises on instantiation.
if nn is not None:

    class ContractSelectorScorer(nn.Module):
        """Tiny MLP: per-candidate utility, success, and optional set-success logits."""

        def __init__(
            self,
            input_dim: int = 5,
            hidden_dim: int = 16,
            *,
            include_feature_count: bool = True,
            include_set_has_success: bool = False,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.include_feature_count = include_feature_count
            self.include_set_has_success = include_set_has_success
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.utility_head = nn.Linear(hidden_dim, 1)
            self.success_head = nn.Linear(hidden_dim, 1)
            if include_set_has_success:
                self.set_success_head = nn.Linear(hidden_dim, 1)

        def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            h = torch.relu(self.fc1(x))
            h = torch.relu(self.fc2(h))
            outputs: dict[str, torch.Tensor] = {
                "utility_logit": self.utility_head(h),
                "contract_success_logit": self.success_head(h),
            }
            if self.include_set_has_success:
                outputs["set_has_success_logit"] = self.set_success_head(h)
            return outputs

else:  # pragma: no cover

    class ContractSelectorScorer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "ContractSelectorScorer requires torch; install torch to use the learned selector"
            )


def _candidate_feature_vector(
    candidate: SelectionCandidate,
    k: int,
    *,
    include_feature_count: bool = True,
) -> list[float]:
    vec = [
        _safe_number(candidate.generator_score),
        _safe_number(candidate.value_score),
        _safe_number(candidate.energy_score),
        float(k),
    ]
    if include_feature_count:
        vec.append(float(len(candidate.available_features)))
    return vec


class LearnedCandidateSelector:
    """Select using a trained ContractSelectorScorer and calibrated threshold."""

    selector_id = "learned_candidate_selector"

    def __init__(
        self,
        scorer: ContractSelectorScorer,
        *,
        threshold_manifest: ThresholdManifestV1 | None = None,
        include_feature_count: bool = True,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("LearnedCandidateSelector requires torch")
        self.scorer = scorer
        self.threshold_manifest = threshold_manifest
        self.include_feature_count = include_feature_count

    def candidate_success_scores(
        self,
        candidates: Sequence[SelectionCandidate],
    ) -> list[float]:
        """Return per-candidate predicted success probabilities (no threshold)."""
        if not candidates:
            return []
        k = len(candidates)
        features = [
            _candidate_feature_vector(c, k, include_feature_count=self.include_feature_count)
            for c in candidates
        ]
        tensor = torch.tensor(features, dtype=torch.float32)
        self.scorer.eval()
        with torch.no_grad():
            outputs = self.scorer(tensor)
        logits = outputs["contract_success_logit"].squeeze(-1)
        probs = torch.sigmoid(logits)
        return [float(p.item()) for p in probs]

    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision:
        if not candidates:
            return SelectionDecision(
                selected_candidate_id=None,
                abstained=True,
                fallback_policy="empty_candidate_set",
                predicted_success=None,
                utility_scores=(),
                selector_id=self.selector_id,
                threshold_id=self.threshold_manifest.threshold_id if self.threshold_manifest else "",
                reason_code="empty_candidate_set",
            )
        scores = self.candidate_success_scores(candidates)
        max_index = max(range(len(scores)), key=lambda i: (scores[i], candidates[i].candidate_id))
        max_score = scores[max_index]
        utility_scores = self._utility_scores(candidates)
        threshold = self.threshold_manifest.threshold if self.threshold_manifest else 0.0
        if self.threshold_manifest is None or max_score > threshold:
            return SelectionDecision(
                selected_candidate_id=candidates[max_index].candidate_id,
                abstained=False,
                fallback_policy=self.threshold_manifest.policy if self.threshold_manifest else "no_abstain",
                predicted_success=max_score,
                utility_scores=utility_scores,
                selector_id=self.selector_id,
                threshold_id=self.threshold_manifest.threshold_id if self.threshold_manifest else "",
                reason_code="selected",
            )
        return SelectionDecision(
            selected_candidate_id=None,
            abstained=True,
            fallback_policy="abstain",
            predicted_success=max_score,
            utility_scores=utility_scores,
            selector_id=self.selector_id,
            threshold_id=self.threshold_manifest.threshold_id if self.threshold_manifest else "",
            reason_code="abstained_risk",
        )

    def _utility_scores(self, candidates: Sequence[SelectionCandidate]) -> tuple[float, ...]:
        if not candidates:
            return ()
        k = len(candidates)
        features = [
            _candidate_feature_vector(c, k, include_feature_count=self.include_feature_count)
            for c in candidates
        ]
        tensor = torch.tensor(features, dtype=torch.float32)
        self.scorer.eval()
        with torch.no_grad():
            outputs = self.scorer(tensor)
        return tuple(float(u.item()) for u in outputs["utility_logit"].squeeze(-1))


def make_fixture_groups(
    *,
    n_groups: int = 8,
    k: int = 4,
    seed: int = 0,
) -> tuple[CandidateSelectionGroupV1, ...]:
    """Generate a tiny synthetic corpus for wiring tests."""
    rng = random.Random(seed)
    generators = ("genA", "genB")
    groups: list[CandidateSelectionGroupV1] = []
    for group_index in range(n_groups):
        split = "train" if group_index < n_groups // 2 else (
            "validation" if group_index < 3 * n_groups // 4 else "test"
        )
        primary_generator = generators[group_index % 2]
        candidates: list[SelectionCandidate] = []
        acceptable_ids: list[str] = []
        # Every group gets at most one acceptable candidate, except a few
        # no-positive groups to exercise abstention.
        has_positive = group_index % 5 != 0
        acceptable_index = rng.randrange(k) if has_positive else -1
        for candidate_index in range(k):
            candidate_id = f"g{group_index}_c{candidate_index}"
            generator_id = generators[candidate_index % 2]
            is_acceptable = candidate_index == acceptable_index
            if is_acceptable:
                acceptable_ids.append(candidate_id)
            # Hard-failed candidates are marked semantic_success=False and given
            # low scores so even score-only baselines avoid them in this fixture.
            hard_failed = not is_acceptable and candidate_index % 3 == 0
            semantic_success = True if is_acceptable else (False if hard_failed else None)
            if is_acceptable:
                generator_score = 0.9 + rng.uniform(0.0, 0.09)
                value_score = 0.85 + rng.uniform(0.0, 0.1)
                energy_score = 0.88 + rng.uniform(0.0, 0.08)
            elif hard_failed:
                generator_score = 0.05 + rng.uniform(0.0, 0.05)
                value_score = 0.05 + rng.uniform(0.0, 0.05)
                energy_score = 0.05 + rng.uniform(0.0, 0.05)
            else:
                generator_score = 0.4 + rng.uniform(0.0, 0.2)
                value_score = 0.4 + rng.uniform(0.0, 0.2)
                energy_score = 0.4 + rng.uniform(0.0, 0.2)
            candidates.append(
                SelectionCandidate(
                    candidate_id=candidate_id,
                    canonical_program=f"(program {candidate_id})",
                    ast_fingerprint=f"fp_{candidate_id}",
                    generator_id=generator_id,
                    generator_score=generator_score,
                    value_score=value_score,
                    energy_score=energy_score,
                    semantic_success=semantic_success,
                    acceptable_set=is_acceptable,
                    available_features={"group_index": group_index, "hard": hard_failed},
                )
            )
        prompt = f"prompt for group {group_index}"
        contract = f"contract for group {group_index}"
        groups.append(
            CandidateSelectionGroupV1(
                group_id=f"fixture_group_{group_index}",
                prompt_hash=content_sha(prompt),
                contract_hash=content_sha(contract),
                generator_id=primary_generator,
                checkpoint_sha=content_sha({"seed": seed, "group": group_index}),
                seed=seed,
                k=k,
                candidates=tuple(candidates),
                acceptable_set=tuple(acceptable_ids),
                oracle_best_id=acceptable_ids[0] if acceptable_ids else None,
                split=split,
            )
        )
    return tuple(groups)


def brier_score(predicted: Sequence[float], actual: Sequence[float]) -> float | None:
    """Mean squared error between predicted probabilities and binary outcomes."""
    if len(predicted) != len(actual) or not predicted:
        return None
    return sum((p - float(a)) ** 2 for p, a in zip(predicted, actual)) / len(predicted)


def expected_calibration_error(
    predicted: Sequence[float],
    actual: Sequence[float],
    *,
    bins: int = 10,
) -> float | None:
    """Expected calibration error using the judge-independence binning helper."""
    if len(predicted) != len(actual) or not predicted:
        return None
    return calibration_error(predicted, [bool(a) for a in actual], bins=bins)


def _group_max_success_scores(
    selector: CandidateSelector,
    candidates: Sequence[SelectionCandidate],
) -> list[float]:
    """Return per-candidate success scores if the selector exposes them."""
    if isinstance(selector, LearnedCandidateSelector):
        return selector.candidate_success_scores(candidates)
    return [0.0] * len(candidates)


def select_threshold_on_validation(
    groups: Sequence[CandidateSelectionGroupV1],
    selector: CandidateSelector,
    *,
    target_risk: float = 0.05,
    metric_label_version: str = "candidate_acceptability_v1",
) -> ThresholdManifestV1:
    """Sweep thresholds on validation groups and pick the lowest risk-bounded one."""
    if not isinstance(selector, LearnedCandidateSelector):
        raise TypeError("threshold calibration requires a LearnedCandidateSelector")
    val_groups = [g for g in groups if g.split == "validation"]
    calibration_set_hash = content_sha(
        [selection_group_to_dict(g) for g in val_groups]
    )
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not val_groups:
        return ThresholdManifestV1(
            threshold_id=f"{selector.selector_id}_empty_validation",
            selector_id=selector.selector_id,
            calibration_set_hash=calibration_set_hash,
            metric_label_version=metric_label_version,
            threshold=1.0,
            validation_coverage=0.0,
            validation_risk=0.0,
            policy="risk_lte_target",
            timestamp=timestamp,
        )
    group_data: list[tuple[float, bool]] = []
    for group in val_groups:
        scores = selector.candidate_success_scores(group.candidates)
        max_index = max(range(len(scores)), key=lambda i: scores[i])
        selected_acceptable = group.candidates[max_index].candidate_id in group.acceptable_set
        group_data.append((scores[max_index], selected_acceptable))
    thresholds = sorted({score for score, _ in group_data})
    best_threshold = 1.0
    best_coverage = 0.0
    best_risk = 0.0
    for threshold in thresholds:
        selected = [acceptable for score, acceptable in group_data if score >= threshold]
        n_selected = len(selected)
        risk = (n_selected - sum(selected)) / n_selected if n_selected else 0.0
        coverage = n_selected / len(val_groups)
        if risk <= target_risk:
            best_threshold = threshold
            best_coverage = coverage
            best_risk = risk
            break
    return ThresholdManifestV1(
        threshold_id=f"{selector.selector_id}_{calibration_set_hash[:8]}",
        selector_id=selector.selector_id,
        calibration_set_hash=calibration_set_hash,
        metric_label_version=metric_label_version,
        threshold=best_threshold,
        validation_coverage=best_coverage,
        validation_risk=best_risk,
        policy="risk_lte_target",
        timestamp=timestamp,
    )


def risk_coverage_curve(
    groups: Sequence[CandidateSelectionGroupV1],
    selector: CandidateSelector,
) -> list[dict[str, float]]:
    """Return (threshold, coverage, risk, n_selected) points over all groups."""
    if not isinstance(selector, LearnedCandidateSelector):
        return []
    if not groups:
        return []
    group_data: list[tuple[float, bool]] = []
    for group in groups:
        scores = selector.candidate_success_scores(group.candidates)
        max_index = max(range(len(scores)), key=lambda i: scores[i])
        selected_acceptable = group.candidates[max_index].candidate_id in group.acceptable_set
        group_data.append((scores[max_index], selected_acceptable))
    all_thresholds = sorted({score for score, _ in group_data} | {0.0, 1.0})
    curve: list[dict[str, float]] = []
    for threshold in all_thresholds:
        selected = [acceptable for score, acceptable in group_data if score >= threshold]
        n_selected = len(selected)
        risk = (n_selected - sum(selected)) / n_selected if n_selected else 0.0
        coverage = n_selected / len(groups)
        curve.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "risk": risk,
                "n_selected": float(n_selected),
            }
        )
    return curve


def evaluate_selector(
    groups: Sequence[CandidateSelectionGroupV1],
    selector: CandidateSelector,
    *,
    threshold_manifest: ThresholdManifestV1 | None = None,
) -> dict[str, Any]:
    """Evaluate a selector across groups and return the metric bundle."""
    if not groups:
        return {
            "selector_id": selector.selector_id,
            "pass_at_k": 0.0,
            "selected_pass_at_k": 0.0,
            "selector_regret": 0.0,
            "top1_acceptable_recall": 0.0,
            "top1_ndcg": 0.0,
            "abstention_rate": 0.0,
            "invalid_over_valid_count": 0,
            "cross_generator_groups": 0,
            "n_groups": 0,
        }
    decisions = [
        selector.select(
            prompt_context={"group_id": group.group_id},
            structured_contract={"contract_hash": group.contract_hash},
            candidates=group.candidates,
        )
        for group in groups
    ]
    pass_at_k = sum(1 for g in groups if g.acceptable_set) / len(groups)
    selections = [
        (decision, group)
        for decision, group in zip(decisions, groups)
        if decision.selected_candidate_id is not None
    ]
    selected_acceptable = sum(
        1 for decision, group in selections
        if decision.selected_candidate_id in group.acceptable_set
    )
    selected_total = len(selections)
    selected_pass_at_k = selected_acceptable / selected_total if selected_total else 0.0
    oracle_positive = [g for g in groups if g.acceptable_set]
    regrets = 0
    invalid_over_valid = 0
    group_to_decision = {group.group_id: decision for group, decision in zip(groups, decisions)}
    for group in oracle_positive:
        decision = group_to_decision[group.group_id]
        if decision.selected_candidate_id is None or decision.selected_candidate_id not in group.acceptable_set:
            regrets += 1
        if decision.selected_candidate_id is not None and decision.selected_candidate_id not in group.acceptable_set:
            invalid_over_valid += 1
    selector_regret = regrets / len(oracle_positive) if oracle_positive else 0.0
    top1_acceptable_recall = selected_acceptable / len(oracle_positive) if oracle_positive else 0.0
    top1_ndcg = (
        sum(
            1
            for decision, group in selections
            if decision.selected_candidate_id in group.acceptable_set
        )
        / len(groups)
    )
    abstention_rate = sum(1 for d in decisions if d.abstained) / len(groups)
    cross_generator_groups = sum(
        1
        for g in groups
        if len({c.generator_id for c in g.candidates}) > 1
    )
    result: dict[str, Any] = {
        "selector_id": selector.selector_id,
        "threshold_id": threshold_manifest.threshold_id if threshold_manifest else None,
        "pass_at_k": pass_at_k,
        "selected_pass_at_k": selected_pass_at_k,
        "selector_regret": selector_regret,
        "top1_acceptable_recall": top1_acceptable_recall,
        "top1_ndcg": top1_ndcg,
        "abstention_rate": abstention_rate,
        "invalid_over_valid_count": invalid_over_valid,
        "cross_generator_groups": cross_generator_groups,
        "n_groups": len(groups),
    }
    predicted = [
        decision.predicted_success
        for decision in decisions
        if decision.predicted_success is not None
    ]
    actual = [
        1.0
        if group_to_decision[group.group_id].selected_candidate_id in group.acceptable_set
        else 0.0
        for group in groups
        if group_to_decision[group.group_id].predicted_success is not None
    ]
    if predicted:
        result["brier_score"] = brier_score(predicted, actual)
        result["expected_calibration_error"] = expected_calibration_error(predicted, actual)
    return result


def train_selector_fixture(
    groups: Sequence[CandidateSelectionGroupV1],
    *,
    hidden_dim: int = 16,
    lr: float = 0.05,
    epochs: int = 20,
    seed: int = 0,
) -> tuple[ContractSelectorScorer, dict[str, Any]]:
    """Train a tiny ContractSelectorScorer on train-split groups."""
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("train_selector_fixture requires torch")
    torch.manual_seed(seed)
    train_groups = [g for g in groups if g.split == "train"]
    include_feature_count = True
    input_dim = 5
    model = ContractSelectorScorer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        include_feature_count=include_feature_count,
        include_set_has_success=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for group in train_groups:
            k = len(group.candidates)
            xs = [
                _candidate_feature_vector(
                    c, k, include_feature_count=include_feature_count
                )
                for c in group.candidates
            ]
            labels = [
                1.0 if c.candidate_id in group.acceptable_set else 0.0
                for c in group.candidates
            ]
            x = torch.tensor(xs, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32)
            outputs = model(x)
            loss = bce(outputs["contract_success_logit"].squeeze(-1), y)
            acc_idx = [i for i, label in enumerate(labels) if label > 0.5]
            rej_idx = [i for i, label in enumerate(labels) if label <= 0.5]
            if acc_idx and rej_idx:
                util = outputs["utility_logit"].squeeze(-1)
                pos = util[acc_idx].unsqueeze(1)
                neg = util[rej_idx].unsqueeze(0)
                pair_loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
                loss = loss + pair_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(epoch_loss / max(n_batches, 1), 6),
            }
        )
    return model, {
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "history": history,
        "input_dim": input_dim,
        "include_feature_count": include_feature_count,
    }


def selection_group_to_dict(group: CandidateSelectionGroupV1) -> dict[str, Any]:
    return group.to_dict()


def selection_group_from_dict(value: Mapping[str, Any]) -> CandidateSelectionGroupV1:
    return CandidateSelectionGroupV1.from_dict(value)


def write_selection_groups(
    path: str,
    groups: Sequence[CandidateSelectionGroupV1],
) -> None:
    """Write groups as deterministic JSONL."""
    with open(path, "w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(
                json.dumps(
                    selection_group_to_dict(group),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )


def load_selection_groups(path: str) -> tuple[CandidateSelectionGroupV1, ...]:
    """Load groups from JSONL."""
    groups: list[CandidateSelectionGroupV1] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            groups.append(selection_group_from_dict(json.loads(line)))
    return tuple(groups)
