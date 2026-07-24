"""SLM-230 recurrence observability and evaluation-only exit contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

OBSERVABILITY_SCHEMA = "RecurrenceObservabilityV1"
EXIT_POLICY_SCHEMA = "RecurrenceExitPolicyV1"
REPORT_SCHEMA = "RecurrenceObservabilityReportV1"
INFERENCE_SIGNALS = frozenset(
    {
        "full_kl_from_previous",
        "legal_kl_from_previous",
        "topk_stable",
        "entropy",
        "d_legal",
        "grammar_valid",
    }
)
ORACLE_SIGNALS = frozenset(
    {
        "target_cross_entropy",
        "target_accuracy",
        "reward_score",
        "structural_similarity",
        "d_good",
    }
)


class RecurrenceVerdict(str, Enum):
    REFINING = "refining"
    WEIGHT_SHARING_ONLY = "weight_sharing_only"
    STAGNANT = "stagnant"
    OSCILLATORY = "oscillatory"
    UNSTABLE = "unstable"
    INCONCLUSIVE = "inconclusive"


class ExitMode(str, Enum):
    FIXED = "fixed"
    KL_PLATEAU = "kl_plateau"
    TOPK_STABLE = "topk_stable"
    DEBT_PLATEAU = "debt_plateau"
    LEARNED_TINY_CALIBRATOR = "learned_tiny_calibrator"
    ORACLE = "oracle"


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RecurrenceExitPolicyV1:
    mode: ExitMode
    minimum_depth: int
    maximum_depth: int
    consecutive_depths: int = 1
    kl_threshold: float | None = None
    topk_stability_k: int | None = None
    debt_threshold: float | None = None
    calibrator_artifact_hash: str | None = None
    fallback_depth: int | None = None
    allowed_signals: tuple[str, ...] = ()
    calibration_split_hash: str | None = None
    diagnostic_only: bool = True
    schema: str = EXIT_POLICY_SCHEMA

    def validate(self) -> None:
        if self.schema != EXIT_POLICY_SCHEMA:
            raise ValueError(f"unsupported exit-policy schema: {self.schema}")
        if self.minimum_depth < 1 or self.maximum_depth < self.minimum_depth:
            raise ValueError("exit-policy depth bounds are invalid")
        if self.consecutive_depths < 1:
            raise ValueError("consecutive_depths must be >= 1")
        fallback = self.fallback_depth or self.maximum_depth
        if not self.minimum_depth <= fallback <= self.maximum_depth:
            raise ValueError("fallback_depth must be inside the policy bounds")
        signals = frozenset(self.allowed_signals)
        if self.mode is not ExitMode.ORACLE and signals & ORACLE_SIGNALS:
            raise ValueError(
                "production-like recurrence exits cannot consume gold, D_good, "
                "reward, structural, or evaluator outcomes"
            )
        if self.mode is not ExitMode.ORACLE and not signals <= INFERENCE_SIGNALS:
            raise ValueError("exit policy declares an unsupported inference signal")
        if self.mode is ExitMode.KL_PLATEAU:
            if self.kl_threshold is None or self.kl_threshold < 0:
                raise ValueError("kl_plateau requires a non-negative threshold")
            if not signals & {
                "full_kl_from_previous",
                "legal_kl_from_previous",
            }:
                raise ValueError("kl_plateau requires a declared KL signal")
        if self.mode is ExitMode.TOPK_STABLE:
            if self.topk_stability_k is None or self.topk_stability_k < 1:
                raise ValueError("topk_stable requires topk_stability_k >= 1")
            if "topk_stable" not in signals:
                raise ValueError("topk_stable requires the topk_stable signal")
        if self.mode is ExitMode.DEBT_PLATEAU:
            if self.debt_threshold is None or self.debt_threshold < 0:
                raise ValueError("debt_plateau requires a non-negative threshold")
            if "d_legal" not in signals:
                raise ValueError("debt_plateau requires inference-available D_legal")
        if (
            self.mode is ExitMode.LEARNED_TINY_CALIBRATOR
            and not self.calibrator_artifact_hash
        ):
            raise ValueError("learned calibrator requires a frozen artifact hash")
        if self.mode is not ExitMode.FIXED and self.mode is not ExitMode.ORACLE:
            if not self.calibration_split_hash:
                raise ValueError("adaptive exits require a frozen calibration split")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["allowed_signals"] = list(self.allowed_signals)
        return payload


@dataclass(frozen=True)
class RecurrenceObservabilityV1:
    checkpoint_sha256: str
    model_config_hash: str
    tokenizer_hash: str
    decode_config_hash: str
    trained_recurrence_depth: int
    evaluated_depth: int
    test_time_extrapolation: bool
    record_id: str
    split: str
    suite: str
    request_fingerprint: str
    candidate_set_hash: str
    candidate_set_scope: str
    target_count: int
    full_entropy: float
    full_top_ids: tuple[int, ...]
    full_top1_stable: bool | None
    full_kl_from_previous: float | None
    full_js_from_previous: float | None
    logit_cosine_from_previous: float | None
    logit_l2_from_previous: float | None
    legal_kl_from_previous: float | None
    legal_js_from_previous: float | None
    target_cross_entropy: float
    target_accuracy: float
    y_residual_norm: float
    z_residual_norm: float | None
    grammar_valid: bool
    decoded_output_sha256: str
    decoded_output: str
    structural_similarity: float
    reward_score: float | None
    latency_ms: float
    forwards: int
    block_evaluations: int
    numerical_status: str
    good_bad_status: str
    runtime_symbol_hash: str | None
    schema: str = OBSERVABILITY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["full_top_ids"] = list(self.full_top_ids)
        return payload


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if not selected.numel():
        raise ValueError("observability mask must select at least one position")
    return float(selected.float().mean().detach().cpu())


def distribution_metrics(
    logits: torch.Tensor,
    *,
    mask: torch.Tensor,
    previous_logits: torch.Tensor | None,
    top_k: int = 5,
    legal_candidate_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute deterministic full and frozen-candidate recurrence metrics."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocabulary]")
    if mask.shape != logits.shape[:1] or mask.dtype is not torch.bool:
        raise ValueError("mask must be a bool tensor over token positions")
    if previous_logits is not None and previous_logits.shape != logits.shape:
        raise ValueError("previous_logits must match logits")
    if top_k < 1 or top_k > logits.size(-1):
        raise ValueError("top_k is outside the vocabulary")

    current_logp = F.log_softmax(logits.detach().float(), dim=-1)
    current_p = current_logp.exp()
    entropy = -(current_p * current_logp).sum(dim=-1)
    top_ids = current_logp.topk(top_k, dim=-1).indices
    selected_top_ids = top_ids[mask]
    top_id_summary = tuple(
        int(value)
        for value in selected_top_ids.mode(dim=0).values.detach().cpu().tolist()
    )
    result: dict[str, Any] = {
        "entropy": _masked_mean(entropy, mask),
        "top_ids": top_id_summary,
        "top1_stable": None,
        "full_kl": None,
        "full_js": None,
        "logit_cosine": None,
        "logit_l2": None,
        "legal_kl": None,
        "legal_js": None,
    }
    if previous_logits is None:
        return result

    previous_logp = F.log_softmax(previous_logits.detach().float(), dim=-1)
    previous_p = previous_logp.exp()
    kl = (current_p * (current_logp - previous_logp)).sum(dim=-1)
    midpoint = (current_p + previous_p) * 0.5
    midpoint_log = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    js = 0.5 * (
        (current_p * (current_logp - midpoint_log)).sum(dim=-1)
        + (previous_p * (previous_logp - midpoint_log)).sum(dim=-1)
    )
    current_selected = logits.detach().float()[mask]
    previous_selected = previous_logits.detach().float()[mask]
    result.update(
        {
            "top1_stable": bool(
                torch.equal(
                    current_logp.argmax(dim=-1)[mask],
                    previous_logp.argmax(dim=-1)[mask],
                )
            ),
            "full_kl": _masked_mean(kl, mask),
            "full_js": _masked_mean(js, mask),
            "logit_cosine": float(
                F.cosine_similarity(
                    current_selected.flatten(),
                    previous_selected.flatten(),
                    dim=0,
                ).cpu()
            ),
            "logit_l2": float(
                (current_selected - previous_selected).norm().detach().cpu()
            ),
        }
    )
    if legal_candidate_ids is not None:
        if (
            legal_candidate_ids.ndim != 1
            or not legal_candidate_ids.numel()
            or legal_candidate_ids.unique().numel() != legal_candidate_ids.numel()
        ):
            raise ValueError("legal_candidate_ids must be a nonempty unique vector")
        current_legal = logits.index_select(-1, legal_candidate_ids)
        previous_legal = previous_logits.index_select(-1, legal_candidate_ids)
        current_legal_logp = F.log_softmax(current_legal.float(), dim=-1)
        previous_legal_logp = F.log_softmax(previous_legal.float(), dim=-1)
        current_legal_p = current_legal_logp.exp()
        previous_legal_p = previous_legal_logp.exp()
        legal_kl = (
            current_legal_p * (current_legal_logp - previous_legal_logp)
        ).sum(dim=-1)
        legal_midpoint = (current_legal_p + previous_legal_p) * 0.5
        legal_midpoint_log = legal_midpoint.clamp_min(
            torch.finfo(legal_midpoint.dtype).tiny
        ).log()
        legal_js = 0.5 * (
            (
                current_legal_p
                * (current_legal_logp - legal_midpoint_log)
            ).sum(dim=-1)
            + (
                previous_legal_p
                * (previous_legal_logp - legal_midpoint_log)
            ).sum(dim=-1)
        )
        result["legal_kl"] = _masked_mean(legal_kl, mask)
        result["legal_js"] = _masked_mean(legal_js, mask)
    return result


def select_exit_depth(
    observations: Sequence[Mapping[str, Any]],
    policy: RecurrenceExitPolicyV1,
) -> int:
    """Select one depth without changing compiler or verifier authority."""
    policy.validate()
    rows = [
        row
        for row in observations
        if policy.minimum_depth
        <= int(row["evaluated_depth"])
        <= policy.maximum_depth
    ]
    if not rows:
        raise ValueError("no observations fall inside the exit-policy bounds")
    rows.sort(key=lambda row: int(row["evaluated_depth"]))
    if policy.mode is ExitMode.FIXED:
        return min(policy.maximum_depth, int(rows[-1]["evaluated_depth"]))
    if policy.mode is ExitMode.ORACLE:
        return min(
            rows,
            key=lambda row: (
                -float(row.get("reward_score") or 0.0),
                -float(row.get("structural_similarity") or 0.0),
                int(row["evaluated_depth"]),
            ),
        )["evaluated_depth"]

    consecutive = 0
    for row in rows:
        depth = int(row["evaluated_depth"])
        if depth <= policy.minimum_depth:
            continue
        satisfied = False
        if policy.mode is ExitMode.KL_PLATEAU:
            key = (
                "legal_kl_from_previous"
                if "legal_kl_from_previous" in policy.allowed_signals
                else "full_kl_from_previous"
            )
            value = row.get(key)
            satisfied = value is not None and float(value) <= float(
                policy.kl_threshold
            )
        elif policy.mode is ExitMode.TOPK_STABLE:
            satisfied = bool(row.get("full_top1_stable"))
        elif policy.mode is ExitMode.DEBT_PLATEAU:
            value = row.get("d_legal")
            satisfied = value is not None and float(value) <= float(
                policy.debt_threshold
            )
        elif policy.mode is ExitMode.LEARNED_TINY_CALIBRATOR:
            raise RuntimeError(
                "the frozen calibrator runtime is not implemented in SLM-230"
            )
        consecutive = consecutive + 1 if satisfied else 0
        if consecutive >= policy.consecutive_depths:
            return depth
    return policy.fallback_depth or policy.maximum_depth


def histogram_matched_control(
    selected_depths: Sequence[int], *, record_ids: Sequence[str]
) -> dict[str, int]:
    """Deterministic time-shuffled control with an identical exit histogram."""
    if len(selected_depths) != len(record_ids) or not selected_depths:
        raise ValueError("selected_depths and record_ids must be nonempty and aligned")
    ordered_ids = sorted(str(value) for value in record_ids)
    rotated = list(selected_depths[1:]) + [int(selected_depths[0])]
    return dict(zip(ordered_ids, rotated, strict=True))


def classify_recurrence(
    rows: Sequence[Mapping[str, Any]],
    *,
    heldout_record_ids: Sequence[str],
    early_exit_qualified: bool,
) -> RecurrenceVerdict:
    """Apply the bounded SLM-230 recurrence verdict without syntax inflation."""
    heldout = [
        row for row in rows if str(row["record_id"]) in set(heldout_record_ids)
    ]
    if not heldout or any(row.get("numerical_status") != "finite" for row in heldout):
        return RecurrenceVerdict.INCONCLUSIVE
    by_record: dict[str, list[Mapping[str, Any]]] = {}
    for row in heldout:
        by_record.setdefault(str(row["record_id"]), []).append(row)
    if any(len(values) < 2 for values in by_record.values()):
        return RecurrenceVerdict.INCONCLUSIVE

    reward_curves = []
    kl_values = []
    for values in by_record.values():
        values.sort(key=lambda row: int(row["evaluated_depth"]))
        reward_curves.append([float(row.get("reward_score") or 0.0) for row in values])
        kl_values.extend(
            float(row["full_kl_from_previous"])
            for row in values[1:]
            if row.get("full_kl_from_previous") is not None
        )
    if not kl_values:
        return RecurrenceVerdict.INCONCLUSIVE
    nondegrading = all(
        all(after + 1e-9 >= before for before, after in zip(curve, curve[1:]))
        for curve in reward_curves
    )
    improving = any(curve[-1] > curve[0] + 1e-6 for curve in reward_curves)
    regressions = sum(
        after + 1e-9 < before
        for curve in reward_curves
        for before, after in zip(curve, curve[1:])
    )
    if max(kl_values) < 1e-8 and not improving:
        return RecurrenceVerdict.WEIGHT_SHARING_ONLY
    if not all(math.isfinite(value) for value in kl_values):
        return RecurrenceVerdict.UNSTABLE
    if regressions and any(
        any(
            (after - before) * (later - after) < 0
            for before, after, later in zip(curve, curve[1:], curve[2:])
        )
        for curve in reward_curves
    ):
        return RecurrenceVerdict.OSCILLATORY
    if improving and nondegrading and early_exit_qualified:
        return RecurrenceVerdict.REFINING
    return RecurrenceVerdict.STAGNANT


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != REPORT_SCHEMA:
        errors.append("wrong report schema")
    rows = report.get("observations")
    if not isinstance(rows, list) or not rows:
        errors.append("observations are required")
    else:
        for index, row in enumerate(rows):
            if row.get("schema") != OBSERVABILITY_SCHEMA:
                errors.append(f"observation {index} has wrong schema")
            if row.get("test_time_extrapolation"):
                errors.append(f"observation {index} cannot enter the default report")
    policies = report.get("exit_policies")
    if not isinstance(policies, list) or not policies:
        errors.append("exit policies are required")
    else:
        for index, value in enumerate(policies):
            try:
                policy = RecurrenceExitPolicyV1(
                    **{
                        **value,
                        "mode": ExitMode(value["mode"]),
                        "allowed_signals": tuple(value.get("allowed_signals", ())),
                    }
                )
                policy.validate()
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"exit policy {index} is invalid: {exc}")
    if set(report.get("split_manifests", {})) != {"calibration", "heldout"}:
        errors.append("calibration and heldout manifests are required")
    elif (
        set(report["split_manifests"]["calibration"]["record_ids"])
        & set(report["split_manifests"]["heldout"]["record_ids"])
    ):
        errors.append("calibration and heldout records overlap")
    expected_hash = report.get("report_hash")
    if expected_hash:
        payload = dict(report)
        payload.pop("report_hash", None)
        payload.pop("generated_at", None)
        stamp = dict(payload.get("version_stamp") or {})
        stamp.pop("stamped_at", None)
        payload["version_stamp"] = stamp
        if stable_hash(payload) != expected_hash:
            errors.append("report_hash mismatch")
    return errors
