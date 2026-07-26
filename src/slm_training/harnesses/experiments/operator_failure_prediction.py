"""Leak-sealed temporal rows for SLM-406 shadow failure prediction.

The owner accepts only the prefix-time projection emitted by ``eval_runner``.
Terminal outcomes are labels, never feature fields, and this module never
changes decode routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from slm_training.harnesses.model_build.decode_outcome import MODEL_VALID

FEATURE_NAMES = ("normalized_position", "legal_candidates", "forced")
FORBIDDEN_FEATURES = frozenset(
    {
        "parse_ok",
        "meaningful_program_v1",
        "binding_aware_meaningful_v2",
        "semantic_meaning_report_v2",
        "error",
        "decode_outcome",
        "stop_reason",
        "fallback_used",
        "timeout",
        "compiler_lattice_termination_reason",
    }
)


@dataclass(frozen=True)
class OperatorFailureFeatureRowV1:
    """One prefix-time observation with an evaluator-only terminal label."""

    request_id: str
    group_id: str
    checkpoint_sha: str
    position: int
    features: dict[str, float]
    outcome: str
    failure: bool
    schema: str = "operator_failure_feature_row/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_operator_failure_feature_rows(
    details: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sha: str,
    canvas_cap: int,
    target_cluster_by_id: Mapping[str, str] | None = None,
) -> tuple[OperatorFailureFeatureRowV1, ...]:
    """Project per-record prefix traces into leak-free shadow rows.

    ``details`` comes from the canonical evaluator.  The terminal
    ``decode_outcome`` is read only after projection to form the label.  The
    grouping key combines the request/target cluster and checkpoint so no
    split can mix related trajectories or checkpoints.
    """
    if not checkpoint_sha:
        raise ValueError("checkpoint_sha is required")
    if canvas_cap <= 0:
        raise ValueError("canvas_cap must be positive")
    rows: list[OperatorFailureFeatureRowV1] = []
    for detail in details:
        request_id = str(detail.get("id") or "")
        outcome = str(detail.get("decode_outcome") or "")
        if not request_id or not outcome:
            raise ValueError("detail must provide id and decode_outcome")
        cluster = (target_cluster_by_id or {}).get(request_id, request_id)
        for trace in detail.get("temporal_decode_evidence") or ():
            if not isinstance(trace, Mapping):
                raise ValueError("temporal decode evidence must be mappings")
            forbidden = FORBIDDEN_FEATURES.intersection(trace)
            if forbidden:
                raise ValueError(
                    "temporal decode evidence contains forbidden fields: "
                    + ", ".join(sorted(forbidden))
                )
            position = trace.get("position")
            if not isinstance(position, int) or position < 0:
                raise ValueError("temporal decode evidence requires position >= 0")
            legal_candidates = trace.get("legal_candidates", 0)
            if not isinstance(legal_candidates, int) or legal_candidates < 0:
                raise ValueError("legal_candidates must be a non-negative integer")
            rows.append(
                OperatorFailureFeatureRowV1(
                    request_id=request_id,
                    group_id=f"{cluster}:{checkpoint_sha}",
                    checkpoint_sha=checkpoint_sha,
                    position=position,
                    features={
                        "normalized_position": min(1.0, (position + 1) / canvas_cap),
                        "legal_candidates": float(legal_candidates),
                        "forced": float(bool(trace.get("forced", False))),
                    },
                    outcome=outcome,
                    failure=outcome != MODEL_VALID,
                )
            )
    return tuple(rows)


def assert_group_disjoint_splits(
    rows: Sequence[OperatorFailureFeatureRowV1],
    split_by_group: Mapping[str, str],
) -> None:
    """Fail if a trajectory group is absent or appears in multiple splits."""
    seen: dict[str, str] = {}
    for row in rows:
        split = split_by_group.get(row.group_id)
        if split not in {"train", "validation", "test"}:
            raise ValueError("every feature group requires train/validation/test split")
        previous = seen.setdefault(row.group_id, split)
        if previous != split:
            raise ValueError("feature group crosses splits")


__all__ = [
    "FEATURE_NAMES",
    "FORBIDDEN_FEATURES",
    "OperatorFailureFeatureRowV1",
    "assert_group_disjoint_splits",
    "build_operator_failure_feature_rows",
]
