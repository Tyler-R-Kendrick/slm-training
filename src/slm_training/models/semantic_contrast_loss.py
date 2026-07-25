"""Read-only semantic-contrast pairs and their default-off margin objective."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.schema import ExampleRecord

_META_KEY = "semantic_contrast"


@dataclass(frozen=True)
class SemanticContrastTrainingPair:
    """One admitted positive/negative training pair from the immutable corpus."""

    pair_id: str
    positive: ExampleRecord
    negative: ExampleRecord
    family: str
    transform_id: str


@dataclass(frozen=True)
class SemanticContrastMargin:
    """Differentiable pairwise margin and scalar diagnostics."""

    loss: torch.Tensor
    violation_rate: torch.Tensor
    score_distance: torch.Tensor


def pairwise_margin_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    margin: float,
) -> SemanticContrastMargin:
    """Require each positive sequence score to exceed its contrast by ``margin``."""
    if positive_scores.ndim != 1 or positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative scores must be matching rank-1 tensors")
    if positive_scores.numel() == 0:
        raise ValueError("semantic contrast objective requires at least one complete pair")
    if margin < 0.0:
        raise ValueError("semantic contrast margin must be non-negative")
    distance = positive_scores - negative_scores
    violations = torch.relu(float(margin) - distance)
    return SemanticContrastMargin(
        loss=violations.mean(),
        violation_rate=violations.gt(0).float().mean(),
        score_distance=distance.mean(),
    )


def load_semantic_contrast_pairs(path: Path) -> list[SemanticContrastTrainingPair]:
    """Load only admitted, train-split pairs without reclassifying their corpus."""
    pairs_path = path / "pairs.jsonl" if path.is_dir() else path
    pairs: list[SemanticContrastTrainingPair] = []
    with pairs_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("admitted") is not True:
                    continue
                positive_row = dict(row["positive"])
                negative_row = dict(row["negative"])
                if positive_row.get("role") != "positive" or negative_row.get("role") != "negative":
                    raise ValueError("pair roles must be positive then negative")
                positive = ExampleRecord.from_dict(positive_row["record"])
                negative = ExampleRecord.from_dict(negative_row["record"])
                if positive.split != "train" or negative.split != "train":
                    # The immutable artifact carries held-out/OOD diagnostic
                    # rows too; never relabel them into training supervision.
                    continue
                if positive.prompt != negative.prompt:
                    raise ValueError("paired records must share their prompt")
                source_program_id = str(row["source_program_id"])
                if str(positive_row["source_program_id"]) != source_program_id:
                    raise ValueError("positive source_program_id mismatch")
                if str(negative_row["source_program_id"]) != source_program_id:
                    raise ValueError("negative source_program_id mismatch")
                pair_id = str(row["pair_id"])
                family = str(row["family"])
                transform_id = str(row["transform_id"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{pairs_path}:{line_no}: invalid semantic contrast pair: {exc}") from exc
            common: dict[str, Any] = {
                "pair_id": pair_id,
                "family": family,
                "transform_id": transform_id,
                "source_program_id": source_program_id,
            }
            pairs.append(
                SemanticContrastTrainingPair(
                    pair_id=pair_id,
                    positive=replace(positive, meta={**positive.meta, _META_KEY: {**common, "role": "positive"}}),
                    negative=replace(negative, meta={**negative.meta, _META_KEY: {**common, "role": "negative"}}),
                    family=family,
                    transform_id=transform_id,
                )
            )
    if not pairs:
        raise ValueError(f"{pairs_path}: no admitted train-split semantic contrast pairs")
    return pairs


def semantic_contrast_metadata(record: ExampleRecord) -> dict[str, str] | None:
    """Return validated batch metadata injected by :func:`load_semantic_contrast_pairs`."""
    value = record.meta.get(_META_KEY)
    if not isinstance(value, dict):
        return None
    required = ("pair_id", "family", "transform_id", "source_program_id", "role")
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise ValueError("semantic contrast record metadata is incomplete")
    if value["role"] not in {"positive", "negative"}:
        raise ValueError("semantic contrast record role must be positive or negative")
    return {key: str(value[key]) for key in required}


__all__ = [
    "SemanticContrastMargin",
    "SemanticContrastTrainingPair",
    "load_semantic_contrast_pairs",
    "pairwise_margin_loss",
    "semantic_contrast_metadata",
]
