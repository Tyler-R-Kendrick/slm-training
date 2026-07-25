"""Objective-neutral ragged batches over exact dynamic legal-edit candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from slm_training.data.flow.bridge_corpus import (
    ExactLegalEditCandidateSetV1,
    LegalEditBridgeRowV1,
)

FEATURE_NAMES = (
    "action_kind",
    "production",
    "arity",
    "cardinality",
    "node_pointer",
    "slot_pointer",
    "literal_kind",
    "enum_value",
    "frame",
    "successor_fingerprint",
)


def _stable_scalar(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


@dataclass(frozen=True)
class PaddedLegalEditBatch:
    candidate_features: torch.Tensor
    membership_mask: torch.Tensor
    positive_mask: torch.Tensor
    supported_mask: torch.Tensor
    unsupported_mask: torch.Tensor
    unknown_mask: torch.Tensor
    target_distribution: torch.Tensor
    candidate_ids: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True)
class LegalEditBatch:
    """One immutable batch shared by any scorer over exact candidate membership."""

    state_features: torch.Tensor
    candidate_features: torch.Tensor
    row_offsets: torch.Tensor
    candidate_to_row: torch.Tensor
    candidate_ids: tuple[str, ...]
    successor_fingerprints: tuple[str, ...]
    positive_mask: torch.Tensor
    supported_mask: torch.Tensor
    unsupported_mask: torch.Tensor
    unknown_mask: torch.Tensor
    target_distribution: torch.Tensor
    row_ids: tuple[str, ...]
    candidate_set_digests: tuple[str, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        n_candidates = len(self.candidate_ids)
        if self.candidate_features.ndim != 2:
            raise ValueError("candidate_features must have shape [N, F]")
        if self.candidate_features.shape[0] != n_candidates:
            raise ValueError("candidate feature count does not match candidate IDs")
        if len(set(zip(self.candidate_to_row.tolist(), self.candidate_ids))) != n_candidates:
            raise ValueError("duplicate candidate ID within a batch row")
        for tensor in (
            self.candidate_to_row,
            self.positive_mask,
            self.supported_mask,
            self.unsupported_mask,
            self.unknown_mask,
            self.target_distribution,
        ):
            if tensor.numel() != n_candidates:
                raise ValueError("ragged candidate tensor has wrong length")
        if self.row_offsets.ndim != 1 or self.row_offsets.numel() != len(self.row_ids) + 1:
            raise ValueError("row_offsets must have shape [B + 1]")
        if int(self.row_offsets[0]) != 0 or int(self.row_offsets[-1]) != n_candidates:
            raise ValueError("row_offsets do not cover all candidates")
        if bool((self.unknown_mask & self.unsupported_mask).any()):
            raise ValueError("UNKNOWN candidates cannot be negatives")
        if bool((self.positive_mask & ~self.supported_mask).any()):
            raise ValueError("positive candidates must be supported")
        for row in range(len(self.row_ids)):
            start, end = int(self.row_offsets[row]), int(self.row_offsets[row + 1])
            expected = sorted(self.candidate_ids[start:end])
            if list(self.candidate_ids[start:end]) != expected:
                raise ValueError("candidate rows must use canonical ID order")
            mass = float(self.target_distribution[start:end].sum())
            if bool(self.positive_mask[start:end].any()) and abs(mass - 1.0) > 1e-6:
                raise ValueError("multi-positive target distribution is not normalized")

    @classmethod
    def pack(
        cls,
        rows: Sequence[LegalEditBridgeRowV1],
        candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
        *,
        device: str | torch.device = "cpu",
    ) -> "LegalEditBatch":
        state_features: list[list[float]] = []
        candidate_features: list[list[float]] = []
        candidate_ids: list[str] = []
        successors: list[str] = []
        candidate_to_row: list[int] = []
        positive: list[bool] = []
        supported: list[bool] = []
        unsupported: list[bool] = []
        unknown: list[bool] = []
        target_distribution: list[float] = []
        offsets = [0]
        row_ids: list[str] = []
        set_digests: list[str] = []

        for row_index, row in enumerate(rows):
            candidate_set = candidate_sets[row.candidate_set_digest]
            row.model_input(candidate_set)
            ordered = sorted(candidate_set.candidates, key=lambda item: item.candidate_id)
            positive_ids = set(row.positive_candidate_ids)
            supported_ids = set(row.supported_candidate_ids)
            unsupported_ids = set(row.unsupported_candidate_ids)
            unknown_ids = set(row.unknown_candidate_ids)
            positive_count = len(positive_ids)
            state_features.append(
                [
                    _stable_scalar(row.state_summary.get("statement_count")),
                    float(row.step_index),
                    float(row.normalized_progress),
                    float(row.sampled_time),
                ]
            )
            for candidate in ordered:
                candidate_ids.append(candidate.candidate_id)
                successors.append(candidate.successor_fingerprint)
                candidate_to_row.append(row_index)
                candidate_features.append(
                    [_stable_scalar(candidate.features.get(name)) for name in FEATURE_NAMES]
                )
                is_positive = candidate.candidate_id in positive_ids
                positive.append(is_positive)
                supported.append(candidate.candidate_id in supported_ids)
                unsupported.append(candidate.candidate_id in unsupported_ids)
                unknown.append(candidate.candidate_id in unknown_ids)
                target_distribution.append(
                    1.0 / positive_count if is_positive and positive_count else 0.0
                )
            offsets.append(len(candidate_ids))
            row_ids.append(row.row_id)
            set_digests.append(row.candidate_set_digest)

        return cls(
            state_features=torch.tensor(state_features, dtype=torch.float32, device=device),
            candidate_features=torch.tensor(
                candidate_features, dtype=torch.float32, device=device
            ),
            row_offsets=torch.tensor(offsets, dtype=torch.long, device=device),
            candidate_to_row=torch.tensor(
                candidate_to_row, dtype=torch.long, device=device
            ),
            candidate_ids=tuple(candidate_ids),
            successor_fingerprints=tuple(successors),
            positive_mask=torch.tensor(positive, dtype=torch.bool, device=device),
            supported_mask=torch.tensor(supported, dtype=torch.bool, device=device),
            unsupported_mask=torch.tensor(unsupported, dtype=torch.bool, device=device),
            unknown_mask=torch.tensor(unknown, dtype=torch.bool, device=device),
            target_distribution=torch.tensor(
                target_distribution, dtype=torch.float32, device=device
            ),
            row_ids=tuple(row_ids),
            candidate_set_digests=tuple(set_digests),
        )

    @classmethod
    def pack_inference(
        cls,
        candidate_set: ExactLegalEditCandidateSetV1,
        *,
        statement_count: int,
        step_index: int,
        device: str | torch.device = "cpu",
    ) -> "LegalEditBatch":
        """Pack one runtime-enumerated set without inventing supervision.

        Runtime candidates are deliberately marked UNKNOWN until a verifier or
        corpus label certifies support. Candidate membership and features use
        the exact same canonical packing path as training batches.
        """
        ordered = tuple(
            sorted(candidate_set.candidates, key=lambda item: item.candidate_id)
        )
        count = len(ordered)
        return cls(
            state_features=torch.tensor(
                [[float(statement_count), float(step_index), 0.0, 0.0]],
                dtype=torch.float32,
                device=device,
            ),
            candidate_features=torch.tensor(
                [
                    [_stable_scalar(item.features.get(name)) for name in FEATURE_NAMES]
                    for item in ordered
                ],
                dtype=torch.float32,
                device=device,
            ).reshape(count, len(FEATURE_NAMES)),
            row_offsets=torch.tensor([0, count], dtype=torch.long, device=device),
            candidate_to_row=torch.zeros(count, dtype=torch.long, device=device),
            candidate_ids=tuple(item.candidate_id for item in ordered),
            successor_fingerprints=tuple(
                item.successor_fingerprint for item in ordered
            ),
            positive_mask=torch.zeros(count, dtype=torch.bool, device=device),
            supported_mask=torch.zeros(count, dtype=torch.bool, device=device),
            unsupported_mask=torch.zeros(count, dtype=torch.bool, device=device),
            unknown_mask=torch.ones(count, dtype=torch.bool, device=device),
            target_distribution=torch.zeros(count, dtype=torch.float32, device=device),
            row_ids=(candidate_set.state_fingerprint,),
            candidate_set_digests=(candidate_set.candidate_set_digest,),
        )

    def to_padded(self) -> PaddedLegalEditBatch:
        batch_size = len(self.row_ids)
        widths = [
            int(self.row_offsets[index + 1] - self.row_offsets[index])
            for index in range(batch_size)
        ]
        max_width = max(widths, default=0)
        feature_width = self.candidate_features.shape[1]
        padded = self.candidate_features.new_zeros(
            (batch_size, max_width, feature_width)
        )
        membership = torch.zeros(
            (batch_size, max_width), dtype=torch.bool, device=self.candidate_features.device
        )
        masks = [
            torch.zeros_like(membership)
            for _ in range(4)
        ]
        targets = self.target_distribution.new_zeros((batch_size, max_width))
        padded_ids: list[tuple[str | None, ...]] = []
        for row, width in enumerate(widths):
            start, end = int(self.row_offsets[row]), int(self.row_offsets[row + 1])
            padded[row, :width] = self.candidate_features[start:end]
            membership[row, :width] = True
            for output, source in zip(
                masks,
                (
                    self.positive_mask,
                    self.supported_mask,
                    self.unsupported_mask,
                    self.unknown_mask,
                ),
                strict=True,
            ):
                output[row, :width] = source[start:end]
            targets[row, :width] = self.target_distribution[start:end]
            padded_ids.append(
                tuple(self.candidate_ids[start:end])
                + (None,) * (max_width - width)
            )
        return PaddedLegalEditBatch(
            candidate_features=padded,
            membership_mask=membership,
            positive_mask=masks[0],
            supported_mask=masks[1],
            unsupported_mask=masks[2],
            unknown_mask=masks[3],
            target_distribution=targets,
            candidate_ids=tuple(padded_ids),
        )

    def gathered_projection(
        self,
        values: torch.Tensor,
        gather: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply a projection hook without permitting membership mutation."""
        if values.ndim != 1 or values.numel() != len(self.candidate_ids):
            raise ValueError("projection values must have exact shape [N]")
        projected = gather(values, self.candidate_to_row)
        if projected.shape != values.shape:
            raise ValueError("projection hook changed exact candidate membership")
        return projected
