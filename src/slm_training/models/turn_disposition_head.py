"""Torch disposition classifier over the operator policy view -- VAR3-02 (SLM-430).

Mirrors ``models/operator_termination.py``'s shape exactly (a torch loss
function plus a training-loss-free selection report reusing
``flow.termination``'s calibration helpers), for the same disposition head
DSH3-22/23's ``OperatorPolicyInputV1`` surface would train.

``out_of_scope`` is never a class this classifier can output: it is derived
solely from an empty accepted legal set
(``dsl.operators.turn_disposition.derive_turn_disposition``), before any
model is consulted -- this repo's I1 ordering (a deterministic bypass
outranks any learned score). ``TurnDispositionLabel`` structurally excludes
it, so a corpus builder cannot even mislabel an ``out_of_scope`` turn into
this classifier's training data by mistake; the label space itself has no
slot for it. This module trains/labels only the three-way choice a
non-empty legal set still leaves open: ``answer`` (state query, no
mutation), ``clarify`` (ambiguous top candidates), and ``emit`` (a
confident single proposal).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch
import torch.nn.functional as F

from slm_training.flow.termination import brier_score, expected_calibration_error

__all__ = [
    "TurnDispositionLabel",
    "TurnDispositionReportV1",
    "TurnDispositionTargetV1",
    "turn_disposition_losses",
]


class TurnDispositionLabel(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    EMIT = "emit"


#: Fixed, documented class order for the classifier's logits -- index i of a
#: 3-way logit vector corresponds to ``_LABEL_ORDER[i]``.
_LABEL_ORDER: tuple[TurnDispositionLabel, ...] = (
    TurnDispositionLabel.ANSWER,
    TurnDispositionLabel.CLARIFY,
    TurnDispositionLabel.EMIT,
)


@dataclass(frozen=True)
class TurnDispositionTargetV1:
    """Corpus-derived label for a non-empty legal set.

    ``forced_singleton`` mirrors ``OperatorTerminationTargetV1.forced_singleton``:
    a ``COMPLETE``-coverage legal set with exactly one candidate is the same
    I1 bypass ``derive_turn_disposition`` itself takes without ever calling a
    score provider, so it is never labeled anything but ``emit`` and its
    loss is maskable (``mask_forced``) rather than trained against.
    """

    label: TurnDispositionLabel
    forced_singleton: bool
    schema: str = "turn_disposition_target/v1"

    def __post_init__(self) -> None:
        if self.forced_singleton and self.label is not TurnDispositionLabel.EMIT:
            raise ValueError("a forced singleton can only be labeled emit")

    @property
    def label_index(self) -> int:
        return _LABEL_ORDER.index(self.label)


def turn_disposition_losses(
    logits: torch.Tensor,
    target: TurnDispositionTargetV1,
    *,
    mask_forced: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy over the three-way (never ``out_of_scope``) label space."""
    if logits.numel() != len(_LABEL_ORDER):
        raise ValueError(
            f"logits must contain exactly {len(_LABEL_ORDER)} disposition scores"
        )
    if mask_forced and target.forced_singleton:
        loss = logits.sum() * 0.0
    else:
        label_tensor = torch.tensor([target.label_index], dtype=torch.long)
        loss = F.cross_entropy(logits.reshape(1, -1), label_tensor)
    return loss, {"disposition": loss}


@dataclass(frozen=True)
class TurnDispositionReportV1:
    """Selection metrics for the learned disposition classifier; no training loss here."""

    accuracy: float
    clarify_brier: float
    clarify_ece: float
    wrong_emit_rate: float
    abstention_rate: float
    schema: str = "turn_disposition_report/v1"

    @classmethod
    def from_predictions(
        cls,
        *,
        predicted: Sequence[TurnDispositionLabel],
        expected: Sequence[TurnDispositionLabel],
        clarify_probabilities: Sequence[float],
        clarify_targets: Sequence[int],
        emit_correct: Sequence[bool],
    ) -> "TurnDispositionReportV1":
        n = max(1, len(expected))
        matches = sum(1 for p, e in zip(predicted, expected) if p == e)
        wrong_emit = sum(
            1
            for p, correct in zip(predicted, emit_correct)
            if p is TurnDispositionLabel.EMIT and not correct
        )
        abstentions = sum(1 for p in predicted if p is TurnDispositionLabel.CLARIFY)
        return cls(
            accuracy=matches / n,
            clarify_brier=brier_score(clarify_probabilities, clarify_targets),
            clarify_ece=expected_calibration_error(
                clarify_probabilities, clarify_targets
            ),
            wrong_emit_rate=wrong_emit / n,
            abstention_rate=abstentions / n,
        )
