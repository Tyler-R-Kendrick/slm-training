"""Train and evaluate a real TurnDispositionHead classifier -- VAR3-03 (SLM-433).

VAR3-02 shipped ``turn_disposition_losses``/``TurnDispositionReportV1``
(``slm_training.models.turn_disposition_head``) as "the training-ready
surface a future corpus-labeling issue would target, tested here against
synthetic tensors only." This module is that future issue: it wires the
first actual trainable classifier over that loss -- a linear head on top of
the existing DSH3-23 ``OperatorFeatureEncoder`` (never a reimplemented
embedding stack) -- and reports it against two matched always-emit
baselines on one identical held-out real-corpus split, reusing
``evals.cap2_operator.score_disposition_predictions`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from slm_training.data.flow.turn_disposition_corpus import TurnDispositionRowV1
from slm_training.evals.cap2_operator import score_disposition_predictions
from slm_training.models.operator_feature_encoder import (
    FeatureArm,
    OperatorFeatureEncoder,
    OperatorFeatureVocabularyV1,
)
from slm_training.models.operator_policy_view import (
    OperatorPolicyInputV1,
    operator_policy_input_from_dict,
)
from slm_training.models.turn_disposition_head import (
    TurnDispositionLabel,
    TurnDispositionTargetV1,
    turn_disposition_losses,
)

__all__ = [
    "TurnDispositionClassifierV1",
    "TurnDispositionExampleV1",
    "evaluate_turn_disposition_arms",
    "train_turn_disposition_head",
]

#: Must match `models.turn_disposition_head._LABEL_ORDER` exactly -- index i
#: of the classifier's 3-way logits corresponds to `_LABEL_ORDER[i]`.
_LABEL_ORDER: tuple[TurnDispositionLabel, ...] = (
    TurnDispositionLabel.ANSWER,
    TurnDispositionLabel.CLARIFY,
    TurnDispositionLabel.EMIT,
)


@dataclass(frozen=True)
class TurnDispositionExampleV1:
    """One trainable example: a rehydrated sanitized view plus its gold target.

    Carries ``top_candidate_serialized``/``top_candidate_correct`` alongside
    the gold ``target`` so every arm in :func:`evaluate_turn_disposition_arms`
    can score an "if this row emits" outcome against the same real corpus
    fact, whether or not the gold disposition itself emits.
    """

    view: OperatorPolicyInputV1
    target: TurnDispositionTargetV1
    accepted_action_serialized: str
    top_candidate_serialized: str
    top_candidate_correct: bool
    trace_id: str
    split: str

    @classmethod
    def from_row(cls, row: TurnDispositionRowV1) -> "TurnDispositionExampleV1":
        return cls(
            view=operator_policy_input_from_dict(row.policy_input),
            target=row.target,
            accepted_action_serialized=row.accepted_action_serialized,
            top_candidate_serialized=row.top_candidate_serialized,
            top_candidate_correct=row.top_candidate_correct,
            trace_id=row.trace_id,
            split=row.split,
        )


class TurnDispositionClassifierV1(nn.Module):
    """Three-way (answer/clarify/emit) classifier over one pooled action
    embedding -- the trainable surface ``turn_disposition_losses`` was built
    to train, wired for the first time by this issue. Reuses
    ``OperatorFeatureEncoder`` (DSH3-23) rather than a new embedding stack;
    ``out_of_scope`` has no output unit because the gold label space itself
    structurally excludes it (see ``models.turn_disposition_head``)."""

    def __init__(
        self,
        vocabulary: OperatorFeatureVocabularyV1,
        *,
        dim: int = 16,
        arm: FeatureArm = FeatureArm.TYPED,
    ) -> None:
        super().__init__()
        self.encoder = OperatorFeatureEncoder(vocabulary, dim, arm)
        self.classifier = nn.Linear(dim, len(_LABEL_ORDER))

    def forward(self, view: OperatorPolicyInputV1) -> torch.Tensor:
        action_embeddings = self.encoder.encode(view)
        pooled = action_embeddings.mean(dim=0)
        return self.classifier(pooled)


def train_turn_disposition_head(
    classifier: TurnDispositionClassifierV1,
    examples: Sequence[TurnDispositionExampleV1],
    *,
    steps: int,
    learning_rate: float,
) -> list[float]:
    """Full-batch Adam over every training example each step.

    Mirrors ``harnesses.experiments.typed_operator_policy.
    train_typed_operator_policy``'s matched schedule. Forced-singleton rows
    are masked (zero loss, no gradient) exactly like ``turn_disposition_
    losses``'s own ``mask_forced`` default -- a corpus made entirely of
    forced rows trains nothing, which is the correct, expected behavior.
    """
    if not examples:
        raise ValueError("training requires at least one example")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        losses = [
            turn_disposition_losses(classifier(example.view), example.target)[0]
            for example in examples
        ]
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()
        history.append(float(total_loss.detach()))
    return history


def _predicted_label(
    classifier: TurnDispositionClassifierV1, example: TurnDispositionExampleV1
) -> TurnDispositionLabel:
    classifier.eval()
    with torch.no_grad():
        logits = classifier(example.view)
    return _LABEL_ORDER[int(torch.argmax(logits).item())]


def _trained_row(
    classifier: TurnDispositionClassifierV1, example: TurnDispositionExampleV1
) -> dict[str, Any]:
    if example.target.forced_singleton:
        # I1: a COMPLETE-coverage legal set with exactly one candidate is a
        # deterministic bypass that outranks any learned score -- the
        # classifier is never even consulted here, exactly like
        # `derive_turn_disposition`'s own forced-singleton path.
        return {
            "predicted_disposition": "emit",
            "emit_correct": example.top_candidate_correct,
        }
    label = _predicted_label(classifier, example)
    if label is not TurnDispositionLabel.EMIT:
        return {"predicted_disposition": label.value}
    return {
        "predicted_disposition": "emit",
        "emit_correct": example.top_candidate_correct,
    }


def evaluate_turn_disposition_arms(
    classifier: TurnDispositionClassifierV1,
    examples: Sequence[TurnDispositionExampleV1],
) -> dict[str, dict[str, Any]]:
    """Score ``disposition_off`` / ``derived_only`` / ``disposition_trained``
    on one identical held-out example set at matched budget, each through
    the unchanged ``score_disposition_predictions``.

    ``disposition_off`` (VAR3-02's own always-emit baseline) and
    ``derived_only`` (the free ``out_of_scope``/``answer``/forced-singleton
    rules, ``clarify`` never firing) are numerically identical on this real
    corpus: it contains no ``out_of_scope``/``answer`` turns by construction
    (``data.flow.turn_disposition_corpus``'s corpus generator only emits
    AST-mutating steps), and a forced-singleton row's only live candidate is
    also the naive always-emit top pick -- so the two named baselines
    coincide here. That is reported honestly rather than manufactured
    apart: the informative comparison in this corpus is emit-vs-clarify
    disposition awareness (``disposition_trained`` against either
    always-emit baseline), not forced-singleton awareness.
    """
    if not examples:
        raise ValueError("evaluation requires at least one example")
    always_emit_rows = [
        {"predicted_disposition": "emit", "emit_correct": example.top_candidate_correct}
        for example in examples
    ]
    trained_rows = [_trained_row(classifier, example) for example in examples]
    return {
        "disposition_off": score_disposition_predictions(always_emit_rows),
        "derived_only": score_disposition_predictions(list(always_emit_rows)),
        "disposition_trained": score_disposition_predictions(trained_rows),
    }
