"""Default-off typed operator-policy scorer for bounded CAP2 experiments.

The scorer deliberately consumes only the persisted ``OperatorPolicyInputV1``
boundary.  Labels are evaluator-only ``OperatorPolicyRowV1`` fields and never
enter the encoder.  This is experiment machinery, not a serving default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.models.operator_feature_encoder import (
    CandidateScoringHead,
    FeatureArm,
    OperatorFeatureEncoder,
    OperatorFeatureVocabularyV1,
)
from slm_training.models.operator_policy_objective import (
    OperatorPolicyObjectiveV1,
    OperatorPolicyRouteV1,
    OperatorPolicyTargetV1,
    TargetUtilityLabel,
    route_operator_policy_inference,
)
from slm_training.models.operator_policy_view import (
    OperatorPolicyInputV1,
    operator_policy_input_from_dict,
)
from slm_training.dsl.operators import LegalSetCoverage, OperatorSupportVerdict


@dataclass(frozen=True)
class TypedOperatorPolicyExampleV1:
    """One label-aligned persisted policy view for bounded local training."""

    row_id: str
    view: OperatorPolicyInputV1
    accepted_action_row: int
    accepted_argument_rows: tuple[tuple[str, int], ...]

    @classmethod
    def from_row(cls, row: Any) -> "TypedOperatorPolicyExampleV1":
        """Accept a corpus row structurally without making models own data IO."""
        return cls(
            row_id=str(row.row_id),
            view=operator_policy_input_from_dict(row.policy_input),
            accepted_action_row=int(row.accepted_action_row),
            accepted_argument_rows=tuple(
                (str(slot_id), int(reference_row))
                for slot_id, reference_row in row.accepted_argument_rows
            ),
        )

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("row_id is required")
        if not 0 <= self.accepted_action_row < len(self.view.action_rows):
            raise ValueError("accepted_action_row is outside the policy view")
        slots = {
            slot.slot_id: slot
            for slot in self.view.action_rows[self.accepted_action_row].argument_slots
        }
        for slot_id, reference_row in self.accepted_argument_rows:
            if slot_id not in slots:
                raise ValueError("accepted argument names an unknown action slot")
            if not 0 <= reference_row < len(self.view.reference_rows):
                raise ValueError("accepted argument row is outside the policy view")
            if reference_row not in slots[slot_id].candidate_rows:
                raise ValueError("accepted argument is outside the live slot domain")

    @property
    def objective(self) -> OperatorPolicyObjectiveV1:
        return OperatorPolicyObjectiveV1(
            coverage=self.view.coverage,
            targets=tuple(
                OperatorPolicyTargetV1(
                    action_key=str(action.row),
                    compiler_support=action.verdict,
                    utility=(
                        TargetUtilityLabel.POSITIVE
                        if action.row == self.accepted_action_row
                        else (
                            TargetUtilityLabel.UNKNOWN
                            if action.verdict is OperatorSupportVerdict.UNKNOWN
                            else TargetUtilityLabel.NEGATIVE
                        )
                    ),
                )
                for action in self.view.action_rows
            ),
        )


@dataclass(frozen=True)
class TypedOperatorPolicyDecisionV1:
    """Auditable default-off decision; selected rows remain compiler-local."""

    route: OperatorPolicyRouteV1
    selected_action_row: int | None
    selected_argument_rows: tuple[tuple[str, int], ...]
    model_forwards: int


class TypedOperatorPolicyScorer(nn.Module):
    """Ragged typed action/argument scorer over sanitized policy views."""

    def __init__(self, vocabulary: OperatorFeatureVocabularyV1, *, dim: int = 16) -> None:
        super().__init__()
        self.encoder = OperatorFeatureEncoder(vocabulary, dim=dim, arm=FeatureArm.TYPED)
        self.action_head = nn.Linear(dim, 1)
        self.argument_head = CandidateScoringHead(dim)

    @classmethod
    def from_examples(
        cls, examples: Sequence[TypedOperatorPolicyExampleV1], *, dim: int = 16
    ) -> "TypedOperatorPolicyScorer":
        if not examples:
            raise ValueError("typed operator policy requires examples")
        return cls(
            OperatorFeatureVocabularyV1.from_inputs([example.view for example in examples]),
            dim=dim,
        )

    def forward(
        self, view: OperatorPolicyInputV1
    ) -> tuple[torch.Tensor, dict[tuple[int, str], tuple[torch.Tensor, tuple[int, ...]]]]:
        """Return scores only for compiler-provided action and slot candidates."""
        reference_embeddings = self.encoder.encode_reference_rows(view)
        action_embeddings = self.encoder.encode(view)
        action_logits = self.action_head(action_embeddings).squeeze(-1)
        argument_logits: dict[tuple[int, str], tuple[torch.Tensor, tuple[int, ...]]] = {}
        for action in view.action_rows:
            for slot in action.argument_slots:
                rows = tuple(slot.candidate_rows)
                candidates = reference_embeddings[
                    torch.tensor(rows, dtype=torch.long, device=reference_embeddings.device)
                ]
                argument_logits[(action.row, slot.slot_id)] = (
                    self.argument_head(action_embeddings[action.row], candidates),
                    rows,
                )
        return action_logits, argument_logits


def typed_operator_policy_loss(
    scorer: TypedOperatorPolicyScorer, example: TypedOperatorPolicyExampleV1
) -> torch.Tensor:
    """Supervise COMPLETE rows only; PARTIAL rows remain defer-only evidence."""
    if example.view.coverage is not LegalSetCoverage.COMPLETE:
        return next(scorer.parameters()).sum() * 0.0
    action_logits, argument_logits = scorer(example.view)
    losses = [F.cross_entropy(action_logits.unsqueeze(0), torch.tensor([example.accepted_action_row]))]
    for slot_id, reference_row in example.accepted_argument_rows:
        logits, candidate_rows = argument_logits[(example.accepted_action_row, slot_id)]
        losses.append(
            F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([candidate_rows.index(reference_row)])
            )
        )
    return torch.stack(losses).mean()


def decide_typed_operator_policy(
    scorer: TypedOperatorPolicyScorer, example: TypedOperatorPolicyExampleV1
) -> TypedOperatorPolicyDecisionV1:
    """Route COMPLETE/PARTIAL decisions without a learned forced-state path."""
    route = route_operator_policy_inference(example.objective)
    if route is OperatorPolicyRouteV1.COMPLETE_SINGLETON:
        return TypedOperatorPolicyDecisionV1(
            route=route,
            selected_action_row=example.accepted_action_row,
            selected_argument_rows=(),
            model_forwards=0,
        )
    if route is not OperatorPolicyRouteV1.COMPLETE_AMBIGUOUS:
        return TypedOperatorPolicyDecisionV1(
            route=route,
            selected_action_row=None,
            selected_argument_rows=(),
            model_forwards=0,
        )
    action_logits, argument_logits = scorer(example.view)
    action_row = int(action_logits.argmax().item())
    arguments = []
    for slot in example.view.action_rows[action_row].argument_slots:
        logits, candidate_rows = argument_logits[(action_row, slot.slot_id)]
        arguments.append((slot.slot_id, candidate_rows[int(logits.argmax().item())]))
    return TypedOperatorPolicyDecisionV1(
        route=route,
        selected_action_row=action_row,
        selected_argument_rows=tuple(arguments),
        model_forwards=1,
    )


def train_typed_operator_policy(
    scorer: TypedOperatorPolicyScorer,
    examples: Sequence[TypedOperatorPolicyExampleV1],
    *,
    steps: int,
    learning_rate: float,
) -> list[float]:
    """Train only COMPLETE examples with one matched full-batch schedule."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    trainable = [
        example
        for example in examples
        if example.view.coverage is LegalSetCoverage.COMPLETE
    ]
    if not trainable:
        raise ValueError("typed operator policy has no COMPLETE training rows")
    optimizer = torch.optim.Adam(scorer.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.stack(
            [typed_operator_policy_loss(scorer, example) for example in trainable]
        ).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


__all__ = [
    "TypedOperatorPolicyDecisionV1",
    "TypedOperatorPolicyExampleV1",
    "TypedOperatorPolicyScorer",
    "decide_typed_operator_policy",
    "train_typed_operator_policy",
    "typed_operator_policy_loss",
]
