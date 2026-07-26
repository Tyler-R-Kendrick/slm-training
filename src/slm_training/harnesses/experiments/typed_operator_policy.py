"""Default-off typed operator-policy scorer for bounded CAP2 experiments.

The scorer deliberately consumes only the persisted ``OperatorPolicyInputV1``
boundary.  Labels are evaluator-only ``OperatorPolicyRowV1`` fields and never
enter the encoder.  This is experiment machinery, not a serving default.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.models.operator_feature_encoder import (
    CandidateScoringHead,
    FeatureArm,
    OperatorFeatureEncoder,
    OperatorFeatureVocabularyV1,
)
from slm_training.models.action_code_registry import ActionCodeRegistry
from slm_training.models.local_action_head import (
    LocalFlatHead,
    StateContext,
    TernaryECOCHead,
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


TypedOperatorPolicyHeadFamily = Literal[
    "local_flat",
    "ternary_ecoc",
    "factorized",
    "independent_set",
    "recurrent_set",
]

TYPED_OPERATOR_POLICY_HEAD_FAMILIES: tuple[TypedOperatorPolicyHeadFamily, ...] = (
    "local_flat",
    "ternary_ecoc",
    "factorized",
    "independent_set",
    "recurrent_set",
)


def _semantic_action_identity(view: OperatorPolicyInputV1, row: int) -> str:
    """Stable parameter/decode key built only from the sanitized action view."""
    action = view.action_rows[row]
    payload = {
        "operator_id": action.operator_id,
        "operator_version": action.operator_version,
        "locality": action.locality,
        "cost": action.cost,
        "effect_signature": sorted(effect.value for effect in action.effect_signature),
        "slots": [
            {
                "slot_id": slot.slot_id,
                "ref_kind": slot.ref_kind.value,
                "binding_phase": slot.binding_phase.value,
                "required": slot.required,
                "repeated": slot.repeated,
            }
            for slot in action.argument_slots
        ],
    }
    # Hex preserves an exact, dot-free semantic key for LocalFlatHead's
    # ParameterDict. It is derived only from fields the policy may observe.
    return "typed_" + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode().hex()


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
    """Ragged typed scorer with one selected legal-action head family.

    All families score only rows supplied by ``OperatorPolicyInputV1``. The
    family switch is an experiment lever, never a legality or routing switch.
    """

    def __init__(
        self,
        vocabulary: OperatorFeatureVocabularyV1,
        *,
        dim: int = 16,
        head_family: TypedOperatorPolicyHeadFamily = "local_flat",
    ) -> None:
        super().__init__()
        if head_family not in TYPED_OPERATOR_POLICY_HEAD_FAMILIES:
            raise ValueError(f"unknown typed policy head family {head_family!r}")
        self.head_family = head_family
        self.encoder = OperatorFeatureEncoder(vocabulary, dim=dim, arm=FeatureArm.TYPED)
        self.argument_head = CandidateScoringHead(dim)
        self.context_projection = nn.Linear(dim * 2, dim)
        self.action_head: nn.Linear | None = None
        self.local_flat: LocalFlatHead | None = None
        self.ecoc: TernaryECOCHead | None = None
        self.factor_projections: nn.ModuleList | None = None
        self.independent_query: nn.Linear | None = None
        self.independent_key: nn.Linear | None = None
        self.independent_value: nn.Linear | None = None
        self.independent_score: nn.Linear | None = None
        self.recurrent: nn.GRU | None = None
        self.recurrent_score: nn.Linear | None = None
        if head_family == "local_flat":
            self.local_flat = LocalFlatHead(dim, unseen_action_policy="unk")
        elif head_family == "ternary_ecoc":
            self.ecoc = TernaryECOCHead(dim, registry=ActionCodeRegistry())
        elif head_family == "factorized":
            self.factor_projections = nn.ModuleList(nn.Linear(dim, 1) for _ in range(3))
        elif head_family == "independent_set":
            self.independent_query = nn.Linear(dim, dim, bias=False)
            self.independent_key = nn.Linear(dim, dim, bias=False)
            self.independent_value = nn.Linear(dim, dim, bias=False)
            self.independent_score = nn.Linear(dim * 2, 1)
        else:
            self.recurrent = nn.GRU(dim, dim, batch_first=True)
            self.recurrent_score = nn.Linear(dim, 1)

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[TypedOperatorPolicyExampleV1],
        *,
        dim: int = 16,
        head_family: TypedOperatorPolicyHeadFamily = "local_flat",
    ) -> "TypedOperatorPolicyScorer":
        if not examples:
            raise ValueError("typed operator policy requires examples")
        scorer = cls(
            OperatorFeatureVocabularyV1.from_inputs([example.view for example in examples]),
            dim=dim,
            head_family=head_family,
        )
        if scorer.local_flat is not None:
            scorer.local_flat.materialize(
                _semantic_action_identity(example.view, action.row)
                for example in examples
                for action in example.view.action_rows
            )
        return scorer

    def _context(
        self, action_embeddings: torch.Tensor, reference_embeddings: torch.Tensor
    ) -> torch.Tensor:
        action_mean = action_embeddings.mean(dim=0)
        reference_mean = reference_embeddings.mean(dim=0)
        return torch.tanh(self.context_projection(torch.cat((action_mean, reference_mean))).unsqueeze(0))

    def _action_logits(
        self,
        view: OperatorPolicyInputV1,
        action_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        identities = [_semantic_action_identity(view, action.row) for action in view.action_rows]
        context = self._context(action_embeddings, reference_embeddings)
        if self.local_flat is not None:
            output = self.local_flat.score(
                context,
                StateContext(state_family_id="typed_operator_policy"),
                identities,
            )
            assert output.logits is not None
            return output.logits.squeeze(0)
        if self.ecoc is not None:
            output = self.ecoc.score(
                context,
                StateContext(state_family_id="typed_operator_policy"),
                identities,
            )
            assert output.trits is not None
            entry = self.ecoc._get_entry(identities)
            log_probs = F.log_softmax(output.trits.squeeze(0), dim=-1)
            return torch.stack(
                [
                    sum(log_probs[position, digit] for position, digit in enumerate(assignment.codeword))
                    for assignment in entry.assignments
                ]
            )
        if self.factor_projections is not None:
            return sum(projection(action_embeddings).squeeze(-1) for projection in self.factor_projections)
        if self.independent_query is not None:
            assert self.independent_key is not None
            assert self.independent_value is not None
            assert self.independent_score is not None
            query = self.independent_query(action_embeddings)
            key = self.independent_key(action_embeddings)
            affinity = query @ key.T / math.sqrt(action_embeddings.shape[-1])
            affinity.fill_diagonal_(float("-inf"))
            neighbors = F.softmax(affinity, dim=-1) @ self.independent_value(action_embeddings)
            return self.independent_score(torch.cat((action_embeddings, neighbors), dim=-1)).squeeze(-1)
        assert self.recurrent is not None
        assert self.recurrent_score is not None
        order = sorted(range(len(identities)), key=identities.__getitem__)
        order_tensor = torch.tensor(order, dtype=torch.long, device=action_embeddings.device)
        outputs, _ = self.recurrent(action_embeddings[order_tensor].unsqueeze(0))
        sorted_logits = self.recurrent_score(outputs.squeeze(0)).squeeze(-1)
        logits = torch.empty_like(sorted_logits)
        logits[order_tensor] = sorted_logits
        return logits

    def forward(
        self, view: OperatorPolicyInputV1
    ) -> tuple[torch.Tensor, dict[tuple[int, str], tuple[torch.Tensor, tuple[int, ...]]]]:
        """Return scores only for compiler-provided action and slot candidates."""
        reference_embeddings = self.encoder.encode_reference_rows(view)
        action_embeddings = self.encoder.encode(view)
        action_logits = self._action_logits(view, action_embeddings, reference_embeddings)
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
