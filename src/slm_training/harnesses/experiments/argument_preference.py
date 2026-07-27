"""SLM-418 (DSH5-10) ninth slice: argument-level preference for the typed policy.

The disposition doc (``docs/design/dsh5-10-replay-preference-rows.md``) has
named ``typed_operator_policy.py``'s ``TypedOperatorPolicyScorer`` -- not the
generic TwoTower pair format the sixth/seventh/eighth slices used -- as the
issue's actual training target since the sixth slice. This module is the
first real step onto that target, deliberately scoped to what is honestly
representable there today.

Only ``pronoun_focus_followup`` rows qualify. The other six named patterns'
``chosen_action``/``rejected_action`` are history controls (``undo``,
``redo:<id>``, ``checkout:<id>``) or ``merge:<pair>`` -- none of which has a
row in ``OperatorPolicyInputV1.action_rows`` (built only from
``legal_set.entries``, i.e. operator-registry actions; see
``build_operator_policy_input`` in
``slm_training.models.operator_policy_view``). ``pronoun_focus_followup`` is
the one pattern whose chosen and rejected actions are both the *same*
operator with different bound arguments for the *same* slot -- a genuine
argument-selection preference, not an action-selection or control-selection
one. Wiring the other six patterns into this head requires a real scope
decision about what "action row" would even mean for a control, which is
left open rather than guessed at here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from slm_training.dsl.operators import (
    LegalSetCoverage,
    OperatorLegalSetV1,
    OperatorLibraryV1,
    OperatorReplayPreferenceRowV1,
    ReferenceTableV1,
    ReplayPreferenceRelation,
    deserialize_operator_action,
)
from slm_training.harnesses.experiments.typed_operator_policy import TypedOperatorPolicyScorer
from slm_training.models.operator_policy_view import (
    OperatorPolicyInputV1,
    build_operator_policy_input,
)

__all__ = [
    "TypedOperatorArgumentPreferenceExampleV1",
    "build_argument_preference_example",
    "train_typed_operator_argument_preference",
    "typed_operator_argument_preference_loss",
]


@dataclass(frozen=True)
class TypedOperatorArgumentPreferenceExampleV1:
    """One replay-grounded argument-selection preference in the sanitized policy-input space.

    ``chosen_reference_row``/``rejected_reference_row`` are row indices into
    ``view.reference_rows`` -- the same opaque-free space
    ``TypedOperatorPolicyScorer.forward`` scores over, never the runtime's
    own ``OperatorRef`` identities.
    """

    row_id: str
    view: OperatorPolicyInputV1
    action_row: int
    slot_id: str
    chosen_reference_row: int
    rejected_reference_row: int

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("row_id is required")
        if not 0 <= self.action_row < len(self.view.action_rows):
            raise ValueError("action_row is outside the policy view")
        slots = {
            slot.slot_id: slot for slot in self.view.action_rows[self.action_row].argument_slots
        }
        if self.slot_id not in slots:
            raise ValueError("slot_id names an unknown action slot")
        candidates = slots[self.slot_id].candidate_rows
        if self.chosen_reference_row not in candidates or self.rejected_reference_row not in candidates:
            raise ValueError("chosen/rejected reference row is outside the live slot domain")
        if self.chosen_reference_row == self.rejected_reference_row:
            raise ValueError("chosen and rejected reference rows must differ")


def build_argument_preference_example(
    row: OperatorReplayPreferenceRowV1,
    *,
    reference_table: ReferenceTableV1,
    legal_set: OperatorLegalSetV1,
    library: OperatorLibraryV1,
) -> TypedOperatorArgumentPreferenceExampleV1 | None:
    """Render one row into an argument-preference example, or ``None``.

    Returns ``None`` -- never fabricates a slot/row pairing -- whenever
    ``row`` is not ``pronoun_focus_followup``, its chosen/rejected actions
    are not the same operator with exactly one differing slot (never true by
    construction for a well-formed row, but checked rather than assumed), or
    either bound ref cannot be resolved against ``reference_table`` (would
    indicate a stale/inconsistent row).
    """
    if row.semantic_relation is not ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP:
        return None
    chosen_operator_id, chosen_arguments = deserialize_operator_action(row.chosen_action)
    rejected_operator_id, rejected_arguments = deserialize_operator_action(row.rejected_action)
    if chosen_operator_id != rejected_operator_id:
        return None
    chosen_by_slot = {argument.slot_id: argument.value for argument in chosen_arguments}
    rejected_by_slot = {argument.slot_id: argument.value for argument in rejected_arguments}
    if set(chosen_by_slot) != set(rejected_by_slot):
        return None
    differing_slots = [
        slot_id
        for slot_id, chosen_ref in chosen_by_slot.items()
        if chosen_ref != rejected_by_slot[slot_id]
    ]
    if len(differing_slots) != 1:
        return None
    slot_id = differing_slots[0]

    view = build_operator_policy_input(reference_table, legal_set, library)
    action_row = next(
        (action.row for action in view.action_rows if action.operator_id == chosen_operator_id), None
    )
    if action_row is None:
        return None

    table_entries = (*reference_table.entries, *reference_table.selectors)
    row_by_ref = {
        (entry.ref.KIND, entry.ref.opaque_id): index for index, entry in enumerate(table_entries)
    }
    chosen_ref = chosen_by_slot[slot_id]
    rejected_ref = rejected_by_slot[slot_id]
    chosen_row = row_by_ref.get((chosen_ref.KIND, chosen_ref.opaque_id))
    rejected_row = row_by_ref.get((rejected_ref.KIND, rejected_ref.opaque_id))
    if chosen_row is None or rejected_row is None:
        return None

    try:
        return TypedOperatorArgumentPreferenceExampleV1(
            row_id=row.input_state_id,
            view=view,
            action_row=action_row,
            slot_id=slot_id,
            chosen_reference_row=chosen_row,
            rejected_reference_row=rejected_row,
        )
    except ValueError:
        return None


def typed_operator_argument_preference_loss(
    scorer: TypedOperatorPolicyScorer, example: TypedOperatorArgumentPreferenceExampleV1
) -> torch.Tensor:
    """Bradley-Terry pairwise margin over ``CandidateScoringHead`` logits.

    Surrogate preference loss on argument-selection logits for one slot --
    not textbook DPO (mirrors the same honesty note
    ``scripts/train_preference.py``'s own ``dpo_loss`` carries for the
    generic TwoTower path). Defers PARTIAL/UNKNOWN-coverage rows exactly
    like ``typed_operator_policy_loss`` does, via a zero-valued,
    gradient-connected loss.
    """
    if example.view.coverage is not LegalSetCoverage.COMPLETE:
        return next(scorer.parameters()).sum() * 0.0
    _, argument_logits = scorer(example.view)
    logits, candidate_rows = argument_logits[(example.action_row, example.slot_id)]
    chosen_index = candidate_rows.index(example.chosen_reference_row)
    rejected_index = candidate_rows.index(example.rejected_reference_row)
    return -F.logsigmoid(logits[chosen_index] - logits[rejected_index])


def train_typed_operator_argument_preference(
    scorer: TypedOperatorPolicyScorer,
    examples: Sequence[TypedOperatorArgumentPreferenceExampleV1],
    *,
    steps: int,
    learning_rate: float,
) -> list[float]:
    """Train only COMPLETE examples with one matched full-batch schedule.

    Mirrors ``train_typed_operator_policy``'s own schedule shape exactly,
    over this module's pairwise loss instead of single-label cross-entropy.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    trainable = [
        example for example in examples if example.view.coverage is LegalSetCoverage.COMPLETE
    ]
    if not trainable:
        raise ValueError("typed operator argument preference has no COMPLETE training rows")
    optimizer = torch.optim.Adam(scorer.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.stack(
            [typed_operator_argument_preference_loss(scorer, example) for example in trainable]
        ).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history
