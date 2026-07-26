"""Typed categorical feature encoder over the sanitized operator view (DSH3-23).

`_stable_scalar` (`slm_training.models.legal_edit_batch`) hashes every
categorical candidate feature — including successor fingerprints — into one
float per field. This module answers whether that hurts operator
discrimination and sample efficiency: it adds a typed-embedding encoder over
`OperatorPolicyInputV1` (SLM-397/DSH3-22) and a shared ragged batch so a
hash-scalar arm, a typed-embedding arm, and a typed-plus-labeled-identity-
bucket control arm can be trained and compared at matched data, parameters,
seeds, and budget.

This is fixture-only wiring evidence (``claim_class: "wiring"``): the
matched-arm comparison this module drives is deliberately run on a small
deterministic fixture in
``scripts/run_dsh3_23_operator_feature_encoder_fixture.py``, not a
production-scale claim. See ``docs/design/dsh3-23-operator-feature-encoder.md``
for the disposition.

Only a minimal "flat" scoring head is implemented here — factorized
(SLM-383/DSH3-15) and ECOC (SLM-404/DSH3-29) heads are separate tickets that
consume the same ``OperatorFeatureBatchV1`` this module builds; see the
design doc's scope notes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.operators import (
    BindingPhase,
    CompilerFact,
    EffectDeltaKind,
    LegalSetCoverage,
    OperatorSupportVerdict,
    RefKind,
    SelectorFact,
)
from slm_training.harnesses.experiments.candidate_selector import (
    brier_score,
    expected_calibration_error,
)
from slm_training.models.legal_edit_batch import _stable_scalar
from slm_training.models.operator_policy_view import (
    OperatorActionViewV1,
    OperatorArgumentSlotViewV1,
    OperatorPolicyInputV1,
    ReferenceModelViewV1,
)

__all__ = [
    "CandidateScoringHead",
    "FeatureArm",
    "FixtureOperatorDecisionV1",
    "OperatorFeatureBatchV1",
    "OperatorFeatureEncoder",
    "OperatorFeatureEncodingReportV1",
    "OperatorFeatureVocabularyV1",
    "build_operator_feature_batch",
    "evaluate_fixture_arm",
    "make_fixture_operator_decisions",
    "permute_fixture_decision",
    "row_identity_bucket",
    "run_matched_arms_fixture",
    "train_fixture_arm",
]

_POSITION_NORMALIZER = 64.0
_HASH_SCALAR_FIELDS = 7  # ref_kind, value_type, has_parent, position, operator_id, binding_phase, locality


class FeatureArm(str, Enum):
    """Which candidate-feature encoding a run uses; matched arms per DSH3-23."""

    HASH_SCALAR = "hash_scalar"
    TYPED = "typed"
    TYPED_IDENTITY_BUCKET = "typed_identity_bucket"


def row_identity_bucket(row: int, n_buckets: int) -> int:
    """A labeled, purely positional "identity" bucket — an anti-generalization
    control only. Under the SLM-397 permutation-invariance contract, row
    order in ``OperatorPolicyInputV1`` carries no real signal; a model that
    benefits from this feature is exploiting spurious build-time order, not
    learning operator semantics."""
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    return row % n_buckets


@dataclass(frozen=True)
class OperatorFeatureVocabularyV1:
    """Closed-enum vocabularies plus fixture-supplied open identifier sets.

    ``ref_kinds``/``binding_phases``/``compiler_facts``/``effect_kinds`` are
    closed runtime enums — their vocabularies never grow. ``value_types``,
    ``localities``, and ``operator_ids`` are open DSL-pack/registry
    identifiers; index 0 in every lookup is reserved for an out-of-vocabulary
    fallback so a held-out identifier never crashes encoding.
    """

    ref_kinds: tuple[str, ...]
    binding_phases: tuple[str, ...]
    compiler_facts: tuple[str, ...]
    effect_kinds: tuple[str, ...]
    value_types: tuple[str, ...] = ()
    localities: tuple[str, ...] = ()
    operator_ids: tuple[str, ...] = ()
    schema: str = "operator_feature_vocabulary/v1"

    @classmethod
    def closed(cls) -> OperatorFeatureVocabularyV1:
        return cls(
            ref_kinds=tuple(kind.value for kind in RefKind),
            binding_phases=tuple(phase.value for phase in BindingPhase),
            compiler_facts=tuple(
                fact.value for fact in (*CompilerFact, *SelectorFact)
            ),
            effect_kinds=tuple(kind.value for kind in EffectDeltaKind),
        )

    @classmethod
    def from_inputs(
        cls, inputs: Sequence[OperatorPolicyInputV1]
    ) -> OperatorFeatureVocabularyV1:
        value_types = sorted(
            {row.value_type for view in inputs for row in view.reference_rows}
        )
        localities = sorted(
            {action.locality for view in inputs for action in view.action_rows}
        )
        operator_ids = sorted(
            {action.operator_id for view in inputs for action in view.action_rows}
        )
        return replace(
            cls.closed(),
            value_types=tuple(value_types),
            localities=tuple(localities),
            operator_ids=tuple(operator_ids),
        )

    def _index(self, vocabulary: tuple[str, ...], value: str) -> int:
        try:
            return vocabulary.index(value) + 1
        except ValueError:
            return 0

    def ref_kind_index(self, value: str) -> int:
        return self._index(self.ref_kinds, value)

    def value_type_index(self, value: str) -> int:
        return self._index(self.value_types, value)

    def binding_phase_index(self, value: str) -> int:
        return self._index(self.binding_phases, value)

    def compiler_fact_indices(self, values: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(self._index(self.compiler_facts, value) for value in values)

    def effect_kind_indices(self, values: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(self._index(self.effect_kinds, value) for value in values)

    def locality_index(self, value: str) -> int:
        return self._index(self.localities, value)

    def operator_id_index(self, value: str) -> int:
        return self._index(self.operator_ids, value)


@dataclass(frozen=True)
class OperatorFeatureBatchV1:
    """One ragged batch of reference/action rows, shared by every scoring head.

    ``inputs`` preserves each source ``OperatorPolicyInputV1`` alongside a
    flat row range so a head can gather per-input action logits without the
    encoder needing to know which head consumes its output.
    """

    inputs: tuple[OperatorPolicyInputV1, ...]
    reference_offsets: tuple[int, ...]
    action_offsets: tuple[int, ...]
    schema: str = "operator_feature_batch/v1"

    def reference_range(self, index: int) -> tuple[int, int]:
        return self.reference_offsets[index], self.reference_offsets[index + 1]

    def action_range(self, index: int) -> tuple[int, int]:
        return self.action_offsets[index], self.action_offsets[index + 1]


def build_operator_feature_batch(
    inputs: Sequence[OperatorPolicyInputV1],
) -> OperatorFeatureBatchV1:
    reference_offsets = [0]
    action_offsets = [0]
    for view in inputs:
        reference_offsets.append(reference_offsets[-1] + len(view.reference_rows))
        action_offsets.append(action_offsets[-1] + len(view.action_rows))
    return OperatorFeatureBatchV1(
        inputs=tuple(inputs),
        reference_offsets=tuple(reference_offsets),
        action_offsets=tuple(action_offsets),
    )


class OperatorFeatureEncoder(nn.Module):
    """Encodes one ``OperatorFeatureBatchV1`` into per-action-row score inputs.

    Three arms share this exact module shape (only the reference-row
    encoding differs), so "matched parameters/optimizer/budget" is structural
    rather than something a caller must separately guarantee:

    * ``HASH_SCALAR`` — every categorical field goes through the existing
      ``_stable_scalar`` hash (the legacy control), concatenated and
      linearly projected to ``dim``.
    * ``TYPED`` — a learned embedding per categorical field, summed for
      multi-valued fields (``compiler_facts``, ``effect_signature``).
    * ``TYPED_IDENTITY_BUCKET`` — ``TYPED`` plus one additional learned
      embedding keyed by :func:`row_identity_bucket`, a labeled
      anti-generalization control only.

    No arm ever consumes a ``semantic_fingerprint``, opaque ID, or proof/
    application hash — none of those fields exist on
    ``ReferenceModelViewV1``/``OperatorActionViewV1`` to begin with.
    """

    def __init__(
        self,
        vocabulary: OperatorFeatureVocabularyV1,
        dim: int,
        arm: FeatureArm,
        *,
        n_identity_buckets: int = 16,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.dim = dim
        self.arm = arm
        self.n_identity_buckets = n_identity_buckets

        if arm is FeatureArm.HASH_SCALAR:
            self.reference_projection = nn.Linear(6, dim)
            self.action_projection = nn.Linear(4, dim)
        else:
            self.ref_kind_embedding = nn.Embedding(len(vocabulary.ref_kinds) + 1, dim)
            self.value_type_embedding = nn.Embedding(
                len(vocabulary.value_types) + 1, dim
            )
            self.compiler_fact_embedding = nn.Embedding(
                len(vocabulary.compiler_facts) + 1, dim
            )
            self.reference_position = nn.Linear(4, dim)
            self.reference_projection = nn.Linear(dim * 3, dim)

            self.operator_embedding = nn.Embedding(len(vocabulary.operator_ids) + 1, dim)
            self.binding_phase_embedding = nn.Embedding(
                len(vocabulary.binding_phases) + 1, dim
            )
            self.effect_kind_embedding = nn.Embedding(
                len(vocabulary.effect_kinds) + 1, dim
            )
            self.locality_embedding = nn.Embedding(len(vocabulary.localities) + 1, dim)
            self.cost_projection = nn.Linear(1, dim)
            self.action_projection = nn.Linear(dim * 5, dim)

            if arm is FeatureArm.TYPED_IDENTITY_BUCKET:
                self.identity_embedding = nn.Embedding(n_identity_buckets, dim)

    def _reference_embeddings(self, view: OperatorPolicyInputV1) -> torch.Tensor:
        if not view.reference_rows:
            return torch.zeros((0, self.dim))
        if self.arm is FeatureArm.HASH_SCALAR:
            rows = []
            for row in view.reference_rows:
                rows.append(
                    [
                        _stable_scalar(row.ref_kind.value),
                        _stable_scalar(row.value_type),
                        _stable_scalar(row.has_parent),
                        (row.relative_position or 0) / _POSITION_NORMALIZER,
                        float(row.selector_cardinality or 0) / _POSITION_NORMALIZER,
                        float(row.selector_max_fanout or 0) / _POSITION_NORMALIZER,
                    ]
                )
            return self.reference_projection(torch.tensor(rows, dtype=torch.float32))

        ref_kind_ids = torch.tensor(
            [self.vocabulary.ref_kind_index(row.ref_kind.value) for row in view.reference_rows],
            dtype=torch.long,
        )
        value_type_ids = torch.tensor(
            [self.vocabulary.value_type_index(row.value_type) for row in view.reference_rows],
            dtype=torch.long,
        )
        fact_sums = torch.stack(
            [
                self.compiler_fact_embedding(
                    torch.tensor(
                        self.vocabulary.compiler_fact_indices(
                            tuple(fact.value for fact in row.compiler_facts)
                        )
                        or (0,),
                        dtype=torch.long,
                    )
                ).sum(dim=0)
                for row in view.reference_rows
            ]
        )
        position_features = torch.tensor(
            [
                [
                    float(row.has_parent),
                    (row.relative_position or 0) / _POSITION_NORMALIZER,
                    float(row.selector_cardinality or 0) / _POSITION_NORMALIZER,
                    float(row.selector_max_fanout or 0) / _POSITION_NORMALIZER,
                ]
                for row in view.reference_rows
            ],
            dtype=torch.float32,
        )
        combined = torch.cat(
            [
                self.ref_kind_embedding(ref_kind_ids),
                self.value_type_embedding(value_type_ids),
                fact_sums + self.reference_position(position_features),
            ],
            dim=-1,
        )
        embeddings = self.reference_projection(combined)
        if self.arm is FeatureArm.TYPED_IDENTITY_BUCKET:
            buckets = torch.tensor(
                [
                    row_identity_bucket(row.row, self.n_identity_buckets)
                    for row in view.reference_rows
                ],
                dtype=torch.long,
            )
            embeddings = embeddings + self.identity_embedding(buckets)
        return embeddings

    def _action_embeddings(
        self, view: OperatorPolicyInputV1, reference_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if not view.action_rows:
            return torch.zeros((0, self.dim))

        def pooled_candidates(action) -> torch.Tensor:
            rows: list[int] = [
                row for slot in action.argument_slots for row in slot.candidate_rows
            ]
            if not rows:
                return torch.zeros(self.dim)
            return reference_embeddings[torch.tensor(rows, dtype=torch.long)].mean(dim=0)

        if self.arm is FeatureArm.HASH_SCALAR:
            rows = []
            for action in view.action_rows:
                rows.append(
                    [
                        _stable_scalar(action.operator_id),
                        _stable_scalar(action.locality),
                        _stable_scalar(action.cost),
                        _stable_scalar(action.verdict.value),
                    ]
                )
            action_features = self.action_projection(torch.tensor(rows, dtype=torch.float32))
            pooled = torch.stack([pooled_candidates(action) for action in view.action_rows])
            return action_features + pooled

        operator_ids = torch.tensor(
            [self.vocabulary.operator_id_index(action.operator_id) for action in view.action_rows],
            dtype=torch.long,
        )
        localities = torch.tensor(
            [self.vocabulary.locality_index(action.locality) for action in view.action_rows],
            dtype=torch.long,
        )
        binding_phases = torch.tensor(
            [
                self.vocabulary.binding_phase_index(
                    action.argument_slots[0].binding_phase.value
                    if action.argument_slots
                    else ""
                )
                for action in view.action_rows
            ],
            dtype=torch.long,
        )
        effect_sums = torch.stack(
            [
                self.effect_kind_embedding(
                    torch.tensor(
                        self.vocabulary.effect_kind_indices(
                            tuple(kind.value for kind in action.effect_signature)
                        )
                        or (0,),
                        dtype=torch.long,
                    )
                ).sum(dim=0)
                for action in view.action_rows
            ]
        )
        costs = torch.tensor([[action.cost] for action in view.action_rows], dtype=torch.float32)
        pooled = torch.stack([pooled_candidates(action) for action in view.action_rows])
        combined = torch.cat(
            [
                self.operator_embedding(operator_ids),
                self.binding_phase_embedding(binding_phases),
                effect_sums,
                self.locality_embedding(localities) + self.cost_projection(costs),
                pooled,
            ],
            dim=-1,
        )
        return self.action_projection(combined)

    def encode_reference_rows(self, view: OperatorPolicyInputV1) -> torch.Tensor:
        """Return one ``(len(view.reference_rows), dim)`` per-reference-row tensor.

        Exposed so a candidate-scoring head can score individual candidates
        within a slot's domain (the unit ``_stable_scalar`` encoded per-row in
        the legacy bridge batch), rather than only the pooled action-level
        summary :meth:`encode` returns.
        """
        return self._reference_embeddings(view)

    def encode(self, view: OperatorPolicyInputV1) -> torch.Tensor:
        """Return one ``(len(view.action_rows), dim)`` action-embedding tensor."""
        reference_embeddings = self._reference_embeddings(view)
        return self._action_embeddings(view, reference_embeddings)


class CandidateScoringHead(nn.Module):
    """Scores each row-local candidate reference against one action context.

    Shared, arm-agnostic: the only thing that differs between arms is which
    :class:`OperatorFeatureEncoder` produced ``action_embedding`` and
    ``candidate_embeddings``, so "matched parameters/optimizer/budget" holds
    by construction rather than by convention a caller must uphold.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim * 2, 1)

    def forward(
        self, action_embedding: torch.Tensor, candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if candidate_embeddings.shape[0] == 0:
            return torch.zeros(0)
        expanded = action_embedding.unsqueeze(0).expand(candidate_embeddings.shape[0], -1)
        return self.linear(torch.cat([expanded, candidate_embeddings], dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class FixtureOperatorDecisionV1:
    """One synthetic sanitized decision: a view plus its held-out-only gold row.

    ``gold_row`` is never part of ``view`` — exactly as a real training
    pipeline would keep labels in a parallel structure alongside the
    sanitized ``OperatorPolicyInputV1``, never inside it.
    """

    view: OperatorPolicyInputV1
    gold_row: int


def make_fixture_operator_decisions(
    n: int, seed: int, *, n_candidates: int = 5
) -> tuple[FixtureOperatorDecisionV1, ...]:
    """Build ``n`` synthetic decisions: one operator, one slot, ``n_candidates``
    VALUE references, exactly one of which (the gold row) uses a distinct
    ``value_type`` at a randomized position — content, never row order,
    distinguishes it."""
    rng = random.Random(seed)
    decisions: list[FixtureOperatorDecisionV1] = []
    for _ in range(n):
        gold_row = rng.randrange(n_candidates)
        reference_rows = tuple(
            ReferenceModelViewV1(
                row=row,
                ref_kind=RefKind.VALUE,
                value_type="openui.number" if row == gold_row else "openui.string",
                compiler_facts=(CompilerFact.VALUE_VISIBLE,),
                has_parent=False,
                parent_row=None,
                relative_position=None,
            )
            for row in range(n_candidates)
        )
        action_rows = (
            OperatorActionViewV1(
                row=0,
                operator_id="openui.fixture_op",
                operator_version="v1",
                locality="node",
                cost=1.0,
                effect_signature=(),
                argument_slots=(
                    OperatorArgumentSlotViewV1(
                        slot_id="value",
                        ref_kind=RefKind.VALUE,
                        binding_phase=BindingPhase.APPLICATION,
                        required=True,
                        repeated=False,
                        candidate_rows=tuple(range(n_candidates)),
                        domain_complete=True,
                    ),
                ),
                verdict=OperatorSupportVerdict.SUPPORTED,
                coverage=LegalSetCoverage.COMPLETE,
            ),
        )
        view = OperatorPolicyInputV1(
            reference_rows=reference_rows,
            action_rows=action_rows,
            ordinary_action_count=0,
            coverage=LegalSetCoverage.COMPLETE,
        )
        decisions.append(FixtureOperatorDecisionV1(view=view, gold_row=gold_row))
    return tuple(decisions)


def permute_fixture_decision(
    decision: FixtureOperatorDecisionV1, seed: int
) -> FixtureOperatorDecisionV1:
    """Reshuffle a decision's row order/numbering, preserving all content and
    which candidate is gold — the opaque-ID/order permutation adversarial
    control from the SLM-397 boundary, replayed at the encoder layer."""
    order = list(range(len(decision.view.reference_rows)))
    random.Random(seed).shuffle(order)
    new_row_of_old = {old: new for new, old in enumerate(order)}
    reference_rows = tuple(
        replace(decision.view.reference_rows[old_row], row=new_row)
        for new_row, old_row in enumerate(order)
    )
    action_rows = tuple(
        replace(
            action,
            argument_slots=tuple(
                replace(
                    slot,
                    candidate_rows=tuple(
                        sorted(new_row_of_old[row] for row in slot.candidate_rows)
                    ),
                )
                for slot in action.argument_slots
            ),
        )
        for action in decision.view.action_rows
    )
    view = replace(decision.view, reference_rows=reference_rows, action_rows=action_rows)
    return FixtureOperatorDecisionV1(view=view, gold_row=new_row_of_old[decision.gold_row])


def _candidate_logits(
    encoder: OperatorFeatureEncoder,
    head: CandidateScoringHead,
    decision: FixtureOperatorDecisionV1,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    reference_embeddings = encoder.encode_reference_rows(decision.view)
    action_embedding = encoder.encode(decision.view)[0]
    candidate_rows = decision.view.action_rows[0].argument_slots[0].candidate_rows
    candidate_embeddings = reference_embeddings[torch.tensor(candidate_rows, dtype=torch.long)]
    return head(action_embedding, candidate_embeddings), candidate_rows


def train_fixture_arm(
    decisions: Sequence[FixtureOperatorDecisionV1],
    *,
    arm: FeatureArm,
    dim: int = 8,
    steps: int = 150,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Train one arm's encoder+head to convergence on ``decisions``.

    Full-batch Adam over every decision each step — deliberately identical
    optimizer, step count, and learning rate across arms so a metric
    difference reflects the encoding, not the optimization schedule.
    """
    vocabulary = OperatorFeatureVocabularyV1.from_inputs([d.view for d in decisions])
    torch.manual_seed(seed)
    encoder = OperatorFeatureEncoder(vocabulary, dim=dim, arm=arm)
    head = CandidateScoringHead(dim)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *head.parameters()], lr=lr
    )
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        losses = []
        for decision in decisions:
            logits, candidate_rows = _candidate_logits(encoder, head, decision)
            gold_position = candidate_rows.index(decision.gold_row)
            losses.append(
                F.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([gold_position], dtype=torch.long)
                )
            )
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()
        history.append(float(total_loss.detach()))
    return {
        "encoder": encoder,
        "head": head,
        "vocabulary": vocabulary,
        "history": history,
        "param_count": sum(p.numel() for p in (*encoder.parameters(), *head.parameters())),
    }


def evaluate_fixture_arm(
    trained: dict[str, Any], decisions: Sequence[FixtureOperatorDecisionV1]
) -> dict[str, Any]:
    encoder: OperatorFeatureEncoder = trained["encoder"]
    head: CandidateScoringHead = trained["head"]
    encoder.eval()
    head.eval()
    correct = 0
    ndcg_sum = 0.0
    accepted_mass_sum = 0.0
    predicted_probs: list[float] = []
    correctness: list[bool] = []
    with torch.no_grad():
        for decision in decisions:
            logits, candidate_rows = _candidate_logits(encoder, head, decision)
            probs = F.softmax(logits, dim=-1)
            gold_position = candidate_rows.index(decision.gold_row)
            ranking = torch.argsort(logits, descending=True).tolist()
            top1_position = ranking[0]
            rank = ranking.index(gold_position) + 1
            correct += int(top1_position == gold_position)
            ndcg_sum += 1.0 / math.log2(rank + 1)
            accepted_mass_sum += float(probs[gold_position].item())
            predicted_probs.append(float(probs[top1_position].item()))
            correctness.append(top1_position == gold_position)
    n = len(decisions)
    return {
        "n": n,
        "top1_recall": correct / n,
        "ndcg": ndcg_sum / n,
        "accepted_set_mass": accepted_mass_sum / n,
        "brier_score": brier_score(predicted_probs, correctness),
        "calibration_error": expected_calibration_error(predicted_probs, correctness),
        "final_train_loss": trained["history"][-1],
        "param_count": trained["param_count"],
    }


def _permutation_robustness(
    trained: dict[str, Any],
    decisions: Sequence[FixtureOperatorDecisionV1],
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare predictions on ``decisions`` versus a row-permuted replay.

    ``TYPED`` is expected to match exactly (permutation-invariant per
    SLM-397); ``TYPED_IDENTITY_BUCKET`` is expected to *not* necessarily
    match, since it is the labeled control that can see row position.
    """
    permuted = [permute_fixture_decision(decision, seed) for decision in decisions]
    encoder: OperatorFeatureEncoder = trained["encoder"]
    head: CandidateScoringHead = trained["head"]
    encoder.eval()
    head.eval()
    matches = 0
    with torch.no_grad():
        for original, shuffled in zip(decisions, permuted):
            original_logits, original_rows = _candidate_logits(encoder, head, original)
            shuffled_logits, shuffled_rows = _candidate_logits(encoder, head, shuffled)
            original_by_row = dict(zip(original_rows, original_logits.tolist()))
            shuffled_by_row = dict(zip(shuffled_rows, shuffled_logits.tolist()))
            original_gold_logit = original_by_row[original.gold_row]
            shuffled_gold_logit = shuffled_by_row[shuffled.gold_row]
            if math.isclose(original_gold_logit, shuffled_gold_logit, abs_tol=1e-5):
                matches += 1
    return {"n": len(decisions), "matches": matches, "invariant": matches == len(decisions)}


@dataclass(frozen=True)
class OperatorFeatureEncodingReportV1:
    """Matched-arm fixture comparison report (DSH3-23). Wiring evidence only."""

    schema: str
    claim_class: str
    arms: dict[str, dict[str, Any]]
    permutation_robustness: dict[str, dict[str, Any]]
    n_train: int
    n_eval: int
    steps: int
    lr: float
    seed: int
    dim: int
    disposition: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_class": self.claim_class,
            "arms": self.arms,
            "permutation_robustness": self.permutation_robustness,
            "recipe": {
                "n_train": self.n_train,
                "n_eval": self.n_eval,
                "steps": self.steps,
                "lr": self.lr,
                "seed": self.seed,
                "dim": self.dim,
            },
            "disposition": self.disposition,
            "note": self.note,
        }


def run_matched_arms_fixture(
    *,
    n_train: int = 30,
    n_eval: int = 20,
    n_candidates: int = 5,
    dim: int = 8,
    steps: int = 150,
    lr: float = 0.05,
    seed: int = 0,
) -> OperatorFeatureEncodingReportV1:
    """Run the DSH3-23 matched-arm comparison on a small deterministic fixture.

    Not a production-scale claim (``claim_class: "wiring"``) — see the
    module docstring and ``docs/design/dsh3-23-operator-feature-encoder.md``.
    """
    train_decisions = make_fixture_operator_decisions(
        n_train, seed=seed, n_candidates=n_candidates
    )
    eval_decisions = make_fixture_operator_decisions(
        n_eval, seed=seed + 1, n_candidates=n_candidates
    )

    arms: dict[str, dict[str, Any]] = {}
    permutation_robustness: dict[str, dict[str, Any]] = {}
    for arm in FeatureArm:
        trained = train_fixture_arm(
            train_decisions, arm=arm, dim=dim, steps=steps, lr=lr, seed=seed
        )
        arms[arm.value] = evaluate_fixture_arm(trained, eval_decisions)
        permutation_robustness[arm.value] = _permutation_robustness(
            trained, eval_decisions, seed=seed + 2
        )

    typed_recall = arms[FeatureArm.TYPED.value]["top1_recall"]
    hash_recall = arms[FeatureArm.HASH_SCALAR.value]["top1_recall"]
    typed_invariant = permutation_robustness[FeatureArm.TYPED.value]["invariant"]
    if not typed_invariant:
        disposition = "invalid_permutation_variance"
        note = (
            "The TYPED arm's predictions changed under a row-order/opaque-ID "
            "permutation that preserves all sanitized content — this would "
            "violate the SLM-397 boundary's permutation-invariance contract "
            "and must be treated as a defect, not adopted."
        )
    elif n_train < 200:
        disposition = "wiring_underpowered"
        note = (
            f"Fixture-scale only ({n_train} train / {n_eval} eval decisions): "
            f"typed top-1 recall {typed_recall:.3f} vs hash-scalar "
            f"{hash_recall:.3f} is directional, not a powered claim. Per the "
            "DSH3-23 stop rule, adopt typed encoding only on a supported "
            "benefit at production scale; this run alone does not clear that "
            "bar in either direction."
        )
    elif typed_recall > hash_recall:
        disposition = "supported"
        note = "Typed embeddings outperformed the hash-scalar control at matched scale."
    else:
        disposition = "closed_negative"
        note = (
            "Hash scalars matched or beat typed embeddings at matched scale; "
            "per the stop rule, retain the simpler legacy encoder."
        )

    return OperatorFeatureEncodingReportV1(
        schema="operator_feature_encoding_report/v1",
        claim_class="wiring",
        arms=arms,
        permutation_robustness=permutation_robustness,
        n_train=n_train,
        n_eval=n_eval,
        steps=steps,
        lr=lr,
        seed=seed,
        dim=dim,
        disposition=disposition,
        note=note,
    )
