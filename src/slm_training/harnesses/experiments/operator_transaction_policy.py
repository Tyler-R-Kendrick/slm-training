"""Default-off bounded set-valued policy over conflict-free transactions (DSH5-06).

Reuses, never reimplements, two proven layers:

* ``dsl.operators.transactions``/``transaction_executor`` (DSH5-04/DSH5-05) —
  the only authority for what is "exactly conflict-free". A candidate action
  set is never judged legal by anything this module computes; it is legal
  only when the real :func:`~slm_training.dsl.operators.build_operator_transaction`
  accepts it, and it is only ever *executed* through the real
  :func:`~slm_training.dsl.operators.commit_operator_transaction`.
* ``harnesses.experiments.typed_operator_policy`` (DSH3-28) — its
  ``independent_set``/``recurrent_set`` head families already compute one
  permutation-equivariant logit per candidate action from the sanitized
  ``OperatorPolicyInputV1`` boundary. This module adds bounded-K (K=2/K=4)
  *set*-valued decoding on top of those same per-action logits; it does not
  fork a new encoder.

Safety invariant (mirrors this repository's "speculation ranks only over
forward-calculated symbol tables and always verifies before commit"
architecture invariant): the learned score only *ranks* candidate subsets.
Every ranked subset is re-verified by the real, ground-truth
``build_operator_transaction`` before it is treated as selected; a subset
that the ranking believed was conflict-free but that the real check rejects
is counted as a ``conflict_attempts`` event and never committed. This holds
even under the ``shuffled_conflict_graph`` control, which deliberately feeds
the *ranking* step a corrupted belief about which pairs conflict — the real
verification step is never bypassed.

This is experiment machinery, not a serving default: set-valued selection is
never invoked outside this module and outside explicit tests/scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import time
from typing import Literal, Sequence

import torch

from slm_training.dsl.operators import (
    ApplicationProvenanceV1,
    LegalSetCoverage,
    OperatorLibraryV1,
    OperatorStateV1,
    OperatorSupportVerdict,
    OperatorTransactionCommitDecisionV1,
    PreparedOperatorActionV1,
    ReferenceTableBuilder,
    build_operator_transaction,
    commit_operator_transaction,
)
from slm_training.dsl.operators.contracts import AstOperatorV1
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.pack import DslPack
from slm_training.harnesses.experiments.typed_operator_policy import (
    TypedOperatorPolicyScorer,
)
from slm_training.models.operator_policy_view import (
    OperatorActionViewV1,
    OperatorPolicyInputV1,
)

OperatorTransactionPolicyHeadFamily = Literal["independent_set", "recurrent_set"]

OPERATOR_TRANSACTION_POLICY_HEAD_FAMILIES: tuple[
    OperatorTransactionPolicyHeadFamily, ...
] = ("independent_set", "recurrent_set")

#: Same bound the executor enforces (``OPERATOR_TRANSACTION_MAX_ACTIONS``);
#: repeated here as a decode-time precondition so a caller with a larger
#: candidate pool fails closed before any scoring happens, rather than
#: silently truncating it.
MAX_TRANSACTION_POLICY_CANDIDATES = 8


def _candidate_view(row: int, declaration: AstOperatorV1) -> OperatorActionViewV1:
    """One sanitized, argument-free view of a prepared action's declaration.

    Arguments are already bound at prepare time (the policy chooses *which*
    prepared actions to combine, never their arguments), so ``argument_slots``
    is always empty here — there is nothing left to bind or leak.
    """
    return OperatorActionViewV1(
        row=row,
        operator_id=declaration.operator_id,
        operator_version=declaration.version,
        locality=declaration.locality,
        cost=declaration.cost,
        effect_signature=declaration.effect_signature,
        argument_slots=(),
        verdict=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.COMPLETE,
    )


@dataclass(frozen=True)
class OperatorTransactionPolicyExampleV1:
    """One base-state candidate pool for bounded set-valued selection.

    ``prepared_actions``/``library``/``state``/``reference_table``/
    ``provenance`` are evaluator-only execution context — the scorer never
    receives them, only :attr:`policy_input`. ``accepted_row_sets`` is the
    evaluator-only positive-equivalence-class label (every acceptable
    conflict-free action set for this base state); it is never a model
    input.
    """

    row_id: str
    pack: DslPack
    library: OperatorLibraryV1
    state: OperatorStateV1
    reference_table: ReferenceTableV1
    provenance: ApplicationProvenanceV1
    prepared_actions: tuple[PreparedOperatorActionV1, ...]
    declarations: tuple[AstOperatorV1, ...]
    coverage: LegalSetCoverage
    accepted_row_sets: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("row_id is required")
        if len(self.prepared_actions) != len(self.declarations):
            raise ValueError("one declaration is required per prepared action")
        if len(self.prepared_actions) > MAX_TRANSACTION_POLICY_CANDIDATES:
            raise ValueError(
                "candidate pool exceeds MAX_TRANSACTION_POLICY_CANDIDATES"
            )
        n = len(self.prepared_actions)
        for accepted in self.accepted_row_sets:
            if any(not 0 <= row < n for row in accepted):
                raise ValueError("accepted_row_sets must index live candidates")

    @property
    def policy_input(self) -> OperatorPolicyInputV1:
        return OperatorPolicyInputV1(
            reference_rows=(),
            action_rows=tuple(
                _candidate_view(row, declaration)
                for row, declaration in enumerate(self.declarations)
            ),
            ordinary_action_count=0,
            coverage=self.coverage,
        )


@dataclass(frozen=True)
class OperatorTransactionPolicyDecisionV1:
    """Auditable bounded set decision; ``selected_rows`` is always ground-truth
    conflict-free (verified by the real executor) or empty (DEFER)."""

    selected_rows: frozenset[int]
    model_forwards: int
    conflict_attempts: int
    deferred: bool


def _pairwise_hard_conflicts(
    example: OperatorTransactionPolicyExampleV1,
) -> frozenset[frozenset[int]]:
    """Ground-truth pairwise incompatibility, computed only by the real builder.

    Any pairwise rejection (conflict, cycle-adjacent footprint issues, etc.)
    marks the pair as jointly infeasible; this is a fast necessary-condition
    prune, never treated as sufficient on its own — see
    :func:`_feasible_ranked_subsets`.
    """
    conflicts: set[frozenset[int]] = set()
    for i, j in itertools.combinations(range(len(example.prepared_actions)), 2):
        decision = build_operator_transaction(
            library=example.library,
            state=example.state,
            reference_table=example.reference_table,
            prepared_actions=(
                example.prepared_actions[i],
                example.prepared_actions[j],
            ),
            provenance=example.provenance,
        )
        if not decision.succeeded:
            conflicts.add(frozenset((i, j)))
    return frozenset(conflicts)


def _prune_feasible_subsets(
    n: int, k: int, hard_conflicts: frozenset[frozenset[int]]
) -> list[frozenset[int]]:
    """Every subset of size 1..k with no pairwise hard conflict (necessary,
    not sufficient — see the module docstring's safety invariant)."""
    subsets: list[frozenset[int]] = []
    for size in range(1, k + 1):
        for combo in itertools.combinations(range(n), size):
            candidate = frozenset(combo)
            if any(
                frozenset(pair) in hard_conflicts
                for pair in itertools.combinations(sorted(candidate), 2)
            ):
                continue
            subsets.append(candidate)
    return subsets


def score_transaction_candidates(
    scorer: TypedOperatorPolicyScorer, example: OperatorTransactionPolicyExampleV1
) -> torch.Tensor:
    """One scorer forward call; per-candidate logits only (no argument head use)."""
    action_logits, _ = scorer(example.policy_input)
    return action_logits


def decode_bounded_transaction_set(
    scorer: TypedOperatorPolicyScorer,
    example: OperatorTransactionPolicyExampleV1,
    *,
    k: int,
    conflict_belief: frozenset[frozenset[int]] | None = None,
) -> OperatorTransactionPolicyDecisionV1:
    """Rank bounded conflict-free-believed subsets, then verify before commit.

    ``conflict_belief`` overrides the pairwise pruning belief only (the
    ``shuffled_conflict_graph`` control); it never substitutes for the real
    per-subset :func:`build_operator_transaction` check every ranked
    candidate must still pass before it can be selected.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if example.coverage is not LegalSetCoverage.COMPLETE:
        # PARTIAL action coverage never implies complete-set knowledge.
        return OperatorTransactionPolicyDecisionV1(
            selected_rows=frozenset(),
            model_forwards=0,
            conflict_attempts=0,
            deferred=True,
        )
    n = len(example.prepared_actions)
    if n == 0:
        return OperatorTransactionPolicyDecisionV1(
            selected_rows=frozenset(),
            model_forwards=0,
            conflict_attempts=0,
            deferred=True,
        )
    hard_conflicts = (
        _pairwise_hard_conflicts(example) if conflict_belief is None else conflict_belief
    )
    ranked_candidates = _prune_feasible_subsets(n, min(k, n), hard_conflicts)
    logits = score_transaction_candidates(scorer, example)
    scored = sorted(
        ranked_candidates,
        key=lambda subset: float(logits[list(subset)].sum()),
        reverse=True,
    )
    conflict_attempts = 0
    for subset in scored:
        ordered = tuple(sorted(subset))
        prepared = tuple(example.prepared_actions[row] for row in ordered)
        decision = build_operator_transaction(
            library=example.library,
            state=example.state,
            reference_table=example.reference_table,
            prepared_actions=prepared,
            provenance=example.provenance,
        )
        if decision.succeeded:
            return OperatorTransactionPolicyDecisionV1(
                selected_rows=frozenset(ordered),
                model_forwards=1,
                conflict_attempts=conflict_attempts,
                deferred=False,
            )
        conflict_attempts += 1
    return OperatorTransactionPolicyDecisionV1(
        selected_rows=frozenset(),
        model_forwards=1,
        conflict_attempts=conflict_attempts,
        deferred=True,
    )


def execute_transaction_policy_decision(
    example: OperatorTransactionPolicyExampleV1,
    decision: OperatorTransactionPolicyDecisionV1,
    *,
    reference_table_builder: ReferenceTableBuilder,
) -> OperatorTransactionCommitDecisionV1 | None:
    """Commit ``decision`` only through the real DSH5-05 executor; ``None`` on DEFER."""
    if decision.deferred or not decision.selected_rows:
        return None
    ordered = tuple(sorted(decision.selected_rows))
    prepared = tuple(example.prepared_actions[row] for row in ordered)
    built = build_operator_transaction(
        library=example.library,
        state=example.state,
        reference_table=example.reference_table,
        prepared_actions=prepared,
        provenance=example.provenance,
    )
    if not built.succeeded:
        raise ValueError(
            "decode_bounded_transaction_set returned an unverified selection"
        )
    assert built.transaction is not None
    return commit_operator_transaction(
        pack=example.pack,
        library=example.library,
        base_state=example.state,
        base_reference_table=example.reference_table,
        transaction=built.transaction,
        provenance=example.provenance,
        reference_table_builder=reference_table_builder,
    )


def transaction_policy_set_loss(
    scorer: TypedOperatorPolicyScorer,
    example: OperatorTransactionPolicyExampleV1,
    *,
    k: int,
) -> torch.Tensor:
    """Multi-positive contrastive loss over enumerated bounded feasible sets.

    Every member of ``accepted_row_sets`` is a positive equivalence class;
    every other enumerated (pairwise-pruned) feasible subset up to size ``k``
    is an implicit negative. COMPLETE coverage only, mirroring
    ``typed_operator_policy_loss``.
    """
    if example.coverage is not LegalSetCoverage.COMPLETE:
        return next(scorer.parameters()).sum() * 0.0
    n = len(example.prepared_actions)
    hard_conflicts = _pairwise_hard_conflicts(example)
    subsets = _prune_feasible_subsets(n, min(k, n), hard_conflicts)
    positive_indices = [
        index
        for index, subset in enumerate(subsets)
        if subset in example.accepted_row_sets
    ]
    if not positive_indices:
        raise ValueError(
            "no accepted_row_sets member is a bounded pairwise-feasible subset"
        )
    logits = score_transaction_candidates(scorer, example)
    subset_scores = torch.stack(
        [logits[list(subset)].sum() for subset in subsets]
    )
    log_probs = torch.log_softmax(subset_scores, dim=-1)
    positive_log_prob = torch.logsumexp(
        log_probs[torch.tensor(positive_indices, dtype=torch.long)], dim=0
    )
    return -positive_log_prob


def train_operator_transaction_policy(
    scorer: TypedOperatorPolicyScorer,
    examples: Sequence[OperatorTransactionPolicyExampleV1],
    *,
    k: int,
    steps: int,
    learning_rate: float,
) -> list[float]:
    """Train only COMPLETE examples with one matched full-batch schedule."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    trainable = [
        example for example in examples if example.coverage is LegalSetCoverage.COMPLETE
    ]
    if not trainable:
        raise ValueError("operator transaction policy has no COMPLETE training rows")
    optimizer = torch.optim.Adam(scorer.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.stack(
            [
                transaction_policy_set_loss(scorer, example, k=k)
                for example in trainable
            ]
        ).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@dataclass(frozen=True)
class OperatorTransactionPolicyReportV1:
    """One arm's measured bounded set-selection result (fixture/control scale;
    never a ship claim on its own — see ``documenting-experiment-results``)."""

    head_family: str
    arm: str
    k: int
    n: int
    exact_set_match_rate: float
    accepted_set_mass: float
    final_state_success_rate: float
    conflict_attempts: int
    actions_per_forward: float
    model_calls: int
    compiler_time_ms: float
    schema: str = "operator_transaction_policy_report/v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "head_family": self.head_family,
            "arm": self.arm,
            "k": self.k,
            "n": self.n,
            "exact_set_match_rate": self.exact_set_match_rate,
            "accepted_set_mass": self.accepted_set_mass,
            "final_state_success_rate": self.final_state_success_rate,
            "conflict_attempts": self.conflict_attempts,
            "actions_per_forward": self.actions_per_forward,
            "model_calls": self.model_calls,
            "compiler_time_ms": self.compiler_time_ms,
        }


def evaluate_operator_transaction_policy(
    scorer: TypedOperatorPolicyScorer,
    examples: Sequence[OperatorTransactionPolicyExampleV1],
    *,
    head_family: str,
    arm: str,
    k: int,
    reference_table_builder: ReferenceTableBuilder,
    conflict_belief_by_row_id: dict[str, frozenset[frozenset[int]]] | None = None,
    selected_rows_override: dict[str, frozenset[int]] | None = None,
) -> OperatorTransactionPolicyReportV1:
    """Run one arm over ``examples`` and reduce it to one report row.

    ``selected_rows_override`` bypasses scoring entirely (the ``oracle_set``
    control): the given rows are still only ever executed through the real
    commit path, never asserted correct without that check.
    """
    exact_matches = 0
    mass_total = 0.0
    committed = 0
    conflict_attempts = 0
    model_calls = 0
    actions_selected = 0
    forwards_with_actions = 0
    compiler_time_ms = 0.0
    for example in examples:
        start = time.perf_counter()
        if selected_rows_override is not None:
            rows = selected_rows_override[example.row_id]
            decision = OperatorTransactionPolicyDecisionV1(
                selected_rows=rows,
                model_forwards=0,
                conflict_attempts=0,
                deferred=not rows,
            )
        else:
            belief = (
                None
                if conflict_belief_by_row_id is None
                else conflict_belief_by_row_id.get(example.row_id)
            )
            decision = decode_bounded_transaction_set(
                scorer, example, k=k, conflict_belief=belief
            )
            model_calls += decision.model_forwards
        conflict_attempts += decision.conflict_attempts
        commit = execute_transaction_policy_decision(
            example, decision, reference_table_builder=reference_table_builder
        )
        compiler_time_ms += (time.perf_counter() - start) * 1000.0
        if commit is not None and commit.succeeded:
            committed += 1
        if decision.selected_rows:
            actions_selected += len(decision.selected_rows)
            forwards_with_actions += 1
        if example.accepted_row_sets:
            if decision.selected_rows in example.accepted_row_sets:
                exact_matches += 1
            best_mass = max(
                (
                    len(decision.selected_rows & accepted) / len(accepted)
                    for accepted in example.accepted_row_sets
                    if accepted
                ),
                default=0.0,
            )
            mass_total += best_mass
    n = len(examples)
    return OperatorTransactionPolicyReportV1(
        head_family=head_family,
        arm=arm,
        k=k,
        n=n,
        exact_set_match_rate=exact_matches / n if n else 0.0,
        accepted_set_mass=mass_total / n if n else 0.0,
        final_state_success_rate=committed / n if n else 0.0,
        conflict_attempts=conflict_attempts,
        actions_per_forward=(
            actions_selected / forwards_with_actions if forwards_with_actions else 0.0
        ),
        model_calls=model_calls,
        compiler_time_ms=compiler_time_ms,
    )


__all__ = [
    "MAX_TRANSACTION_POLICY_CANDIDATES",
    "OPERATOR_TRANSACTION_POLICY_HEAD_FAMILIES",
    "OperatorTransactionPolicyDecisionV1",
    "OperatorTransactionPolicyExampleV1",
    "OperatorTransactionPolicyHeadFamily",
    "OperatorTransactionPolicyReportV1",
    "decode_bounded_transaction_set",
    "evaluate_operator_transaction_policy",
    "execute_transaction_policy_decision",
    "score_transaction_candidates",
    "train_operator_transaction_policy",
    "transaction_policy_set_loss",
]
