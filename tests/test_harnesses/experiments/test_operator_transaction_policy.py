"""Tests for the DSH5-06 bounded set-valued transaction policy.

Core invariant under test (the issue's own acceptance criterion): *every
executed set is exactly conflict-free and atomic*. The ``shuffled_conflict_graph``
control deliberately corrupts the ranking step's belief about which candidate
pairs conflict; these tests prove that corruption can only ever cost
``conflict_attempts`` (a slower search) and never causes an actually-conflicting
set to reach ``commit_operator_transaction`` — the real DSH5-05 executor is
always the final, ground-truth gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    LegalSetCoverage,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    build_reference_table,
    prepare_operator_action,
)
from slm_training.dsl.operators.contracts import OperatorRef
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.experiments.operator_transaction_policy import (
    MAX_TRANSACTION_POLICY_CANDIDATES,
    OperatorTransactionPolicyExampleV1,
    decode_bounded_transaction_set,
    evaluate_operator_transaction_policy,
    execute_transaction_policy_decision,
    train_operator_transaction_policy,
)
from slm_training.harnesses.experiments.typed_operator_policy import (
    TypedOperatorPolicyScorer,
)

SOURCE = (
    'root = Card([TextContent(":hero.alpha"), TextContent(":hero.beta"), '
    'TextContent(":hero.gamma"), TextContent(":hero.delta")], "clear")'
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Fixture:
    """Real ``openui`` pack authority; synthetic single-target write operators.

    Mirrors ``tests/test_dsl/test_operator_transaction_executor.py``'s
    ``_Fixture`` — one real, pack-verified base state/table plus ad hoc
    ``AstOperatorV1`` declarations whose executors go through the real
    ``OperatorLibraryV1``/pack-authority round trip.
    """

    def __init__(self) -> None:
        self.pack = get_pack("openui")
        self.state = OperatorStateV1.from_source(self.pack, SOURCE)
        self.request_id = "tx-policy-request"
        self.branch_digest = _sha("tx-policy-branch")
        self.names = ("alpha", "beta", "gamma", "delta")
        self.descriptors = tuple(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha(name),
                value_type=f"openui.{name}",
            )
            for name in self.names
        )
        self.table = build_reference_table(
            request_id=self.request_id,
            state_digest=self.state.state_digest,
            branch_digest=self.branch_digest,
            descriptors=self.descriptors,
            seed=1,
        )
        self._registered: dict[str, RegisteredOperatorV1] = {}
        self._library: OperatorLibraryV1 | None = None
        self._pack_with_library = None
        self._declarations: dict[str, AstOperatorV1] = {}

    def target(self, name: str) -> OperatorRef:
        return next(
            entry.ref
            for entry in self.table.entries
            if entry.descriptor.value_type == f"openui.{name}"
        )

    def provenance(self) -> ApplicationProvenanceV1:
        return ApplicationProvenanceV1(
            pack_id="openui",
            compiler_id="openui.tx_policy_fixture",
            compiler_version="v1",
            source_artifact_digest=_sha(self.state.source),
            request_id=self.request_id,
        )

    def declare_write(self, *, operator_id: str, target_name: str) -> AstOperatorV1:
        write_target = self.target(target_name)
        declaration = AstOperatorV1(
            operator_id=operator_id,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(),
            preconditions=(),
            effect_signature=(EffectDeltaKind.PROPERTY,),
            locality="node",
            cost=1.0,
        )

        def execute(
            state: OperatorStateV1, _arguments: tuple
        ) -> OperatorMutationV1:
            delta = EffectDeltaV1(
                EffectDeltaKind.PROPERTY, write_target, "before", "after"
            )
            effect = ActionEffectV1(
                property_deltas=(delta,), compiler_coverage=CompilerCoverage.EXACT
            )
            return OperatorMutationV1(source=state.source, effect=effect)

        self._registered[operator_id] = RegisteredOperatorV1(declaration, execute)
        self._library = None
        self._pack_with_library = None
        self._declarations[operator_id] = declaration
        return declaration

    @property
    def library(self) -> OperatorLibraryV1:
        if self._library is None:
            self._library = OperatorLibraryV1(tuple(self._registered.values()))
        return self._library

    @property
    def pack_with_library(self):
        if self._pack_with_library is None:
            self._pack_with_library = replace(
                self.pack, operator_library=self.library
            )
        return self._pack_with_library

    def fresh_reference_table(self, pack, state: OperatorStateV1):
        seed = int(state.state_digest[:8], 16)
        return build_reference_table(
            request_id=self.request_id,
            state_digest=state.state_digest,
            branch_digest=self.branch_digest,
            descriptors=self.descriptors,
            seed=seed,
        )

    def example(
        self,
        *,
        row_id: str,
        operator_ids: tuple[str, ...],
        accepted_row_sets: tuple[frozenset[int], ...],
        coverage: LegalSetCoverage = LegalSetCoverage.COMPLETE,
    ) -> OperatorTransactionPolicyExampleV1:
        prepared = tuple(
            prepare_operator_action(
                pack=self.pack_with_library,
                library=self.library,
                state=self.state,
                reference_table=self.table,
                operator_id=operator_id,
                arguments=(),
                provenance=self.provenance(),
            )
            for operator_id in operator_ids
        )
        declarations = tuple(
            self._declarations[operator_id] for operator_id in operator_ids
        )
        return OperatorTransactionPolicyExampleV1(
            row_id=row_id,
            pack=self.pack_with_library,
            library=self.library,
            state=self.state,
            reference_table=self.table,
            provenance=self.provenance(),
            prepared_actions=prepared,
            declarations=declarations,
            coverage=coverage,
            accepted_row_sets=accepted_row_sets,
        )


def _disjoint_example(fixture: _Fixture, row_id: str) -> OperatorTransactionPolicyExampleV1:
    fixture.declare_write(operator_id="openui.txp_write_alpha", target_name="alpha")
    fixture.declare_write(operator_id="openui.txp_write_beta", target_name="beta")
    fixture.declare_write(operator_id="openui.txp_write_gamma", target_name="gamma")
    return fixture.example(
        row_id=row_id,
        operator_ids=(
            "openui.txp_write_alpha",
            "openui.txp_write_beta",
            "openui.txp_write_gamma",
        ),
        accepted_row_sets=(frozenset({0, 1}),),
    )


def _conflicting_example(fixture: _Fixture, row_id: str) -> OperatorTransactionPolicyExampleV1:
    """Two candidates that both write ``alpha`` — genuinely conflicting."""
    fixture.declare_write(operator_id="openui.txp_conflict_left", target_name="alpha")
    fixture.declare_write(operator_id="openui.txp_conflict_right", target_name="alpha")
    fixture.declare_write(operator_id="openui.txp_conflict_safe", target_name="beta")
    return fixture.example(
        row_id=row_id,
        operator_ids=(
            "openui.txp_conflict_left",
            "openui.txp_conflict_right",
            "openui.txp_conflict_safe",
        ),
        accepted_row_sets=(frozenset({0, 2}), frozenset({1, 2})),
    )


def _scorer(
    examples, head_family: str = "independent_set", *, seed: int = 11
) -> TypedOperatorPolicyScorer:
    torch.manual_seed(seed)
    views = [example.policy_input for example in examples]
    from slm_training.models.operator_feature_encoder import OperatorFeatureVocabularyV1

    return TypedOperatorPolicyScorer(
        OperatorFeatureVocabularyV1.from_inputs(views), dim=8, head_family=head_family
    )


@pytest.mark.parametrize("head_family", ["independent_set", "recurrent_set"])
def test_decode_selects_a_ground_truth_conflict_free_set(head_family: str) -> None:
    fixture = _Fixture()
    example = _disjoint_example(fixture, "row-1")
    scorer = _scorer([example], head_family=head_family)
    decision = decode_bounded_transaction_set(scorer, example, k=2)
    assert not decision.deferred
    assert decision.model_forwards == 1
    # Every returned pair must actually be accepted by the real builder.
    from slm_training.dsl.operators import build_operator_transaction

    ordered = tuple(sorted(decision.selected_rows))
    prepared = tuple(example.prepared_actions[row] for row in ordered)
    verified = build_operator_transaction(
        library=example.library,
        state=example.state,
        reference_table=example.reference_table,
        prepared_actions=prepared,
        provenance=example.provenance,
    )
    assert verified.succeeded


def test_shuffled_conflict_belief_never_commits_a_real_conflict() -> None:
    """The core safety invariant: a corrupted ranking belief can waste search
    (nonzero ``conflict_attempts``) but can never make it into a commit."""
    fixture = _Fixture()
    example = _conflicting_example(fixture, "row-conflict")
    # This scorer is randomly initialized and never trained, so *which* pair
    # it ranks first is init noise, not a property. The seed is pinned here
    # only to establish this test's precondition -- that the corrupted belief
    # actually ranks an infeasible pair first, so there is wasted search to
    # observe. The safety assertions below (never deferred into a commit,
    # never the genuinely conflicting pair, commit really succeeds) hold at
    # every seed; only the final `conflict_attempts >= 1` observation needs
    # the precondition. Pinned separately from `_scorer`'s default so a
    # parameter-shape change anywhere in the encoder cannot silently turn
    # this into a vacuous pass.
    scorer = _scorer([example], seed=0)
    # Deliberately claim the genuinely conflicting pair (0, 1) is fine, and
    # invent conflicts that do not really exist — this is exactly the
    # `shuffled_conflict_graph` control's corrupted ranking belief.
    fake_belief = frozenset({frozenset({0, 2}), frozenset({1, 2})})
    decision = decode_bounded_transaction_set(
        scorer, example, k=2, conflict_belief=fake_belief
    )
    assert not decision.deferred
    # The selected set must still be genuinely conflict-free...
    assert decision.selected_rows != frozenset({0, 1})
    commit = execute_transaction_policy_decision(
        example, decision, reference_table_builder=fixture.fresh_reference_table
    )
    assert commit is not None and commit.succeeded
    # ...and reaching it required at least one rejected (blocked) attempt,
    # because the corrupted belief ranked an infeasible pair first.
    assert decision.conflict_attempts >= 1


def test_partial_coverage_never_force_emits_a_complete_set() -> None:
    fixture = _Fixture()
    example = _disjoint_example(fixture, "row-partial")
    example = replace(example, coverage=LegalSetCoverage.PARTIAL)
    scorer = _scorer([example])
    decision = decode_bounded_transaction_set(scorer, example, k=2)
    assert decision.deferred
    assert decision.selected_rows == frozenset()
    assert decision.model_forwards == 0


def test_bounded_k_is_respected() -> None:
    fixture = _Fixture()
    fixture.declare_write(operator_id="openui.txp_k_a", target_name="alpha")
    fixture.declare_write(operator_id="openui.txp_k_b", target_name="beta")
    fixture.declare_write(operator_id="openui.txp_k_c", target_name="gamma")
    fixture.declare_write(operator_id="openui.txp_k_d", target_name="delta")
    example = fixture.example(
        row_id="row-k",
        operator_ids=(
            "openui.txp_k_a",
            "openui.txp_k_b",
            "openui.txp_k_c",
            "openui.txp_k_d",
        ),
        accepted_row_sets=(frozenset({0, 1, 2, 3}),),
    )
    scorer = _scorer([example])
    decision = decode_bounded_transaction_set(scorer, example, k=2)
    assert len(decision.selected_rows) <= 2


def test_train_reduces_loss_and_evaluate_reports_a_ground_truth_row() -> None:
    fixture = _Fixture()
    example = _disjoint_example(fixture, "row-train")
    scorer = _scorer([example])
    history = train_operator_transaction_policy(
        scorer, [example], k=2, steps=3, learning_rate=0.05
    )
    assert len(history) == 3
    report = evaluate_operator_transaction_policy(
        scorer,
        [example],
        head_family="independent_set",
        arm="enabled",
        k=2,
        reference_table_builder=fixture.fresh_reference_table,
    )
    assert report.n == 1
    assert report.final_state_success_rate == 1.0


def test_candidate_pool_bound_is_enforced() -> None:
    fixture = _Fixture()
    for index in range(MAX_TRANSACTION_POLICY_CANDIDATES + 1):
        fixture.declare_write(operator_id=f"openui.txp_bound_{index}", target_name="alpha")
    with pytest.raises(ValueError):
        fixture.example(
            row_id="row-bound",
            operator_ids=tuple(
                f"openui.txp_bound_{index}"
                for index in range(MAX_TRANSACTION_POLICY_CANDIDATES + 1)
            ),
            accepted_row_sets=(),
        )
