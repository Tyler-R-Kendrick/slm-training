"""Tests for VAR3-03 (SLM-433) real TurnDispositionHead training/evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import torch

from slm_training.data.flow.turn_disposition_corpus import build_turn_disposition_rows
from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RegisteredOperatorV1,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    collapse_conversation_trace,
    create_conversation_trace,
)
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.experiments.turn_disposition_training import (
    TurnDispositionClassifierV1,
    TurnDispositionExampleV1,
    evaluate_turn_disposition_arms,
    train_turn_disposition_head,
)
from slm_training.models.operator_feature_encoder import OperatorFeatureVocabularyV1
from slm_training.models.turn_disposition_head import (
    TurnDispositionLabel,
    TurnDispositionTargetV1,
)

SOURCE = 'root = TextContent(":hero.title")'


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.disposition_training_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="request-1",
    )


def _table(state: OperatorStateV1, branch: str, *, seed: int):
    return build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(),
        seed=seed,
    )


def _declaration(operator_id: str) -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=operator_id,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )


def _ambiguous_rows(seed_offset: int = 0):
    """Two always-legal operators (real >=2-candidate rows every step)."""

    def execute_set_x(_state, _arguments):
        return OperatorMutationV1(
            source='root = TextContent(":set.x")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    def execute_set_y(_state, _arguments):
        return OperatorMutationV1(
            source='root = TextContent(":set.y")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(_declaration("openui.set_x"), execute_set_x),
            RegisteredOperatorV1(_declaration("openui.set_y"), execute_set_y),
        )
    )
    pack = replace(get_pack("openui"), operator_library=library)
    root_state = OperatorStateV1.from_source(pack, SOURCE)
    branch = branch_fingerprint(root_state.state_digest, _sha(f"root-branch-{seed_offset}"))
    root_table = _table(root_state, branch, seed=1 + seed_offset)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state),
    )

    def append(trace, operator_id, *, seed):
        state = trace.current.state
        result = library.apply(pack, state, operator_id, (), _provenance(state))
        assert result.succeeded, result.application.rejection
        output_table = _table(result.state, trace.current.branch_digest, seed=seed)
        return append_operator_turn(
            trace,
            pack=pack,
            library=library,
            application=result.application,
            output_reference_table=output_table,
        )

    trace = append(trace, "openui.set_x", seed=2 + seed_offset)
    trace = append(trace, "openui.set_y", seed=3 + seed_offset)

    def authority_resolver(_node):
        return pack, library

    decision = collapse_conversation_trace(
        pack=pack, authority_resolver=authority_resolver, trace=trace
    )
    assert decision.rejection is None, decision.rejection

    rows, rejected = build_turn_disposition_rows(
        trace=trace,
        collapse=decision.collapse,
        authority_resolver=authority_resolver,
        action_frequency={},
    )
    assert rejected == ()
    return rows


def test_classifier_trains_and_reports_all_three_arms_on_a_real_corpus() -> None:
    torch.manual_seed(0)
    train_rows = [*_ambiguous_rows(seed_offset=0), *_ambiguous_rows(seed_offset=10)]
    train_examples = [TurnDispositionExampleV1.from_row(row) for row in train_rows]
    vocabulary = OperatorFeatureVocabularyV1.from_inputs(
        [example.view for example in train_examples]
    )
    classifier = TurnDispositionClassifierV1(vocabulary, dim=8)

    history = train_turn_disposition_head(
        classifier, train_examples, steps=20, learning_rate=0.05
    )
    assert len(history) == 20
    assert all(value >= 0.0 for value in history)

    arms = evaluate_turn_disposition_arms(classifier, train_examples)
    assert set(arms) == {"disposition_off", "derived_only", "disposition_trained"}
    for arm_report in arms.values():
        assert arm_report["case_count"] == len(train_examples)
        assert 0.0 <= arm_report["wrong_op_rate"] <= 1.0
        assert 0.0 <= arm_report["abstention_rate"] <= 1.0

    # disposition_off and derived_only are expected to coincide exactly on
    # this real corpus (see the module docstring): it has no
    # out_of_scope/answer turns, and every row's top-ranked candidate is the
    # same regardless of which of the two baselines "emits" it.
    assert arms["disposition_off"] == arms["derived_only"]


def test_masked_forced_singleton_examples_train_to_a_flat_zero_loss() -> None:
    """A corpus made entirely of forced-singleton rows carries no learning
    signal -- turn_disposition_losses masks it, so training loss stays
    exactly zero throughout, proving the mask is actually wired end to end."""
    torch.manual_seed(0)
    rows = _ambiguous_rows()
    forced_only = [row for row in rows if row.target.forced_singleton]
    # None of these rows are forced (both operators always legal), so
    # synthesize the forced case directly from one non-forced example's
    # view, overriding only the target -- the encoder/classifier plumbing
    # under test does not care where the target came from.
    examples = [
        replace(
            TurnDispositionExampleV1.from_row(row),
            target=TurnDispositionTargetV1(
                label=TurnDispositionLabel.EMIT, forced_singleton=True
            ),
        )
        for row in rows
    ]
    vocabulary = OperatorFeatureVocabularyV1.from_inputs(
        [example.view for example in examples]
    )
    classifier = TurnDispositionClassifierV1(vocabulary, dim=8)
    history = train_turn_disposition_head(
        classifier, examples, steps=5, learning_rate=0.05
    )
    assert history == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert forced_only == []  # sanity: the real rows were indeed non-forced


class _ClassifierThatMustNeverBeCalled(TurnDispositionClassifierV1):
    """A poisoned classifier double: raises if ``forward`` is ever invoked."""

    def forward(self, view):  # noqa: D102 - test double, no docstring needed
        raise AssertionError("the classifier must not be consulted for a forced-singleton row")


def test_forced_singleton_examples_never_consult_the_classifier() -> None:
    """I1: a COMPLETE-coverage singleton is a deterministic bypass that
    outranks any learned score. ``evaluate_turn_disposition_arms`` must
    report ``emit`` for a forced-singleton example without ever calling the
    classifier -- proven here with a classifier double that raises if
    invoked, mirroring this repo's own
    ``test_out_of_scope_never_consults_score_provider`` convention."""
    rows = _ambiguous_rows()
    forced_examples = [
        replace(
            TurnDispositionExampleV1.from_row(row),
            target=TurnDispositionTargetV1(
                label=TurnDispositionLabel.EMIT, forced_singleton=True
            ),
        )
        for row in rows
    ]
    vocabulary = OperatorFeatureVocabularyV1.from_inputs(
        [example.view for example in forced_examples]
    )
    poisoned_classifier = _ClassifierThatMustNeverBeCalled(vocabulary, dim=8)

    # No AssertionError from the poisoned classifier's `forward` is the
    # actual invariant under test: `_trained_row` must bypass it entirely
    # for every forced-singleton example.
    arms = evaluate_turn_disposition_arms(poisoned_classifier, forced_examples)
    assert arms["disposition_trained"]["case_count"] == len(forced_examples)
    assert arms["disposition_trained"]["abstention_rate"] == 0.0
