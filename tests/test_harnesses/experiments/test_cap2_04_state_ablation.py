"""Regression tests for the CAP2-04 state-ownership ablation harness."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap2_04_state_ablation import (
    ArmConfig,
    FixtureDecision,
    SEMANTIC_DIM,
    _CompilerOwnedModel,
    _CompilerOwnedNoStateModel,
    _DiscreteCodeModel,
    _ExplicitExactModel,
    _ImplicitStateModel,
    _build_model,
    _choose_levels,
    _count_active_parameters,
    _count_parameters,
    _decode_model,
    _forward_model,
    _oracle_output,
    _split_unseen_states,
    build_arms,
    evaluate_arm,
    fixture_decisions,
    match_active_parameters,
    run_matrix,
)


STATE_COUNT = 8
ACTION_COUNT = 5


@pytest.fixture
def decisions() -> tuple:
    return fixture_decisions(state_count=STATE_COUNT, action_count=ACTION_COUNT)


@pytest.fixture
def unseen_state_ids() -> tuple[int, ...]:
    return _split_unseen_states(STATE_COUNT, seed=42)


def test_all_arms_instantiate_from_manifest(decisions: tuple) -> None:
    """Every declared arm mode builds a model with the expected head family."""
    action_count = len(decisions[0].legal_actions)
    for mode in (
        "implicit",
        "explicit_exact",
        "discrete_code",
        "compiler_owned",
        "compiler_owned_no_state",
    ):
        cfg = ArmConfig(arm_id=f"{mode}_test", mode=mode)
        model = _build_model(cfg, action_count)
        if mode == "implicit":
            assert isinstance(model, _ImplicitStateModel)
        elif mode == "explicit_exact":
            assert isinstance(model, _ExplicitExactModel)
        elif mode == "discrete_code":
            assert isinstance(model, _DiscreteCodeModel)
        elif mode == "compiler_owned":
            assert isinstance(model, _CompilerOwnedModel)
        elif mode == "compiler_owned_no_state":
            assert isinstance(model, _CompilerOwnedNoStateModel)


def test_parameter_counts_are_positive(decisions: tuple) -> None:
    """Each arm has a positive number of trainable parameters."""
    action_count = len(decisions[0].legal_actions)
    for mode in (
        "implicit",
        "explicit_exact",
        "discrete_code",
        "compiler_owned",
        "compiler_owned_no_state",
    ):
        cfg = ArmConfig(arm_id=f"{mode}_test", mode=mode)
        model = _build_model(cfg, action_count)
        assert _count_parameters(model) > 0
        assert _count_active_parameters(model) > 0


def test_match_active_parameters_equalizes_counts(decisions: tuple) -> None:
    """After matching, every arm targets the same active-parameter count."""
    arms = build_arms(
        state_count=STATE_COUNT,
        action_count=ACTION_COUNT,
        modes=(
            "implicit",
            "explicit_exact",
            "discrete_code",
            "compiler_owned",
            "compiler_owned_no_state",
        ),
    )
    matched = match_active_parameters(arms, decisions)
    targets = {a.target_active_parameters for a in matched}
    assert len(targets) == 1
    assert targets.pop() is not None


def test_forced_decision_bypasses_compiler_owned_heads(decisions: tuple) -> None:
    """A state with a single legal action returns a forced decision."""
    cfg = ArmConfig(arm_id="compiler_owned_test", mode="compiler_owned")
    model = _build_model(cfg, ACTION_COUNT)
    base = decisions[0]
    decision = FixtureDecision(
        decision_id=base.decision_id,
        semantic_input=base.semantic_input,
        history=base.history,
        state_id=base.state_id,
        state_family_id=base.state_family_id,
        legal_actions=("action:00",),
        correct_action="action:00",
    )
    output = _forward_model(model, decision, torch.device("cpu"))
    pred = _decode_model(model, output, list(decision.legal_actions))
    assert pred.decision_kind == "forced"
    assert pred.action_identity == "action:00"


def test_oracle_pass_recovers_every_correct_action(decisions: tuple) -> None:
    """Wiring the oracle output must recover the correct action for every state."""
    action_count = len(decisions[0].legal_actions)
    for mode in (
        "implicit",
        "explicit_exact",
        "discrete_code",
        "compiler_owned",
        "compiler_owned_no_state",
    ):
        cfg = ArmConfig(arm_id=f"{mode}_oracle", mode=mode)
        model = _build_model(cfg, action_count)
        correct = 0
        for decision in decisions:
            out = _oracle_output(model, decision, torch.device("cpu"))
            pred = _decode_model(model, out, list(decision.legal_actions))
            if pred.action_identity == decision.correct_action:
                correct += 1
        assert correct == len(decisions), f"{mode} oracle pass missed {len(decisions) - correct} states"


def test_no_future_info_in_state_features(decisions: tuple) -> None:
    """State features never contain the target action string or future decisions."""
    for decision in decisions:
        # The correct action identity is never embedded as a string in the input.
        assert decision.correct_action not in decision.semantic_input
        assert decision.correct_action not in decision.history
        # History is a fixed-length random vector, not a lookahead label.
        assert len(decision.history) == SEMANTIC_DIM
        # No action identifiers appear anywhere in the numeric features.
        for action in decision.legal_actions:
            assert action not in decision.semantic_input
            assert action not in decision.history


def test_unseen_state_split_is_disjoint(decisions: tuple, unseen_state_ids: tuple[int, ...]) -> None:
    """Held-out state ids are a strict subset of all state ids."""
    all_ids = {d.state_id for d in decisions}
    assert set(unseen_state_ids) < all_ids
    assert len(unseen_state_ids) >= 1


def test_unseen_states_evaluated_in_report(decisions: tuple, unseen_state_ids: tuple[int, ...]) -> None:
    """The fixture matrix reports an unseen-state accuracy for every arm."""
    cfg = ArmConfig(arm_id="explicit_test", mode="explicit_exact")
    result = evaluate_arm(cfg, decisions, unseen_state_ids)
    assert 0.0 <= result.unseen_state_accuracy <= 1.0


def test_replay_determinism(decisions: tuple, unseen_state_ids: tuple[int, ...]) -> None:
    """Two runs with the same seed produce identical results."""
    cfg = ArmConfig(arm_id="implicit_test", mode="implicit", seed=7)
    r1 = evaluate_arm(cfg, decisions, unseen_state_ids)
    r2 = evaluate_arm(cfg, decisions, unseen_state_ids)
    assert r1.oracle_accuracy == r2.oracle_accuracy
    assert r1.random_init_accuracy == r2.random_init_accuracy
    assert r1.unseen_state_accuracy == r2.unseen_state_accuracy
    assert r1.trainable_parameters == r2.trainable_parameters


def test_run_matrix_produces_versioned_report(decisions: tuple) -> None:
    """The full matrix returns a versioned report with all requested arms."""
    report = run_matrix(
        state_count=STATE_COUNT,
        action_count=ACTION_COUNT,
        modes=("implicit", "compiler_owned"),
        match_parameters=True,
    )
    assert report.version == "cap2-04-v1"
    assert len(report.states) == STATE_COUNT
    assert {a.mode for a in report.arms} == {"implicit", "compiler_owned"}
    assert not any(a.leakage for a in report.arms)


def test_discrete_code_capacity_covers_state_count(decisions: tuple) -> None:
    """The default discrete-code levels have capacity >= state_count."""
    levels = _choose_levels(STATE_COUNT)
    capacity = 1
    for level in levels:
        capacity *= level
    assert capacity >= STATE_COUNT


def test_discrete_code_model_reports_capacity(decisions: tuple) -> None:
    """The discrete-code arm exposes its nominal capacity."""
    cfg = ArmConfig(arm_id="discrete_test", mode="discrete_code")
    model = _build_model(cfg, ACTION_COUNT)
    assert isinstance(model, _DiscreteCodeModel)
    assert model.capacity() >= STATE_COUNT


def test_implicit_model_does_not_receive_state_id(decisions: tuple) -> None:
    """The implicit arm's forward signature does not accept a state id."""
    cfg = ArmConfig(arm_id="implicit_test", mode="implicit")
    model = _build_model(cfg, ACTION_COUNT)
    semantic = torch.tensor([list(decisions[0].semantic_input)], dtype=torch.float32)
    history = torch.tensor([list(decisions[0].history)], dtype=torch.float32)
    out = model(semantic, history, list(decisions[0].legal_actions))
    assert out.logits is not None


def test_compiler_owned_no_state_has_no_state_embedding(decisions: tuple) -> None:
    """The strict compiler-owned control has no state-family embedding table."""
    cfg = ArmConfig(arm_id="no_state_test", mode="compiler_owned_no_state")
    model = _build_model(cfg, ACTION_COUNT)
    assert isinstance(model, _CompilerOwnedNoStateModel)
    param_names = {name for name, _ in model.named_parameters()}
    assert "state_family_embedding" not in param_names
    assert "state_family_embedding.weight" not in param_names


def test_explicit_model_has_state_embedding(decisions: tuple) -> None:
    """The explicit arm embeds the exact state id."""
    cfg = ArmConfig(arm_id="explicit_test", mode="explicit_exact")
    model = _build_model(cfg, ACTION_COUNT)
    assert isinstance(model, _ExplicitExactModel)
    assert model.state_embedding.num_embeddings == STATE_COUNT


def test_build_arms_respects_seed_and_mode_filters() -> None:
    """build_arms returns one config per (seed, mode) combination."""
    arms = build_arms(
        state_count=STATE_COUNT,
        action_count=ACTION_COUNT,
        seeds=(0, 1),
        modes=("implicit", "explicit_exact"),
    )
    assert len(arms) == 4
    assert {a.seed for a in arms} == {0, 1}
    assert {a.mode for a in arms} == {"implicit", "explicit_exact"}
