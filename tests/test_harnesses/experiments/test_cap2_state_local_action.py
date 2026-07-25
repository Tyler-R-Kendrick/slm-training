"""Regression tests for the CAP2-03 state-local action-head fixture harness.

DSH3-21 adds fixture-parity and optimizer-visibility coverage for
``LocalFlatHead`` on top of the original CAP2-03 wiring tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap2_state_local_action import (
    HIDDEN_DIM,
    StateLocalActionReport,
    _evaluate_head,
    _make_head,
    fixture_states,
    run_fixture,
)
from slm_training.models.local_action_head import LocalFlatHead


HEAD_FAMILIES = (
    "global_masked",
    "local_flat",
    "ternary_digit",
    "ternary_ecoc",
    "grammar_factorized",
)


def test_fixture_states_are_deterministic() -> None:
    """The fixture state set is a fixed, hard-coded tuple across calls."""
    assert fixture_states() == fixture_states()


def test_run_fixture_produces_versioned_report() -> None:
    """The default matrix runs all five head families and returns a versioned report."""
    report = run_fixture()
    assert isinstance(report, StateLocalActionReport)
    assert report.version == "cap2-03-v1"
    assert report.hidden_dim == HIDDEN_DIM
    assert {r.head_family for r in report.results} == set(HEAD_FAMILIES)


def test_local_flat_head_zero_step_fixture_parity() -> None:
    """DSH3-21 trainability plumbing must not change fixture-mode behavior.

    Before materialize() is ever called, LocalFlatHead lazily allocates
    embeddings exactly as it did before DSH3-21, so the oracle-wiring and
    random-init fixture pass must recover the same accuracy as the CAP2-03
    baseline report (docs/design/iter-cap2-03-state-local-action-heads-20260718.md).
    """
    states = fixture_states()
    result = _evaluate_head("local_flat", states)
    assert result.oracle_accuracy == 1.0
    assert result.random_init_accuracy == 1.0
    assert result.forced_states_recovered == 1
    assert result.abstain_count == 0
    assert result.detected_error_count == 0


def test_local_flat_head_from_make_head_is_a_plain_module() -> None:
    """_make_head builds an un-materialized LocalFlatHead (fixture/wiring mode)."""
    head = _make_head("local_flat")
    assert isinstance(head, LocalFlatHead)
    assert not head.materialized


def test_local_flat_head_can_be_materialized_over_fixture_vocabulary() -> None:
    """The full fixture action vocabulary can be materialized before an optimizer.

    This proves the head used by this harness is optimizer-visible once the
    harness (or a future training loop built on it) commits to a fixed action
    vocabulary, without changing the harness's own fixture-mode code path.
    """
    import torch

    states = fixture_states()
    flat_actions = sorted({action for state in states for action in state.legal_actions})
    head = LocalFlatHead(HIDDEN_DIM)
    head.materialize(flat_actions)
    assert head.materialized
    names = {name for name, _ in head.named_parameters()}
    for action in flat_actions:
        assert f"action_embeddings.{action}" in names
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    assert sum(p.numel() for group in optimizer.param_groups for p in group["params"]) == sum(
        p.numel() for p in head.parameters()
    )


def test_all_head_families_replay_deterministically() -> None:
    """Two evaluations of the same head family produce identical oracle accuracy."""
    states = fixture_states()
    for family in HEAD_FAMILIES:
        r1 = _evaluate_head(family, states)
        r2 = _evaluate_head(family, states)
        assert r1.oracle_accuracy == r2.oracle_accuracy == 1.0
