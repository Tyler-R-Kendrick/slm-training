"""Tests for the VSS3-02 energy ranker (SLM-70)."""

from __future__ import annotations

import torch

from slm_training.dsl.solver.energy_ranker import (
    CandidateEnergyOutput,
    EnergyCandidateRanker,
    make_stub_energy_scorer,
)
from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleId, SolverBounds


def _hole_id(name: str) -> HoleId:
    return HoleId(namespace=name, path=(), kind="test")


def _state(values: tuple[DomainValue, ...]) -> FiniteDomainState:
    return FiniteDomainState(
        problem_id="p1",
        pack_id="openui",
        constraint_version="v1",
        bounds=SolverBounds(
            max_tokens=128,
            max_nodes=64,
            max_depth=8,
            max_backtracks=4,
            max_verifier_calls=16,
        ),
        holes=(),
        decision_level=0,
    )


def _value(tag: str, token_ids: list[int]) -> DomainValue:
    return DomainValue.create(tag, {"token_ids": token_ids})


def test_energy_ranker_returns_permutation() -> None:
    values = (_value("a", [1, 2]), _value("b", [3]), _value("c", [4, 5, 6]))
    ranker = EnergyCandidateRanker(make_stub_energy_scorer())
    ranked = ranker.rank(_state(values), _hole_id("h1"), values)
    assert set(ranked) == set(values)
    assert len(ranked) == len(values)
    assert ranker.fallback_count == 0


def test_energy_ranker_sorts_by_ascending_energy() -> None:
    values = (_value("long", [1] * 10), _value("short", [1]), _value("mid", [1] * 5))
    ranker = EnergyCandidateRanker(make_stub_energy_scorer())
    ranked = ranker.rank(_state(values), _hole_id("h1"), values)
    # Shorter payload -> lower energy.
    assert ranked[0].tag == "short"
    assert ranked[1].tag == "mid"
    assert ranked[2].tag == "long"


def test_energy_ranker_falls_back_on_wrong_length() -> None:
    values = (_value("a", [1]), _value("b", [2]))

    def bad_scorer(state, hole_id, values):
        return CandidateEnergyOutput(
            energies=torch.tensor([0.0]),
            candidate_ids=values[:1],
            scorer_id="bad",
        )

    ranker = EnergyCandidateRanker(bad_scorer)
    ranked = ranker.rank(_state(values), _hole_id("h1"), values)
    assert ranked == values
    assert ranker.fallback_count == 1


def test_energy_ranker_falls_back_on_nan() -> None:
    values = (_value("a", [1]), _value("b", [2]))

    def nan_scorer(state, hole_id, values):
        return CandidateEnergyOutput(
            energies=torch.tensor([0.0, float("nan")]),
            candidate_ids=values,
            scorer_id="nan",
        )

    ranker = EnergyCandidateRanker(nan_scorer)
    ranked = ranker.rank(_state(values), _hole_id("h1"), values)
    assert ranked == values
    assert ranker.fallback_count == 1


def test_energy_ranker_falls_back_on_membership_change() -> None:
    values = (_value("a", [1]), _value("b", [2]))
    other = _value("c", [3])

    def swap_scorer(state, hole_id, values):
        return CandidateEnergyOutput(
            energies=torch.tensor([0.0, 1.0]),
            candidate_ids=(other, values[1]),
            scorer_id="swap",
        )

    ranker = EnergyCandidateRanker(swap_scorer)
    ranked = ranker.rank(_state(values), _hole_id("h1"), values)
    assert ranked == values
    assert ranker.fallback_count == 1


def test_energy_ranker_empty_values() -> None:
    ranker = EnergyCandidateRanker(make_stub_energy_scorer())
    assert ranker.rank(_state(()), _hole_id("h1"), ()) == ()
