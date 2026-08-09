"""VSS3-02 (SLM-70): cost-to-go energy scorer tests.

Exercises the guarantee-critical contract directly: the scorer is a
permutation-only ranking signal that can never alter the hard candidate set,
UNKNOWN/censored rows are masked from hard losses, multiple supported alternatives
are not forced into a one-hot order, and a tiny fixture overfits deterministically.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from slm_training.dsl.solver.state import DomainValue, HoleId
from slm_training.models.solver_energy import (
    CandidateEnergyInput,
    CandidateEnergyRanker,
    CandidateEnergyScorer,
    cost_target_from_row,
    energy_pairwise_loss,
    energy_regression_loss,
    work_scalar,
)

HOLE = HoleId(namespace="root", path=("a",), kind="slot")


def val(payload: int) -> DomainValue:
    return DomainValue.create("t", payload)


def feature_fn(state, hole_id, values):
    cand = torch.tensor([[float(v.payload), 1.0] for v in values], dtype=torch.float32)
    return CandidateEnergyInput(
        state_fingerprint="fp",
        state_features=torch.zeros(2),
        hole_features=torch.zeros(1),
        candidate_ids=tuple(values),
        candidate_features=cand,
    )


def make_scorer(seed: int = 0) -> CandidateEnergyScorer:
    torch.manual_seed(seed)
    return CandidateEnergyScorer(state_dim=2, hole_dim=1, candidate_dim=2, hidden_dim=16)


class _BadScorer:
    scorer_id = "bad"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def energies(self, inp: CandidateEnergyInput) -> torch.Tensor:
        n = inp.candidate_features.shape[0]
        if self.mode == "count":
            return torch.zeros(n + 1)
        if self.mode == "nan":
            out = torch.zeros(n)
            out[0] = float("nan")
            return out
        if self.mode == "inf":
            out = torch.zeros(n)
            out[0] = float("inf")
            return out
        return torch.zeros(n)


# 1. exactly one energy per live candidate
def test_scorer_one_energy_per_candidate():
    scorer = make_scorer()
    values = (val(1), val(2), val(3))
    out = scorer(feature_fn(None, HOLE, values))
    assert out.energies.shape == (3,)
    assert out.candidate_ids == values
    assert out.scorer_id == "solver-energy-v1"


# 2. ranking is a permutation — hard set identical before/after
def test_ranker_output_is_a_permutation():
    ranker = CandidateEnergyRanker(make_scorer(), feature_fn)
    values = (val(1), val(2), val(3), val(4))
    ranked = ranker.rank(None, HOLE, values)
    assert set(v.to_dict()["value"] for v in ranked) == {1, 2, 3, 4}
    assert len(ranked) == len(values)


# 3. bad scorer output triggers deterministic fallback, membership unchanged
def test_bad_energy_falls_back_without_altering_membership():
    values = (val(1), val(2), val(3))
    for mode in ("count", "nan", "inf"):
        ranker = CandidateEnergyRanker(_BadScorer(mode), feature_fn)
        ranked = ranker.rank(None, HOLE, values)
        assert ranked == values  # deterministic identity fallback
        assert ranker.fallback_count == 1


# 4. certified-removed values (absent from the live set) can't be reintroduced
def test_ranker_cannot_reintroduce_removed_values():
    ranker = CandidateEnergyRanker(make_scorer(), feature_fn)
    live_subset = (val(2), val(5))  # 1, 3, 4 were certified-removed upstream
    ranked = ranker.rank(None, HOLE, live_subset)
    assert set(v.payload for v in ranked) == {2, 5}


# 5. regression loss masks unobserved (UNKNOWN/censored) rows
def test_regression_loss_masks_unobserved():
    energies = torch.tensor([0.0, 5.0, 5.0])
    targets = torch.tensor([0.0, 1.0, 1.0])
    all_masked = torch.tensor([0.0, 0.0, 0.0])
    assert float(energy_regression_loss(energies, targets, all_masked)) == 0.0
    only_first = torch.tensor([1.0, 0.0, 0.0])
    # Only the (matching) first row counts -> zero; the mismatched rows are masked.
    assert float(energy_regression_loss(energies, targets, only_first)) == 0.0


# 6. pairwise orders by lower observed cost
def test_pairwise_prefers_lower_cost():
    # energy[0] high but its cost is low -> the pair is penalized.
    energies = torch.tensor([2.0, 0.0], requires_grad=True)
    targets = torch.tensor([0.5, 2.0])  # candidate 0 is cheaper
    observed = torch.tensor([1.0, 1.0])
    loss = energy_pairwise_loss(energies, targets, ["g", "g"], observed, margin=0.1)
    assert float(loss) > 0.0
    # Correctly ordered energies incur no penalty.
    good = torch.tensor([0.0, 2.0])
    assert float(energy_pairwise_loss(good, targets, ["g", "g"], observed, margin=0.1)) == 0.0


# 7. equal-cost alternatives are not forced apart (no one-hot collapse)
def test_equal_cost_alternatives_not_forced_apart():
    energies = torch.tensor([0.3, 0.3])
    targets = torch.tensor([1.0, 1.0])  # equal observed cost
    observed = torch.tensor([1.0, 1.0])
    # Equal-cost pairs are skipped by the pairwise loss.
    assert float(energy_pairwise_loss(energies, targets, ["g", "g"], observed)) == 0.0


# 8. tiny fixture overfits deterministically and orders by cost
def test_tiny_fixture_overfits_and_orders_by_cost():
    torch.manual_seed(0)
    scorer = CandidateEnergyScorer(state_dim=2, hole_dim=1, candidate_dim=2, hidden_dim=16)
    values = (val(1), val(2), val(3))
    inp = feature_fn(None, HOLE, values)
    targets = torch.tensor([math.log1p(10.0), math.log1p(5.0), math.log1p(20.0)])
    observed = torch.ones(3)
    opt = torch.optim.Adam(scorer.parameters(), lr=0.05)
    first = None
    for _ in range(300):
        opt.zero_grad()
        energies = scorer.energies(inp)
        loss = energy_regression_loss(energies, targets, observed)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < first * 0.1  # loss dropped by >10x
    ranker = CandidateEnergyRanker(scorer, feature_fn)
    ranked = ranker.rank(None, HOLE, values)
    # cheapest (payload 2, work 5) first, dearest (payload 3, work 20) last.
    assert [v.payload for v in ranked] == [2, 1, 3]


# 9. fixed seed produces stable ordering
def test_fixed_seed_stable_ordering():
    values = (val(1), val(2), val(3), val(4))
    a = CandidateEnergyRanker(make_scorer(7), feature_fn).rank(None, HOLE, values)
    b = CandidateEnergyRanker(make_scorer(7), feature_fn).rank(None, HOLE, values)
    assert [v.payload for v in a] == [v.payload for v in b]


# 10. work-target v1 from a VSS3-01 candidate_cost row
def test_work_target_v1():
    row = {
        "remaining_expanded_nodes": 4,
        "remaining_verifier_calls": 3,
        "remaining_backtracks": 1,
        "remaining_decisions": 2,
    }
    assert work_scalar(
        expanded_nodes=4, verifier_calls=3, backtracks=1, decisions=2
    ) == 10.0
    assert cost_target_from_row(row) == math.log1p(10.0)


# 11. config is disabled by default (old configs/checkpoints load unchanged)
def test_config_energy_disabled_by_default():
    from slm_training.models.twotower import TwoTowerConfig

    defaults = {f.name: f.default for f in dataclasses.fields(TwoTowerConfig)}
    assert defaults["solver_energy_head"] is False
    assert defaults["solver_ranker"] == "deterministic"
    assert defaults["solver_energy_loss_weight"] == 0.0
    assert defaults["solver_energy_cost_version"] == "v1"


# 12. no final/witness text can enter the scorer input contract
def test_no_text_leak_in_scorer_input():
    fields = {f.name for f in dataclasses.fields(CandidateEnergyInput)}
    assert fields == {
        "state_fingerprint",
        "state_features",
        "hole_features",
        "candidate_ids",
        "candidate_features",
    }
    assert not (fields & {"text", "final_text", "witness", "witness_text", "source"})


# ------------------------------------------------------------------------- #
# M1: energy-reranker lever over the compiler-decision seam (twotower).
# ------------------------------------------------------------------------- #

_HERO = (
    'root = Stack([b1], "column")\n'
    "b1 = Card([b2, b3])\n"
    'b2 = TextContent(":slot_0")\n'
    'b3 = TextContent(":slot_1")'
)


def _hero_records():
    from slm_training.dsl.schema import ExampleRecord

    return [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=_HERO,
            placeholders=[":slot_0", ":slot_1"],
            split="train",
        )
    ]


def _lever_model(**kwargs):
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    torch.manual_seed(123)
    return TwoTowerModel.from_records(
        _hero_records(),
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            **kwargs,
        ),
    )


# 13. the profile arm and its zero-weight control build the identical head
def test_solver_energy_profile_prebuild_parity():
    control = _lever_model(structural_aux_head_profile="solver-energy")
    candidate = _lever_model(
        structural_aux_head_profile="solver-energy",
        solver_energy_loss_weight=1.0,
        solver_energy_decode_weight=1.0,
    )
    assert control.solver_energy_head is not None
    assert candidate.solver_energy_head is not None
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )


# 14. bias covers exactly the supplied candidate ids and respects the clamp
def test_solver_energy_bias_covers_only_supplied_candidates_and_is_bounded():
    model = _lever_model(
        structural_aux_head_profile="solver-energy",
        solver_energy_loss_weight=1.0,
        solver_energy_decode_weight=0.5,
    )
    ctx = torch.randn(1, 4, 32)
    ctx_pad = torch.zeros(1, 4, dtype=torch.bool)
    hidden = torch.randn(32)
    candidates = (2, 3, 5)
    bias = model._solver_energy_bias(ctx, ctx_pad, hidden, candidates)
    assert bias is not None
    assert bias.shape == (len(candidates),)
    assert bool((bias.abs() <= 0.5 + 1e-6).all())


# 15. weight <= 0 (or a missing head) contributes nothing
def test_solver_energy_bias_disabled_returns_none():
    model = _lever_model(structural_aux_head_profile="solver-energy")
    ctx = torch.randn(1, 4, 32)
    ctx_pad = torch.zeros(1, 4, dtype=torch.bool)
    assert model._solver_energy_bias(ctx, ctx_pad, torch.randn(32), (2, 3)) is None


# 16. non-finite scorer output degrades to identity order with a counter
def test_solver_energy_bias_non_finite_falls_back_with_counter():
    from slm_training.models.decode_stats import collect_decode_stats

    model = _lever_model(
        structural_aux_head_profile="solver-energy",
        solver_energy_loss_weight=1.0,
        solver_energy_decode_weight=1.0,
    )

    class _NaNHead(torch.nn.Module):
        def energies(self, inp):
            out = torch.zeros(len(inp.candidate_ids))
            out[0] = float("nan")
            return out

    model.solver_energy_head = _NaNHead()
    ctx = torch.randn(1, 4, 32)
    ctx_pad = torch.zeros(1, 4, dtype=torch.bool)
    with collect_decode_stats() as stats:
        bias = model._solver_energy_bias(ctx, ctx_pad, torch.randn(32), (2, 3))
    assert bias is None
    assert stats.solver_energy_fallbacks == 1


# 17. enabling decode without the owning training loss is a hard error
def test_solver_energy_decode_weight_requires_training_loss():
    import pytest

    from slm_training.models.twotower import TwoTowerConfig

    with pytest.raises(ValueError, match="solver_energy_decode_weight"):
        TwoTowerConfig(
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            solver_energy_decode_weight=1.0,
        )


# 18. the pairwise objective on the seam decreases under head-only steps
def test_solver_energy_training_loss_decreases():
    model = _lever_model(
        structural_aux_head_profile="solver-energy",
        solver_energy_loss_weight=1.0,
        solver_energy_decode_weight=1.0,
    )
    records = _hero_records()
    optimizer = torch.optim.Adam(model.solver_energy_head.parameters(), lr=0.05)
    losses: list[float] = []
    for _ in range(5):
        model.training_loss(records)
        auxiliary = model.take_detached_auxiliary_loss()
        assert auxiliary is not None
        optimizer.zero_grad()
        auxiliary.backward()
        optimizer.step()
        losses.append(float(auxiliary))
    assert losses[-1] < losses[0]
