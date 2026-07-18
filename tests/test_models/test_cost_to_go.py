"""Tests for the VSS3-02 cost-to-go head on TwoTower (SLM-70)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.dsl.solver.state import DomainValue, HoleId
from slm_training.harnesses.distill.solver_supervision import CandidateCostRow, SearchCounters
from slm_training.models.twotower import (
    TwoTowerConfig,
    TwoTowerModel,
    _load_checkpoint_state,
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def _hole_id(name: str) -> HoleId:
    return HoleId(namespace=name, path=(), kind="test")


def _make_rows() -> list[CandidateCostRow]:
    """Return candidate_cost rows where lower token-count candidates are cheaper."""
    counters = SearchCounters(
        nodes=1, tokens=1, depth=1, backtracks=0, verifier_calls=0
    )
    base = {
        "schema_version": 1,
        "row_kind": "candidate_cost",
        "state_fingerprint": "fp1",
        "parent_fingerprint": None,
        "problem_id": "p1",
        "pack_id": "openui",
        "constraint_version": "v1",
        "program_family_id": "fam1",
        "lineage_id": "lin1",
        "split_group_id": "sg1",
        "split": "train",
        "capsule_id": None,
        "hole_id": _hole_id("h1"),
        "ranker_id": "baseline",
        "chosen": False,
        "support_verdict": "unknown",
        "terminal_success": False,
        "cost_observed": True,
        "censor_reason": None,
        "conflict_reason": None,
        "counters": counters,
        "final_trajectory_status": "unknown",
        "trace_id": None,
        "trajectory_id": None,
    }

    def row(candidate: DomainValue, *, nodes: int, tokens: int, observed: bool = True) -> CandidateCostRow:
        return CandidateCostRow(
            **{
                **base,
                "candidate": candidate,
                "nodes": nodes,
                "tokens": tokens,
                "depth": nodes,
                "backtracks": 0,
                "verifier_calls": 0,
                "cost_observed": observed,
            }
        )

    # Two candidates in the same (state, hole): short is cheaper.
    a = DomainValue.create("short", {"token_ids": [10, 11]})
    b = DomainValue.create("long", {"token_ids": [10, 11, 12, 13]})
    return [
        row(a, nodes=2, tokens=2),
        row(b, nodes=8, tokens=8),
        # Another state with a censored row.
        CandidateCostRow(
            **{
                **base,
                "state_fingerprint": "fp2",
                "candidate": DomainValue.create("c", {"token_ids": [20]}),
                "nodes": 0,
                "tokens": 0,
                "depth": 0,
                "backtracks": 0,
                "verifier_calls": 0,
                "cost_observed": False,
            }
        ),
    ]


def test_head_created_when_enabled() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            cost_to_go_hidden_dim=16,
        ),
        device="cpu",
    )
    assert model.cost_to_go_head is not None


def test_head_absent_when_disabled() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    assert model.cost_to_go_head is None


def test_loss_decreases_on_synthetic_rows() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            cost_to_go_hidden_dim=16,
            cost_to_go_loss_weight=1.0,
        ),
        device="cpu",
    )
    rows = _make_rows()
    opt = torch.optim.AdamW(model.cost_to_go_head.parameters(), lr=1e-2)
    losses: list[float] = []
    for _ in range(20):
        loss = model.cost_to_go_loss(rows)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    assert losses[-1] < losses[0]
    assert model.last_training_metrics["cost_to_go_observed_rows"] == 2


def test_censored_rows_are_masked_from_regression() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            cost_to_go_hidden_dim=16,
            cost_to_go_loss_weight=1.0,
        ),
        device="cpu",
    )
    rows = _make_rows()
    loss = model.cost_to_go_loss(rows)
    loss.backward()
    # Censored row is at index 2; observed mask should only include 0,1.
    assert model.last_training_metrics["cost_to_go_observed_rows"] == 2


def test_old_checkpoint_loads_with_missing_head(tmp_path: Path) -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    old_model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
        ),
        device="cpu",
    )
    ckpt = tmp_path / "old.pt"
    old_model.save(ckpt)

    new_model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            cost_to_go_hidden_dim=16,
        ),
        device="cpu",
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    _load_checkpoint_state(new_model, payload["state_dict"])
    assert new_model.cost_to_go_head is not None


def test_pairwise_ordering_follows_cost() -> None:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            cost_to_go_hidden_dim=16,
            cost_to_go_loss_weight=1.0,
        ),
        device="cpu",
    )
    rows = _make_rows()
    opt = torch.optim.AdamW(model.cost_to_go_head.parameters(), lr=1e-2)
    for _ in range(30):
        loss = model.cost_to_go_loss(rows)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    short = DomainValue.create("short", {"token_ids": [10, 11]})
    long = DomainValue.create("long", {"token_ids": [10, 11, 12, 13]})
    state = None
    output = model.score_candidates(
        state, _hole_id("h1"), (short, long), context_prompt="Hero"
    )
    energies = output.energies.detach()
    assert energies[0] < energies[1]
