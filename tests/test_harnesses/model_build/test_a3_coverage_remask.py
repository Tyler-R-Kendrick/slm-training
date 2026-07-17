"""A3 coverage-energy remask policy tests."""

from __future__ import annotations

import torch

from slm_training.models.parallel_decode import (
    select_remask_coverage_indices,
    select_remask_indices,
)


def test_coverage_remask_prefers_high_deficit_positions() -> None:
    conf = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    known = torch.tensor([[True, True, True, True]])
    # Position 2 sits in the most under-covered region.
    deficit = torch.tensor([[0.0, 0.1, 0.9, 0.2]])
    idxs = select_remask_coverage_indices(
        conf, known, remask_ratio=0.5, coverage_deficit=deficit
    )
    assert 0 not in idxs  # BOS protected
    assert 2 in idxs
    assert len(idxs) >= 1


def test_coverage_remask_falls_back_to_confidence() -> None:
    conf = torch.tensor([[0.9, 0.1, 0.8, 0.05]])
    known = torch.tensor([[True, True, True, True]])
    idxs = select_remask_coverage_indices(
        conf, known, remask_ratio=0.5, coverage_deficit=None
    )
    assert 0 not in idxs
    classic = select_remask_indices(conf, known, remask_ratio=0.5)
    assert set(idxs) & set(classic)


def test_coverage_remask_empty_when_ratio_zero() -> None:
    conf = torch.tensor([[0.9, 0.9]])
    known = torch.tensor([[True, True]])
    assert select_remask_coverage_indices(conf, known, remask_ratio=0.0) == []


def test_coverage_remask_respects_budget_and_bos() -> None:
    conf = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
    known = torch.tensor([[True, True, True, True, True, True]])
    deficit = torch.tensor([[0.0, 0.5, 0.9, 0.1, 0.8, 0.3]])
    idxs = select_remask_coverage_indices(
        conf, known, remask_ratio=0.34, coverage_deficit=deficit
    )
    # ceil(0.34 * 5 eligible) == 2; BOS never chosen; highest deficit first.
    assert 0 not in idxs
    assert set(idxs) <= {1, 2, 3, 4, 5}
    assert idxs[0] == 2  # highest deficit


def test_model_coverage_deficit_scores_filler_over_content() -> None:
    import pytest

    pytest.importorskip("torch")
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    model = TwoTowerModel.from_records(
        [
            ExampleRecord(
                id="a",
                prompt="Hero",
                openui='root = Stack([t])\nt = TextContent(":hero.title")',
                placeholders=[":hero.title"],
            )
        ],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=0,
            remask_policy="coverage", output_tokenizer="lexer",
        ),
        device="cpu",
    )
    tok = model.tokenizer
    if not hasattr(tok, "kind_ids"):
        pytest.skip("active tokenizer exposes no kind_ids")
    # One component token (content) + three structural filler tokens.
    comp_ids = sorted(tok.kind_ids("component"))
    if not comp_ids:
        pytest.skip("tokenizer exposes no component kind")
    equal_id = int(tok.token_to_id["="])
    ids = torch.tensor([[tok.bos_id, comp_ids[0], equal_id, equal_id]])
    known = torch.tensor([[True, True, True, True]])
    deficit = model._coverage_deficit(ids, known)
    # Content position (the component) scores 0; filler positions score > 0.
    assert float(deficit[0, 1]) == 0.0
    assert float(deficit[0, 2]) > 0.0
    assert float(deficit[0, 3]) > 0.0
