"""Deterministic denoising-NLL suite tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.evals.denoising_nll import (
    DenoisingNLLConfig,
    evaluate_denoising_nll,
    fixed_mask_positions,
)
from slm_training.evals.loss_suites import (
    CATEGORY_WEIGHTS,
    LOSS_SUITE_VERSION,
    binding_positions,
    evaluate_loss_suites,
    evaluate_repair_nll,
    load_suite_spec,
    structural_positions,
)
from slm_training.models.choice_tokenizer import ChoiceTokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _records(split: str = "held_out") -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id="h1",
            prompt="Hero",
            openui=HERO,
            split=split,
            placeholders=[":hero.title", ":hero.body"],
        ),
        ExampleRecord(
            id="h2",
            prompt="CTA",
            openui=CTA,
            split=split,
            placeholders=[":cta.label"],
        ),
    ]


def _model() -> TwoTowerModel:
    return TwoTowerModel.from_records(
        _records(),
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=0
        ),
        device="cpu",
    )


def test_fixed_mask_positions_deterministic_and_order_free() -> None:
    a = fixed_mask_positions(
        "rec", 0.5, suite_version="v1", mask_seed=0, eligible=[1, 2, 3, 4, 5, 6]
    )
    b = fixed_mask_positions(
        "rec", 0.5, suite_version="v1", mask_seed=0, eligible=[6, 5, 4, 3, 2, 1]
    )
    assert a == b
    assert len(a) == 3
    # Different record / seed / version give different draws (typically).
    c = fixed_mask_positions(
        "rec2", 0.5, suite_version="v1", mask_seed=0, eligible=[1, 2, 3, 4, 5, 6]
    )
    d = fixed_mask_positions(
        "rec", 0.5, suite_version="v2", mask_seed=0, eligible=[1, 2, 3, 4, 5, 6]
    )
    assert len(c) == 3 and len(d) == 3


def test_denoising_nll_is_deterministic() -> None:
    model = _model()
    cfg = DenoisingNLLConfig()
    r1 = evaluate_denoising_nll(model, _records(), config=cfg)
    r2 = evaluate_denoising_nll(model, _records(), config=cfg)
    assert r1 == r2
    assert r1["aggregate"]["mean_nll"] is not None
    assert r1["aggregate"]["masked_tokens"] > 0
    assert r1["bits_per_char"] is not None


def test_denoising_nll_emits_reconcilable_family_and_task_slices() -> None:
    records = _records()
    records[0].meta = {"source_family": "programspec_generated", "task": "generation"}
    records[1].meta = {"source_family": "corruption_repair", "task": "repair"}
    report = evaluate_denoising_nll(_model(), records, config=DenoisingNLLConfig())
    assert [row["id"] for row in report["per_record"]] == ["h1", "h2"]
    assert set(report["by_family"]) == {
        "corruption_repair",
        "programspec_generated",
    }
    assert set(report["by_task"]) == {"generation", "repair"}
    assert set(report["memorization_by_family"]) == set(report["by_family"])
    total_tokens = sum(row["masked_tokens"] for row in report["by_family"].values())
    assert total_tokens == report["aggregate"]["masked_tokens"]


def test_denoising_nll_invariant_to_training_options() -> None:
    """Training-only levers must not change the eval number for a fixed model."""
    model = _model()
    cfg = DenoisingNLLConfig()
    base = evaluate_denoising_nll(model, _records(), config=cfg)
    model.config.mdlm_schedule = True
    model.config.ltr_loss_weight = 0.9
    model.config.fidelity_loss_weight = 0.9
    model.config.visible_corrupt_rate = 0.5
    model.config.mask_min = 0.01
    model.config.mask_max = 0.99
    toggled = evaluate_denoising_nll(model, _records(), config=cfg)
    assert toggled == base


def test_denoising_nll_does_not_disturb_rng_or_mode() -> None:
    model = _model()
    model.train()
    state_before = model._rng.getstate()
    evaluate_denoising_nll(model, _records(), config=DenoisingNLLConfig())
    assert model._rng.getstate() == state_before
    assert model.training  # restored to train mode


def test_legal_support_decomposition() -> None:
    model = _model()
    report = evaluate_denoising_nll(model, _records(), config=DenoisingNLLConfig())
    if not report["legal_support_available"]:
        pytest.skip("grammar engine unavailable")
    agg = report["aggregate"]
    assert agg["legal_mean_nll"] is not None
    # Renormalizing over a subset support can only shrink NLL at constrained
    # positions, so the rescue gap is non-negative (up to float noise).
    assert agg["constraint_rescue_gap"] >= -1e-6


def test_position_filters_partition_sensibly() -> None:
    model = _model()
    ids = model._encode_openui(HERO, placeholders=[":hero.title", ":hero.body"])
    binding = binding_positions(model, _records()[0], ids)
    structural = structural_positions(model, _records()[0], ids)
    assert binding and structural
    assert not (set(binding) & set(structural))


def test_position_filters_support_choice_tokens() -> None:
    tokenizer = ChoiceTokenizer.build()
    model = type("ChoiceModel", (), {"tokenizer": tokenizer})()
    ids = [
        tokenizer.bos_id,
        tokenizer.token_to_id["r="],
        tokenizer.token_to_id["+Stack"],
        tokenizer.token_to_id["&0"],
        tokenizer.token_to_id["@0"],
        tokenizer.eos_id,
    ]
    record = _records()[0]
    assert binding_positions(model, record, ids) == [3, 4]
    assert structural_positions(model, record, ids) == [1, 2]


def test_repair_nll_deterministic() -> None:
    model = _model()
    r1 = evaluate_repair_nll(model, _records())
    r2 = evaluate_repair_nll(model, _records())
    assert r1 == r2
    assert r1["n_edits"] > 0
    assert r1["aggregate"]["mean_nll"] is not None
    assert 0.0 <= r1["aggregate"]["restore_top1"] <= 1.0


def _write_suite(test_dir: Path, suite: str, records: list[ExampleRecord]) -> None:
    suite_dir = test_dir / "suites" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(suite_dir / "records.jsonl", records)


def test_frozen_suite_spec_matches_weights() -> None:
    spec = load_suite_spec(LOSS_SUITE_VERSION)
    assert spec["loss_suite_version"] == LOSS_SUITE_VERSION
    assert {k: float(v) for k, v in spec["weights"].items()} == CATEGORY_WEIGHTS
    assert len(spec["mask_rates"]) == 5


def test_loss_suites_clear_stale_runtime_symbol_features(tmp_path: Path) -> None:
    """Request-local features from a training batch must not leak into the
    teacher-forced suites: a stale batch dimension crashes the batched NLL
    forward, and stale content would silently bias it."""
    import torch

    model = _model()
    test_dir = tmp_path / "test_data"
    _write_suite(test_dir, "held_out", _records("held_out"))
    stale = torch.zeros((4, model.tokenizer.vocab_size, 32))
    stale[:, 0, :] = 1.0
    model.denoiser.set_runtime_symbol_features(stale)
    report = evaluate_loss_suites(model, test_dir)
    assert report["aggregate"]["weighted_nll"] is not None
    assert model.denoiser._runtime_symbol_features is None


def test_loss_suites_full_report(tmp_path: Path) -> None:
    model = _model()
    test_dir = tmp_path / "test_data"
    _write_suite(test_dir, "held_out", _records("held_out"))
    _write_suite(test_dir, "ood", _records("ood"))
    report = evaluate_loss_suites(model, test_dir)
    assert report["aggregate"]["complete"] is True
    assert report["aggregate"]["missing_categories"] == []
    assert report["aggregate"]["weighted_nll"] is not None
    assert set(report["categories"]) == set(CATEGORY_WEIGHTS)
    definition = report["definition"]
    assert definition["base_record_ids"] == ["h1", "h2"]
    assert definition["weights"] == CATEGORY_WEIGHTS


def test_loss_suites_missing_suite_is_explicit(tmp_path: Path) -> None:
    model = _model()
    test_dir = tmp_path / "test_data"
    _write_suite(test_dir, "held_out", _records("held_out"))
    # No ood suite on disk.
    report = evaluate_loss_suites(model, test_dir)
    assert report["categories"]["schema_ood"] is None
    assert report["aggregate"]["complete"] is False
    assert "schema_ood" in report["aggregate"]["missing_categories"]
    # Aggregate still computed over present categories (renormalized).
    assert report["aggregate"]["weighted_nll"] is not None
