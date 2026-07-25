"""Integration checks for the SLM-230 runner."""

from __future__ import annotations

import json

import pytest

from scripts.run_slm230_recurrence_observability import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    _evaluation_depth,
    _markdown,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm230_recurrence_observability import (
    validate_report,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _model() -> TwoTowerModel:
    record = ExampleRecord(
        id="fixture",
        prompt="One text node",
        openui='root = TextContent(":fixture")',
        split="train",
    )
    return TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            d_model=16,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_transition_layers=1,
            grammar_constrained=False,
            gen_steps=1,
            seed=230,
        ),
        device="cpu",
    )


def test_evaluation_depth_restores_default_after_success_and_error() -> None:
    model = _model()
    assert model.config.recursive_steps == 3
    with _evaluation_depth(model, 1):
        assert model.config.recursive_steps == 1
        assert model.denoiser.recursive_steps == 1
    assert model.config.recursive_steps == 3
    assert model.denoiser.recursive_steps == 3

    with pytest.raises(RuntimeError):
        with _evaluation_depth(model, 2):
            raise RuntimeError("fixture")
    assert model.config.recursive_steps == 3
    assert model.denoiser.recursive_steps == 3


def test_test_r_extrapolation_is_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="does not authorize"):
        with _evaluation_depth(model, 4):
            pass


def test_committed_report_and_markdown_are_consistent() -> None:
    if not DEFAULT_JSON.is_file() or not DEFAULT_MARKDOWN.is_file():
        pytest.skip("generated evidence follows the clean implementation commit")
    report = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    assert validate_report(report) == []
    assert DEFAULT_MARKDOWN.read_text(encoding="utf-8") == _markdown(report)
    assert report["training_default_changed"] is False
    assert report["production_default_changed"] is False
    assert report["ship_gate_claim"] is False
