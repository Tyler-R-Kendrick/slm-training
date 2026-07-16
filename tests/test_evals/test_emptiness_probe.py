"""A1 emptiness-probe wiring tests (fixture/scratch = wiring evidence only)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.emptiness_probe import (
    EmptinessProbeConfig,
    evaluate_emptiness,
    minimal_valid_program,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id="h1",
            prompt="Hero",
            openui=HERO,
            split="held_out",
            placeholders=[":hero.title", ":hero.body"],
        ),
        ExampleRecord(
            id="h2",
            prompt="CTA",
            openui=CTA,
            split="held_out",
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


def test_minimal_valid_program_exists_for_default_dsl() -> None:
    program = minimal_valid_program()
    assert program is not None
    # It must actually validate through the parser, not just be a string.
    from slm_training.dsl.parser import validate

    validate(program)


def test_emptiness_probe_reports_decomposed_margins() -> None:
    model = _model()
    report = evaluate_emptiness(model, _records(), config=EmptinessProbeConfig())
    assert report["n_records"] == 2
    assert report["empty_program"] is not None
    assert report["verdict"] in {
        "length_bias_constraint_distortion",
        "content_modeling_failure",
        "populated_preferred",
    }
    # Every scored record carries both total and per-token margins.
    for row in report["per_record"]:
        assert "margin_total" in row and "margin_per_token" in row
        assert row["pop_tokens"] >= row["empty_tokens"]
        assert isinstance(row["empty_preferred_total"], bool)


def test_emptiness_probe_is_deterministic() -> None:
    a = evaluate_emptiness(_model(), _records())
    b = evaluate_emptiness(_model(), _records())
    assert a["mean_margin_total"] == pytest.approx(b["mean_margin_total"])
    assert a["mean_margin_per_token"] == pytest.approx(b["mean_margin_per_token"])


def test_emptiness_probe_handles_no_minimal_program(monkeypatch) -> None:
    import slm_training.evals.emptiness_probe as probe

    monkeypatch.setattr(probe, "minimal_valid_program", lambda dsl=None: None)
    report = probe.evaluate_emptiness(_model(), _records())
    assert report["n_records"] == 0
    assert report["empty_program"] is None
    assert report["per_record"] == []
