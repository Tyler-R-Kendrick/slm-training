"""Data-adequacy ladder: nested subsets, marginal gains, honest classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.data_adequacy_ladder import (
    DATA_ADEQUACY_LADDER_SCHEMA,
    RungMeasurement,
    build_ladder_artifact,
    classify_marginal_gain,
    load_ladder_classification,
    marginal_gains_per_100,
    materialize_nested_subsets,
    plan_nested_rungs,
)


def _rungs(*points: tuple[int, float], suite_n: int = 100):
    return [
        RungMeasurement(records=records, metric=metric, suite_n=suite_n)
        for records, metric in points
    ]


def test_plan_nested_rungs_halves_down_to_floor() -> None:
    assert plan_nested_rungs(101) == (25, 50, 101)
    assert plan_nested_rungs(101, rung_count=2) == (50, 101)
    with pytest.raises(ValueError, match="below min_records"):
        plan_nested_rungs(8)


def test_materialize_nested_subsets_are_prefixes(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    lines = [json.dumps({"i": i}) for i in range(10)]
    (train / "records.jsonl").write_text("\n".join(lines) + "\n")
    (train / "manifest.json").write_text(
        json.dumps({"content_fingerprint": "abc123"})
    )
    subsets = materialize_nested_subsets(train, tmp_path / "out", [4, 8])
    assert [records for records, _ in subsets] == [4, 8]
    small = (subsets[0][1] / "records.jsonl").read_text().splitlines()
    large = (subsets[1][1] / "records.jsonl").read_text().splitlines()
    assert large[:4] == small
    manifest = json.loads((subsets[0][1] / "manifest.json").read_text())
    assert manifest["parent_content_fingerprint"] == "abc123"
    assert manifest["claim_class"] == "fixture"
    assert manifest["promotion_authorized"] is False
    with pytest.raises(ValueError, match="exceeds corpus size"):
        materialize_nested_subsets(train, tmp_path / "out2", [999])


def test_marginal_gains_per_100_direction_aware() -> None:
    rising = _rungs((50, 0.4), (100, 0.6))
    assert marginal_gains_per_100(rising) == [pytest.approx(0.4)]
    # Lower-is-better metrics (NLL) flip the sign.
    nll = _rungs((50, 2.0), (100, 1.5))
    assert marginal_gains_per_100(nll, higher_is_better=False) == [
        pytest.approx(1.0)
    ]
    assert marginal_gains_per_100(_rungs((50, 0.4))) == []


def test_classification_refuses_undecidable_flat_claims() -> None:
    points = _rungs((50, 0.50), (100, 0.51), suite_n=3)
    # Underpowered suite: never flat, whatever the gain looks like.
    verdict = classify_marginal_gain(points, mde=0.05, sd=0.2)
    assert verdict.classification == "undecidable"
    assert "below the powered floor" in verdict.reason
    # No sd estimate: undecidable.
    no_sd = classify_marginal_gain(
        _rungs((50, 0.50), (100, 0.51)), mde=0.05, sd=None
    )
    assert no_sd.classification == "undecidable"
    # One rung: undecidable.
    single = classify_marginal_gain(_rungs((100, 0.5)), mde=0.05, sd=0.1)
    assert single.classification == "undecidable"


def test_classification_flat_and_rising_when_powered() -> None:
    flat = classify_marginal_gain(
        _rungs((50, 0.50), (100, 0.51), suite_n=500), mde=0.05, sd=0.1
    )
    assert flat.classification == "flat"
    assert flat.required_suite_n is not None
    rising = classify_marginal_gain(
        _rungs((50, 0.40), (100, 0.60), suite_n=500), mde=0.05, sd=0.1
    )
    assert rising.classification == "rising"


def test_artifact_round_trip_feeds_adequacy(tmp_path: Path) -> None:
    measurements = _rungs((50, 2.0), (100, 1.98), suite_n=500)
    verdict = classify_marginal_gain(
        measurements, mde=0.05, sd=0.1, higher_is_better=False
    )
    artifact = build_ladder_artifact(
        train_dir="outputs/data/train/x",
        measurements=measurements,
        verdict=verdict,
        metric_name="best_weighted_nll",
        higher_is_better=False,
    )
    assert artifact["schema"] == DATA_ADEQUACY_LADDER_SCHEMA
    assert artifact["claim_class"] == "fixture"
    assert artifact["promotion_authorized"] is False
    path = tmp_path / "data_adequacy_ladder.json"
    path.write_text(json.dumps(artifact, indent=2))
    flat, source = load_ladder_classification(path)
    assert flat is True
    assert source == str(path)
    with pytest.raises(ValueError, match="not a data_adequacy_ladder"):
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"schema": "nope"}))
        load_ladder_classification(other)


def test_component_coverage_minimum_override_reaches_generator() -> None:
    from slm_training.harnesses.synthesis_plan import (
        apply_corpus_generation_overrides,
        load_synthesis_plan,
    )

    plan = load_synthesis_plan(
        Path("src/slm_training/resources/synthesis_plans/corpus/cap0_tiny_v2.json")
    )
    policy = apply_corpus_generation_overrides(
        plan.corpus_generation,
        generation_mode="until_coverage",
        component_coverage_minimum=4,
    )
    minima = dict(policy.generator.coverage_minimums)
    assert minima["component"] == 4
    assert policy.mode.value == "until_coverage"
    # No override leaves the plan untouched.
    same = apply_corpus_generation_overrides(plan.corpus_generation)
    assert same is plan.corpus_generation
