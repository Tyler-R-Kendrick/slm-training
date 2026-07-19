"""Regression tests for the EFS0-05 rejected-lever registry and campaign."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from slm_training.autoresearch.evidence import collect_evidence
from slm_training.harnesses.experiments.rejected_lever_registry import (
    CONFOUNDS,
    PairedSeedObservation,
    ReAdjudicationRowV1,
    RejectedLeverRegistryV1,
    RejectedLeverV1,
    build_preregistered_campaign,
    check_duplicate_experiment_ids,
    check_duplicate_run_ids,
    classify_verdict,
    closed_lever_signatures,
    lever_signature,
    load_registry,
    paired_seed_result,
    save_registry,
    to_evidence_items,
)


def _example_entry(**overrides) -> RejectedLeverV1:
    base = {
        "entry_id": "E999_test",
        "experiment_ids": ("E999",),
        "run_ids": ("E999_run",),
        "hypothesis": "test lever",
        "original_matrix": "quality",
        "original_source_commit": "a" * 40,
        "original_train_commit": "a" * 40,
        "original_eval_commit": "a" * 40,
        "status": "provisional_negative",
    }
    base.update(overrides)
    return RejectedLeverV1.model_validate(base)


def test_schema_rejects_unknown_confound() -> None:
    with pytest.raises(ValueError, match="unknown confounds"):
        _example_entry(confounds=("not_a_confound",))


def test_schema_accepts_known_confounds() -> None:
    entry = _example_entry(confounds=tuple(sorted(CONFOUNDS)))
    assert set(entry.confounds) == CONFOUNDS


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="test-registry",
        entries=(_example_entry(),),
    )
    path = tmp_path / "registry.json"
    save_registry(registry, path)
    loaded = load_registry(path)
    assert loaded.registry_id == registry.registry_id
    assert len(loaded.entries) == 1
    assert loaded.entries[0].entry_id == "E999_test"


def test_duplicate_run_detection() -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="dup-test",
        entries=(
            _example_entry(entry_id="A", run_ids=("run_1",)),
            _example_entry(entry_id="B", run_ids=("run_1", "run_2")),
        ),
    )
    dups = check_duplicate_run_ids(registry)
    assert dups == {"run_1": ["A", "B"]}


def test_duplicate_experiment_detection() -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="dup-exp-test",
        entries=(
            _example_entry(entry_id="A", experiment_ids=("E1",)),
            _example_entry(entry_id="B", experiment_ids=("E1", "E2")),
        ),
    )
    dups = check_duplicate_experiment_ids(registry)
    assert dups == {"E1": ["A", "B"]}


def test_campaign_selects_only_eligible_levers() -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="campaign-test",
        entries=(
            _example_entry(entry_id="open_1", status="reopen_candidate"),
            _example_entry(entry_id="open_2", status="provisional_negative"),
            _example_entry(entry_id="closed", status="closed"),
            _example_entry(entry_id="invalid", status="invalidated"),
        ),
    )
    rows = build_preregistered_campaign(registry, required_levers=5, seed_count=5)
    assert len(rows) == 2
    assert {row.lever_id for row in rows} == {"open_1", "open_2"}
    assert rows[0].seeds == tuple(range(5))


def test_campaign_respects_required_lever_cap() -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="cap-test",
        entries=tuple(
            _example_entry(entry_id=f"open_{i}", status="reopen_candidate")
            for i in range(10)
        ),
    )
    rows = build_preregistered_campaign(registry, required_levers=3, seed_count=4)
    assert len(rows) == 3
    assert rows[0].seeds == tuple(range(4))


def test_seed_completeness_checked() -> None:
    row = ReAdjudicationRowV1(
        row_id="r1",
        lever_id="E999_test",
        control_run_id="c1",
        treatment_run_id="t1",
        seeds=(0, 1, 2, 3, 4),
        primary_metric="binding_aware_meaningful_v2",
        cost_metric="wall_seconds",
        decoder_path="current_exact_or_compiler",
    )
    observations = [
        PairedSeedObservation(
            seed=seed,
            control_value=0.4,
            treatment_value=0.45,
            control_cost=1.0,
            treatment_cost=1.1,
        )
        for seed in row.seeds
    ]
    result = paired_seed_result(row, observations)
    assert len(result.observations) == 5
    assert result.observations[0].delta == pytest.approx(0.05)


def test_failure_and_timeout_are_retained_but_excluded() -> None:
    row = ReAdjudicationRowV1(
        row_id="r1",
        lever_id="E999_test",
        control_run_id="c1",
        treatment_run_id="t1",
        seeds=(0, 1),
        primary_metric="binding_aware_meaningful_v2",
        cost_metric="wall_seconds",
        decoder_path="current_exact_or_compiler",
    )
    observations = [
        PairedSeedObservation(
            seed=0,
            control_value=0.4,
            treatment_value=0.45,
            control_cost=1.0,
            treatment_cost=1.1,
        ),
        PairedSeedObservation(
            seed=1,
            control_value=0.4,
            treatment_value=0.45,
            control_cost=1.0,
            treatment_cost=1.1,
            timeout=True,
        ),
    ]
    result = paired_seed_result(row, observations)
    assert len(result.observations) == 2
    assert result.observations[1].timeout
    # Only one valid observation -> bootstrap CI is NaN and verdict inconclusive.
    assert math.isnan(result.ci_low)
    assert result.verdict == "inconclusive"


def test_verdict_classification() -> None:
    # CI above min_effect -> reopened.
    assert classify_verdict(0.10, 0.06, 0.14, 0.05, 0.02) == "reopened_positive"
    # CI entirely below min_effect -> confirmed negative.
    assert classify_verdict(-0.01, -0.04, 0.01, 0.05, 0.02) == "confirmed_negative"
    # CI inside equivalence band -> equivalent.
    assert classify_verdict(0.005, -0.01, 0.02, 0.05, 0.02) == "equivalent"
    # Overlapping both thresholds -> inconclusive.
    assert classify_verdict(0.03, -0.01, 0.07, 0.05, 0.02) == "inconclusive"
    # Non-finite mean -> inconclusive.
    assert classify_verdict(float("nan"), 0.0, 0.0, 0.05, 0.02) == "inconclusive"


def test_paired_result_is_deterministic() -> None:
    row = ReAdjudicationRowV1(
        row_id="det",
        lever_id="E999_test",
        control_run_id="c1",
        treatment_run_id="t1",
        seeds=(0, 1, 2, 3, 4),
        primary_metric="binding_aware_meaningful_v2",
        cost_metric="wall_seconds",
        decoder_path="current_exact_or_compiler",
    )
    obs = [
        PairedSeedObservation(
            seed=seed,
            control_value=0.40 + seed * 0.01,
            treatment_value=0.45 + seed * 0.01,
            control_cost=1.0,
            treatment_cost=1.05,
        )
        for seed in row.seeds
    ]
    r1 = paired_seed_result(row, obs)
    r2 = paired_seed_result(row, obs)
    assert r1.mean_delta == r2.mean_delta
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high
    assert r1.verdict == r2.verdict


def test_evidence_items_have_rejected_lever_kind() -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="ev-test",
        entries=(_example_entry(entry_id="E999_test", status="closed"),),
    )
    items = to_evidence_items(registry)
    assert len(items) == 1
    assert items[0].kind == "rejected_lever"
    assert "E999_test" in items[0].summary
    assert "signature=" in items[0].summary


def test_closed_signatures_only_include_closed_and_invalidated() -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="closed-test",
        entries=(
            _example_entry(
                entry_id="closed",
                status="closed",
                experiment_ids=("E_closed",),
                confounds=("tiny_n",),
            ),
            _example_entry(
                entry_id="invalid",
                status="invalidated",
                experiment_ids=("E_invalid",),
                confounds=("decoder_bug",),
            ),
            _example_entry(
                entry_id="reopen",
                status="reopen_candidate",
                experiment_ids=("E_reopen",),
            ),
            _example_entry(
                entry_id="provisional",
                status="provisional_negative",
                experiment_ids=("E_provisional",),
            ),
        ),
    )
    closed = closed_lever_signatures(registry)
    assert len(closed) == 2
    assert lever_signature(registry.entries[0]) in closed
    assert lever_signature(registry.entries[1]) in closed


def test_evidence_intake_classifies_registry(tmp_path: Path) -> None:
    registry = RejectedLeverRegistryV1(
        registry_id="intake-test",
        entries=(_example_entry(entry_id="E999_test"),),
    )
    path = tmp_path / "rejected_lever_registry.json"
    save_registry(registry, path)
    snapshot = collect_evidence([tmp_path], repo_root=tmp_path.parent)
    kinds = {item.kind for item in snapshot.items}
    assert "rejected_lever" in kinds


def test_lever_signature_is_stable() -> None:
    e1 = _example_entry(
        entry_id="A",
        experiment_ids=("E1", "E2"),
        confounds=("tiny_n", "decoder_bug"),
    )
    e2 = _example_entry(
        entry_id="B",
        experiment_ids=("E2", "E1"),
        confounds=("decoder_bug", "tiny_n"),
    )
    assert lever_signature(e1) == lever_signature(e2)
