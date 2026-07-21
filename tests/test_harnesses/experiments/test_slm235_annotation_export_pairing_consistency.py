"""Tests for the SLM-235 (AEP0-01) annotation export pairing-mechanism
consistency probe."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm235_annotation_export_pairing_consistency import (
    EXPERIMENT_ID,
    MATRIX_SET,
    AnnotationExportPairingReport,
    FlipScenario,
    build_default_scenarios,
    render_markdown,
    run_pairing_consistency_fixture,
)


def _result(report, name):
    return next(r for r in report.results if r.scenario.name == name)


def test_default_scenarios_shape() -> None:
    scenarios = build_default_scenarios()
    names = {s.name for s in scenarios}
    assert "single_flip_control" in names
    assert "alternating_four_events" in names
    assert "alternating_five_events_up_first" in names
    assert sum(1 for n in names if n.startswith("organic_multi_prompt_seed")) == 5
    controls = {s.name for s in scenarios if s.is_negative_control}
    assert controls == {"single_flip_control"}


def test_fixture_runs_all_scenarios() -> None:
    report = run_pairing_consistency_fixture()
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.results) == 8
    assert report.gate_hash


def test_control_scenario_matches_exactly() -> None:
    report = run_pairing_consistency_fixture()
    control = _result(report, "single_flip_control")
    assert control.incremental_pairs_in_file == 1
    assert control.batch_pairs_written == 1
    assert control.pairs_retained == 1
    assert control.pairs_lost == 0
    assert control.divergent is False


def test_alternating_four_events_loses_two_pairs() -> None:
    report = run_pairing_consistency_fixture()
    result = _result(report, "alternating_four_events")
    assert result.incremental_pairs_in_file == 3
    assert result.batch_pairs_written == 1
    assert result.pairs_retained == 1
    assert result.pairs_lost == 2
    assert result.divergent is True


def test_alternating_five_events_loses_three_pairs() -> None:
    report = run_pairing_consistency_fixture()
    result = _result(report, "alternating_five_events_up_first")
    assert result.incremental_pairs_in_file == 4
    assert result.batch_pairs_written == 1
    assert result.pairs_retained == 1
    assert result.pairs_lost == 3
    assert result.divergent is True


def test_organic_seeds_with_multi_flip_prompts_lose_pairs() -> None:
    report = run_pairing_consistency_fixture()
    organic = [r for r in report.results if r.scenario.seed is not None]
    assert len(organic) == 5
    for result in organic:
        if result.prompts_with_multi_events > 0:
            assert result.divergent is True
            assert result.pairs_lost > 0


def test_static_audit_confirms_shared_default_path() -> None:
    report = run_pairing_consistency_fixture()
    audit = report.static_shared_default_path_audit
    assert audit["paths_identical"] is True
    assert audit["export_cli_default_references_shared_constant"] is True
    assert audit["live_store_default_references_shared_constant"] is True


def test_disposition_confirms_the_gap() -> None:
    report = run_pairing_consistency_fixture()
    assert report.disposition == "gap_confirmed"
    assert "lost" in report.disposition_rationale.lower()


def test_report_roundtrips_through_dict() -> None:
    report = run_pairing_consistency_fixture()
    payload = report.to_dict()
    restored = AnnotationExportPairingReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_pairing_consistency_fixture()
    b = run_pairing_consistency_fixture()
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_custom_scenario_list_is_respected() -> None:
    scenarios = [
        FlipScenario(
            name="only_scenario",
            description="d",
            prompts=(("p", ("down", "up")),),
            is_negative_control=True,
        )
    ]
    report = run_pairing_consistency_fixture(scenarios=scenarios)
    assert len(report.results) == 1
    assert report.results[0].scenario.name == "only_scenario"


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_pairing_consistency_fixture()
    text = render_markdown(report)
    assert report.disposition in text
    assert "| scenario | prompts |" in text
    assert "No-go for promotion" in text
    assert "alternating_four_events" in text
