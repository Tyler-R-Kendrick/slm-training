"""Tests for SLM-190 exact CTMC reference fixture harness."""

from __future__ import annotations


from slm_training.harnesses.experiments.slm190_exact_flow import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    ExactFlowReport,
    build_canonical_edit_adapter,
    build_choice_sequence_adapter,
    build_toy_layout_adapter,
    render_markdown,
    run_exact_flow_fixture,
    validate_report,
)


def test_constants() -> None:
    assert EXPERIMENT_ID == "slm190-exact-flow"
    assert MATRIX_SET == "slm190_exact_flow"
    assert MATRIX_VERSION == "ffe2-02-v1"
    assert len(ARM_NAMES) == 4


def test_build_adapters_torch_free() -> None:
    assert build_toy_layout_adapter().domain_id == "toy_layout"
    assert build_choice_sequence_adapter().domain_id == "choice_sequence"
    assert build_canonical_edit_adapter().domain_id == "canonical_edit_graph"


def test_run_exact_flow_fixture_produces_report(tmp_path) -> None:
    report = run_exact_flow_fixture(
        output_dir=tmp_path,
        rate_fn_names=("uniform_rate",),
        times=(1.0,),
        seed=0,
        write_design_docs=False,
    )
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.n_domains == 3
    assert len(report.cases) == 3
    assert all(c.mass_conservation_error < 1e-4 for c in report.cases)
    assert all(c.illegal_edge_rate_sum < 1e-8 for c in report.cases)


def test_report_round_trip(tmp_path) -> None:
    report = run_exact_flow_fixture(
        output_dir=tmp_path,
        rate_fn_names=("uniform_rate",),
        times=(1.0,),
        seed=1,
        write_design_docs=False,
    )
    data = report.to_dict()
    restored = ExactFlowReport.from_dict(data)
    assert restored.matrix_set == report.matrix_set
    assert len(restored.cases) == len(report.cases)


def test_render_markdown_contains_caveats(tmp_path) -> None:
    report = run_exact_flow_fixture(
        output_dir=tmp_path,
        rate_fn_names=("uniform_rate",),
        times=(1.0,),
        seed=2,
        write_design_docs=False,
    )
    md = render_markdown(report)
    assert "wiring / fixture only" in md
    assert "No-go for promotion" in md
    assert "SLM-190" in md
    assert "lumpable" in md.lower() or "not_lumpable" in md.lower()


def test_validate_report_passes(tmp_path) -> None:
    report = run_exact_flow_fixture(
        output_dir=tmp_path,
        rate_fn_names=("uniform_rate",),
        times=(1.0,),
        seed=3,
        write_design_docs=False,
    )
    errors = validate_report(report)
    assert not errors
