"""Regression tests for the CAP0-02 arity analyzer."""

from __future__ import annotations

import pytest

from slm_training.dsl.analysis.arity import (
    AnalysisBounds,
    ArityAnalyzer,
    CodingMetadata,
    StateAtom,
    SupportQuery,
    SupportVerdict,
)


def _toy_source(a: str = ":hero.title", b: str = ":cta.label") -> str:
    return f"root = row(title, action)\ntitle = text(\"{a}\")\naction = button(\"{b}\")"


def test_report_is_reproducible():
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    report1 = analyzer.analyze([_toy_source()])
    report2 = analyzer.analyze([_toy_source()])
    assert report1.digest == report2.digest
    assert report1.minimized_states == report2.minimized_states
    assert report1.total_states == 1
    assert report1.minimized_states == 1


def test_signature_invariant_under_renaming():
    """Placeholder surface names and binding names must not affect the signature."""
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    report_a = analyzer.analyze([_toy_source(":hero.title", ":cta.label")])
    report_b = analyzer.analyze([_toy_source(":page.blurb", ":hero.body")])
    assert report_a.minimized_states == report_b.minimized_states
    # The signatures differ only in placeholder indices, so the count is the same.
    # The digest differs because placeholder identities are part of the frame.


def test_support_oracle_recognizes_contained_atom():
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    report = analyzer.analyze([_toy_source()])
    state = report.continuation_summaries[0].state_signature
    query = SupportQuery(
        state_fingerprint=state.fingerprint(),
        hole_id="title.text",
        candidate=StateAtom.placeholder(0),
    )
    result = analyzer.check(state, query)
    assert result.verdict == SupportVerdict.SUPPORTED


def test_support_oracle_is_conservatively_unknown_for_foreign_atom():
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    report = analyzer.analyze([_toy_source()])
    state = report.continuation_summaries[0].state_signature
    query = SupportQuery(
        state_fingerprint=state.fingerprint(),
        hole_id="action.label",
        candidate=StateAtom.literal("never-seen"),
    )
    result = analyzer.check(state, query)
    assert result.verdict == SupportVerdict.UNKNOWN


def test_bounded_generation_deduplicates_identical_programs():
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    source = _toy_source(":hero.title", ":cta.label")
    report = analyzer.analyze([source, source, source])
    assert report.total_states == 3
    assert report.minimized_states == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_ast_nodes": -1},
        {"max_ast_nodes": 1, "max_ast_depth": -1},
        {"max_ast_nodes": 1, "max_live_bindings": -1},
    ],
)
def test_bounds_validation_rejects_negative_values(kwargs):
    with pytest.raises(ValueError):
        AnalysisBounds(**kwargs)


def test_report_with_coding_metadata_serializes():
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    report = analyzer.analyze([_toy_source()], include_coding_metadata=True)
    assert report.coding_metadata is not None
    data = report.to_dict()
    assert "coding_metadata" in data
    assert data["coding_metadata"]["state_count"] == report.minimized_states


def test_coding_metadata_changes_digest():
    bounds = AnalysisBounds(max_ast_nodes=4)
    analyzer = ArityAnalyzer("toy-layout", bounds)
    report_without = analyzer.analyze([_toy_source()], include_coding_metadata=False)
    report_with = analyzer.analyze([_toy_source()], include_coding_metadata=True)
    assert report_without.digest != report_with.digest


def test_coding_metadata_round_trip_dict_keys():
    metadata = CodingMetadata(
        state_count=41,
        dimensions=4,
        alphabet_size=7,
        min_distance=3,
        feasible=True,
        status="feasible",
        bound_name="singleton_upper_bound",
        bound_value=49,
        construction="mds_7_4_2_3",
        proof_status="local_verified_construction",
        source_citation=None,
        utilization=41 / 49,
        scale_mode=None,
        ecoc_width=None,
        margin_planes=None,
    )
    report = ArityAnalyzer("toy-layout", AnalysisBounds(max_ast_nodes=4)).analyze(
        [_toy_source()], include_coding_metadata=False
    )
    report_with_meta = report  # immutable; build a new one below
    from slm_training.dsl.analysis.arity.report import ArityReport

    report_with_meta = ArityReport(
        frame_id=report.frame_id,
        bounds=report.bounds,
        exact=report.exact,
        total_states=report.total_states,
        minimized_states=report.minimized_states,
        continuation_summaries=report.continuation_summaries,
        version=report.version,
        digest=report.digest,
        coding_metadata=metadata,
    )
    data = report_with_meta.to_dict()["coding_metadata"]
    assert data["alphabet_size"] == 7
    assert data["construction"] == "mds_7_4_2_3"
    assert data["proof_status"] == "local_verified_construction"
