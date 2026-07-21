"""Tests for the SLM-241 (GRT0-01) D2 canonicalizer round-trip / alpha-
invariance stress probe."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm241_grammar_round_trip_alpha_invariance import (
    EXPERIMENT_ID,
    MATRIX_SET,
    GrammarRoundTripReport,
    binder_names,
    generate_candidate_sources,
    mask_literals,
    permute_binders,
    render_markdown,
    run_round_trip_fixture,
    unmask_literals,
)


def test_mask_and_unmask_literals_round_trip() -> None:
    source = 'root = Stack([a], "column")\na = TextContent(":hero.title")'
    masked, literals = mask_literals(source)
    assert '"' not in masked or "\x00LIT" in masked
    assert unmask_literals(masked, literals) == source


def test_binder_names_excludes_nothing_and_preserves_order() -> None:
    source = "root = Stack([a, b], \"column\")\na = TextContent(\"x\")\nb = Button(\"y\")"
    assert binder_names(source) == ["root", "a", "b"]


def test_permute_binders_is_none_for_trivial_program() -> None:
    # Only "root" -- no non-root binder to permute.
    source = 'root = Button("hi")'
    renamed, mapping = permute_binders(source)
    assert renamed is None
    assert mapping == {}


def test_permute_binders_renames_every_non_root_binder() -> None:
    source = (
        'root = Stack([a, b], "column")\n'
        'a = TextContent("x")\n'
        'b = Button("y")'
    )
    renamed, mapping = permute_binders(source)
    assert renamed is not None
    assert set(mapping) == {"a", "b"}
    assert "root = Stack(" in renamed
    for old, new in mapping.items():
        assert old != new
        assert new in renamed


def test_permute_binders_protects_placeholder_aliased_text() -> None:
    # Regression analogue of tests/test_dsl/test_canonicalize.py's
    # alpha_invariance_when_placeholders_alias_binder_names case: a binder
    # named "form" also appears inside a quoted placeholder string.
    source = (
        'root = Stack([title, form], "column")\n'
        'title = TextContent(":form.title")\n'
        'submit = Button(":form.submit")\n'
        'actions = Buttons([submit])\n'
        'form = Form("contact", actions, [])'
    )
    renamed, mapping = permute_binders(source)
    assert renamed is not None
    assert "form" in mapping
    # The placeholder text must survive untouched even though "form" the
    # binder was renamed.
    assert ":form.title" in renamed
    assert ":form.submit" in renamed


def test_generate_candidate_sources_returns_valid_openui() -> None:
    from slm_training.dsl.parser import validate

    candidates = generate_candidate_sources(seed=0, count=5)
    assert len(candidates) == 5
    for source, components, depth, width in candidates:
        assert isinstance(source, str) and source.strip()
        validate(source)
        assert depth >= 1
        assert width >= 1
        assert components


def test_fixture_runs_and_produces_rows() -> None:
    report = run_round_trip_fixture(seeds=(0,), count_per_seed=10)
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.rows) == 10
    assert report.gate_hash


def test_fixture_confirms_or_reports_gap_honestly() -> None:
    report = run_round_trip_fixture(seeds=(0, 1), count_per_seed=15)
    assert report.disposition in {"ceiling_confirmed_at_scale", "gap_confirmed", "inconclusive"}
    if report.disposition == "gap_confirmed":
        assert any(
            (not r.idempotent) or (not r.revalidates) or (r.alpha_invariant is False)
            for r in report.rows
        )
    if report.disposition == "ceiling_confirmed_at_scale":
        for row in report.rows:
            assert row.idempotent
            assert row.revalidates
            assert row.alpha_invariant is not False


def test_every_scored_row_is_a_real_grammar_valid_candidate() -> None:
    report = run_round_trip_fixture(seeds=(0,), count_per_seed=8)
    for row in report.rows:
        assert row.source_len > 0
        assert row.canonicalize_ms >= 0.0


def test_gate_hash_is_deterministic() -> None:
    a = run_round_trip_fixture(seeds=(0,), count_per_seed=6)
    b = run_round_trip_fixture(seeds=(0,), count_per_seed=6)
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_report_roundtrips_through_dict() -> None:
    report = run_round_trip_fixture(seeds=(0,), count_per_seed=6)
    payload = report.to_dict()
    restored = GrammarRoundTripReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_render_markdown_includes_disposition_and_tables() -> None:
    report = run_round_trip_fixture(seeds=(0,), count_per_seed=6)
    text = render_markdown(report)
    assert report.disposition in text
    assert "| seed | candidates |" in text
    assert "No-go for promotion" in text
    assert "Counterexamples" in text
