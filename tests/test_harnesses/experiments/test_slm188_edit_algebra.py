"""Tests for SLM-188 (FFE1-02) edit-algebra reachability fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm188_edit_algebra import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    CanonicalEdit,
    EditReachabilityReport,
    TransitionCertificateV1,
    apply_canonical_edit,
    build_fixture_codec,
    build_seed_target_pairs,
    build_sketch_seed,
    permute_slot_contract,
    plan_edit_sequence,
    replay_transition_certificate,
    render_markdown,
    run_edit_reachability_fixture,
    run_invariance_suite,
    topology_tree_from_openui,
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def test_build_fixture_codec_is_torch_free() -> None:
    codec = build_fixture_codec()
    assert codec.bos_id is not None
    assert "=" in codec.production_to_id
    assert "+Stack" in codec.production_to_id


def test_topology_tree_from_openui_builds_tree() -> None:
    codec = build_fixture_codec()
    tree = topology_tree_from_openui(HERO, codec, [":hero.title", ":hero.body"])
    assert tree.node_type == "document"
    assert len(tree.children) == 4


def test_build_sketch_seed_preserves_structure() -> None:
    from slm_training.dsl.canonicalize import canonicalize

    canonical_target = canonicalize(HERO, validate=True)
    sketch = build_sketch_seed(canonical_target)
    assert 'TextContent(":slot")' in sketch
    assert "Stack" in sketch
    assert "root" in sketch  # structural skeleton preserved


def test_plan_edit_sequence_reaches_hero() -> None:
    from slm_training.dsl.canonicalize import canonicalize

    canonical_target = canonicalize(HERO, validate=True)
    sketch = build_sketch_seed(canonical_target)
    edits, stop_reason = plan_edit_sequence(sketch, canonical_target)
    assert stop_reason == "planned"
    assert edits
    current = sketch
    for edit in edits:
        nxt = apply_canonical_edit(current, edit)
        assert nxt is not None
        current = nxt
    assert "Card" in current
    assert ":hero.title" in current


def test_apply_canonical_edit_replace_production() -> None:
    source = 'root = Stack([n0], "column")\nn0 = TextContent(":slot")'
    edit = CanonicalEdit(
        edit_id="e1",
        action="ReplaceProduction",
        target_name="n0",
        production="Button",
    )
    result = apply_canonical_edit(source, edit)
    assert result is not None
    assert "Button" in result


def test_apply_insert_missing_statement_is_replayable() -> None:
    source = 'title = TextContent(":title")\nroot = Stack([title], "column")'
    edit = CanonicalEdit(
        edit_id="insert-body",
        action="InsertStatement",
        target_name="body",
        production="TextContent",
        slot=":body",
    )
    result = apply_canonical_edit(source, edit)
    assert result is not None
    assert 'body = TextContent(":body")' in result


def test_apply_canonical_edit_bind_slot() -> None:
    source = 'root = Stack([n0], "column")\nn0 = TextContent(":slot")'
    edit = CanonicalEdit(
        edit_id="e1",
        action="BindSlotPointer",
        target_name="n0",
        slot=':new',
    )
    result = apply_canonical_edit(source, edit)
    assert result is not None
    assert ':new' in result


def test_permute_slot_contract_is_safe() -> None:
    source = 'root = Stack([a], "column")\na = TextContent(":x")\nb = TextContent(":y")'
    permuted = permute_slot_contract(source, [":x", ":y"], [":y", ":x"])
    assert ":y" in permuted
    assert ":x" in permuted
    # No chain-collapse: both placeholders still present.
    assert permuted.count(":y") == source.count(":y")
    assert permuted.count(":x") == source.count(":x")


def test_run_edit_reachability_fixture_produces_report() -> None:
    report = run_edit_reachability_fixture()
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.cases
    assert "harness.experiments.slm188_edit_algebra" in report.version_stamp["components"]


def test_report_round_trip() -> None:
    report = run_edit_reachability_fixture()
    recovered = EditReachabilityReport.from_dict(report.to_dict())
    assert recovered.matrix_set == MATRIX_SET
    assert recovered.experiment_id == EXPERIMENT_ID
    assert len(recovered.cases) == len(report.cases)


def test_render_markdown_contains_caveats() -> None:
    report = run_edit_reachability_fixture()
    md = render_markdown(report)
    assert "SLM-188" in md
    assert "Claim class:" in md
    assert "wiring / fixture only" in md
    assert "No-go for promotion" in md
    assert "Honest caveats" in md


def test_transition_certificate_round_trip() -> None:
    cert = TransitionCertificateV1(
        source_fingerprint="a",
        target_fingerprint="b",
        edit=CanonicalEdit(edit_id="e", action="Stop", target_name="root").to_dict(),
    )
    recovered = TransitionCertificateV1.from_dict(cert.to_dict())
    assert recovered.source_fingerprint == "a"
    assert recovered.certificate_digest == cert.certificate_digest


def test_replay_transition_certificate() -> None:
    from slm_training.dsl.canonicalize import canonical_fingerprint

    source = 'root = Stack([n0], "column")\nn0 = TextContent(":slot")'
    edit = CanonicalEdit(
        edit_id="e1",
        action="BindSlotPointer",
        target_name="n0",
        slot=':new',
    )
    target = apply_canonical_edit(source, edit)
    assert target is not None
    cert = TransitionCertificateV1(
        source_fingerprint=canonical_fingerprint(source),
        target_fingerprint=canonical_fingerprint(target),
        edit=edit.to_dict(),
        source_program=source,
        target_program=target,
        verifier_accepted=True,
    )
    ok, detail = replay_transition_certificate(cert)
    assert ok, detail


def test_invariance_suite_passes_on_hero() -> None:
    sketch = build_sketch_seed(HERO)
    inv = run_invariance_suite(sketch, HERO, [":hero.title", ":hero.body"])
    assert inv.canonical_idempotent
    assert inv.alpha_equivalent
    assert inv.slot_permutation_equivalent


def test_build_seed_target_pairs() -> None:
    pairs = build_seed_target_pairs(
        "minimal_stack", 'root = Stack([], "column")', [("hero", HERO, [":hero.title", ":hero.body"])]
    )
    assert len(pairs) == 1
    case_id, seed_id, target, slots = pairs[0]
    assert case_id == "minimal_stack__hero"
    assert "Card" in target
