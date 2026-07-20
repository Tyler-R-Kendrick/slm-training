"""Tests for SLM-133 AST-sketch × choice-native retrieval factorial harness."""

from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
    AST_SKETCH_RETRIEVAL_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    build_ast_sketch_retrieval_manifest,
    build_ast_training_sketch,
    build_choice_exemplar_bank,
    build_choice_retrieval_exemplar,
    format_choice_exemplar_context,
    nearest_choice_exemplars,
    random_choice_exemplars,
    render_markdown,
    run_fixture_matrix,
    validate_manifest,
)


SIMPLE_TEXT = 'root = Stack([x])\nx = TextContent(":title")'
ALPHA_EQUIV = 'root = Stack([y])\ny = TextContent(":title")'
DISTINCT_CARD = 'root = Stack([z])\nz = Card([t])\nt = TextContent(":body")'
DIFFERENT_BINDING = 'root = Stack([x])\nx = Button(":action")'


def _record(record_id: str, prompt: str, openui: str, **meta: object) -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        split="train",
        source="fixture",
        meta=dict(meta),
    )


def test_build_ast_training_sketch_hashes_valid_program() -> None:
    sketch = build_ast_training_sketch(SIMPLE_TEXT)
    assert sketch.schema == "AstTrainingSketchV1"
    assert len(sketch.sketch_hash) == 64
    assert sketch.sketch_version == "ast_sketch_v1"
    assert sketch.canonical_fingerprint
    assert sketch.structural_fingerprint


def test_sketch_collapses_alpha_equivalent_surface_variants() -> None:
    a = build_ast_training_sketch(SIMPLE_TEXT)
    b = build_ast_training_sketch(ALPHA_EQUIV)
    assert a.sketch_hash == b.sketch_hash
    assert a.topology_hash == b.topology_hash
    assert a.binding_hash == b.binding_hash


def test_sketch_preserves_distinct_component_types() -> None:
    text = build_ast_training_sketch(SIMPLE_TEXT)
    card = build_ast_training_sketch(DISTINCT_CARD)
    assert text.sketch_hash != card.sketch_hash


def test_sketch_preserves_distinct_bindings() -> None:
    text = build_ast_training_sketch(SIMPLE_TEXT)
    button = build_ast_training_sketch(DIFFERENT_BINDING)
    assert text.sketch_hash != button.sketch_hash


def test_sketch_preserves_distinct_arities() -> None:
    one = build_ast_training_sketch('root = Stack([x])\nx = TextContent(":title")')
    two = build_ast_training_sketch(
        'root = Stack([x, y])\nx = TextContent(":title")\ny = TextContent(":body")'
    )
    assert one.sketch_hash != two.sketch_hash


def test_build_choice_retrieval_exemplar_round_trips() -> None:
    record = _record("r1", "a title prompt", SIMPLE_TEXT, quality_tier="gold")
    exemplar = build_choice_retrieval_exemplar(record)
    assert exemplar.schema == "ChoiceRetrievalExemplarV1"
    assert exemplar.record_id == "r1"
    assert exemplar.quality_tier == "gold"
    assert exemplar.choice_sequence
    assert exemplar.sequence_hash
    assert exemplar.ast_sketch.sketch_hash
    data = exemplar.to_dict()
    # Use public from_dict for schema round-trip coverage.
    from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
        ChoiceRetrievalExemplarV1,
    )

    restored_ex = ChoiceRetrievalExemplarV1.from_dict(data)
    assert restored_ex.record_id == exemplar.record_id
    assert restored_ex.sequence_hash == exemplar.sequence_hash


def test_choice_exemplar_bank_excludes_empty_records() -> None:
    records = [
        _record("r1", "prompt one", SIMPLE_TEXT),
        _record("r2", "prompt two", DISTINCT_CARD),
    ]
    bank = build_choice_exemplar_bank(records)
    assert len(bank) == 2
    assert {ex.record_id for ex in bank} == {"r1", "r2"}


def test_nearest_choice_exemplars_excludes_same_record() -> None:
    records = [
        _record("r1", "show a title", SIMPLE_TEXT),
        _record("r2", "show a body", DISTINCT_CARD),
    ]
    bank = build_choice_exemplar_bank(records)
    hits = nearest_choice_exemplars(bank, records[0], k=4)
    assert all(ex.record_id != "r1" for ex, _score, _rank in hits)


def test_nearest_choice_exemplars_ranks_by_prompt_overlap() -> None:
    records = [
        _record("r1", "show a title card", SIMPLE_TEXT),
        _record("r2", "show a title stack", ALPHA_EQUIV),
        _record("r3", "show a body", DISTINCT_CARD),
    ]
    bank = build_choice_exemplar_bank(records)
    hits = nearest_choice_exemplars(bank, records[0], k=2)
    assert len(hits) == 2
    # r2 shares more prompt tokens with r1 than r3 does.
    assert hits[0][0].record_id == "r2"


def test_nearest_choice_exemplars_excludes_shared_root_parent() -> None:
    records = [
        _record("r1", "show a title", SIMPLE_TEXT, root_parent_id="parent_a"),
        _record("r2", "show a body", DISTINCT_CARD, root_parent_id="parent_a"),
        _record("r3", "show a title again", SIMPLE_TEXT, root_parent_id="parent_b"),
    ]
    bank = build_choice_exemplar_bank(records)
    hits = nearest_choice_exemplars(bank, records[0], k=4)
    assert all(ex.record_id != "r2" for ex, _score, _rank in hits)


def test_random_choice_exemplars_is_deterministic() -> None:
    records = [
        _record("r1", "a", SIMPLE_TEXT),
        _record("r2", "b", DISTINCT_CARD),
        _record("r3", "c", DIFFERENT_BINDING),
    ]
    bank = build_choice_exemplar_bank(records)
    first = random_choice_exemplars(bank, records[0], k=2, seed=7)
    second = random_choice_exemplars(bank, records[0], k=2, seed=7)
    assert [ex.record_id for ex, _s, _r in first] == [
        ex.record_id for ex, _s, _r in second
    ]


def test_format_choice_exemplar_context_includes_section_header() -> None:
    records = [
        _record("r1", "show a title", SIMPLE_TEXT),
        _record("r2", "show a body", DISTINCT_CARD),
    ]
    bank = build_choice_exemplar_bank(records)
    hits = nearest_choice_exemplars(bank, records[0], k=4)
    context = format_choice_exemplar_context(hits, budget=2000)
    assert context.startswith("---RETRIEVED_CHOICE_EXEMPLARS v1---")
    assert "rank=1" in context


def test_default_manifest() -> None:
    manifest = build_ast_sketch_retrieval_manifest()
    assert manifest.experiment_id == AST_SKETCH_RETRIEVAL_ID
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert manifest.status == "not_run"
    assert manifest.claim_class == "wiring"
    assert len(manifest.arms) == 6


def test_manifest_without_controls() -> None:
    manifest = build_ast_sketch_retrieval_manifest(include_controls=False)
    assert len(manifest.arms) == 4


def test_validate_manifest_ok() -> None:
    manifest = build_ast_sketch_retrieval_manifest()
    assert validate_manifest(manifest) == []


def test_validate_manifest_rejects_duplicate_arm() -> None:
    manifest = build_ast_sketch_retrieval_manifest(include_controls=False)
    duplicate = manifest.arms[0]
    manifest = manifest.__class__(arms=(*manifest.arms, duplicate))
    errors = validate_manifest(manifest)
    assert any("duplicate arm" in e for e in errors)


def test_validate_frontier_requires_parent() -> None:
    from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
        AstSketchRetrievalManifest,
    )

    manifest = AstSketchRetrievalManifest(
        arms=build_ast_sketch_retrieval_manifest(include_controls=False).arms,
        claim_class="frontier",
        parent_checkpoint_uri=None,
        checkpoint_bucket=None,
    )
    errors = validate_manifest(manifest)
    assert any("parent_checkpoint_uri" in e for e in errors)
    assert any("checkpoint_bucket" in e for e in errors)


def test_run_fixture_matrix(tmp_path) -> None:
    manifest = build_ast_sketch_retrieval_manifest(seeds=(0, 1), include_controls=False)
    report = run_fixture_matrix(manifest, run_id="test", output_dir=tmp_path)
    assert report.status == "fixture"
    assert len(report.rows) == 8  # 4 arms * 2 seeds
    assert all(r.status == "fixture_planned" for r in report.rows)
    assert (tmp_path / "ast_sketch_retrieval_report.json").exists()


def test_render_markdown_includes_hypothesis() -> None:
    manifest = build_ast_sketch_retrieval_manifest(seeds=(0,), include_controls=False)
    report = run_fixture_matrix(manifest, run_id="md_test")
    md = render_markdown(report)
    assert "SLM-133" in md
    assert manifest.hypothesis[:20] in md
    assert "raw_stratified" in md
    assert "Fixture/plan only" in md
