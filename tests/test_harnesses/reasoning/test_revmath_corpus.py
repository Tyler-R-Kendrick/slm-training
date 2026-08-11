"""HARN-11 / SLM-554: frozen content-addressed revmath fixture corpus."""

from __future__ import annotations

from slm_training.harnesses.reasoning.revmath.corpus import (
    CORPUS_ID,
    CORPUS_SCHEMA,
    REQUIRED_CORPUS_CLASSES,
    assert_no_undeclared_train_eval_leakage,
    build_fixture_corpus_manifest,
    coverage_table,
    demonstrate_mutation_failures,
    detect_train_eval_leakage,
    load_fixture_corpus_manifest,
    verify_corpus_replay,
)

# Paired pos/neg cases intentionally share a root family across train/eval;
# leakage must remain detectable via detect_train_eval_leakage.
_ALLOWED_SHARED_FAMILIES = frozenset({"family.add_zero", "family.wkl_path"})


def test_manifest_loads_and_covers_required_classes() -> None:
    manifest = load_fixture_corpus_manifest()
    assert manifest.schema_version == CORPUS_SCHEMA
    assert manifest.corpus_id == CORPUS_ID
    assert manifest.entry_count == len(manifest.entries) == 19
    classes = {e.corpus_class for e in manifest.entries}
    assert REQUIRED_CORPUS_CLASSES <= classes
    rows = coverage_table(manifest)
    assert len(rows) == manifest.entry_count
    assert {r["corpus_class"] for r in rows} == classes


def test_train_eval_root_family_leakage_is_detectable() -> None:
    manifest = load_fixture_corpus_manifest()
    leaked = detect_train_eval_leakage(manifest)
    assert set(leaked) == _ALLOWED_SHARED_FAMILIES
    assert_no_undeclared_train_eval_leakage(
        manifest, allowed_shared_families=_ALLOWED_SHARED_FAMILIES
    )


def test_declared_mutations_fail_intended_gates() -> None:
    manifest = load_fixture_corpus_manifest()
    mutation_entries = [
        e for e in manifest.entries if e.corpus_class.startswith("mutation_")
    ]
    assert len(mutation_entries) == 4
    for entry in mutation_entries:
        hits = demonstrate_mutation_failures(entry)
        assert hits
        assert all(h.split(":", 1)[0] for h in hits)


def test_frozen_corpus_replays_exactly() -> None:
    manifest = load_fixture_corpus_manifest()
    verify_corpus_replay(manifest)


def test_unknown_and_invalid_remain_distinct_from_refutation() -> None:
    manifest = load_fixture_corpus_manifest()
    by_class = {e.corpus_class: e for e in manifest.entries}
    assert by_class["timeout"].expected_outcome == "unknown"
    assert by_class["incomplete_domain"].expected_outcome == "unknown"
    assert by_class["malformed_evidence"].expected_outcome == "invalid"
    assert by_class["counterexample_checked"].expected_outcome == "refuted"
    assert by_class["constructivization_unknown"].expected_outcome == "unknown"
    assert by_class["missing_tool"].expected_unknown_reason == "missing_tool"
    assert by_class["unsupported_feature"].expected_unknown_reason == (
        "unsupported_capability"
    )


def test_rebuild_matches_committed_manifest_bytes_identity() -> None:
    frozen = load_fixture_corpus_manifest()
    live = build_fixture_corpus_manifest()
    assert live.corpus_sha256 == frozen.corpus_sha256
    assert [e.to_dict() for e in live.entries] == [e.to_dict() for e in frozen.entries]
