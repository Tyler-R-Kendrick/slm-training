"""Regression tests for SLM-130 canonical AST deduplication harness."""

from __future__ import annotations

from slm_training.dsl.canonicalize import canonicalize
from slm_training.harnesses.experiments.canonical_ast_dedup import (
    RepresentativePolicy,
    build_abstract_mode_signature,
    build_canonical_ast_fingerprint,
    compute_diversity_coverage,
    dedup_arms_for_pool,
    group_candidates_by_canonical_ast,
    unique_slot_truncation,
)


SIMPLE_TEXT = 'root = Stack([x])\nx = TextContent(":title")'
ALPHA_EQUIV = 'root = Stack([y])\ny = TextContent(":title")'
DISTINCT_CARD = 'root = Stack([z])\nz = Card([t])\nt = TextContent(":body")'


def test_canonical_fingerprint_collapses_surface_variants() -> None:
    base = SIMPLE_TEXT
    # Alpha-equivalent surface variants canonicalize to the same program.
    variant_a = 'root = Stack([x])\nx = TextContent(":title")'
    variant_b = 'root = Stack([y])\ny = TextContent(":title")'
    fp_a = build_canonical_ast_fingerprint(canonicalize(variant_a))
    fp_b = build_canonical_ast_fingerprint(canonicalize(variant_b))
    fp_base = build_canonical_ast_fingerprint(canonicalize(base))
    assert fp_a.canonical_fingerprint == fp_base.canonical_fingerprint
    assert fp_b.canonical_fingerprint == fp_base.canonical_fingerprint


def test_canonical_fingerprint_preserves_distinct_programs() -> None:
    fp_text = build_canonical_ast_fingerprint(canonicalize(SIMPLE_TEXT))
    fp_card = build_canonical_ast_fingerprint(canonicalize(DISTINCT_CARD))
    assert fp_text.canonical_fingerprint != fp_card.canonical_fingerprint


def test_abstract_mode_signature_is_diagnostic_only() -> None:
    sig_text = build_abstract_mode_signature(canonicalize(SIMPLE_TEXT))
    sig_alpha = build_abstract_mode_signature(canonicalize(ALPHA_EQUIV))
    # Abstract sigs may differ or coincide; they are never hard-equivalence keys.
    assert sig_text.signature
    assert sig_alpha.signature
    assert "collapse_literal_payloads" in sig_text.normalization_rules


def test_grouping_detects_canonical_duplicates() -> None:
    canonical_base = canonicalize(SIMPLE_TEXT)
    canonical_alpha = canonicalize(ALPHA_EQUIV)
    candidates = (
        ("c0", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.9}),
        ("c1", canonical_alpha, {"valid": True, "contract_satisfied": True, "generator_score": 0.8}),
        ("c2", canonicalize(DISTINCT_CARD), {"valid": True, "contract_satisfied": False, "generator_score": 0.7}),
    )
    groups = group_candidates_by_canonical_ast(candidates)
    assert len(groups) == 2
    group_ids = {g.canonical_fingerprint.canonical_fingerprint for g in groups}
    assert len(group_ids) == 2
    # The duplicate group has multiplicity 2.
    multiplicities = {g.multiplicity for g in groups}
    assert 2 in multiplicities


def test_representative_policy_prefers_contract_satisfied() -> None:
    canonical_base = canonicalize(SIMPLE_TEXT)
    candidates = (
        ("weak", canonical_base, {"valid": True, "contract_satisfied": False, "generator_score": 1.0}),
        ("strong", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.5}),
    )
    groups = group_candidates_by_canonical_ast(
        candidates, policy=RepresentativePolicy.DETERMINISTIC_LEXICOGRAPHIC
    )
    assert len(groups) == 1
    assert groups[0].selected_representative_id == "strong"


def test_invalid_candidates_never_merge_with_valid() -> None:
    canonical_base = canonicalize(SIMPLE_TEXT)
    candidates = (
        ("valid", canonical_base, {"valid": True, "contract_satisfied": True}),
        ("invalid", "not valid openui {", {"valid": False, "contract_satisfied": False}),
    )
    groups = group_candidates_by_canonical_ast(candidates)
    assert len(groups) == 2
    invalid_group = next(
        g for g in groups if g.member_candidate_ids == ("invalid",)
    )
    assert all(level == "INVALID" for level in invalid_group.member_hard_levels)


def test_unique_slot_truncation_refills_without_extra_generation() -> None:
    canonical_base = canonicalize(SIMPLE_TEXT)
    candidates = (
        ("a1", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.9}),
        ("a2", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.8}),
        ("b", canonicalize(DISTINCT_CARD), {"valid": True, "contract_satisfied": False, "generator_score": 0.7}),
    )
    selected = unique_slot_truncation(candidates, k=2)
    assert len(selected) == 2
    assert "a1" in selected
    assert "b" in selected
    assert "a2" not in selected


def test_dedup_arms_increase_unique_ast_coverage() -> None:
    canonical_base = canonicalize(SIMPLE_TEXT)
    candidates = (
        ("c0", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.9, "semantic_success": True}),
        ("c1", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.85, "semantic_success": True}),
        ("c2", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.84, "semantic_success": True}),
        ("c3", canonicalize(DISTINCT_CARD), {"valid": True, "contract_satisfied": False, "generator_score": 0.7, "semantic_success": False}),
    )
    arms = dedup_arms_for_pool(candidates, prompt_hash="test_prompt")
    assert arms["A_raw_no_dedup"].pool_size == 4
    assert arms["C_terminal_canonical_ast"].unique_canonical_ast == 2
    assert arms["C_terminal_canonical_ast"].duplicate_multiplicity == 2
    # Abstract-mode arm is a diagnostic, not hard authority.
    assert arms["E_abstract_mode_spread"].unique_abstract_mode_signatures >= 1


def test_compute_diversity_coverage_counts_within_prompt() -> None:
    canonical_base = canonicalize(SIMPLE_TEXT)
    candidates = (
        ("c0", canonical_base, {"valid": True, "contract_satisfied": True, "semantic_success": True}),
        ("c1", canonical_base, {"valid": True, "contract_satisfied": True, "semantic_success": True}),
        ("c2", canonicalize(DISTINCT_CARD), {"valid": True, "contract_satisfied": False, "semantic_success": False}),
    )
    report = compute_diversity_coverage(
        candidates, arm="test", prompt_hash="prompt_1"
    )
    assert report.pool_size == 3
    assert report.raw_valid_count == 3
    assert report.unique_canonical_ast == 2
    assert report.duplicate_multiplicity == 1
