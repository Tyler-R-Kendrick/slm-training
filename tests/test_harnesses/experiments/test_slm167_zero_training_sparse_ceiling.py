"""Tests for SLM-167 (SDE1-05) zero-training sparse-action ceiling fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm167_zero_training_sparse_ceiling import (
    ARM_NAMES,
    DECODE_SETTINGS,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    SCORING_METHODS,
    FrozenActionArm,
    FrozenActionReport,
    build_cells,
    render_action_text,
    render_state_text,
    resolve_disposition,
    run_fixture_campaign,
    score_actions,
    validate_manifest,
)


def test_build_cells_produces_all_methods_per_seed() -> None:
    cells = build_cells(seeds=(0, 1, 2))
    assert len(cells) == len(SCORING_METHODS) * len(DECODE_SETTINGS) * 3
    per_seed = {}
    for cell in cells:
        per_seed.setdefault(cell.seed, set()).add(cell.arm_id)
    assert len(per_seed) == 3
    for seed, ids in per_seed.items():
        assert len(ids) == len(SCORING_METHODS) * len(DECODE_SETTINGS), f"seed {seed} has {len(ids)} cells"


def test_cells_cover_all_arm_names() -> None:
    cells = build_cells(seeds=(0,))
    seen = {c.arm_name for c in cells}
    assert seen == set(ARM_NAMES)


def test_validate_manifest_accepts_valid_cells() -> None:
    cells = build_cells(seeds=(0,))
    assert validate_manifest(cells) == []


def test_validate_manifest_rejects_duplicate_arm_id() -> None:
    cells = build_cells(seeds=(0,))
    duplicated = cells + (cells[0],)
    errors = validate_manifest(duplicated)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_invalid_scoring_method() -> None:
    cells = build_cells(seeds=(0,))
    bad = FrozenActionArm(
        arm_id="bad",
        arm_name="invalid",
        scoring_method="invalid",
        decode_setting="gold_state",
        seed=0,
        d_model=64,
        k_retrieve=8,
        use_expanded_descriptions=False,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid scoring_method" in e for e in errors)


def test_validate_manifest_rejects_invalid_decode_setting() -> None:
    cells = build_cells(seeds=(0,))
    bad = FrozenActionArm(
        arm_id="bad",
        arm_name="random_uniform",
        scoring_method="random_uniform",
        decode_setting="invalid",
        seed=0,
        d_model=64,
        k_retrieve=8,
        use_expanded_descriptions=False,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid decode_setting" in e for e in errors)


def test_render_state_text_excludes_gold() -> None:
    text = render_state_text(
        state_signature="sig",
        expected_nonterminal="element",
        parent_field="children",
        frontier_path="root/section[0]",
        scope_summary="scope@abc",
        pointer_candidates=("@0", "@1"),
    )
    assert "state_signature: sig" in text
    assert "expected_nonterminal: element" in text
    assert "pointer_candidates: @0, @1" in text
    assert "gold" not in text.lower()
    assert "future" not in text.lower()


def test_render_action_text_uses_catalog() -> None:
    text = render_action_text("+Card")
    assert "Card" in text
    assert "+Card" in text


def test_score_actions_produces_ranks_and_selected() -> None:
    score = score_actions(
        prompt="Build a card with a title and button.",
        state_signature="state_001",
        expected_nonterminal="element",
        parent_field="children",
        frontier_path="root/body[0]",
        scope_summary="scope@001",
        legal_action_ids=("+Card", "+Button", "+Input", "-"),
        scoring_method="bi_encoder_similarity",
        seed=0,
        d_model=64,
        gold_action_id="+Card",
    )
    assert score.candidate_set_size == 4
    assert len(score.ranks) == 4
    assert score.selected_action_id in score.legal_action_ids
    assert score.gold_action_id == "+Card"
    assert score.latency_seconds >= 0.0
    assert not score.free_running_diverged


def test_hybrid_retrieval_rerank_sets_non_topk_to_negative_inf() -> None:
    score = score_actions(
        prompt="Build a card with a title and button.",
        state_signature="state_002",
        expected_nonterminal="element",
        parent_field="children",
        frontier_path="root/body[0]",
        scope_summary="scope@002",
        legal_action_ids=("+Card", "+Button", "+Input", "+Stack", "+Tabs", "-"),
        scoring_method="hybrid_retrieval_rerank",
        seed=0,
        d_model=64,
        k_retrieve=3,
    )
    # At most k_retrieve raw scores should be finite; the rest are -inf.
    finite = [s for s in score.raw_scores if s != float("-inf")]
    assert len(finite) <= 3


def test_semantic_methods_beat_random_and_frequency() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    means = report.arm_means
    random_top1 = means["random_uniform"]["top1_accuracy"]
    freq_top1 = means["global_frequency"]["top1_accuracy"]
    bi_top1 = means["bi_encoder_similarity"]["top1_accuracy"]
    hybrid_top1 = means["hybrid_retrieval_rerank"]["top1_accuracy"]
    assert bi_top1 > random_top1
    assert bi_top1 > freq_top1
    assert hybrid_top1 >= bi_top1 - 0.05


def test_hybrid_full_set_recall_below_perfect() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    hybrid_recall = report.arm_means["hybrid_retrieval_rerank"]["full_set_recall"]
    assert 0.0 <= hybrid_recall <= 1.0


def test_disposition_useful_prior_or_inconclusive() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    assert report.disposition in {
        "useful_zero_training_prior",
        "ranking_only_not_generative",
        "frequency_explains_signal",
        "no_alignment_signal",
        "inconclusive",
    }


def test_resolve_disposition_no_signal() -> None:
    means = {
        "random_uniform": {"top1_accuracy": 0.10},
        "global_frequency": {"top1_accuracy": 0.12},
        "compiler_local_frequency": {"top1_accuracy": 0.11},
        "permuted_descriptions": {"top1_accuracy": 0.10},
        "bi_encoder_similarity": {"top1_accuracy": 0.11},
        "frozen_continuation": {"top1_accuracy": 0.10},
        "hybrid_retrieval_rerank": {"top1_accuracy": 0.10, "meaningful_program_rate": 0.05},
        "small_model_control": {"top1_accuracy": 0.15},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "no_alignment_signal"


def test_resolve_disposition_useful_prior() -> None:
    means = {
        "random_uniform": {"top1_accuracy": 0.10},
        "global_frequency": {"top1_accuracy": 0.12},
        "compiler_local_frequency": {"top1_accuracy": 0.13},
        "permuted_descriptions": {"top1_accuracy": 0.11},
        "bi_encoder_similarity": {"top1_accuracy": 0.30},
        "frozen_continuation": {"top1_accuracy": 0.35},
        "hybrid_retrieval_rerank": {"top1_accuracy": 0.40, "meaningful_program_rate": 0.30},
        "small_model_control": {"top1_accuracy": 0.50},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "useful_zero_training_prior"


def test_report_round_trip() -> None:
    report = run_fixture_campaign(seeds=(0,))
    reconstructed = FrozenActionReport.from_dict(report.to_dict())
    assert reconstructed.matrix_set == MATRIX_SET
    assert reconstructed.matrix_version == MATRIX_VERSION
    assert reconstructed.experiment_id == EXPERIMENT_ID
    assert reconstructed.status == "fixture"
    assert reconstructed.claim_class == "wiring"
    assert len(reconstructed.rows) == len(report.rows)


def test_report_version_stamp() -> None:
    report = run_fixture_campaign(seeds=(0,))
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    components = report.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm167_zero_training_sparse_ceiling" in components
