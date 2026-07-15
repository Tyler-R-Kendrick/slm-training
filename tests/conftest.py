"""Shared pytest policy."""

from __future__ import annotations

import pytest


TRAINING_TESTS = {
    "tests/test_harnesses/distill/test_select_sft.py::test_self_distill_sft_smoke",
    "tests/test_harnesses/model_build/test_full_state_resume.py::test_full_state_resume_is_bit_exact",
    "tests/test_harnesses/model_build/test_full_state_resume.py::test_loss_eval_wiring",
    "tests/test_harnesses/model_build/test_full_state_resume.py::test_resume_rejects_different_corpus",
    "tests/test_harnesses/model_build/test_full_state_resume.py::test_token_accounting_and_track_block",
    "tests/test_harnesses/model_build/test_full_state_resume.py::test_token_budget_stops_before_steps",
    "tests/test_harnesses/model_build/test_grammar_hf.py::test_generate_with_grammar_constrained_overfit",
    "tests/test_harnesses/model_build/test_grammar_hf.py::test_hf_context_tower_optional",
    "tests/test_harnesses/model_build/test_lexer_smoke.py::test_lexer_train_eval_smoke",
    "tests/test_harnesses/model_build/test_twotower.py::test_twotower_save_load_generate",
    "tests/test_harnesses/model_build/test_twotower.py::test_twotower_train_eval_overfit",
    "tests/test_harnesses/model_build/test_twotower.py::test_twotower_training_loss_decreases",
    "tests/test_harnesses/model_build/test_v7_speculative.py::test_train_survival_gate_updates_head_and_flags",
    "tests/test_harnesses/quality/test_rl_curriculum_telemetry.py::test_grpo_smoke",
    "tests/test_harnesses/rl/test_trajectory.py::test_trajectory_rl_smoke",
    "tests/test_models/test_grammar_diffusion.py::test_grammar_diffusion_train_eval_overfit",
    "tests/test_models/test_grammar_diffusion.py::test_save_load_and_generate_batch_requests",
    "tests/test_models/test_grammar_diffusion.py::test_training_loss_decreases",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep iterative model training out of the default lightweight suite."""
    training = pytest.mark.training
    for item in items:
        if item.nodeid.split("[", 1)[0] in TRAINING_TESTS:
            item.add_marker(training)
