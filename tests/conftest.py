"""Shared pytest policy."""

from __future__ import annotations

import pytest

from slm_training.autoresearch.schemas import RLReadinessReport


@pytest.fixture(autouse=True)
def _disable_remote_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests offline — Hub bucket/jobs inventory is opt-in per test."""
    monkeypatch.setenv("SLM_DISABLE_REMOTE_INVENTORY", "1")


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


@pytest.fixture
def approved_rl_report() -> RLReadinessReport:
    return RLReadinessReport(
        report_id="rl-ready-test",
        evaluation_sha256="a" * 64,
        frozen_snapshot=True,
        required_suites=("smoke", "held_out", "adversarial", "ood", "rico_held"),
        suite_sizes={
            "smoke": 8,
            "held_out": 8,
            "adversarial": 8,
            "ood": 8,
            "rico_held": 1500,
        },
        ship_gates_pass=True,
        agentv_pass=True,
        reward_sample_count=3,
        reward_variance=0.02,
        approved=True,
    )


@pytest.fixture
def approved_rl_report_path(tmp_path, approved_rl_report):
    path = tmp_path / "rl_readiness.json"
    path.write_text(approved_rl_report.model_dump_json(indent=2) + "\n")
    return path
