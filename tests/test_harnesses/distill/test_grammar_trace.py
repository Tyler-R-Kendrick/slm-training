"""Tests for CAP1-02 grammar-state decision traces."""

from __future__ import annotations

import math

from slm_training.harnesses.distill.grammar_trace import (
    GrammarTraceRecorder,
    compute_entropy,
    compute_margin,
    grammar_trace_coverage_report,
    grammar_trace_replay_violations,
    normalize_legal_probs,
    state_fingerprint,
)


def test_normalize_legal_probs_sums_to_one() -> None:
    probs = normalize_legal_probs([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 1e-6
    assert len(probs) == 3


def test_normalize_legal_probs_for_energy_convention() -> None:
    probs = normalize_legal_probs([1.0, 2.0, 3.0], convention="energy")
    # Lower energy should get higher probability.
    assert probs[0] > probs[2]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_compute_entropy_in_bits() -> None:
    entropy = compute_entropy([0.5, 0.5])
    assert entropy is not None
    assert abs(entropy - 1.0) < 1e-6
    assert compute_entropy([1.0]) == 0.0


def test_compute_margin_logit_convention() -> None:
    margin = compute_margin([1.0, 2.0, 3.0], selected_index=2, convention="logit")
    assert margin is not None
    assert margin == 3.0 - 2.0


def test_compute_margin_energy_convention() -> None:
    margin = compute_margin([1.0, 2.0, 3.0], selected_index=2, convention="energy")
    assert margin is not None
    # Best competing energy is 1.0; accepted energy is 3.0.
    assert margin == 1.0 - 3.0


def test_compute_margin_multi_target_uses_best_competitor() -> None:
    # Selected is not the best; margin compares selected to best competitor.
    margin = compute_margin([1.0, 5.0, 3.0], selected_index=0, convention="logit")
    assert margin == 1.0 - 5.0


def test_state_fingerprint_is_stable() -> None:
    fp1 = state_fingerprint(
        prefix_ids=[1, 2, 3],
        legal_action_ids=["a", "b"],
        coverage="complete",
    )
    fp2 = state_fingerprint(
        prefix_ids=[1, 2, 3],
        legal_action_ids=["b", "a"],
        coverage="complete",
    )
    assert fp1 == fp2
    assert len(fp1) == 64


def test_recorder_creates_trace_with_margin_and_entropy() -> None:
    recorder = GrammarTraceRecorder(
        run_id="r1",
        checkpoint_id="ckpt",
        dataset_id="ds",
        example_id="ex",
        seed=7,
        capture_logits=True,
    )
    trace = recorder.record(
        state_fingerprint="fp",
        legal_action_ids=["a", "b", "c"],
        selected_action_id="b",
        logits_or_energies=[1.0, 3.0, 2.0],
        compiler_coverage="complete",
    )
    assert trace is not None
    assert trace.selected_action_id == "b"
    assert trace.top1_margin is not None
    assert trace.posterior_entropy_bits is not None
    assert 0.0 <= trace.posterior_entropy_bits <= math.log2(3)


def test_recorder_state_stratified_skips_duplicates() -> None:
    recorder = GrammarTraceRecorder(state_stratified=True)
    t1 = recorder.record(
        state_fingerprint="fp",
        legal_action_ids=["a"],
        selected_action_id="a",
    )
    t2 = recorder.record(
        state_fingerprint="fp",
        legal_action_ids=["a", "b"],
        selected_action_id="b",
    )
    assert t1 is not None
    assert t2 is None


def test_replay_violations_detect_invalid_selection() -> None:
    records = [
        {
            "compiler_coverage": "complete",
            "legal_action_ids": ["a", "b"],
            "selected_action_id": "c",
            "posterior_entropy_bits": 0.5,
        }
    ]
    violations = grammar_trace_replay_violations(records)
    assert any("selected_action_id" in v for v in violations)


def test_replay_violations_detect_bad_entropy() -> None:
    records = [
        {
            "compiler_coverage": "complete",
            "legal_action_ids": ["a", "b"],
            "selected_action_id": "a",
            "posterior_entropy_bits": 5.0,
        }
    ]
    violations = grammar_trace_replay_violations(records)
    assert any("entropy" in v for v in violations)


def test_replay_violations_detect_bad_coverage() -> None:
    records = [
        {
            "compiler_coverage": "unknown",
            "legal_action_ids": ["a"],
            "selected_action_id": "a",
        }
    ]
    violations = grammar_trace_replay_violations(records)
    assert any("compiler_coverage" in v for v in violations)


def test_coverage_report_contains_required_keys() -> None:
    recorder = GrammarTraceRecorder(
        run_id="r1", checkpoint_id="ckpt", dataset_id="ds", capture_logits=True
    )
    recorder.record(
        state_fingerprint="fp1",
        legal_action_ids=["a", "b"],
        selected_action_id="a",
        compiler_coverage="complete",
        logits_or_energies=[1.0, 2.0],
    )
    recorder.record(
        state_fingerprint="fp2",
        legal_action_ids=["c"],
        selected_action_id="c",
        compiler_coverage="partial",
        logits_or_energies=[1.0],
    )
    report = grammar_trace_coverage_report(recorder.finalize())
    assert report["n"] == 2
    assert report["unique_states"] == 2
    assert report["partial_coverage_records"] == 1
    assert "branching" in report
    assert "provenance" in report


def test_finalize_returns_json_roundtrippable_dicts() -> None:
    recorder = GrammarTraceRecorder(run_id="r1", checkpoint_id="ckpt")
    recorder.record(
        state_fingerprint="fp",
        legal_action_ids=["a"],
        selected_action_id="a",
        compiler_coverage="complete",
    )
    dicts = recorder.finalize()
    assert len(dicts) == 1
    assert dicts[0]["trace_schema_version"] is not None
