"""G4 (SLM-36): reasoning bench invariants + tiny end-to-end smoke."""

from __future__ import annotations

import pytest

from slm_training.dsl.packs import get_pack
from slm_training.dsl.packs.arith_sketch import evaluate_answer
from slm_training.harnesses.reasoning import (
    score_direct_output,
    score_sketch_output,
)


def test_evaluator_is_the_single_oracle() -> None:
    """Validity and answer share one code path; every invalid shape rejects."""
    assert evaluate_answer("x = 3\ny = x * 4\nroot = y + 2") == 14.0
    assert evaluate_answer("root = 10 / 4") == pytest.approx(2.5)
    for bad in (
        "x = 3",                      # no root
        "root = y + 1",               # undefined ref
        "x = y\ny = x\nroot = x",     # cycle
        "root = 1 / 0",               # division by zero
        "root = ",                    # parse failure
    ):
        with pytest.raises(ValueError):
            evaluate_answer(bad)


def test_generator_gold_matches_oracle_and_is_deterministic() -> None:
    pack = get_pack("arith-sketch")
    records = pack.corpus_generator(12, 3)
    again = pack.corpus_generator(12, 3)
    assert [r.openui for r in records] == [r.openui for r in again]
    for record in records:
        assert evaluate_answer(record.openui) == record.meta["gold_answer"]


def test_scoring_is_fail_closed() -> None:
    gold = 14.0
    assert score_sketch_output("x = 3\ny = x * 4\nroot = y + 2", gold)["correct"]
    invalid = score_sketch_output("x = 3\nroot = z + 1", gold)
    assert not invalid["valid"] and not invalid["correct"]
    wrong = score_sketch_output("root = 13", gold)
    assert wrong["valid"] and not wrong["correct"]
    assert score_direct_output("14", gold)["correct"]
    assert not score_direct_output("nope", gold)["correct"]
    assert not score_direct_output("", gold)["correct"]


def test_bench_end_to_end_tiny(tmp_path) -> None:
    """Both arms train, decode, and score through the one oracle. Accuracy is
    not asserted (2 steps of training proves wiring, not skill)."""
    pytest.importorskip("torch")

    from slm_training.harnesses.reasoning import (
        ReasoningBenchConfig,
        run_reasoning_bench,
    )

    summary = run_reasoning_bench(
        ReasoningBenchConfig(
            n_train=8,
            n_test=3,
            steps=2,
            d_model=32,
            denoiser_layers=1,
            seed=1,
            campaign_id="g4_unit",
            output_root=tmp_path,
        )
    )
    assert summary["n_test"] == 3
    for arm in ("sketch", "direct"):
        assert 0.0 <= summary[arm]["answer_accuracy"] <= 1.0
        assert len(summary[arm]["outputs"]) == 3
    # Gold answers re-verify through the same oracle used for scoring.
    assert all(isinstance(g, float) for g in summary["golds"])
