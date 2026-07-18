"""Undefined-vs-zero metric semantics: no vacuous 1.0s, no fabricated 0.0s."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.parser import ParseError
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.evals.agentv import model_ship_gate_cases
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build import eval_runner
from slm_training.harnesses.model_build.eval_runner import (
    _contract_precision,
    _contract_recall,
    _placeholder_fidelity,
    _placeholder_fidelity_normalized,
    _placeholder_validity,
    _tree_match,
    component_type_recall,
    evaluate,
)

_GOLD = 'root = Stack([cta])\ncta = Button(":cta")'


def _record(**overrides: object) -> ExampleRecord:
    data = {
        "id": "r1",
        "prompt": "CTA",
        "openui": _GOLD,
        "placeholders": [":cta"],
        **overrides,
    }
    return ExampleRecord.from_dict(data)


def test_empty_set_metrics_are_undefined_not_perfect() -> None:
    bare = _record(openui="root = Stack([])", placeholders=[])
    pred = "root = Stack([])"
    assert _contract_precision(pred, bare) is None
    assert _contract_recall(pred, bare) is None
    assert _placeholder_fidelity(pred, bare) is None
    assert _placeholder_fidelity_normalized(pred, bare) is None
    assert _placeholder_validity(pred, bare) is None
    # Gold with only Stacks: type recall is undefined, not a free 1.0.
    assert component_type_recall(pred, "root = Stack([])") is None


def test_real_mismatches_still_score_zero() -> None:
    gold = _record()
    # Empty prediction against a non-empty contract is a real failure.
    assert _contract_recall("", gold) == 0.0
    assert _placeholder_fidelity("", gold) == 0.0
    # Spurious placeholders against an empty contract are a real failure.
    bare = _record(openui="root = Stack([])", placeholders=[])
    assert _contract_precision('root = Button(":x")', bare) == 0.0
    assert component_type_recall("root = Stack([])", _GOLD) == 0.0


def test_tree_match_splits_model_failure_from_harness_failure() -> None:
    # Unparseable prediction: a genuine 0.0.
    assert _tree_match("garbage(", _GOLD) == 0.0
    # Unparseable *gold* is harness/data breakage and must raise, not score.
    with pytest.raises(ParseError):
        _tree_match(_GOLD, "not a program (")


def _smoke_config(tmp_path: Path) -> ModelBuildConfig:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    write_jsonl(train_dir / "records.jsonl", [_record(split="train")])
    write_jsonl(
        test_dir / "suites" / "smoke" / "records.jsonl",
        [_record(split="smoke", meta={"suite": "smoke"})],
    )
    return ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="metric-semantics",
        model_name="stub",
    )


class _EchoGoldModel:
    def generate_batch_requests(self, requests: list[object]) -> list[str]:
        return [_GOLD for _ in requests]


def test_empty_suite_aggregates_to_none_not_zero(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    write_jsonl(config.test_dir / "suites" / "smoke" / "records.jsonl", [])
    metrics = evaluate(config, model=_EchoGoldModel(), publish_agentv=False)
    assert metrics["n"] == 0
    assert metrics["document_n"] == 0
    for name in (
        "parse_rate",
        "meaningful_program_rate",
        "syntax_parse_rate",
        "raw_syntax_validity",
        "contract_precision",
        "contract_recall",
        "placeholder_fidelity",
        "structural_similarity",
        "component_type_recall",
        "reward_score",
    ):
        assert metrics[name] is None, name
    assert metrics["metric_defined_n"]["contract_precision"] == 0


def test_reward_harness_error_is_counted_not_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(pred: str, record: ExampleRecord) -> float:
        raise RuntimeError("reward harness broke")

    monkeypatch.setattr(eval_runner, "_reward_for_prediction", _boom)
    metrics = evaluate(_smoke_config(tmp_path), model=_EchoGoldModel(), publish_agentv=False)
    assert metrics["reward_error_count"] == 1
    assert metrics["reward_score"] is None
    assert metrics["metric_defined_n"]["reward_score"] == 0
    # The rest of the scoreboard is unaffected by the reward failure.
    assert metrics["parse_rate"] == 1.0
    assert metrics["contract_recall"] == 1.0


def test_measured_metrics_report_defined_counts_and_fallbacks(tmp_path: Path) -> None:
    metrics = evaluate(_smoke_config(tmp_path), model=_EchoGoldModel(), publish_agentv=False)
    assert metrics["parse_rate"] == 1.0
    assert metrics["contract_precision"] == 1.0
    assert metrics["metric_defined_n"]["contract_precision"] == 1
    # generate_batch_requests collects decode stats → measured zero fallbacks.
    assert metrics["fallback_count"] == 0
    assert metrics["empty_prediction_count"] == 0
    # Default policy is grammar-constrained → syntax metrics are labeled as
    # decoder-guaranteed; the slot contract is not injected by default.
    assert "parse_rate" in metrics["decoder_guaranteed"]
    assert "contract_precision" not in metrics["decoder_guaranteed"]


def test_agentv_single_suite_publishes_one_case() -> None:
    suites = {
        "smoke": {
            "n": 3,
            "meaningful_program_rate": 0.0,
            "syntax_parse_rate": 1.0,
            "structural_similarity": 0.1,
            "component_type_recall": 0.1,
            "placeholder_fidelity": 0.1,
            "reward_score": 0.1,
            "fallback_count": 0,
        }
    }
    cases = model_ship_gate_cases(suites, include_missing_suites=False)
    assert len(cases) == 1
    assert cases[0]["id"] == "smoke"
    # The one real suite failed its bars; no missing_suite noise is attached.
    assert cases[0]["pass"] is False
    assert all("missing_suite" not in failure for failure in cases[0]["failures"])
