"""Undefined-vs-zero metric semantics: no vacuous 1.0s, no fabricated 0.0s."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.dsl.parser import ParseError
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.evals.agentv import model_ship_gate_cases
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build import eval_runner
from slm_training.harnesses.model_build.eval_runner import (
    _cache_replay_payload,
    _contract_precision,
    _contract_recall,
    _effective_record_decode_timeout,
    _placeholder_fidelity,
    _placeholder_fidelity_normalized,
    _placeholder_validity,
    _tree_match,
    component_type_recall,
    evaluate,
)

_GOLD = 'root = Stack([cta])\ncta = Button(":slot_0")'


def test_cache_replay_does_not_reuse_runtime_telemetry(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    replay = _cache_replay_payload(
        {
            "parse_rate": 1.0,
            "latency_ms_p50": 999.0,
            "decode_timeout_count": 3,
            "evaluated_at": "stale",
            "code_git_sha": "old",
            "details": [
                {
                    "id": "r1",
                    "latency_ms": 999.0,
                    "request_completion_latency_ms": 999.0,
                    "amortized_batch_latency_ms": 99.0,
                }
            ],
        },
        config=config,
        suite_path=config.run_dir / "eval_smoke.json",
        version_stamp={"code_commit": "current"},
    )
    assert replay["parse_rate"] == 1.0
    assert replay["latency_ms_p50"] is None
    assert replay["decode_timeout_count"] is None
    assert replay["runtime_measurement_source"] == "cache_not_measured"
    assert replay["evaluated_at"] != "stale"
    assert replay["code_git_sha"] != "old"
    assert "latency_ms" not in replay["details"][0]


def test_eval_wall_fairly_caps_each_remaining_record() -> None:
    assert _effective_record_decode_timeout(
        24.0,
        evaluation_deadline=100.0,
        remaining_record_n=8,
        chunk_record_n=1,
        now=60.0,
    ) == pytest.approx(4.75)
    assert (
        _effective_record_decode_timeout(
            3.0,
            evaluation_deadline=100.0,
            remaining_record_n=8,
            chunk_record_n=1,
            now=60.0,
        )
        == 3.0
    )
    assert (
        _effective_record_decode_timeout(
            24.0,
            evaluation_deadline=None,
            remaining_record_n=8,
            chunk_record_n=1,
            now=60.0,
        )
        == 24.0
    )
    assert (
        _effective_record_decode_timeout(
            24.0,
            evaluation_deadline=None,
            remaining_record_n=3,
            chunk_record_n=3,
            now=60.0,
        )
        == 72.0
    )
    assert (
        _effective_record_decode_timeout(
            3.0,
            evaluation_deadline=100.0,
            remaining_record_n=8,
            chunk_record_n=2,
            now=60.0,
        )
        == 6.0
    )


def _record(**overrides: object) -> ExampleRecord:
    data = {
        "id": "r1",
        "prompt": "CTA",
        "openui": _GOLD,
        "placeholders": [":slot_0"],
        **overrides,
    }
    return ExampleRecord.from_dict(data)


def test_decode_trace_annotation_preserves_eval_record_identity() -> None:
    from slm_training.models.decode_stats import DecodeStats

    stats = DecodeStats()
    stats.constrained_selection_traces.extend(
        [{"row": 0, "phase": "first"}, {"row": 1, "phase": "second"}]
    )
    records = [_record(id="held-a"), _record(id="held-b")]

    eval_runner._annotate_decode_trace_records(stats, records)

    assert [trace["record_id"] for trace in stats.constrained_selection_traces] == [
        "held-a",
        "held-b",
    ]


def test_temporal_decode_evidence_is_per_record_and_excludes_terminal_fields() -> None:
    from slm_training.models.decode_stats import DecodeStats

    stats = DecodeStats()
    stats.constrained_selection_traces.append(
        {
            "position": 3,
            "legal_candidates": 2,
            "forced": False,
            "phase": "ltr_repair",
            "parse_ok": True,
            "decode_outcome": "model_valid",
            "record_id": "held-a",
        }
    )

    assert eval_runner._temporal_decode_evidence(stats, "held-a") == [
        {
            "position": 3,
            "legal_candidates": 2,
            "forced": False,
            "phase": "ltr_repair",
        }
    ]
    assert eval_runner._temporal_decode_evidence(stats, "held-b") == []


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


def test_dual_interface_model_keeps_batched_decode_and_stats(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    records = [
        _record(id=f"smoke-{index}", split="smoke", meta={"suite": "smoke"})
        for index in range(3)
    ]
    write_jsonl(config.test_dir / "suites" / "smoke" / "records.jsonl", records)

    class DualInterfaceModel:
        config = SimpleNamespace(generate_batch_size=8)

        def __init__(self) -> None:
            self.batch_calls: list[int] = []

        def generate_batch_requests(self, requests: list[object]) -> list[str]:
            self.batch_calls.append(len(requests))
            return [_GOLD for _ in requests]

        def generate_with_stats(self, _prompt: str) -> tuple[str, object]:
            raise AssertionError("single-record fallback must not disable batching")

    model = DualInterfaceModel()
    metrics = evaluate(config, model=model, publish_agentv=False)

    assert model.batch_calls == [3]
    assert metrics["decode_batch_size_configured"] == 8
    assert metrics["decode_batch_size_max"] == 3
    assert metrics["decode_chunk_n"] == 1
    # Request latency is the observed batch wall; amortized latency is a
    # separate throughput signal and must not be conflated with it.
    assert metrics["decode_batch_wall_ms_p50"] == pytest.approx(
        metrics["latency_ms_p50"]
    )
    assert metrics["decode_amortized_ms_per_record_p50"] is not None
    assert metrics["decode_records_per_second"] is not None
    assert metrics["details"][0]["request_completion_latency_ms"] >= metrics[
        "details"
    ][0]["amortized_batch_latency_ms"]


def test_config_generate_batch_size_overrides_plugin_default(tmp_path: Path) -> None:
    """A small suite (n < baked generate_batch_size) must not be bundled into
    one all-or-nothing decode chunk when the eval config explicitly requests
    per-record chunking -- a single compiler-heavy record would otherwise
    time out its batch-mates too (autotrain-cycle continuous-openui-local
    8c0b60dd-c2, session ixpohr)."""
    config = _smoke_config(tmp_path)
    config = dataclasses.replace(config, generate_batch_size=1)
    records = [
        _record(id=f"smoke-{index}", split="smoke", meta={"suite": "smoke"})
        for index in range(3)
    ]
    write_jsonl(config.test_dir / "suites" / "smoke" / "records.jsonl", records)

    class DualInterfaceModel:
        config = SimpleNamespace(generate_batch_size=8)

        def __init__(self) -> None:
            self.batch_calls: list[int] = []

        def generate_batch_requests(self, requests: list[object]) -> list[str]:
            self.batch_calls.append(len(requests))
            return [_GOLD for _ in requests]

    model = DualInterfaceModel()
    metrics = evaluate(config, model=model, publish_agentv=False)

    assert model.batch_calls == [1, 1, 1]
    assert metrics["decode_batch_size_configured"] == 1
    assert metrics["decode_chunk_n"] == 3


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
    metrics = evaluate(
        _smoke_config(tmp_path), model=_EchoGoldModel(), publish_agentv=False
    )
    assert metrics["reward_error_count"] == 1
    assert metrics["reward_score"] is None
    assert metrics["metric_defined_n"]["reward_score"] == 0
    # The rest of the scoreboard is unaffected by the reward failure.
    assert metrics["parse_rate"] == 1.0
    assert metrics["contract_recall"] == 1.0


def test_measured_metrics_report_defined_counts_and_fallbacks(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    metrics = evaluate(config, model=_EchoGoldModel(), publish_agentv=False)
    assert metrics["parse_rate"] == 1.0
    assert metrics["contract_precision"] == 1.0
    assert metrics["metric_defined_n"]["contract_precision"] == 1
    # generate_batch_requests collects decode stats → measured zero fallbacks.
    assert metrics["fallback_count"] == 0
    assert metrics["empty_prediction_count"] == 0
    assert not (config.run_dir / "decode_progress.json").exists()
    # Default policy is grammar-constrained → syntax metrics are labeled as
    # decoder-guaranteed; the slot contract is not injected by default.
    assert "parse_rate" in metrics["decoder_guaranteed"]
    assert "contract_precision" not in metrics["decoder_guaranteed"]


def test_decode_timeout_does_not_degrade_quality_rates(tmp_path: Path) -> None:
    """Timeouts are incompleteness metrics, never false parse/quality failures."""
    from slm_training.models.decode_stats import DecodeStats

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    records = [
        _record(id="ok-1", split="smoke", meta={"suite": "smoke"}),
        _record(id="timeout-1", split="smoke", meta={"suite": "smoke"}),
        _record(id="ok-2", split="smoke", meta={"suite": "smoke"}),
    ]
    write_jsonl(train_dir / "records.jsonl", [_record(id="train-1", split="train")])
    write_jsonl(test_dir / "suites" / "smoke" / "records.jsonl", records)

    class MixedTimeoutModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate_with_stats(self, prompt: str) -> tuple[str, DecodeStats]:
            self.calls += 1
            if self.calls == 2:
                error = TimeoutError("decode exceeded")
                error.decode_stats = DecodeStats(tokens_emitted=3)  # type: ignore[attr-defined]
                raise error
            return _GOLD, DecodeStats(tokens_emitted=1)

    def _config(run_id: str) -> ModelBuildConfig:
        return ModelBuildConfig(
            train_dir=train_dir,
            test_dir=test_dir,
            suite="smoke",
            run_root=tmp_path / "runs",
            run_id=run_id,
            model_name="stub",
            decode_timeout_seconds=1,
        )

    class AlwaysGold:
        def generate_with_stats(self, prompt: str) -> tuple[str, DecodeStats]:
            return _GOLD, DecodeStats(tokens_emitted=1)

    metrics = evaluate(
        _config("timeout-metric-semantics"),
        model=MixedTimeoutModel(),
        publish_agentv=False,
    )
    # Baseline: same gold echo with no timeouts — quality must match mixed.
    baseline = evaluate(
        _config("timeout-metric-baseline"),
        model=AlwaysGold(),
        publish_agentv=False,
    )

    assert metrics["document_n"] == 3
    assert metrics["completed_document_n"] == 2
    assert metrics["incomplete_document_n"] == 1
    assert metrics["decode_timeout_count"] == 1
    assert metrics["decode_timeout_document_count"] == 1
    assert metrics["decode_timeout_rate"] == pytest.approx(1 / 3)
    # Every quality metric is over completed rows only — match no-timeout gold.
    quality_keys = (
        "parse_rate",
        "meaningful_program_rate",
        "syntax_parse_rate",
        "raw_syntax_validity",
        "contract_precision",
        "contract_recall",
        "binder_reference_f1",
        "placeholder_fidelity",
        "placeholder_fidelity_normalized",
        "placeholder_validity",
        "exact_match",
        "ast_beq_rate",
        "canonical_beq_rate",
        "structural_similarity",
        "tree_edit_similarity",
        "component_type_recall",
        "reward_score",
    )
    for name in quality_keys:
        assert metrics[name] is not None, name
        assert metrics[name] == pytest.approx(baseline[name]), name
    for name in quality_keys:
        defined = metrics["metric_defined_n"].get(name)
        if defined is None:
            # rate-style keys share defined_n under related names
            continue
        assert defined == 2, name
    assert metrics["metric_defined_n"]["placeholder_fidelity"] == 2
    assert metrics["metric_defined_n"]["structural_similarity"] == 2
    assert metrics["metric_defined_n"]["reward_score"] == 2
    assert metrics["empty_prediction_count"] == 0
    assert metrics["completed_latency_n"] == 2
    assert metrics["incomplete_latency_n"] == 1
    assert metrics["latency_ms_p50"] is not None
    assert metrics["decode_outcome_counts"]["runtime_timeout"] == 1
    assert metrics["decode_outcome_counts"]["model_valid"] == 2
    timeout_rows = [row for row in metrics["details"] if row.get("incomplete")]
    assert len(timeout_rows) == 1
    assert timeout_rows[0]["parse_ok"] is None
    assert timeout_rows[0]["structural_similarity"] is None
    assert timeout_rows[0]["reward_score"] is None
    assert timeout_rows[0]["decode_outcome"] == "runtime_timeout"
    assert metrics["rate_evidence"]["decode_timeout_rate"]["denominator"] == 3
    assert metrics["rate_evidence"]["parse_rate"]["denominator"] == 2


def test_all_timeouts_leave_quality_rates_unmeasured(tmp_path: Path) -> None:
    from slm_training.models.decode_stats import DecodeStats

    config = _smoke_config(tmp_path)
    config = ModelBuildConfig(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        suite="smoke",
        run_root=config.run_root,
        run_id="all-timeout-metric-semantics",
        model_name="stub",
        decode_timeout_seconds=1,
    )

    class AlwaysTimeout:
        def generate_with_stats(self, prompt: str) -> tuple[str, DecodeStats]:
            error = TimeoutError("decode exceeded")
            error.decode_stats = DecodeStats(tokens_emitted=1)  # type: ignore[attr-defined]
            raise error

    metrics = evaluate(config, model=AlwaysTimeout(), publish_agentv=False)
    assert metrics["decode_timeout_rate"] == 1.0
    assert metrics["completed_document_n"] == 0
    assert metrics["incomplete_latency_n"] == 1
    assert metrics["completed_latency_n"] == 0
    # All quality fields unmeasured — not fabricated zeros.
    for name in (
        "parse_rate",
        "meaningful_program_rate",
        "syntax_parse_rate",
        "raw_syntax_validity",
        "contract_precision",
        "contract_recall",
        "placeholder_fidelity",
        "placeholder_validity",
        "exact_match",
        "ast_beq_rate",
        "canonical_beq_rate",
        "structural_similarity",
        "tree_edit_similarity",
        "component_type_recall",
        "reward_score",
        "latency_ms_p50",
        "latency_ms_p95",
    ):
        assert metrics[name] is None, name
    assert metrics["empty_prediction_count"] == 0
    # Runtime completeness metrics still report the wall.
    assert metrics["latency_ms_p50_including_incomplete"] is not None
    assert metrics["decode_timeout_count"] == 1


def test_supervisor_interrupt_persists_non_scoreable_decode_progress(
    tmp_path: Path,
) -> None:
    from slm_training.models.decode_stats import get_active_stats

    config = _smoke_config(tmp_path)
    config.decode_timeout_seconds = 1

    class InterruptedBatchModel:
        def generate_batch_requests(self, requests, **_kwargs):
            assert requests
            stats = get_active_stats()
            assert stats is not None
            stats.completion_witness_states_expanded = 11
            stats.completion_edges_built = 13
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate(config, model=InterruptedBatchModel(), publish_agentv=False)

    progress = json.loads((config.run_dir / "decode_progress.json").read_text())
    assert progress["schema_version"] == "DecodeProgressV1"
    assert progress["status"] == "interrupted"
    assert progress["measurement_complete"] is False
    assert progress["scoreable"] is False
    assert progress["processed_record_n"] == 0
    assert progress["active_record_ids"]
    assert progress["decode_stats"]["completion_witness_states_expanded_sum"] == 11
    assert progress["decode_stats"]["completion_edges_built_sum"] == 13
    assert progress["version_stamp"]["stamp_schema"] == "version_stamp/v1"


def test_decode_initialization_runs_before_document_generation(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)

    class PreparedModel:
        def __init__(self) -> None:
            self.prepared = False

        def prepare_generation(self) -> None:
            self.prepared = True

        def generate_with_stats(
            self, prompt: str
        ) -> tuple[str, eval_runner.DecodeStats]:
            assert prompt
            assert self.prepared
            return _GOLD, eval_runner.DecodeStats(tokens_emitted=1)

    metrics = evaluate(config, model=PreparedModel(), publish_agentv=False)

    assert metrics["decode_initialization_ms"] >= 0.0


def test_repeating_decode_alarm_is_blocked_while_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    previous = object()

    monkeypatch.setattr(
        eval_runner.signal,
        "pthread_sigmask",
        lambda how, mask: events.append(("mask", how)) or {"old"},
    )
    monkeypatch.setattr(
        eval_runner.signal,
        "signal",
        lambda _sig, handler: events.append(("handler", handler)),
    )
    monkeypatch.setattr(
        eval_runner.signal,
        "setitimer",
        lambda _which, seconds, *_args: events.append(("timer", seconds)),
    )

    eval_runner._clear_repeating_decode_alarm(previous)

    assert events[0] == ("mask", eval_runner.signal.SIG_BLOCK)
    assert events[1] == ("handler", eval_runner.signal.SIG_IGN)
    assert events[2] == ("timer", 0)
    assert events[3] == ("handler", previous)
    assert events[4] == ("mask", eval_runner.signal.SIG_SETMASK)


def test_progress_persistence_failure_does_not_mask_supervisor_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _smoke_config(tmp_path)

    class InterruptedBatchModel:
        def generate_batch_requests(self, requests, **_kwargs):
            assert requests
            raise KeyboardInterrupt

    def fail_persistence(*_args, **_kwargs) -> None:
        raise TypeError("not JSON serializable")

    monkeypatch.setattr(eval_runner, "_persist_decode_progress", fail_persistence)
    with pytest.raises(KeyboardInterrupt):
        evaluate(config, model=InterruptedBatchModel(), publish_agentv=False)

    assert "decode progress persistence failed" in capsys.readouterr().err


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
    # Raw metrics, not a precomputed pass boolean, reach AgentEvals.
    assert "pass" not in cases[0]
    assert all(
        "missing_suite" not in criterion["id"] for criterion in cases[0]["assertions"]
    )
