"""Resumable per-record eval chunks: persist, resume without re-decoding, merge.

The fixture suite has 24 records and a stub decoder that costs 30 mocked
seconds per record, i.e. the promotion geometry from ``policy.v3.json``
(``promotion_suite_n: 24``) against the 155 s harness wall.  Every test drives
``time`` through a fake clock: no real decode and no real waiting.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build import eval_runner
from slm_training.harnesses.model_build.eval_runner import (
    _suite_result_cacheable,
    evaluate,
    evaluate_suites,
    load_partial_scoreboard,
    partial_scoreboard_path,
)
from slm_training.models.decode_stats import DecodeStats

_GOLD = 'root = Stack([cta])\ncta = Button(":slot_0")'
SUITE_N = 24
DECODE_SECONDS = 30.0
HARNESS_WALL_SECONDS = 155.0


class FakeClock:
    """Monotonic + perf_counter stand-in advanced explicitly by the stub decoder."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class SlowStubDecoder:
    """30 mocked seconds per record; optional interrupt on the n-th decode."""

    def __init__(self, clock: FakeClock, *, interrupt_on_call: int | None = None) -> None:
        self.clock = clock
        self.calls = 0
        self.decoded_ids: list[str] = []
        self.interrupt_on_call = interrupt_on_call

    def generate_with_stats(self, prompt: str, **_kwargs: object) -> tuple[str, DecodeStats]:
        self.calls += 1
        if self.interrupt_on_call is not None and self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        self.clock.advance(DECODE_SECONDS)
        self.decoded_ids.append(prompt)
        return _GOLD, DecodeStats(tokens_emitted=7)


def _record(**overrides: object) -> ExampleRecord:
    data = {
        "id": "r1",
        "prompt": "CTA",
        "openui": _GOLD,
        "placeholders": [":slot_0"],
        **overrides,
    }
    return ExampleRecord.from_dict(data)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(time, "monotonic", fake)
    monkeypatch.setattr(time, "perf_counter", fake)
    return fake


@pytest.fixture
def suite(tmp_path: Path) -> dict[str, Path]:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites" / "smoke").mkdir(parents=True)
    (test_dir / "suites" / "held_out").mkdir(parents=True)
    write_jsonl(train_dir / "records.jsonl", [_record(id="train-1", split="train")])
    write_jsonl(
        test_dir / "suites" / "smoke" / "records.jsonl",
        [
            _record(id=f"smoke-{index:02d}", prompt=f"smoke-{index:02d}", split="smoke")
            for index in range(SUITE_N)
        ],
    )
    write_jsonl(
        test_dir / "suites" / "held_out" / "records.jsonl",
        [
            _record(id=f"held-{index:02d}", prompt=f"held-{index:02d}", split="held_out")
            for index in range(5)
        ],
    )
    checkpoint = tmp_path / "locked.pt"
    checkpoint.write_bytes(b"locked-checkpoint-bytes")
    return {"train": train_dir, "test": test_dir, "checkpoint": checkpoint}


def _config(suite: dict[str, Path], tmp_path: Path, run_id: str, **extra: object) -> ModelBuildConfig:
    kwargs: dict[str, object] = {"decode_timeout_seconds": 40, **extra}
    return ModelBuildConfig(
        train_dir=suite["train"],
        test_dir=suite["test"],
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id=run_id,
        model_name="stub",
        **kwargs,  # type: ignore[arg-type]
    )


def _identity(suite: dict[str, Path]) -> dict[str, object]:
    return {
        "model_checkpoint_sha256": hashlib.sha256(
            suite["checkpoint"].read_bytes()
        ).hexdigest(),
        "model_checkpoint_path": suite["checkpoint"],
    }


def test_24_records_complete_in_five_chunks_with_one_merged_scoreboard(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    """records_per_run = floor(155 / 30) = 5 -> 5 chunk runs, each <= 155 s."""
    records_per_run = int(HARNESS_WALL_SECONDS // DECODE_SECONDS)
    assert records_per_run == 5
    config = _config(suite, tmp_path, "promotion-arm")
    decoder = SlowStubDecoder(clock)
    metrics: dict[str, object] = {}
    runs = 0
    while True:
        started = clock.now
        metrics = evaluate(
            config,
            model=decoder,
            publish_agentv=False,
            partial_scoreboard=True,
            max_records_this_run=records_per_run,
            **_identity(suite),
        )
        runs += 1
        assert clock.now - started <= HARNESS_WALL_SECONDS
        if metrics["measurement_complete"]:
            break
        assert metrics["resume"]["stop_reason"] == "record_budget"
        assert not _suite_result_cacheable(metrics)
        assert runs < 10
    assert runs == 5
    assert decoder.calls == SUITE_N  # every record decoded exactly once
    assert metrics["completed_document_n"] == SUITE_N
    assert metrics["incomplete_document_n"] == 0
    assert metrics["decode_timeout_count"] == 0
    assert metrics["resume"]["replayed_record_n"] == 20
    assert metrics["resume"]["decoded_this_run_n"] == 4
    assert metrics["structural_similarity"] == pytest.approx(1.0)
    assert metrics["parse_rate"] == pytest.approx(1.0)
    assert metrics["latency_ms_p95"] == pytest.approx(DECODE_SECONDS * 1000.0)
    assert len(metrics["details"]) == SUITE_N
    assert _suite_result_cacheable(metrics)
    partial = load_partial_scoreboard(config.run_dir, "smoke")
    assert partial is not None
    assert partial["measurement_complete"] is True
    assert len(partial["records"]) == SUITE_N
    assert partial["pending_record_ids"] == []
    entry = partial["records"]["smoke-03"]
    assert entry["timed_out"] is False
    assert entry["decode_ms"] == pytest.approx(DECODE_SECONDS * 1000.0)
    assert entry["metrics"]["structural_similarity"] == pytest.approx(1.0)
    assert entry["decode_stats"]["tokens_emitted"] == 7
    written = json.loads((config.run_dir / "eval_smoke.json").read_text())
    assert written["completed_document_n"] == SUITE_N
    assert written["measurement_complete"] is True


def test_merged_scoreboard_matches_one_shot_evaluation(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    one_shot = evaluate(
        _config(suite, tmp_path, "one-shot"),
        model=SlowStubDecoder(clock),
        publish_agentv=False,
        **_identity(suite),
    )
    chunked_config = _config(suite, tmp_path, "chunked")
    decoder = SlowStubDecoder(clock)
    chunked = evaluate(
        chunked_config,
        model=decoder,
        publish_agentv=False,
        max_records_this_run=5,
        **_identity(suite),
    )
    while not chunked["measurement_complete"]:
        chunked = evaluate(
            chunked_config,
            model=decoder,
            publish_agentv=False,
            max_records_this_run=5,
            **_identity(suite),
        )
    for key in (
        "n",
        "document_n",
        "completed_document_n",
        "incomplete_document_n",
        "parse_rate",
        "meaningful_program_rate",
        "structural_similarity",
        "placeholder_fidelity",
        "reward_score",
        "decode_timeout_count",
        "latency_ms_p50",
        "latency_ms_p95",
        "decode_outcome_counts",
    ):
        assert chunked[key] == one_shot[key], key
    assert [row["id"] for row in chunked["details"]] == [
        row["id"] for row in one_shot["details"]
    ]


def test_run_killed_mid_chunk_resumes_without_re_decoding(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(suite, tmp_path, "killed")
    first = SlowStubDecoder(clock, interrupt_on_call=4)
    with pytest.raises(KeyboardInterrupt):
        evaluate(
            config,
            model=first,
            publish_agentv=False,
            partial_scoreboard=True,
            max_records_this_run=5,
            **_identity(suite),
        )
    partial = load_partial_scoreboard(config.run_dir, "smoke")
    assert partial is not None
    assert sorted(partial["records"]) == ["smoke-00", "smoke-01", "smoke-02"]
    assert partial["measurement_complete"] is False
    assert not (config.run_dir / "eval_smoke.json").exists()

    second = SlowStubDecoder(clock)
    metrics = evaluate(
        config,
        model=second,
        publish_agentv=False,
        partial_scoreboard=True,
        max_records_this_run=5,
        **_identity(suite),
    )
    assert second.decoded_ids == [f"smoke-{index:02d}" for index in range(3, 8)]
    assert metrics["resume"]["replayed_record_n"] == 3
    assert metrics["resume"]["decoded_this_run_n"] == 5
    assert metrics["resume"]["pending_record_n"] == SUITE_N - 8
    assert metrics["completed_document_n"] == 8
    assert metrics["incomplete_document_n"] == SUITE_N - 8
    assert metrics["measurement_complete"] is False
    replayed = [row for row in metrics["details"] if row.get("resumed_from_partial")]
    assert [row["id"] for row in replayed] == ["smoke-00", "smoke-01", "smoke-02"]
    assert all(row["structural_similarity"] == pytest.approx(1.0) for row in replayed)


def test_resume_from_another_run_dir_and_stop_at_evaluation_wall(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    """A later run may resume a partial written by an earlier run directory."""
    # Timeout == p95 here: the wall stop reserves one full timeout per record.
    first_config = _config(suite, tmp_path, "chunk-1", decode_timeout_seconds=30)
    decoder = SlowStubDecoder(clock)
    first = evaluate(
        first_config,
        model=decoder,
        publish_agentv=False,
        partial_scoreboard=True,
        evaluation_deadline=clock.now + HARNESS_WALL_SECONDS,
        **_identity(suite),
    )
    assert first["resume"]["stop_reason"] == "evaluation_wall"
    assert first["resume"]["decoded_this_run_n"] == 5
    assert first["effective_decode_timeout_seconds_min"] == pytest.approx(30.0)
    second_config = _config(suite, tmp_path, "chunk-2", decode_timeout_seconds=30)
    second = evaluate(
        second_config,
        model=decoder,
        publish_agentv=False,
        resume_from=first_config.run_dir,
        evaluation_deadline=clock.now + HARNESS_WALL_SECONDS,
        **_identity(suite),
    )
    assert second["resume"]["replayed_record_n"] == 5
    assert second["resume"]["decoded_this_run_n"] == 5
    assert second["completed_document_n"] == 10
    assert decoder.calls == 10
    assert partial_scoreboard_path(second_config.run_dir, "smoke").is_file()


def test_partial_identity_mismatch_restarts_without_reusing_evidence(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(suite, tmp_path, "identity")
    evaluate(
        config,
        model=SlowStubDecoder(clock),
        publish_agentv=False,
        max_records_this_run=3,
        **_identity(suite),
    )
    other = tmp_path / "other.pt"
    other.write_bytes(b"different-checkpoint")
    decoder = SlowStubDecoder(clock)
    metrics = evaluate(
        config,
        model=decoder,
        publish_agentv=False,
        max_records_this_run=3,
        model_checkpoint_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
        model_checkpoint_path=other,
    )
    assert metrics["resume"]["resume_rejected"] == "checkpoint_sha256_mismatch"
    assert metrics["resume"]["replayed_record_n"] == 0
    assert decoder.decoded_ids == ["smoke-00", "smoke-01", "smoke-02"]


def test_exhausted_chunk_budget_reports_measurement_incomplete(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    """Three runs of five records cannot cover 24: the merge stays incomplete."""
    config = _config(suite, tmp_path, "exhausted")
    decoder = SlowStubDecoder(clock)
    metrics: dict[str, object] = {}
    for _ in range(3):
        metrics = evaluate(
            config,
            model=decoder,
            publish_agentv=False,
            max_records_this_run=5,
            **_identity(suite),
        )
    assert decoder.calls == 15
    assert metrics["measurement_complete"] is False
    assert metrics["completed_document_n"] == 15
    assert metrics["pending_document_n"] == 9
    assert metrics["incomplete_document_n"] == 9
    assert metrics["completed_document_n"] + metrics["incomplete_document_n"] == (
        metrics["document_n"]
    )
    assert metrics["decode_timeout_count"] == 0
    assert not _suite_result_cacheable(metrics)
    assert metrics["structural_similarity"] == pytest.approx(1.0)
    assert metrics["metric_defined_n"]["structural_similarity"] == 15


def test_timeouts_persist_as_incomplete_and_replay_as_timeouts(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(suite, tmp_path, "timeouts")

    class TimeoutOnSecond(SlowStubDecoder):
        def generate_with_stats(self, prompt: str, **kwargs: object):
            if prompt == "smoke-01":
                self.calls += 1
                error = TimeoutError("decode exceeded")
                error.decode_stats = DecodeStats(tokens_emitted=2)  # type: ignore[attr-defined]
                raise error
            return super().generate_with_stats(prompt, **kwargs)

    first = evaluate(
        config,
        model=TimeoutOnSecond(clock),
        publish_agentv=False,
        max_records_this_run=3,
        **_identity(suite),
    )
    assert first["decode_timeout_count"] == 1
    partial = load_partial_scoreboard(config.run_dir, "smoke")
    assert partial is not None
    assert partial["records"]["smoke-01"]["timed_out"] is True
    second = evaluate(
        config,
        model=SlowStubDecoder(clock),
        publish_agentv=False,
        max_records_this_run=SUITE_N,
        **_identity(suite),
    )
    assert second["measurement_complete"] is True
    assert second["decode_timeout_count"] == 1
    assert second["decode_timeout_document_count"] == 1
    assert second["completed_document_n"] == SUITE_N - 1
    assert second["incomplete_document_n"] == 1
    assert not _suite_result_cacheable(second)


def test_evaluate_suites_shares_budget_and_defers_gates_until_complete(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = SlowStubDecoder(clock)
    monkeypatch.setattr(eval_runner, "build_model", lambda *_a, **_k: decoder)
    monkeypatch.setattr(
        eval_runner, "_evaluation_version_components", lambda _config: ()
    )
    config = _config(suite, tmp_path, "suites", eval_limit=4)
    first = evaluate_suites(
        config,
        ["smoke", "held_out"],
        checkpoint=suite["checkpoint"],
        write_gates=True,
        max_records_this_run=6,
    )
    assert first["measurement_complete"] is False
    assert first["resume"]["decoded_this_run_n"] == {"smoke": 4, "held_out": 2}
    assert first["resume"]["pending_record_n"] == {"smoke": 0, "held_out": 2}
    assert "gates" not in first
    assert first["evals"]["skipped"].startswith("measurement incomplete")
    assert json.loads((config.run_dir / "scoreboard.json").read_text())[
        "measurement_complete"
    ] is False
    second = evaluate_suites(
        replace(config, run_id="suites-2"),
        ["smoke", "held_out"],
        checkpoint=suite["checkpoint"],
        write_gates=True,
        resume_from=config.run_dir,
        max_records_this_run=6,
    )
    assert second["measurement_complete"] is True
    assert second["resume"]["decoded_this_run_n"] == {"smoke": 0, "held_out": 2}
    assert decoder.calls == 8
    assert second["suites"]["held_out"]["completed_document_n"] == 4
    assert "gates" in second


class BatchedStubDecoder:
    """Batched API (``generate_batch``) with a large baked batch size."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.batches: list[list[str]] = []

    def generate_batch(self, prompts: list[str], **_kwargs: object) -> list[str]:
        self.batches.append(list(prompts))
        self.clock.advance(DECODE_SECONDS * len(prompts))
        return [_GOLD for _ in prompts]


def test_batched_chunk_is_capped_by_record_budget_not_refused(
    clock: FakeClock, suite: dict[str, Path], tmp_path: Path
) -> None:
    """A 16-record baked batch decodes the two budgeted records, never zero."""
    config = _config(suite, tmp_path, "batched", generate_batch_size=16, eval_limit=3)
    decoder = BatchedStubDecoder(clock)
    first = evaluate(
        config,
        model=decoder,
        publish_agentv=False,
        partial_scoreboard=True,
        max_records_this_run=2,
        **_identity(suite),
    )
    assert decoder.batches == [["smoke-00", "smoke-01"]]
    assert first["resume"]["decoded_this_run_n"] == 2
    assert first["resume"]["pending_record_n"] == 1
    assert first["resume"]["stop_reason"] == "record_budget"
    assert first["measurement_complete"] is False
    second = evaluate(
        config,
        model=decoder,
        publish_agentv=False,
        partial_scoreboard=True,
        max_records_this_run=2,
        **_identity(suite),
    )
    assert decoder.batches == [["smoke-00", "smoke-01"], ["smoke-02"]]
    assert second["resume"]["replayed_record_n"] == 2
    assert second["measurement_complete"] is True
    assert second["completed_document_n"] == 3
    # Wall fit caps a batch the same way: 155 s at a 30 s timeout fits 5.
    walled = _config(suite, tmp_path, "batched-wall", generate_batch_size=16, decode_timeout_seconds=30)
    wall_decoder = BatchedStubDecoder(clock)
    metrics = evaluate(
        walled,
        model=wall_decoder,
        publish_agentv=False,
        partial_scoreboard=True,
        evaluation_deadline=clock.now + HARNESS_WALL_SECONDS,
        **_identity(suite),
    )
    assert [len(batch) for batch in wall_decoder.batches] == [5]
    assert metrics["resume"]["stop_reason"] == "evaluation_wall"
    assert metrics["resume"]["decoded_this_run_n"] == 5
