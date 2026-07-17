"""G5 (SLM-37): meta-model trace capture — schema, replay, bucket layout."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.distill.trace_store import (
    DecodeTraceRecorder,
    TraceStore,
    record_harness_decision,
    record_matrix_outcome,
    replay_violations,
    sync_traces,
    trace_bucket_uri,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    "hero = Card([hero_title])"
)


def _decode_trace() -> dict:
    model = TwoTowerModel.from_records(
        [
            ExampleRecord(
                id="a",
                prompt="Hero",
                openui=HERO,
                placeholders=[":hero.title"],
            )
        ],
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            gen_steps=4,
            seed=0,
        ),
        device="cpu",
    )
    recorder = DecodeTraceRecorder()
    model.trace_recorder = recorder
    model.generate("Hero", grammar_constrained=False)
    model.trace_recorder = None
    return recorder.finalize(labels={"fixture": True})


def test_fixture_decode_trace_is_replayable(tmp_path: Path) -> None:
    trace = _decode_trace()
    assert trace["steps"], "decode produced no recorded steps"
    assert replay_violations(trace) == []
    # A corrupted canvas is caught — the invariant is not vacuous.
    corrupted = {**trace, "steps": [dict(s) for s in trace["steps"]]}
    for step in corrupted["steps"]:
        if step.get("commits") and step.get("canvas"):
            t = int(step["commits"][0]["t"])
            canvas = list(step["canvas"])
            canvas[t] = canvas[t] + 1
            step["canvas"] = canvas
            break
    assert replay_violations(corrupted)


def test_meta_trace_kinds_share_one_store(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces", run_id="g5-fixture")
    store.append(_decode_trace())
    record_harness_decision(
        store,
        harness="quality_matrix",
        decision="select_experiment",
        inputs={"matrix_set": "v12", "candidates": ["E259"]},
        outcome={"selected": "E259"},
    )
    record_matrix_outcome(
        store,
        {
            "id": "E259",
            "run_id": "qx_e277_a2_asap_decode",
            "pass": False,
            "failures": ["smoke:meaningful_program_rate actual=0.0 need>=0.66"],
            "suites": {"smoke": {"n": 3, "parse_rate": 0.0}},
        },
        matrix_set="v12",
    )
    assert len(store) == 3
    decode_rows = list(store.iter_kind("decode"))
    decisions = list(store.iter_kind("harness_decision"))
    outcomes = list(store.iter_kind("matrix_outcome"))
    assert len(decode_rows) == 1 and decode_rows[0]["steps"]
    assert decisions[0]["harness"] == "quality_matrix"
    assert outcomes[0]["experiment_id"] == "E259"
    assert outcomes[0]["passed"] is False
    # Identity envelope present on every row.
    for row in store.iter_traces():
        assert row["run_id"] == "g5-fixture"
        assert row["trajectory_id"]
    with pytest.raises(ValueError, match="experiment id"):
        record_matrix_outcome(store, {"pass": True}, matrix_set="v12")


def test_trace_bucket_layout_matches_existing_persistence(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces", run_id="g5-fixture")
    record_harness_decision(store, harness="fixture", decision="noop")
    plan = sync_traces(tmp_path / "traces", "g5-fixture", push=False)
    assert plan["push"] is False
    assert plan["remote_uri"] == trace_bucket_uri("g5-fixture")
    assert plan["remote_uri"].startswith("hf://buckets/TKendrick/OpenUI/traces/")
    assert plan["command"][:3] == ["hf", "buckets", "sync"]
    assert "--no-delete" in plan["command"]
    with pytest.raises(FileNotFoundError):
        sync_traces(tmp_path / "missing", "nope")
