"""Decode trajectory recorder + append-only trace store tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.distill.trace_store import (
    DecodeTraceRecorder,
    TraceStore,
    decode_config_hash,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([b3], "column")\n'
    'b1 = TextContent(":slot_0")\n'
    'b2 = TextContent(":slot_1")\n'
    "b3 = Card([b1, b2])"
)


def _model() -> TwoTowerModel:
    records = [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            split="train",
            placeholders=[":slot_0", ":slot_1"],
        )
    ]
    return TwoTowerModel.from_records(
        records,
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


def test_recorder_captures_maskgit_trajectory() -> None:
    model = _model()
    recorder = DecodeTraceRecorder()
    model.trace_recorder = recorder
    # Unconstrained decode: one pass, no grammar retry — final state is exact.
    text = model.generate("Hero", gold=None, grammar_constrained=False)
    model.trace_recorder = None

    assert recorder.nfe > 0
    assert recorder.commit_count > 0
    assert recorder.steps
    assert recorder.final is not None
    assert recorder.final["text"] == text
    # The final canvas decodes to the returned text (trajectory is faithful).
    assert model._decode_openui(recorder.final["canvas"]).strip() == text.strip()

    # Every commit is reflected in its step's canvas (unless later padded/EOS-cut).
    pad_id = model.tokenizer.pad_id
    mask_id = model.tokenizer.mask_id
    for step_row in recorder.steps:
        canvas = step_row.get("canvas")
        if canvas is None:
            continue
        for commit in step_row.get("commits") or []:
            value = canvas[commit["t"]]
            assert value in {commit["id"], pad_id, mask_id}
            assert commit["lp"] <= 0.0


def test_recorder_zero_cost_when_absent() -> None:
    model = _model()
    assert model.trace_recorder is None
    text = model.generate("Hero", gold=None, grammar_constrained=False)
    assert isinstance(text, str)


def test_recorder_support_captures_exact_precommit_state() -> None:
    model = _model()
    model.config.grammar_ltr_primary = False
    recorder = DecodeTraceRecorder(record_support=True)
    model.trace_recorder = recorder
    model.generate("Hero", gold=None, grammar_constrained=True)
    model.trace_recorder = None

    commits = [
        commit
        for step in recorder.steps
        for commit in step.get("commits", [])
        if commit.get("phase") == "maskgit"
    ]
    assert commits
    assert all("pre_canvas" in commit and "raw_id" in commit for commit in commits)
    assert all(commit["pre_canvas"][commit["t"]] == model.tokenizer.mask_id for commit in commits)


def test_trace_store_append_only(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces")
    recorder = DecodeTraceRecorder()
    recorder.begin(length=4, use_grammar=False)
    recorder.step(0, canvas=[1, 2, 3, 4], unknown=[False] * 4, commits=[])
    recorder.end(canvas=[1, 2, 3, 4], text="x")
    trace = recorder.finalize(final_text="x", reward={"grammar": 0.0}, labels={})

    tid1 = store.append(trace)
    tid2 = store.append(trace)
    assert tid1 != tid2
    assert len(store) == 2
    rows = list(store.iter_traces())
    assert [r["trajectory_id"] for r in rows] == [tid1, tid2]

    # Re-opening the store keeps existing traces (append-only contract).
    reopened = TraceStore(tmp_path / "traces")
    tid3 = reopened.append(trace)
    assert len(reopened) == 3
    assert tid3.startswith("00000002-")
    manifest = json.loads(
        (tmp_path / "traces" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["append_only"] is True
    assert manifest["count"] == 3


def test_decode_config_hash_tracks_decode_knobs() -> None:
    model = _model()
    base = decode_config_hash(model.config)
    assert decode_config_hash(model.config) == base
    model.config.gen_steps = 16
    assert decode_config_hash(model.config) != base
    # Training-only knobs do not change decode identity.
    model.config.gen_steps = 4
    model.config.ltr_loss_weight = 0.9
    assert decode_config_hash(model.config) == base
