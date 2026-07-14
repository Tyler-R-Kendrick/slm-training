"""P2 distill selection / repair / SFT smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.distill.repair import extract_failure_cone, repair_records_from_traces
from slm_training.harnesses.distill.select import SelectConfig, corpus_label, select_traces
from slm_training.harnesses.distill.sft import DistillSFTConfig, traces_to_records, train_self_distill
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _trace(
    *,
    tid: str,
    prompt: str,
    text: str,
    accepted: bool = True,
    exact_gold: bool = False,
    sha: str = "abc",
) -> dict:
    return {
        "trace_id": tid,
        "meta": {"prompt": prompt, "record_id": tid, "policy_checkpoint_sha": sha},
        "final": {"text": text, "canvas": [1, 2, 3]},
        "labels": {"accepted": accepted, "exact_gold": exact_gold},
        "reward": {"grammar": 1.0 if accepted else 0.0, "composite": 0.5},
        "steps": [
            {
                "step": 0,
                "canvas": [1, 0, 3],
                "unknown_positions": [1],
                "commits": [{"t": 1, "id": 2, "lp": -0.5}],
                "remasks": [{"positions": [1], "reason": "grammar_stream"}],
            }
        ],
    }


def test_select_prefers_coverage_and_filters_gold() -> None:
    traces = [
        _trace(tid="a1", prompt="hero card", text="root = Card([])", exact_gold=True),
        _trace(tid="a2", prompt="hero card", text="root = Card([x])"),
        _trace(tid="b1", prompt="cta button", text="root = Button(:x)"),
        _trace(tid="c1", prompt="form", text="root = Stack([])", accepted=False),
    ]
    selected = select_traces(
        traces, config=SelectConfig(budget=10, per_stratum=1, seed=0)
    )
    ids = {t["trace_id"] for t in selected}
    assert "a1" not in ids  # exact gold excluded
    assert "c1" not in ids  # not accepted
    assert "a2" in ids and "b1" in ids
    assert corpus_label(traces[1]) == "self_distilled_success"


def test_failure_cone_and_repair_records() -> None:
    trace = _trace(tid="r1", prompt="hero", text="root = Card([])")
    cones = extract_failure_cone(trace)
    assert cones and cones[0]["cone_positions"]
    records = repair_records_from_traces([trace])
    assert records and records[0].meta["source_family"] == "self_distilled_repair"


def test_self_distill_sft_smoke(tmp_path: Path) -> None:
    openui = 'root = Stack([cta])\ncta = Button(":cta.label")'
    records = [
        ExampleRecord(
            id="t1",
            prompt="CTA",
            openui=openui,
            split="train",
            placeholders=[":cta.label"],
        )
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=0
        ),
        device="cpu",
    )
    traces = [
        _trace(tid="d1", prompt="CTA", text=openui),
        _trace(tid="d2", prompt="CTA", text=openui),
    ]
    # Provide a commit against a real canvas length from the model tokenizer.
    ids = model.tokenizer.encode(openui)
    for tr in traces:
        tr["steps"] = [
            {
                "step": 0,
                "canvas": list(ids),
                "commits": [{"t": min(2, len(ids) - 1), "id": ids[min(2, len(ids) - 1)], "lp": -0.1}],
                "remasks": [],
            }
        ]
    assert traces_to_records(traces)
    summary = train_self_distill(
        model,
        traces,
        anchor_records=records,
        config=DistillSFTConfig(steps=2, batch_size=1, lambda_traj=0.5, lambda_anchor=0.2),
        out_dir=tmp_path / "distill",
    )
    assert summary["n_traces"] == 2
    assert (tmp_path / "distill" / "model.pt").exists()
