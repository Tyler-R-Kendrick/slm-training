"""P3 E64 trajectory RL tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.harnesses.rl.trajectory import (
    TrajectoryRLConfig,
    importance_weighted_loss,
    lexicographic_reward,
    train_trajectory_rl,
)


def test_lexicographic_reward_gates_invalid() -> None:
    assert lexicographic_reward({"grammar": 1.0}, {"accepted": False}) == 0.0
    assert lexicographic_reward({"grammar": 0.0}, {"accepted": True}) == 0.0
    good = lexicographic_reward(
        {"grammar": 1.0, "placeholder": 0.5, "layout": 0.5, "composite": 0.4},
        {"accepted": True},
    )
    assert good > 1_000_000


def test_trajectory_rl_smoke(tmp_path: Path) -> None:
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
    ids = model.tokenizer.encode(openui)
    commit = {"t": min(2, len(ids) - 1), "id": ids[min(2, len(ids) - 1)], "lp": -0.2}

    def make_trace(sample: int, accepted: bool, reward: float) -> dict:
        return {
            "meta": {
                "prompt": "CTA",
                "record_id": "t1",
                "policy_checkpoint_sha": "sha1",
            },
            "labels": {"accepted": accepted},
            "reward": {
                "grammar": 1.0 if accepted else 0.0,
                "placeholder": reward,
                "layout": reward,
                "composite": reward,
            },
            "final": {"text": openui, "canvas": ids},
            "steps": [{"step": 0, "canvas": list(ids), "commits": [commit], "remasks": []}],
        }

    traces = [
        make_trace(0, True, 0.9),
        make_trace(1, True, 0.2),
        make_trace(2, False, 0.0),
        make_trace(3, True, 0.5),
    ]
    loss = importance_weighted_loss(model, traces[:2])
    assert loss is not None
    summary = train_trajectory_rl(
        model,
        traces,
        config=TrajectoryRLConfig(steps=2, group_size=2, seed=0),
        out_dir=tmp_path / "rl",
        base_policy_sha="sha1",
    )
    assert summary["n_groups"] >= 1
    assert (tmp_path / "rl" / "model.pt").exists()
