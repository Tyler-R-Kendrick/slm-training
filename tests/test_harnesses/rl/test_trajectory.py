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


def test_trajectory_rl_smoke(tmp_path: Path, approved_rl_report) -> None:
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
    commit = {
        "t": min(2, len(ids) - 1),
        "id": ids[min(2, len(ids) - 1)],
        "lp": -0.2,
        "allowed_id_set": list(range(model.tokenizer.vocab_size)),
    }

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
        readiness_report=approved_rl_report,
    )
    assert summary["n_groups"] >= 1
    assert summary["require_recorded_support"] is True
    assert (tmp_path / "rl" / "model.pt").exists()


def test_trajectory_rl_refuses_support_mismatch() -> None:
    openui = 'root = Button(":slot_0")'
    record = ExampleRecord(id="t", prompt="CTA", openui=openui, split="train")
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=0
        ),
        device="cpu",
    )
    ids = model.tokenizer.encode(openui)
    trace = {
        "meta": {"prompt": "CTA", "record_id": "t", "policy_checkpoint_sha": "s"},
        "labels": {"accepted": True},
        "reward": {"grammar": 1.0, "placeholder": 1.0},
        "steps": [{"canvas": ids, "commits": [{"t": 1, "id": ids[1], "lp": -0.1}]}],
    }
    with pytest.raises(ValueError, match="allowed_id_set"):
        importance_weighted_loss(
            model,
            [trace, {**trace, "reward": {"grammar": 1.0, "placeholder": 0.1}}],
        )


def test_trajectory_rl_refuses_missing_behavior_logprob() -> None:
    openui = 'root = Button(":slot_0")'
    record = ExampleRecord(id="t", prompt="CTA", openui=openui, split="train")
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=0
        ),
        device="cpu",
    )
    ids = model.tokenizer.encode(openui)
    commit = {
        "t": 1,
        "id": ids[1],
        "allowed_id_set": list(range(model.tokenizer.vocab_size)),
    }
    trace = {
        "meta": {"prompt": "CTA", "record_id": "t"},
        "labels": {"accepted": True},
        "reward": {"grammar": 1.0, "placeholder": 1.0},
        "steps": [{"canvas": ids, "commits": [commit]}],
    }
    with pytest.raises(ValueError, match="log-probability"):
        importance_weighted_loss(
            model,
            [trace, {**trace, "reward": {"grammar": 1.0, "placeholder": 0.1}}],
        )


def test_importance_ratio_uses_matching_trajectory_sum(monkeypatch) -> None:
    from slm_training.harnesses.rl import trajectory

    commit = {"lp": -1.0}
    low = {
        "labels": {"accepted": True},
        "reward": {"grammar": 1.0, "placeholder": 0.1},
        "steps": [{"commits": [commit, commit]}],
    }
    high = {
        "labels": {"accepted": True},
        "reward": {"grammar": 1.0, "placeholder": 0.9},
        "steps": [{"commits": [commit, commit]}],
    }
    monkeypatch.setattr(
        trajectory,
        "trajectory_logprob",
        lambda *_args, **_kwargs: torch.tensor(-2.0, requires_grad=True),
    )
    loss = importance_weighted_loss(None, [low, high])  # type: ignore[arg-type]
    assert loss is not None
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)
