"""Token accounting, token-budget stop, and bit-exact full-state resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.train_loop import train
from slm_training.models.twotower import TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


@pytest.fixture()
def train_dir(tmp_path: Path) -> Path:
    out = tmp_path / "train_data"
    out.mkdir(parents=True)
    write_jsonl(
        out / "records.jsonl",
        [
            ExampleRecord(
                id="a",
                prompt="Hero",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="b",
                prompt="CTA",
                openui=CTA,
                split="train",
                placeholders=[":cta.label"],
            ),
            ExampleRecord(
                id="c",
                prompt="Hero two",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
        ],
    )
    return out


def _cfg(train_dir: Path, tmp_path: Path, run_id: str, steps: int, **kw):
    return ModelBuildConfig(
        train_dir=train_dir,
        run_root=tmp_path / "runs",
        run_id=run_id,
        steps=steps,
        batch_size=2,
        lr=3e-3,
        seed=0,
        model_name="twotower",
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        freeze_context=False,
        telemetry=False,
        **kw,
    )


def test_token_accounting_and_track_block(train_dir: Path, tmp_path: Path) -> None:
    summary = train(_cfg(train_dir, tmp_path, "tok", 4))
    assert summary["seen_target_tokens"] > 0
    assert summary["seen_prompt_tokens"] > 0
    assert summary["stopped_on"] == "steps"
    assert summary["max_wall_minutes"] == 5.0
    track = summary["track"]
    assert track["context_backend"] == "scratch"
    assert track["trainable_params"] > 0
    assert track["frozen_params"] == 0
    assert track["tokens_per_trainable_param"] > 0
    # metrics.jsonl rows carry the counters.
    rows = [
        json.loads(line)
        for line in (tmp_path / "runs" / "tok" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[-1]["seen_target_tokens"] == summary["seen_target_tokens"]


def test_token_budget_stops_before_steps(train_dir: Path, tmp_path: Path) -> None:
    summary = train(_cfg(train_dir, tmp_path, "budget", 1000, target_token_budget=200))
    assert summary["stopped_on"] == "token_budget"
    assert summary["steps"] < 1000
    assert summary["seen_target_tokens"] >= 200


def test_wall_budget_stops_before_steps(train_dir: Path, tmp_path: Path) -> None:
    summary = train(
        _cfg(train_dir, tmp_path, "wall", 1000, max_wall_minutes=1e-9)
    )
    assert summary["stopped_on"] == "wall_time_budget"
    assert summary["steps"] == 0
    assert summary["max_wall_minutes"] == 1e-9


def test_full_state_resume_is_bit_exact(train_dir: Path, tmp_path: Path) -> None:
    full = train(_cfg(train_dir, tmp_path, "full", 6))

    part_a = train(_cfg(train_dir, tmp_path, "part_a", 3))
    assert part_a["steps"] == 3
    full_state = (
        tmp_path / "runs" / "part_a" / "checkpoints" / "last_full_state.pt"
    )
    assert full_state.exists()

    part_b = train(_cfg(train_dir, tmp_path, "part_b", 6, resume_from=full_state))
    assert part_b["resumed_from"] == str(full_state)
    assert part_b["steps"] == 6
    assert part_b["last_loss"] == pytest.approx(full["last_loss"], abs=0.0)
    assert part_b["seen_target_tokens"] == full["seen_target_tokens"]
    assert part_b["seen_prompt_tokens"] == full["seen_prompt_tokens"]

    m_full = TwoTowerModel.from_checkpoint(
        tmp_path / "runs" / "full" / "checkpoints" / "last.pt"
    )
    m_resumed = TwoTowerModel.from_checkpoint(
        tmp_path / "runs" / "part_b" / "checkpoints" / "last.pt"
    )
    full_sd = m_full.state_dict()
    resumed_sd = m_resumed.state_dict()
    assert set(full_sd) == set(resumed_sd)
    for key, value in full_sd.items():
        assert torch.equal(value, resumed_sd[key]), key


def test_resume_rejects_different_corpus(train_dir: Path, tmp_path: Path) -> None:
    train(_cfg(train_dir, tmp_path, "orig", 2))
    full_state = tmp_path / "runs" / "orig" / "checkpoints" / "last_full_state.pt"

    other_dir = tmp_path / "other_train"
    other_dir.mkdir()
    write_jsonl(
        other_dir / "records.jsonl",
        [
            ExampleRecord(
                id="a",
                prompt="Hero",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="z",
                prompt="Different",
                openui=CTA,
                split="train",
                placeholders=[":cta.label"],
            ),
        ],
    )
    with pytest.raises(ValueError, match="resume_from data mismatch"):
        train(_cfg(other_dir, tmp_path, "bad_resume", 4, resume_from=full_state))


def test_init_from_starts_new_run_on_different_corpus(
    train_dir: Path, tmp_path: Path
) -> None:
    source = train(_cfg(train_dir, tmp_path, "source", 2))
    source_checkpoint = Path(source["checkpoint"])
    other_dir = tmp_path / "other_train"
    other_dir.mkdir()
    write_jsonl(
        other_dir / "records.jsonl",
        [
            ExampleRecord(
                id="different-a",
                prompt="Hero",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="different-b",
                prompt="CTA",
                openui=CTA,
                split="train",
                placeholders=[":cta.label"],
            ),
            ExampleRecord(
                id="different-c",
                prompt="Hero two",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
        ],
    )

    initialized = train(
        _cfg(other_dir, tmp_path, "initialized", 0, init_from=source_checkpoint)
    )

    assert initialized["initialized_from"] == str(source_checkpoint)
    assert initialized["resumed_from"] is None
    assert initialized["steps"] == 0
    source_model = TwoTowerModel.from_checkpoint(source_checkpoint)
    initialized_model = TwoTowerModel.from_checkpoint(initialized["checkpoint"])
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, initialized_model.state_dict()[key]), key


def test_loss_eval_wiring(train_dir: Path, tmp_path: Path) -> None:
    test_dir = tmp_path / "test_data"
    held = test_dir / "suites" / "held_out"
    held.mkdir(parents=True)
    write_jsonl(
        held / "records.jsonl",
        [
            ExampleRecord(
                id="h1",
                prompt="Hero held",
                openui=HERO,
                split="held_out",
                placeholders=[":hero.title", ":hero.body"],
            )
        ],
    )
    summary = train(
        _cfg(
            train_dir,
            tmp_path,
            "nll",
            4,
            test_dir=test_dir,
            loss_eval_every=2,
        )
    )
    assert len(summary["nll_history"]) >= 2
    assert summary["best_weighted_nll"] is not None
    run_dir = tmp_path / "runs" / "nll"
    assert (run_dir / "nll_history.jsonl").exists()
    assert (run_dir / "loss_suites.json").exists()
    assert (run_dir / "checkpoints" / "best_weighted_nll.pt").exists()
    first, last = summary["nll_history"][0], summary["nll_history"][-1]
    assert first["base_suite"] == "held_out"
    assert first["weighted_nll"] is not None
    assert last["seen_target_tokens"] > 0
