"""Token accounting, token-budget stop, and bit-exact full-state resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.train_loop import train
from slm_training.levers import MAX_HARNESS_WALL_MINUTES
from slm_training.models.twotower import TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":slot_0")'


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
                placeholders=[":slot_0", ":slot_1"],
            ),
            ExampleRecord(
                id="b",
                prompt="CTA",
                openui=CTA,
                split="train",
                placeholders=[":slot_0"],
            ),
            ExampleRecord(
                id="c",
                prompt="Hero two",
                openui=HERO,
                split="train",
                placeholders=[":slot_0", ":slot_1"],
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
    summary = train(_cfg(train_dir, tmp_path, "wall", 1000, max_wall_minutes=1e-9))
    assert summary["stopped_on"] == "wall_time_budget"
    assert summary["steps"] == 0
    assert summary["max_wall_minutes"] == 1e-9


def test_wall_budget_reserves_time_for_finalization(
    train_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=f"at most {MAX_HARNESS_WALL_MINUTES}"):
        train(
            _cfg(
                train_dir,
                tmp_path,
                "over_wall",
                1,
                max_wall_minutes=MAX_HARNESS_WALL_MINUTES + 0.01,
            )
        )


def test_full_state_resume_is_bit_exact(train_dir: Path, tmp_path: Path) -> None:
    full = train(_cfg(train_dir, tmp_path, "full", 6))

    part_a = train(_cfg(train_dir, tmp_path, "part_a", 3))
    assert part_a["steps"] == 3
    full_state = tmp_path / "runs" / "part_a" / "checkpoints" / "last_full_state.pt"
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
                placeholders=[":slot_0", ":slot_1"],
            ),
            ExampleRecord(
                id="z",
                prompt="Different",
                openui=CTA,
                split="train",
                placeholders=[":slot_0"],
            ),
        ],
    )
    with pytest.raises(ValueError, match="resume_from data mismatch"):
        train(_cfg(other_dir, tmp_path, "bad_resume", 4, resume_from=full_state))


def test_initialize_from_resets_state_for_new_corpus(
    train_dir: Path, tmp_path: Path
) -> None:
    prior_recipe = {
        "output_tokenizer": "choice",
        "slot_component_loss_weight": 1.0,
        "slot_component_lexeme_prior_weight": 1.0,
        "slot_component_next_context": True,
        "slot_component_pair_interaction": True,
    }
    source = train(_cfg(train_dir, tmp_path, "source", 2, **prior_recipe))
    source_checkpoint = Path(source["checkpoint"])

    other_dir = tmp_path / "other_train"
    other_dir.mkdir()
    rows = [
        ExampleRecord(
            id=f"new-{index}",
            prompt=prompt,
            openui=openui,
            split="train",
            placeholders=placeholders,
        )
        for index, (prompt, openui, placeholders) in enumerate(
            [
                (
                    "Hero",
                    HERO,
                    [":slot_0", ":slot_1"],
                ),
                (
                    "CTA",
                    CTA,
                    [":slot_0"],
                ),
                (
                    "Hero two",
                    HERO,
                    [":slot_0", ":slot_1"],
                ),
            ]
        )
    ]
    write_jsonl(other_dir / "records.jsonl", rows)

    initialized = train(
        _cfg(
            other_dir,
            tmp_path,
            "initialized",
            0,
            initialize_from=source_checkpoint,
            **prior_recipe,
        )
    )
    assert initialized["initialized_from"] == str(source_checkpoint)
    assert initialized["resumed_from"] is None
    assert initialized["steps"] == 0
    assert initialized["seen_target_tokens"] == 0
    assert initialized["recipe"]["slot_component_loss_weight"] == 1.0
    assert initialized["recipe"]["slot_component_lexeme_prior_weight"] == 1.0
    assert initialized["recipe"]["slot_component_next_context"] is True
    assert initialized["recipe"]["slot_component_pair_interaction"] is True
    assert initialized["initialized_prior_fields"] == [
        "slot_component_lexeme_priors",
        "slot_component_span_priors",
    ]
    assert initialized["rebuilt_prior_fields"] == [
        "slot_component_lexeme_priors",
        "slot_component_span_priors",
    ]
    assert initialized["initialized_weight_count"] > 0
    assert initialized["initialized_weight_rms_drift"] == 0.0

    source_model = TwoTowerModel.from_checkpoint(source_checkpoint)
    initialized_model = TwoTowerModel.from_checkpoint(Path(initialized["checkpoint"]))
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, initialized_model.state_dict()[key]), key
    assert source_model.config.slot_component_lexeme_priors
    assert any(
        key == "slot_0"
        for key, _scores in initialized_model.config.slot_component_lexeme_priors
    )
    assert (
        source_model.config.slot_component_lexeme_priors
        == initialized_model.config.slot_component_lexeme_priors
    )

    retained = train(
        _cfg(
            other_dir,
            tmp_path,
            "retained",
            2,
            initialize_from=source_checkpoint,
            initialization_weight_retention=1.0,
            **prior_recipe,
        )
    )
    retained_model = TwoTowerModel.from_checkpoint(Path(retained["checkpoint"]))
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, retained_model.state_dict()[key]), key
    assert retained["initialized_weight_count"] > 0
    assert retained["initialized_weight_rms_drift"] == 0.0
    assert retained["recipe"]["initialization_weight_retention"] == 1.0


def test_initialize_from_cannot_mix_with_resume(
    train_dir: Path, tmp_path: Path
) -> None:
    source = train(_cfg(train_dir, tmp_path, "source_for_conflict", 1))
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(
            _cfg(
                train_dir,
                tmp_path,
                "conflict",
                2,
                resume_from=(
                    tmp_path
                    / "runs"
                    / "source_for_conflict"
                    / "checkpoints"
                    / "last_full_state.pt"
                ),
                initialize_from=Path(source["checkpoint"]),
            )
        )
    with pytest.raises(
        ValueError, match="initialization_weight_retention requires initialize_from"
    ):
        train(
            _cfg(
                train_dir,
                tmp_path,
                "retention_without_initialization",
                1,
                initialization_weight_retention=0.1,
            )
        )


def test_parent_replay_is_deterministic_and_provenanced(
    train_dir: Path, tmp_path: Path
) -> None:
    replay_dir = tmp_path / "replay_data"
    replay_dir.mkdir()
    write_jsonl(
        replay_dir / "records.jsonl",
        [
            ExampleRecord(
                id="a",
                prompt="Parent hero",
                openui=HERO,
                split="train",
                placeholders=[":slot_0", ":slot_1"],
            ),
            ExampleRecord(
                id="parent-cta",
                prompt="Parent CTA",
                openui=CTA,
                split="train",
                placeholders=[":slot_0"],
            ),
        ],
    )
    summary = train(
        _cfg(
            train_dir,
            tmp_path,
            "replay",
            8,
            replay_train_dir=replay_dir,
            replay_fraction=0.5,
        )
    )
    replay = summary["replay"]
    assert replay["enabled"] is True
    assert replay["requested_fraction"] == 0.5
    assert replay["effective_fraction"] == 0.5
    assert replay["seen_primary_examples"] == 8
    assert replay["seen_replay_examples"] == 8
    assert replay["primary_data_manifest_sha"]
    assert replay["replay_data_manifest_sha"]
    assert summary["data_manifest_sha"] == replay["combined_data_manifest_sha"]
    assert summary["recipe"]["replay_fraction"] == 0.5
    loss_proxy = replay["example_token_loss_proxy"]
    assert loss_proxy["primary"]["count"] == 8
    assert loss_proxy["replay"]["count"] == 8
    assert loss_proxy["primary"]["mean"] > 0
    assert loss_proxy["replay"]["mean"] > 0
    assert loss_proxy["primary"]["first_20_mean"] == loss_proxy["primary"]["mean"]
    assert loss_proxy["replay"]["last_20_mean"] == loss_proxy["replay"]["mean"]


def test_parent_replay_requires_a_corpus(train_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires replay_train_dir"):
        train(_cfg(train_dir, tmp_path, "missing_replay", 1, replay_fraction=0.25))


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
                placeholders=[":slot_0", ":slot_1"],
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
