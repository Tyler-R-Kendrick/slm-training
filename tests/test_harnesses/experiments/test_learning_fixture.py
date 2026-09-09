"""Fixture CLI and real training-effect replay; not evidence of model improvement."""

from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.autonomous_learning.cli import main
from slm_training.harnesses.experiments.autonomous_learning.learner_correction import (
    train_once,
)
from tests.test_harnesses.model_build import test_full_state_resume as resume_fixtures
from tests.test_harnesses.model_build.test_full_state_resume import _cfg

train_dir = resume_fixtures.train_dir


def test_fixture_cli_requires_explicit_activation():
    with pytest.raises(SystemExit) as result:
        main(["prepare", "--run-id", "must-not-run"])
    assert result.value.code == 2


def test_canonical_cli_reaches_research_subgroup(capsys):
    from scripts.slm import COMMANDS
    from scripts.slm import main as slm

    assert COMMANDS[("experiments", "learning-comparison")].module == (
        "slm_training.harnesses.experiments.autonomous_learning.cli"
    )
    assert slm(["experiments", "learning-comparison", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--enable-fixture-experiment" in help_text and "--run-root" in help_text


def test_efficiency_compatibility_path_is_canonical_module():
    import importlib

    assert importlib.import_module(
        "slm_training.harnesses.experiments.efficiency_gain"
    ) is importlib.import_module("slm_training.harness_core.efficiency_gain")


def test_manifest_creation_binds_preparation_time(tmp_path):
    from slm_training.harnesses.experiments.autonomous_learning.learner_correction import (
        comparison_manifest,
        fixture_config,
    )

    store = CampaignStore("same-preparation", root=tmp_path / "campaigns")
    prepared = {
        "bootstrap": {"finished_at": "2026-09-07T00:00:00+00:00"},
        "selection": {"selection_sha256": "a" * 64},
    }
    configs = {
        name: fixture_config(tmp_path, tmp_path / name, run_id=name, steps=2)
        for name in ("control", "candidate")
    }
    first = comparison_manifest(store, prepared, configs)
    second = comparison_manifest(store, prepared, configs)
    assert first.created_at == prepared["bootstrap"]["finished_at"]
    assert first.model_dump() == second.model_dump()


def test_actual_trial_reused_after_downstream_failure(train_dir, tmp_path, monkeypatch):
    import slm_training.harnesses.model_build as owner

    store = CampaignStore("training-replay", root=tmp_path / "campaigns")
    config = _cfg(train_dir, tmp_path, "trial", 2)
    first = train_once(store, config)

    def forbidden(*args, **kwargs):
        pytest.fail("completed training repeated during an evaluation retry")

    monkeypatch.setattr(owner, "train", forbidden)
    assert train_once(store, config) == first
    attempts = [
        row
        for row in store.verify_event_chain()
        if row["event_type"] == "learning_training_attempt_observed"
    ]
    assert len(attempts) == 1 and attempts[0]["detail"]["spent_seconds"] > 0
    with pytest.raises(ValueError, match="idempotency"):
        train_once(store, replace(config, steps=3))


@pytest.mark.parametrize("entrypoint", ["learning", "trainer"])
def test_actual_completed_chunk_resume_preserves_known_exposure(
    train_dir, tmp_path, entrypoint
):
    from slm_training.harness_core.checkpoint_exposure import checkpoint_exposure
    from slm_training.harnesses.model_build.train_loop import train

    store = CampaignStore("published-resume", root=tmp_path / "campaigns")
    execute = (
        (lambda config: train_once(store, config))
        if entrypoint == "learning"
        else train
    )
    first_config = _cfg(
        train_dir,
        tmp_path,
        "first",
        2,
        full_state_checkpoint=True,
        sync_checkpoints=False,
    )
    first = execute(first_config)
    assert checkpoint_exposure(Path(first["checkpoint"])) is not None
    # The real fidelity caller supplied this loose alias despite publishing a bundle.
    resumed = execute(
        _cfg(
            train_dir,
            tmp_path,
            "second",
            4,
            resume_from=first_config.checkpoint_dir / "last_full_state.pt",
            full_state_checkpoint=True,
            sync_checkpoints=False,
        )
    )
    exposure = checkpoint_exposure(Path(resumed["checkpoint"]))
    assert exposure is not None, "published continuation lost known ancestry"
    assert exposure == resumed["exposure_by_snapshot"]
    assert sum(row["consumed_examples"] for row in exposure) == (
        resumed["replay"]["seen_primary_examples"]
        + resumed["replay"]["seen_replay_examples"]
    )


def test_trial_release_successor_preserves_inputs_and_retries(tmp_path):
    import json
    from slm_training.autoresearch.campaign_events import lock_learning_trial

    store = CampaignStore("release-successor", root=tmp_path)
    inputs = dict(
        config="recipe",
        training="data",
        parent="state",
        versions={"train": "v43"},
        context_layout_contract="frozen",
    )
    original = lock_learning_trial(store, "trial", inputs, resumable=True)
    successor = {**inputs, "versions": {"train": "v44"}}
    with pytest.raises(ValueError, match="idempotency"):
        lock_learning_trial(store, "trial", successor, resumable=False)
    updated = lock_learning_trial(store, "trial", successor, resumable=True)
    assert updated != original and json.loads(original.read_text()) == inputs
    assert lock_learning_trial(store, "trial", successor, resumable=True) == updated
    events = store.verify_event_chain()
    assert len(events) == 2
    assert events[-1]["detail"]["previous_input_sha256"] == original.stem
    assert events[-1]["detail"]["resume_verified"] is False
    for key in ("config", "training", "parent", "context_layout_contract"):
        with pytest.raises(ValueError, match="idempotency"):
            lock_learning_trial(
                store, "trial", {**successor, key: "changed"}, resumable=True
            )
    updated.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        lock_learning_trial(store, "trial", inputs, resumable=True)


def test_actual_data_warm_starts_freeze_context_size(train_dir, tmp_path):
    from slm_training.dsl.schema import write_jsonl
    from slm_training.harnesses.model_build.data import load_train_records
    from slm_training.models.twotower import TwoTowerModel

    store = CampaignStore("matched-context", root=tmp_path / "campaigns")
    ancestor = train_once(store, _cfg(train_dir, tmp_path, "ancestor", 1))
    changed = tmp_path / "changed"
    changed.mkdir()
    write_jsonl(
        changed / "records.jsonl",
        [
            replace(row, prompt=row.prompt + " novelcontextword")
            for row in load_train_records(train_dir)
        ],
    )
    models = []
    for name, data in (("control", train_dir), ("candidate", changed)):
        cfg = _cfg(
            data, tmp_path, name, 1, initialize_from=Path(ancestor["checkpoint"])
        )
        result = train_once(store, cfg)
        models.append(
            TwoTowerModel.from_checkpoint(Path(result["checkpoint"]), device="cpu")
        )
        assert (
            result["track"]["trainable_params"] == ancestor["track"]["trainable_params"]
        )
    assert (
        models[0].context_tokenizer.token_to_id
        == models[1].context_tokenizer.token_to_id
    )
    assert "novelcontextword" not in models[1].context_tokenizer.token_to_id
