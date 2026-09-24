"""Shared public six-case contract fixture, with no neural capability claim."""

import pytest

from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.autonomous_learning import measurement_fixture


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    inputs = {
        "train_version": "train-fixture",
        "eval_version": "eval-fixture",
        "test_dir": str(tmp_path / "data"),
        "data_manifests": {},
        "selection": {
            "selected_record_ids": [f"case-{i}" for i in range(6)],
            "selection_sha256": "e" * 64,
        },
        "arms": {
            arm: {
                "checkpoint": str(tmp_path / arm / "last.pt"),
                "checkpoint_sha256": "a" * 64,
                "trainable_parameters": 65826,
            }
            for arm in ("control", "candidate")
        },
    }
    identity = {
        "components": {
            "harness.model_build.eval": "v109",
            "evals.scoring": "v28",
            "harness.experiments": "v169",
        }
    }
    monkeypatch.setattr(measurement_fixture, "_inputs", lambda *args: inputs)
    monkeypatch.setattr(measurement_fixture, "source_identity", lambda: identity)
    store = CampaignStore("measurement-test", tmp_path / "campaigns")
    plan = measurement_fixture.prepare(
        store, tmp_path / "retained.json", "train-fixture", "eval-fixture"
    )
    return store, plan
