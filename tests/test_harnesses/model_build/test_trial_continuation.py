"""Actual CPU training: continuation, safe yields, immutable trial cursors."""

import json

import pytest

from tests.test_harnesses.model_build.test_full_state_resume import (
    _cfg,
    train_dir as corpus_fixture,
)
from slm_training.harnesses.model_build.train_loop import train
from slm_training.harnesses.model_build.full_state import load_full_state
from slm_training.harnesses.model_build.resume_contract import exposure_by_snapshot

train_dir = corpus_fixture


@pytest.mark.parametrize("warm", [False, True])
def test_real_accumulated_trial_resume_preserves_optimizer_and_rng(
    train_dir, tmp_path, warm
):
    import random
    import numpy as np
    import torch
    from pathlib import Path

    initialization = {}
    if warm:
        ancestor = train(_cfg(train_dir, tmp_path, "ancestor", 1))
        initialization = {"initialize_from": Path(ancestor["checkpoint"])}
    random.seed(43)
    np.random.seed(43)
    full = train(
        _cfg(train_dir, tmp_path, "whole", 6, grad_accum_steps=2, **initialization)
    )
    random.seed(78)
    np.random.seed(78)
    first = train(
        _cfg(train_dir, tmp_path, "first", 3, grad_accum_steps=2, **initialization)
    )
    cursor = json.loads(
        (tmp_path / "runs/first/checkpoints/trial_cursor.json").read_text()
    )
    resumed = train(
        _cfg(
            train_dir,
            tmp_path,
            "second",
            6,
            grad_accum_steps=2,
            resume_from=cursor["resume_from"],
        )
    )
    a = load_full_state(tmp_path / "runs/whole/checkpoints/last_full_state.pt")
    b = load_full_state(tmp_path / "runs/second/checkpoints/last_full_state.pt")

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for x, y in zip(left, right, strict=True):
                equal(x, y)
        else:
            assert left == right

    for key in (
        "model",
        "optimizer",
        "scaler",
        "torch_rng",
        "python_rng",
        "numpy_rng",
        "cuda_rng",
        "loop_rng",
        "model_mask_rng",
        "pending_batch_ids",
    ):
        equal(a[key], b[key])
    assert full["exposure_by_snapshot"] == resumed["exposure_by_snapshot"]
    assert full["seen_target_tokens"] == resumed["seen_target_tokens"]
    assert first["steps"] == 3 and resumed["steps"] == 6
    assert resumed["continuation_kind"] == "exact_same_environment"
    assert cursor["role"] == "trial_cursor"
    assert not (tmp_path / "champion").exists()
    if warm:
        assert resumed["resume_compatibility"]["original_initializer"] == str(
            initialization["initialize_from"]
        )


@pytest.mark.parametrize("predecessor", [0, 1])
def test_source_migration_is_exact_and_preserves_original_contract(predecessor):
    import hashlib
    from copy import deepcopy
    from pathlib import Path
    from slm_training.harnesses.model_build import resume_contract as owner

    migration = json.loads(
        (
            Path(owner.__file__).parents[2]
            / "resources/model_build/resume_compatibility_v1.json"
        ).read_text()
    )
    for name, digest in migration["to_source"].items():
        assert (
            hashlib.sha256(
                Path(owner.__file__).with_name(name).read_bytes()
            ).hexdigest()
            == digest
        )
    previous = {
        "recipe": {"initialize_from": "ancestor.pt", "lr": 0.01},
        "source": {
            **migration["from_sources"][predecessor],
            "model.py": "model-digest",
        },
    }
    current = {
        "recipe": {"initialize_from": None, "lr": 0.01},
        "source": {**migration["to_source"], "model.py": "model-digest"},
    }
    unchanged = deepcopy(previous)
    receipt = owner._match_resume_identity(previous, current)
    assert receipt["source_migration"]["from_source"] == previous["source"]
    assert previous == unchanged
    unapproved = {
        **previous,
        "source": {**previous["source"], "train_loop.py": "unapproved"},
    }
    with pytest.raises(ValueError, match="mismatch"):
        owner._match_resume_identity(unapproved, current)
    for field in ("model.py", "train_loop.py", "resume_contract.py"):
        wrong = {**current, "source": {**current["source"], field: "unapproved"}}
        with pytest.raises(ValueError, match="mismatch"):
            owner._match_resume_identity(previous, wrong)
    with pytest.raises(ValueError, match="mismatch"):
        owner._match_resume_identity(
            previous, {**current, "recipe": {**current["recipe"], "lr": 0.02}}
        )


@pytest.mark.parametrize(
    "change", [{"batch_size": 1}, {"lr": 0.9}, {"grad_accum_steps": 3}]
)
def test_exact_resume_rejects_recipe_drift(train_dir, tmp_path, change):
    from dataclasses import replace

    train(_cfg(train_dir, tmp_path, "first", 1))
    config = _cfg(
        train_dir,
        tmp_path,
        "second",
        2,
        resume_from=tmp_path / "runs/first/checkpoints/last_full_state.pt",
    )
    with pytest.raises(ValueError, match="recipe/tokenizer/source/runtime mismatch"):
        train(replace(config, **change))


def test_token_yield_finishes_accumulation_and_counts_update(train_dir, tmp_path):
    summary = train(
        _cfg(
            train_dir, tmp_path, "yield", 100, grad_accum_steps=3, target_token_budget=1
        )
    )
    state = load_full_state(tmp_path / "runs/yield/checkpoints/last_full_state.pt")
    assert summary["steps"] == state["step"] == 1
    assert state["accumulation_position"] == 0
    assert state["seen_primary_examples"] == 5  # actual batches 2 + 1 + 2


def test_exposure_keeps_snapshot_denominators():
    rows = exposure_by_snapshot("old", "new", 10, 100, 200, 200)
    assert rows[0]["example_exposure_epochs"] == 20
    assert rows[1]["example_exposure_epochs"] == 2
    with pytest.raises(ValueError, match="invalid_count"):
        exposure_by_snapshot("old", None, 0, 0, 200, 0)


def test_legacy_state_loads_without_inventing_exact_resume(tmp_path):
    import torch
    from slm_training.harnesses.model_build.resume_contract import (
        validate_resume_contract,
    )

    path = tmp_path / "legacy.pt"
    torch.save({"kind": "full_train_state", "version": 1}, path)
    legacy = load_full_state(path)
    assert legacy["version"] == 1
    with pytest.raises(ValueError, match="legacy state lacks"):
        validate_resume_contract(legacy, None, None, "some-data")


def test_exact_resume_rejects_missing_rng_and_scaler_fields(train_dir, tmp_path):
    import torch

    train(_cfg(train_dir, tmp_path, "original", 1))
    state = load_full_state(tmp_path / "runs/original/checkpoints/last_full_state.pt")
    for key in (
        "python_rng",
        "numpy_rng",
        "model_mask_rng",
        "cuda_rng",
        "scaler",
        "scheduler",
    ):
        incomplete = {name: value for name, value in state.items() if name != key}
        path = tmp_path / f"missing-{key}.pt"
        torch.save(incomplete, path)
        with pytest.raises(ValueError, match="missing exact state"):
            train(_cfg(train_dir, tmp_path, f"retry-{key}", 2, resume_from=path))


def test_exact_resume_uses_effective_scaler_not_requested_cpu_amp(train_dir, tmp_path):
    train(_cfg(train_dir, tmp_path, "cpu-amp", 1, use_amp=True))
    resumed = train(
        _cfg(
            train_dir,
            tmp_path,
            "cpu-amp-resumed",
            2,
            use_amp=True,
            resume_from=tmp_path / "runs/cpu-amp/checkpoints/last_full_state.pt",
        )
    )
    assert resumed["continuation_kind"] == "exact_same_environment"
