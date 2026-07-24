"""CLI coverage for publishing SpectralDispositionV1."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.model_cycle import cmd_promote, main as model_cycle_main
from scripts.publish_spectral_disposition import main
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.train_loop import train
from slm_training.lineage.store import LineageStore


def test_publish_and_check_are_consistent(tmp_path) -> None:
    args = [
        "--json",
        str(tmp_path / "report.json"),
        "--markdown",
        str(tmp_path / "report.md"),
    ]
    assert main(args) == 0
    assert main([*args, "--check"]) == 0


def _init_args(root, run_id: str, recipe: str | None = None) -> list[str]:
    args = [
        "--lineage-root",
        str(root),
        "init",
        "--track",
        "twotower",
        "--run-id",
        run_id,
        "--data-snapshot-sha",
        "data",
        "--eval-snapshot-sha",
        "eval",
    ]
    if recipe is not None:
        args.extend(["--recipe-json", recipe])
    return args


def test_lineage_init_branch_and_promote_enforce_disposition(tmp_path) -> None:
    root = tmp_path / "lineage"
    with pytest.raises(ValueError, match="not authorized"):
        model_cycle_main(_init_args(root, "absolute", '{"alpha_target": 2.0}'))

    assert model_cycle_main(_init_args(root, "parent")) == 0
    store = LineageStore(root)
    store.transition_run("parent", "validated")
    with pytest.raises(ValueError, match="not authorized"):
        model_cycle_main(
            [
                "--lineage-root",
                str(root),
                "branch",
                "--parent",
                "parent",
                "--run-id",
                "bad-branch",
                "--recipe-json",
                '{"trace_log_enabled": true}',
            ]
        )

    assert (
        model_cycle_main(
            _init_args(root, "muon", '{"optimizer_name": "muon_hybrid"}')
        )
        == 0
    )
    store.transition_run("muon", "validated")
    with pytest.raises(ValueError, match="fixture/research-only"):
        cmd_promote(SimpleNamespace(lineage_root=root, run_id="muon"))


def test_model_build_config_rejects_unknown_optimizer(tmp_path) -> None:
    with pytest.raises(ValueError, match="optimizer_name"):
        ModelBuildConfig(train_dir=tmp_path, optimizer_name="silent-adamw")


def test_direct_promotion_training_blocks_muon_before_artifacts(tmp_path) -> None:
    config = ModelBuildConfig(
        train_dir=tmp_path,
        optimizer_name="muon_hybrid",
        register_promoted=True,
    )
    with pytest.raises(ValueError, match="fixture/research-only"):
        train(config)
