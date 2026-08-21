"""Persistent climb champion warm-start (SWARM-CHAMPION)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.hillclimb import (
    CHAMPION_EPOCHS_EXHAUSTED,
    ClimbChampionSidecar,
    HillClimbError,
    assert_champion_eval_disjoint,
    assert_warm_start_launch,
    attach_initialize_from,
    champion_epoch_park_reason,
    dump_climb_champion,
    load_climb_champion,
    maybe_advance_climb_champion,
    seed_climb_champion,
    write_climb_champion,
)
from slm_training.autoresearch.thrash_regime import (
    REGIME_CLIMB,
    REGIME_ISOLATE,
    decide_screening_regime,
)


def _ckpt(path: Path, marker: str = "w") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(marker.encode("utf-8"))
    return path


def test_sidecar_round_trip(tmp_path: Path) -> None:
    loop_dir = tmp_path / "loop"
    ckpt = _ckpt(tmp_path / "src.pt")
    sidecar = ClimbChampionSidecar(
        source_campaign="camp-1",
        cumulative_steps=22,
        train_data_manifest_sha="a" * 64,
        cumulative_epochs=0.2,
        trainable_params=1984,
        knobs={"binder_arity_loss_weight": 1.0},
    )
    write_climb_champion(loop_dir, checkpoint=ckpt, sidecar=sidecar)
    loaded = load_climb_champion(loop_dir)
    assert loaded is not None
    assert loaded.source_campaign == "camp-1"
    assert loaded.cumulative_steps == 22
    assert loaded.train_data_manifest_sha == "a" * 64
    assert loaded.knobs["binder_arity_loss_weight"] == 1.0
    dump_climb_champion(loaded, loop_dir)
    again = load_climb_champion(loop_dir)
    assert again is not None
    assert again.as_dict()["cumulative_steps"] == 22


def test_attach_initialize_from_both_arms(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path / "champion" / "last.pt")
    control = {"steps": 40, "seed": 7, "train_version": "wf_smoke_v2"}
    candidate = {
        "steps": 40,
        "seed": 7,
        "train_version": "wf_smoke_v2",
        "binder_arity_loss_weight": 1.0,
    }
    control, candidate = attach_initialize_from(control, candidate, ckpt)
    assert control["initialize_from"] == str(ckpt)
    assert candidate["initialize_from"] == str(ckpt)
    assert_warm_start_launch(control, candidate, trainable_params=(100, 100))


def test_params_equality_assertion() -> None:
    knobs = {"steps": 8, "seed": 1, "train_version": "t", "initialize_from": "/c.pt"}
    with pytest.raises(HillClimbError, match="unequal_params"):
        assert_warm_start_launch(knobs, knobs, trainable_params=(1, 2))


def test_advance_on_confirm_only(tmp_path: Path) -> None:
    loop_dir = tmp_path / "loop"
    base = _ckpt(tmp_path / "base.pt", "b")
    seed_climb_champion(
        loop_dir,
        baseline_checkpoint=base,
        source_campaign="seed",
        extra_steps=10,
        train_data_manifest_sha="t" * 64,
        record_count=10,
    )
    before = load_climb_champion(loop_dir)
    assert before is not None
    win = _ckpt(tmp_path / "win.pt", "w")
    maybe_advance_climb_champion(
        loop_dir,
        confirmed=False,
        checkpoint=win,
        source_campaign="reject",
        extra_steps=40,
        train_data_manifest_sha="t" * 64,
        record_count=10,
    )
    after_reject = load_climb_champion(loop_dir)
    assert after_reject is not None
    assert after_reject.source_campaign == "seed"
    maybe_advance_climb_champion(
        loop_dir,
        confirmed=True,
        checkpoint=win,
        source_campaign="confirm-win",
        extra_steps=40,
        train_data_manifest_sha="t" * 64,
        record_count=10,
        knobs={"fidelity_loss_weight": 1.5},
    )
    after = load_climb_champion(loop_dir)
    assert after is not None
    assert after.source_campaign == "confirm-win"
    assert after.cumulative_steps == 50
    assert (loop_dir / "champion" / "last.pt").read_bytes() == b"w"


def test_epoch_park_at_cap() -> None:
    sidecar = ClimbChampionSidecar(
        source_campaign="c",
        cumulative_steps=1,
        train_data_manifest_sha="t" * 64,
        cumulative_epochs=50.0,
    )
    assert champion_epoch_park_reason(sidecar, max_cumulative_epochs=50) is None
    sidecar.cumulative_epochs = 50.01
    assert (
        champion_epoch_park_reason(sidecar, max_cumulative_epochs=50)
        == CHAMPION_EPOCHS_EXHAUSTED
    )


def test_leakage_fail_closed() -> None:
    with pytest.raises(HillClimbError, match="missing_eval"):
        assert_champion_eval_disjoint(
            train_data_manifest_sha="aa", eval_data_manifest_sha=None
        )
    with pytest.raises(HillClimbError, match="collision"):
        assert_champion_eval_disjoint(
            train_data_manifest_sha="same", eval_data_manifest_sha="same"
        )


def test_climb_regime_activates_on_champion_checkpoint() -> None:
    isolate = decide_screening_regime(
        climb_baseline_knobs=None, compiler_ms_timeout=False
    )
    assert isolate.regime == REGIME_ISOLATE
    climb = decide_screening_regime(
        climb_baseline_knobs=None,
        compiler_ms_timeout=False,
        climb_champion_available=True,
    )
    assert climb.regime == REGIME_CLIMB
    assert climb.climb_baseline is not None
    assert climb.reason == "climb_champion_checkpoint_available"
