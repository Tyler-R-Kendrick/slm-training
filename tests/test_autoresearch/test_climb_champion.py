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


def test_baseline_seed_status_never_confirmed(tmp_path: Path) -> None:
    from slm_training.autoresearch.hillclimb import (
        CLIMB_CHAMPION_ADVANCE_STATUSES,
        CLIMB_CHAMPION_STATUS_BASELINE_SEED,
        CLIMB_CHAMPION_STATUS_CONFIRMED,
        CLIMB_CHAMPION_STATUSES,
        is_baseline_seed_champion,
    )
    from slm_training.autoresearch.thrash_regime import CLIMB_BASELINE_STATUSES

    # Declared sidecar status, excluded from every confirmed/promote set.
    assert CLIMB_CHAMPION_STATUS_BASELINE_SEED in CLIMB_CHAMPION_STATUSES
    assert CLIMB_CHAMPION_STATUS_BASELINE_SEED not in CLIMB_CHAMPION_ADVANCE_STATUSES
    assert CLIMB_CHAMPION_STATUS_BASELINE_SEED not in CLIMB_BASELINE_STATUSES
    assert CLIMB_CHAMPION_STATUS_CONFIRMED in CLIMB_CHAMPION_ADVANCE_STATUSES

    loop_dir = tmp_path / "loop"
    base = _ckpt(tmp_path / "runs" / "c1-control" / "checkpoints" / "last.pt", "b")
    # A queue row carrying baseline_seed never seeds a confirmed champion.
    assert (
        seed_climb_champion(
            loop_dir,
            confirmed_artifacts=[
                {"status": "baseline_seed", "checkpoint": str(base)},
                {"status": "rejected", "checkpoint": str(base)},
            ],
        )
        is None
    )
    assert load_climb_champion(loop_dir) is None
    seeded = seed_climb_champion(
        loop_dir,
        baseline_checkpoint=base,
        source_campaign="c1",
        extra_steps=22,
        train_data_manifest_sha="t" * 64,
        record_count=101,
        train_dir=str(tmp_path / "train"),
    )
    assert seeded is not None
    assert seeded.status == CLIMB_CHAMPION_STATUS_BASELINE_SEED
    assert is_baseline_seed_champion(seeded)
    loaded = load_climb_champion(loop_dir)
    assert loaded is not None
    assert loaded.status == "baseline_seed"
    assert loaded.record_count == 101
    assert loaded.train_dir == str(tmp_path / "train")
    assert loaded.cumulative_steps == 22
    # A confirmed artifact seeds a confirmed champion (unchanged semantics).
    other = tmp_path / "other"
    win = _ckpt(tmp_path / "win.pt", "w")
    confirmed = seed_climb_champion(
        other,
        confirmed_artifacts=[{"status": "confirmed", "checkpoint": str(win)}],
        baseline_checkpoint=base,
    )
    assert confirmed is not None
    assert confirmed.status == CLIMB_CHAMPION_STATUS_CONFIRMED
    assert not is_baseline_seed_champion(confirmed)


def test_baseline_seed_never_advances_on_screening_win(tmp_path: Path) -> None:
    loop_dir = tmp_path / "loop"
    base = _ckpt(tmp_path / "base.pt", "b")
    seed_climb_champion(
        loop_dir,
        baseline_checkpoint=base,
        source_campaign="seed",
        extra_steps=22,
        train_data_manifest_sha="t" * 64,
        record_count=101,
    )
    win = _ckpt(tmp_path / "win.pt", "w")
    # Screening win (confirmed=False): explicit no-advance path.
    kept = maybe_advance_climb_champion(
        loop_dir,
        confirmed=False,
        checkpoint=win,
        source_campaign="screen-win",
        extra_steps=40,
        train_data_manifest_sha="t" * 64,
        record_count=101,
    )
    assert kept is not None
    assert kept.status == "baseline_seed"
    assert kept.source_campaign == "seed"
    assert (loop_dir / "champion" / "last.pt").read_bytes() == b"b"
    # Confirmed win replaces it with a confirmed champion.
    advanced = maybe_advance_climb_champion(
        loop_dir,
        confirmed=True,
        checkpoint=win,
        source_campaign="confirm-win",
        extra_steps=40,
        train_data_manifest_sha="t" * 64,
        record_count=101,
        train_dir=str(tmp_path / "train"),
    )
    assert advanced is not None
    assert advanced.status == "confirmed"
    assert advanced.cumulative_steps == 62
    assert advanced.train_dir == str(tmp_path / "train")
    assert (loop_dir / "champion" / "last.pt").read_bytes() == b"w"


def test_legacy_sidecar_defaults_to_confirmed(tmp_path: Path) -> None:
    import json

    loop_dir = tmp_path / "loop"
    _ckpt(loop_dir / "champion" / "last.pt", "c")
    (loop_dir / "champion" / "last.json").write_text(
        json.dumps(
            {
                "schema": "climb_champion/v1",
                "source_campaign": "old",
                "cumulative_steps": 5,
                "train_data_manifest_sha": "a" * 64,
                "cumulative_epochs": 0.05,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_climb_champion(loop_dir)
    assert loaded is not None
    assert loaded.status == "confirmed"
    assert loaded.train_dir is None
    (loop_dir / "champion" / "last.json").write_text(
        json.dumps({"schema": "climb_champion/v2", "status": "bogus"}),
        encoding="utf-8",
    )
    with pytest.raises(HillClimbError, match="unknown_status"):
        load_climb_champion(loop_dir)


def test_epoch_cap_follows_train_manifest_record_count(tmp_path: Path) -> None:
    import json

    from slm_training.autoresearch.hillclimb import (
        champion_cumulative_epochs,
        train_manifest_record_count,
    )

    train_dir = tmp_path / "train" / "corpus_v9"
    train_dir.mkdir(parents=True)
    (train_dir / "manifest.json").write_text(
        json.dumps({"record_count": 676}), encoding="utf-8"
    )
    assert train_manifest_record_count(train_dir) == 676
    assert train_manifest_record_count(tmp_path / "missing") is None
    assert train_manifest_record_count(None) is None
    sidecar = ClimbChampionSidecar(
        source_campaign="c",
        cumulative_steps=5100,
        train_data_manifest_sha="t" * 64,
        # Accumulated against a 101-record corpus: 50.5 epochs.
        cumulative_epochs=5100 / 101,
        train_dir=str(train_dir),
        record_count=101,
    )
    # Fixed count would park; the current 676-record corpus does not.
    assert champion_epoch_park_reason(sidecar, max_cumulative_epochs=50) == (
        CHAMPION_EPOCHS_EXHAUSTED
    )
    assert champion_cumulative_epochs(sidecar, record_count=676) == pytest.approx(
        5100 / 676
    )
    assert (
        champion_epoch_park_reason(sidecar, max_cumulative_epochs=50, record_count=676)
        is None
    )
    # A smaller corpus lowers the cap in steps.
    assert (
        champion_epoch_park_reason(sidecar, max_cumulative_epochs=50, record_count=100)
        == CHAMPION_EPOCHS_EXHAUSTED
    )


def test_warm_start_launch_rejects_unequal_pair() -> None:
    base = {
        "steps": 40,
        "seed": 7,
        "train_version": "wf_smoke_v2",
        "initialize_from": "/c.pt",
    }
    with pytest.raises(HillClimbError, match="unequal_extra_steps"):
        assert_warm_start_launch(base, {**base, "steps": 41})
    with pytest.raises(HillClimbError, match="unequal_train_data"):
        assert_warm_start_launch(base, {**base, "train_version": "other"})
    with pytest.raises(HillClimbError, match="unequal_seed"):
        assert_warm_start_launch(base, {**base, "seed": 8})
    with pytest.raises(HillClimbError, match="unequal_checkpoint"):
        assert_warm_start_launch(base, {**base, "initialize_from": "/d.pt"})
