"""Actual bundle producer to champion exposure consumer, without training mocks."""

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.hillclimb import (
    champion_cumulative_epochs,
    champion_epoch_park_reason,
    dump_climb_champion,
    load_climb_champion,
    maybe_advance_climb_champion,
    seed_climb_champion,
)
from slm_training.harness_core.checkpoint_bundle import stage_checkpoint_bundle
from slm_training.harness_core.checkpoint_exposure import exposure_epochs
from tests.test_harnesses.model_build.test_checkpoint_bundle import _checkpoint


def _row(sha="a" * 64, count=10, examples=200):
    return dict(
        snapshot_sha256=sha,
        record_count=count,
        consumed_examples=examples,
        example_exposure_epochs=examples / count,
        tokens={"prompt": 600, "target": 1000},
    )


def _trial(tmp_path, name, rows, ancestors=()):
    checkpoint = _checkpoint(tmp_path / name / "source")
    root = tmp_path / name / "trial"
    digest = stage_checkpoint_bundle(
        root,
        checkpoint,
        {
            "role": "trial_cursor",
            "optimizer_updates": 100,
            "exposure": rows,
            "ancestor_exposure": list(ancestors),
        },
    )
    return root / "bundles" / digest / "last.pt"


def test_actual_examples_not_optimizer_updates_and_new_corpus(tmp_path):
    first = _trial(tmp_path, "first", [_row()])
    loop = tmp_path / "loop"
    seed = seed_climb_champion(
        loop,
        baseline_checkpoint=first,
        extra_steps=100,
        train_data_manifest_sha="a" * 64,
        record_count=10,
    )
    assert seed.cumulative_epochs == 20  # 100 updates x 2 examples / 10 records.
    assert champion_cumulative_epochs(seed, record_count=1000) == 20
    second_row = _row("b" * 64, count=100, examples=100)
    second = _trial(tmp_path, "second", [second_row], ancestors=[_row()])
    promoted = maybe_advance_climb_champion(
        loop,
        confirmed=True,
        checkpoint=second,
        source_campaign="confirmed",
        extra_steps=100,
        train_data_manifest_sha="b" * 64,
        record_count=100,
    )
    assert promoted.cumulative_epochs == 21
    assert load_climb_champion(loop).exposure_history == [_row(), second_row]
    assert (
        champion_epoch_park_reason(promoted, max_cumulative_epochs=20)
        == "champion_epochs_exhausted"
    )


def test_exact_resume_cumulative_counts_are_not_added_again(tmp_path):
    loop = tmp_path / "loop"
    first = _trial(tmp_path, "first", [_row(examples=100)])
    seed_climb_champion(loop, baseline_checkpoint=first, extra_steps=50)
    resumed = _trial(tmp_path, "resumed", [_row(examples=200)])
    kwargs = dict(
        confirmed=True,
        checkpoint=resumed,
        source_campaign="same_trial",
        extra_steps=50,
        train_data_manifest_sha="a" * 64,
        record_count=10,
    )
    first_result = maybe_advance_climb_champion(loop, **kwargs)
    dump_climb_champion(first_result, loop)
    replayed = maybe_advance_climb_champion(loop, **kwargs)
    assert first_result.cumulative_epochs == replayed.cumulative_epochs == 20
    assert replayed.cumulative_steps == first_result.cumulative_steps


def test_unknown_legacy_checkpoint_never_invents_examples(tmp_path):
    seed = seed_climb_champion(
        tmp_path / "loop",
        baseline_checkpoint=_checkpoint(tmp_path / "old"),
        extra_steps=100,
        record_count=10,
    )
    assert seed.exposure_history is None
    assert seed.cumulative_epochs == 0
    assert champion_epoch_park_reason(seed) == "champion_exposure_unavailable"


def test_confirmation_of_existing_bundle_does_not_retrain_or_recharge(tmp_path):
    loop = tmp_path / "loop"
    checkpoint = _trial(tmp_path, "same", [_row()])
    seed_climb_champion(loop, baseline_checkpoint=checkpoint, extra_steps=100)
    confirmed = maybe_advance_climb_champion(
        loop,
        confirmed=True,
        checkpoint=checkpoint,
        source_campaign="confirm-only",
        extra_steps=100,
        train_data_manifest_sha="a" * 64,
        record_count=10,
    )
    assert confirmed.status == "confirmed"
    assert confirmed.cumulative_steps == 100
    assert confirmed.cumulative_epochs == 20


@pytest.mark.parametrize(
    "field,value",
    [
        ("consumed_examples", -1),
        ("record_count", 0),
        ("example_exposure_epochs", float("nan")),
        ("consumed_examples", True),
    ],
)
def test_malformed_exposure_refused(field, value):
    row = _row()
    row[field] = value
    with pytest.raises(ValueError, match="champion_exposure"):
        exposure_epochs([row])


@pytest.mark.parametrize(
    "change",
    case_values(__file__, "test_published_resume_alias_requires_actual_bundle_identity"),
)
def test_published_resume_alias_requires_actual_bundle_identity(tmp_path, change):
    """Real bundle protocol with opaque state bytes, not optimizer-resume evidence."""
    import json
    from types import SimpleNamespace
    from slm_training.harness_core.checkpoint_bundle import (
        published_resume_state,
        seal_trial_checkpoint,
    )

    checkpoint = _checkpoint(tmp_path / "run")
    state = checkpoint.parent / "last_full_state.pt"
    state.write_bytes(b"state fixture")
    config = SimpleNamespace(
        full_state_checkpoint=True,
        initialize_from=None,
        resume_from=None,
        run_id="fixture",
    )
    sealed = seal_trial_checkpoint(checkpoint, config, "a" * 64, 100, [_row()])
    published = sealed.parent / state.name
    cursor = checkpoint.parent / "trial_cursor.json"
    pointer = json.loads(cursor.read_text())
    if change == "newer":
        state.write_bytes(b"new unsealed state")
    elif change == "legacy":
        cursor.unlink()
    elif change == "corrupt":
        published.write_bytes(b"corrupt published state")
    elif change == "missing":
        published.unlink()
    elif change == "linked":
        cursor.unlink()
        cursor.symlink_to(tmp_path / "missing-pointer")
    elif change == "malformed":
        cursor.write_text("[]")
    elif change in {"path", "count", "role"}:
        field, value = {
            "path": ("resume_from", str(state)),
            "count": ("optimizer_updates", True),
            "role": ("role", "champion"),
        }[change]
        pointer[field] = value
        cursor.write_text(json.dumps(pointer))
    if change in {"identical", "newer", "legacy"}:
        assert published_resume_state(state) == (
            published if change == "identical" else state
        )
        assert published_resume_state(None) is None
    else:
        with pytest.raises(ValueError, match="bundle:|trial_cursor:"):
            published_resume_state(state)
