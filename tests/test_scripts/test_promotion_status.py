from pathlib import Path

import scripts.run_autotrain_continuous as module


def test_baseline_seed_excluded_from_promotion_status_sets() -> None:
    from slm_training.autoresearch.hillclimb import (
        CLIMB_CHAMPION_ADVANCE_STATUSES,
    )
    from slm_training.autoresearch.hillclimb import (
        CLIMB_CHAMPION_STATUS_BASELINE_SEED as seed,
    )
    from slm_training.autoresearch.thrash_regime import CLIMB_BASELINE_STATUSES

    assert seed not in module._PROMOTE_AUTHORITY_STATUSES
    assert seed not in module._CHAMPION_STATUSES
    assert seed not in module._RETRYABLE_PROMOTE_STATUSES
    assert seed not in CLIMB_CHAMPION_ADVANCE_STATUSES
    assert seed not in CLIMB_BASELINE_STATUSES
    assert not module._is_decisive_causal_terminal(
        {"status": seed, "resolve_reasons": ["primary_metric_win_rejected"]}
    )
    assert not module._should_enqueue_champion(
        {"status": seed, "positive": False, "reasons": []}
    )
    assert module.seed_climb_champion(
        Path("/nonexistent-p8"),
        confirmed_artifacts=[{"status": seed, "checkpoint": __file__}],
    ) is None
