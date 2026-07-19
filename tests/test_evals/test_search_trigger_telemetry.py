"""Regression tests for EFS2-02 observe-only trigger telemetry wiring."""

from __future__ import annotations

from slm_training.evals.search_trigger_telemetry import (
    DecisionStep,
    TriggerObserver,
    TriggerPredicate,
    TriggerRegime,
    TriggerThresholdManifest,
    compare_trigger_regimes,
)


def _steps_repeating_state(n: int, fingerprint: str = "s0") -> list[DecisionStep]:
    """Return ``n`` identical decision steps to exercise stagnation.

    Scores have a large margin and low entropy so uncertainty does not fire.
    """
    return [
        DecisionStep(
            state_fingerprint=fingerprint,
            decision_depth=1,
            live_action_scores=(0.9, 0.1),
            certified_reductions=0,
        )
        for _ in range(n)
    ]


def test_observe_does_not_mutate_input_step() -> None:
    step = DecisionStep(
        state_fingerprint="a",
        decision_depth=2,
        live_action_scores=(1.0, 0.9, 0.1),
        certified_reductions=1,
    )
    observer = TriggerObserver(TriggerRegime.GREEDY, TriggerThresholdManifest())
    observer.observe(0, step)
    assert step.state_fingerprint == "a"
    assert step.decision_depth == 2
    assert step.live_action_scores == (1.0, 0.9, 0.1)
    assert step.certified_reductions == 1


def test_bottom_is_recorded_as_retraction_event() -> None:
    observer = TriggerObserver(TriggerRegime.GREEDY, TriggerThresholdManifest())
    obs = observer.observe(
        0,
        DecisionStep(
            state_fingerprint="conflict",
            decision_depth=0,
            live_action_scores=(0.0,),
            is_bottom=True,
        ),
    )
    assert obs is not None
    assert obs.predicate == TriggerPredicate.BOTTOM
    assert obs.triggered is True


def test_stagnation_fires_after_repeated_state() -> None:
    # Default repeat_window=3 -> StagnationTracker patience=2, fires on third repeat.
    observer = TriggerObserver(TriggerRegime.GREEDY, TriggerThresholdManifest())
    steps = _steps_repeating_state(4)
    observations = [observer.observe(i, step) for i, step in enumerate(steps)]
    assert not observations[0].triggered
    assert not observations[1].triggered
    assert observations[2].triggered
    assert observations[2].predicate == TriggerPredicate.STAGNATION
    assert observations[2].repeated_state_count == 2
    assert observations[3].triggered


def test_uncertainty_fires_on_low_margin_high_entropy() -> None:
    thresholds = TriggerThresholdManifest(
        margin_quantile=0.1,
        entropy_quantile=0.5,
    )
    observer = TriggerObserver(TriggerRegime.TEMPERATURE, thresholds)
    # Two nearly-equal high-probability actions -> low margin, high entropy.
    obs = observer.observe(
        0,
        DecisionStep(
            state_fingerprint="u",
            decision_depth=1,
            live_action_scores=(0.51, 0.50),
        ),
    )
    assert obs.triggered is True
    assert obs.predicate == TriggerPredicate.UNCERTAINTY
    assert obs.margin < thresholds.margin_quantile
    assert obs.entropy > thresholds.entropy_quantile


def test_budget_pressure_fires_above_forward_limit() -> None:
    thresholds = TriggerThresholdManifest(budget_pressure_forward_limit=5)
    observer = TriggerObserver(TriggerRegime.BEAM, thresholds)
    obs = observer.observe(
        0,
        DecisionStep(
            state_fingerprint="bp",
            decision_depth=1,
            live_action_scores=(0.8, 0.2),
            model_forwards=7,
        ),
    )
    assert obs.triggered is True
    assert obs.predicate == TriggerPredicate.BUDGET_PRESSURE


def test_outcome_labels_attach_to_all_observations() -> None:
    observer = TriggerObserver(TriggerRegime.GREEDY, TriggerThresholdManifest())
    for i, step in enumerate(_steps_repeating_state(3)):
        observer.observe(i, step)
    observer.label_outcomes(final_pass=False, recoverable=True, remaining_cost=12.5)
    for obs in observer.result.observations:
        assert obs.outcome_final_pass is False
        assert obs.outcome_recoverable is True
        assert obs.outcome_remaining_cost == 12.5


def test_observations_are_byte_identical_across_runs() -> None:
    examples = [
        (
            "ex1",
            [
                DecisionStep(
                    state_fingerprint="s",
                    decision_depth=1,
                    live_action_scores=(0.7, 0.3),
                ),
                DecisionStep(
                    state_fingerprint="s",
                    decision_depth=2,
                    live_action_scores=(0.6, 0.4),
                ),
            ],
            True,
            False,
        ),
    ]
    result_a = compare_trigger_regimes(examples, seed=7)
    result_b = compare_trigger_regimes(examples, seed=7)
    assert len(result_a.runs) == len(result_b.runs)
    for ra, rb in zip(result_a.runs, result_b.runs):
        assert ra.regime == rb.regime
        assert len(ra.observations) == len(rb.observations)
        for oa, ob in zip(ra.observations, rb.observations):
            assert oa.to_dict() == ob.to_dict()


def test_compare_includes_all_default_regimes() -> None:
    examples = [
        (
            "ex1",
            [DecisionStep("s", 1, (0.8, 0.2))],
            True,
            False,
        ),
    ]
    result = compare_trigger_regimes(examples)
    regimes = {r.regime for r in result.runs}
    assert regimes == {TriggerRegime.GREEDY, TriggerRegime.TEMPERATURE, TriggerRegime.BEAM}


def test_to_dict_round_trip_keys() -> None:
    observer = TriggerObserver(TriggerRegime.GREEDY, TriggerThresholdManifest())
    observer.observe(0, DecisionStep("s", 1, (0.8, 0.2)))
    observer.label_outcomes(final_pass=True, recoverable=False)
    data = observer.result.to_dict()
    assert data["regime"] == "greedy"
    assert data["example_id"] == ""
    assert "thresholds" in data
    assert len(data["observations"]) == 1
    assert data["observations"][0]["triggered"] is False
    assert data["observations"][0]["outcome_final_pass"] is True


def test_threshold_manifest_is_frozen_dict() -> None:
    manifest = TriggerThresholdManifest(
        repeat_window=2,
        no_progress_window=4,
        margin_quantile=0.05,
        entropy_quantile=0.9,
    )
    data = manifest.to_dict()
    assert data["repeat_window"] == 2
    assert data["no_progress_window"] == 4
    assert data["margin_quantile"] == 0.05
    assert data["entropy_quantile"] == 0.9
    assert data["budget_pressure_forward_limit"] is None


def test_empty_action_scores_yield_zero_margin_entropy() -> None:
    observer = TriggerObserver(TriggerRegime.GREEDY, TriggerThresholdManifest())
    obs = observer.observe(0, DecisionStep("empty", 1, ()))
    assert obs.margin == 0.0
    assert obs.entropy == 0.0
    assert obs.live_action_count == 0


def test_regime_perturbation_changes_only_scores() -> None:
    """Temperature/beam regimes must keep metadata identical; only scores differ."""
    step = DecisionStep(
        state_fingerprint="meta",
        decision_depth=3,
        live_action_scores=(0.9, 0.1),
        certified_reductions=2,
        value_score=0.5,
        verifier_calls=1,
        model_forwards=4,
        wall_ms=10.0,
    )
    greedy = TriggerObserver(TriggerRegime.GREEDY, TriggerThresholdManifest())
    temp = TriggerObserver(TriggerRegime.TEMPERATURE, TriggerThresholdManifest())
    beam = TriggerObserver(TriggerRegime.BEAM, TriggerThresholdManifest())

    g_obs = greedy.observe(0, step)
    t_obs = temp.observe(0, step)
    b_obs = beam.observe(0, step)

    for obs in (t_obs, b_obs):
        assert obs.state_fingerprint == g_obs.state_fingerprint
        assert obs.decision_depth == g_obs.decision_depth
        assert obs.certified_reductions_since_prior == g_obs.certified_reductions_since_prior
        assert obs.value_score == g_obs.value_score
        assert obs.verifier_calls_since_progress == g_obs.verifier_calls_since_progress
        assert obs.model_forwards_since_progress == g_obs.model_forwards_since_progress
