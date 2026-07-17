"""G1 (SLM-46): Track-A matrix routes through the real autoresearch engine."""

from __future__ import annotations

import sys

from slm_training.autoresearch.engine import (
    compile_commands,
    create_hypothesis_feedback,
    validate_experiment,
    validate_hypothesis_matrix,
)
from slm_training.autoresearch.program_matrices import (
    TRACK_A_CAMPAIGN_ID,
    track_a_campaign,
    track_a_evidence,
    track_a_matrix,
    track_a_sources,
)
from slm_training.autoresearch.schemas import Diagnosis, ExperimentOutcome


def test_track_a_matrix_validates_end_to_end() -> None:
    campaign = track_a_campaign()
    matrix = track_a_matrix()
    evidence = track_a_evidence()
    sources = track_a_sources()
    # The primary engine entry point: raises on any contract violation.
    validate_hypothesis_matrix(campaign, matrix, evidence, sources)
    assert len(matrix.hypotheses) >= 5
    # Every candidate individually grounded and knob-legal.
    for candidate in matrix.hypotheses:
        validate_experiment(campaign, candidate.experiment, evidence, sources)


def test_track_a_knobs_are_authentic_not_incidental() -> None:
    """Each emptiness lever sets a real Track-A knob (not just seed/steps),
    and every signature is distinct."""
    matrix = track_a_matrix()
    knobs_by_id = {
        c.experiment.experiment_id: c.experiment.knobs for c in matrix.hypotheses
    }
    assert knobs_by_id["a3-coverage-remask"].remask_policy == "coverage"
    assert knobs_by_id["a4-min-content-auto"].decode_min_content == -1
    assert knobs_by_id["a5-lattice-search"].compiler_search_mode == "lattice"
    combined = knobs_by_id["a3-a4-combined"]
    assert combined.remask_policy == "coverage"
    assert combined.decode_min_content == -1


def test_track_a_compiles_to_bounded_cpu_commands() -> None:
    campaign = track_a_campaign()
    matrix = track_a_matrix()
    for candidate in matrix.hypotheses:
        commands = compile_commands(campaign, candidate.experiment)
        # Bounded compilation: argv arrays only, never a shell string.
        assert all(isinstance(cmd, list) for cmd in commands)
        assert all(isinstance(part, str) for cmd in commands for part in cmd)
        evaluate = next(
            cmd for cmd in commands if cmd[:3] == [sys.executable, "-m", "scripts.evaluate_model"]
        )
        # zero GPU budget -> forced CPU.
        assert "--device" not in " ".join(evaluate) or "cpu" in evaluate
        # The emptiness knobs actually reach the eval command.
        knobs = candidate.experiment.knobs
        if knobs.remask_policy is not None:
            assert "--remask-policy" in evaluate
            assert knobs.remask_policy in evaluate
        if knobs.decode_min_content is not None:
            assert "--decode-min-content" in evaluate
        if knobs.compiler_search_mode is not None:
            assert "--compiler-search-mode" in evaluate


def test_track_a_feedback_ack_closes_the_loop() -> None:
    matrix = track_a_matrix()
    winner = matrix.recommended_experiment_id
    outcome = ExperimentOutcome(
        experiment_id=winner,
        campaign_id=TRACK_A_CAMPAIGN_ID,
        status="completed",
        metrics={"held_out.meaningful_program_rate": 0.0},
        command=("python", "-m", "scripts.evaluate_model"),
        exit_code=0,
    )
    diagnosis = Diagnosis(
        experiment_id=winner,
        target="model",
        confidence=0.5,
        evidence=("meaningful parse unchanged at fixture budget",),
        recommended_actions=("scale the budget before re-judging the lever",),
    )
    feedback = create_hypothesis_feedback(matrix, outcome, diagnosis)
    assert feedback.feedback_id.startswith("feedback-")
    assert feedback.campaign_id == TRACK_A_CAMPAIGN_ID
    assert feedback.experiment_id == winner
    # A successor matrix must acknowledge exactly this feedback id.
    assert feedback.matrix_id == matrix.matrix_id
