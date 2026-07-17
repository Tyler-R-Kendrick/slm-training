"""G1 (SLM-46): Track A program matrix through the engine end-to-end."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.engine import (
    compile_commands,
    create_hypothesis_feedback,
    validate_experiment,
    validate_hypothesis_matrix,
)
from slm_training.autoresearch.program_matrix import (
    CAMPAIGN_ID,
    build_track_a_campaign,
    build_track_a_evidence,
    build_track_a_matrix,
    build_track_a_sources,
)
from slm_training.autoresearch.schemas import (
    Diagnosis,
    ExperimentKnobs,
    ExperimentOutcome,
)


def test_track_a_matrix_validates_end_to_end() -> None:
    campaign = build_track_a_campaign()
    evidence = build_track_a_evidence()
    sources = build_track_a_sources()
    matrix = build_track_a_matrix()
    # Engine validation: grounding, novelty, feedback contract, knob allowlist.
    validate_hypothesis_matrix(campaign, matrix, evidence, sources)
    for candidate in matrix.hypotheses:
        validate_experiment(campaign, candidate.experiment, evidence, sources)
    # Every knob signature in the matrix is distinct (matched-pair honesty).
    signatures = {
        tuple(
            sorted(
                candidate.experiment.knobs.model_dump(exclude_none=True).items()
            )
        )
        for candidate in matrix.hypotheses
    }
    assert len(signatures) == len(matrix.hypotheses)


def test_track_a_program_knobs_compile_to_bounded_commands() -> None:
    campaign = build_track_a_campaign()
    matrix = build_track_a_matrix()
    by_id = {c.experiment.experiment_id: c.experiment for c in matrix.hypotheses}
    commands = {
        eid: compile_commands(campaign, experiment)
        for eid, experiment in by_id.items()
    }
    train = {eid: cmds[-2] for eid, cmds in commands.items()}
    # Typed knob -> bounded flag; no researcher-authored shell anywhere.
    assert "--asap-decode" in train["trackA-a2-asap"]
    assert "--asap-decode" not in train["trackA-control"]
    floor = train["trackA-a4-floor"]
    assert floor[floor.index("--decode-min-content") + 1] == "-1"
    stack = train["trackA-a2a4-stack"]
    assert "--asap-decode" in stack and "--decode-min-content" in stack
    dose = train["trackA-a2-dose"]
    assert dose[dose.index("--steps") + 1] == "400"
    for cmds in commands.values():
        for argv in cmds:
            assert all(isinstance(token, str) for token in argv)
            assert not any(ch in token for token in argv for ch in ";|&$`")


def test_track_a_feedback_ack_round_trip() -> None:
    campaign = build_track_a_campaign()
    evidence = build_track_a_evidence()
    sources = build_track_a_sources()
    matrix = build_track_a_matrix()
    executed = matrix.recommended_experiment_id
    outcome = ExperimentOutcome(
        experiment_id=executed,
        campaign_id=CAMPAIGN_ID,
        status="completed",
        metrics={"held_out.meaningful_program_rate": 0.0},
    )
    diagnosis = Diagnosis(
        experiment_id=executed,
        target="model",
        confidence=0.6,
        evidence=("fixture run reproduces the empty wall",),
        recommended_actions=("rerun at frontier scale",),
    )
    feedback = create_hypothesis_feedback(matrix, outcome, diagnosis)
    assert feedback.matrix_id == matrix.matrix_id

    # The successor matrix must acknowledge the feedback and not repeat the
    # executed knob signature.
    executed_spec = next(
        c.experiment for c in matrix.hypotheses if c.experiment.experiment_id == executed
    )
    successor = matrix.model_copy(
        update={
            "matrix_id": "dsl-program-track-a-m2",
            "predecessor_matrix_id": matrix.matrix_id,
            "feedback_ids": (feedback.feedback_id,),
            "hypotheses": tuple(
                candidate.model_copy(
                    update={
                        "experiment": candidate.experiment.model_copy(
                            update={
                                "experiment_id": f"m2-{candidate.experiment.experiment_id}",
                                "knobs": candidate.experiment.knobs.model_copy(
                                    update={"seed": 1}
                                ),
                            }
                        )
                    }
                )
                for candidate in matrix.hypotheses
            ),
            "recommended_experiment_id": "m2-trackA-a2-asap",
        }
    )
    validate_hypothesis_matrix(
        campaign,
        successor,
        evidence,
        sources,
        prior_experiments=(executed_spec,),
        prior_experiment_ids=frozenset({executed}),
        feedback=(feedback,),
        previous_matrix=matrix,
    )
    # Unacknowledged feedback is rejected.
    with pytest.raises(ValueError, match="acknowledge"):
        validate_hypothesis_matrix(
            campaign,
            successor.model_copy(update={"feedback_ids": ()}),
            evidence,
            sources,
            feedback=(feedback,),
            previous_matrix=matrix,
        )


def test_program_knobs_stay_strict_and_allowlisted() -> None:
    # Strict schema: unknown fields still rejected.
    with pytest.raises(ValidationError):
        ExperimentKnobs.model_validate({"asap_decode": True, "shell": "rm -rf /"})
    with pytest.raises(ValidationError):
        ExperimentKnobs(decode_min_content=-2)
    # A campaign that does not allow the program levers rejects them.
    campaign = build_track_a_campaign().model_copy(
        update={"allowed_knobs": frozenset({"steps"})}
    )
    matrix = build_track_a_matrix()
    evidence = build_track_a_evidence()
    sources = build_track_a_sources()
    with pytest.raises(ValueError, match="forbidden"):
        validate_experiment(
            campaign, matrix.hypotheses[0].experiment, evidence, sources
        )
