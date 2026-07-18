"""VSS4-02 (SLM-75): verified-scope-solver campaign through the engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.engine import (
    compile_commands,
    create_hypothesis_feedback,
    validate_experiment,
    validate_hypothesis_matrix,
)
from slm_training.autoresearch.schemas import (
    Diagnosis,
    ExperimentKnobs,
    ExperimentOutcome,
)
from slm_training.autoresearch.verified_scope_matrix import (
    CAMPAIGN_ID,
    MATRIX_ID,
    build_vss_campaign,
    build_vss_evidence,
    build_vss_matrix,
    build_vss_sources,
)


def test_vss_matrix_validates_end_to_end() -> None:
    campaign = build_vss_campaign()
    evidence = build_vss_evidence()
    sources = build_vss_sources()
    matrix = build_vss_matrix()
    validate_hypothesis_matrix(campaign, matrix, evidence, sources)
    for candidate in matrix.hypotheses:
        validate_experiment(campaign, candidate.experiment, evidence, sources)
    # Every knob signature is distinct — order-only matched-pair honesty.
    signatures = {
        tuple(
            sorted(candidate.experiment.knobs.model_dump(exclude_none=True).items())
        )
        for candidate in matrix.hypotheses
    }
    assert len(signatures) == len(matrix.hypotheses)
    # The four spec hypotheses plus a matched control.
    ids = {c.experiment.experiment_id for c in matrix.hypotheses}
    assert ids == {
        "vss402-control",
        "vss402-exact-closure",
        "vss402-capsules",
        "vss402-energy",
        "vss402-late-realization",
    }
    # At least one regime-transition candidate (the learned energy ranker).
    assert any(
        c.novelty.transition_kind == "regime_transition_candidate"
        for c in matrix.hypotheses
    )


def test_vss_knobs_compile_to_bounded_grammar_diffusion_commands() -> None:
    campaign = build_vss_campaign()
    matrix = build_vss_matrix()
    by_id = {c.experiment.experiment_id: c.experiment for c in matrix.hypotheses}
    commands = {
        eid: compile_commands(campaign, experiment)
        for eid, experiment in by_id.items()
    }
    train = {eid: cmds[-2] for eid, cmds in commands.items()}
    # Typed scope/topology knobs -> bounded flags on the grammar_diffusion track.
    assert "--model" in train["vss402-exact-closure"]
    assert "--scope-contracts" in train["vss402-exact-closure"]
    assert "--scope-local-oracle" in train["vss402-exact-closure"]
    assert "--scope-contracts" not in train["vss402-control"]
    assert "--topology-actions" in train["vss402-capsules"]
    assert "--scope-contract-negatives" in train["vss402-energy"]
    # Evaluation always appends the honest ship gates; nothing weakens them.
    for eid, cmds in commands.items():
        evaluate = cmds[-1]
        assert "--ship-gates" in evaluate
    # No researcher-authored shell anywhere: every token is a plain string.
    for cmds in commands.values():
        for argv in cmds:
            assert all(isinstance(token, str) for token in argv)
            assert not any(ch in token for token in argv for ch in ";|&$`")


def test_vss_scope_knobs_require_programspec_data_source() -> None:
    # Strict schema: scope knobs without a ProgramSpec-backed source are rejected.
    with pytest.raises(ValidationError):
        ExperimentKnobs(scope_contracts=True)
    # Unknown fields are still rejected outright.
    with pytest.raises(ValidationError):
        ExperimentKnobs.model_validate({"scope_contracts": True, "shell": "x"})


def test_vss_forbidden_knobs_rejected_by_restricted_campaign() -> None:
    campaign = build_vss_campaign().model_copy(
        update={"allowed_knobs": frozenset({"steps"})}
    )
    matrix = build_vss_matrix()
    evidence = build_vss_evidence()
    sources = build_vss_sources()
    with pytest.raises(ValueError, match="forbidden"):
        validate_experiment(
            campaign,
            matrix.hypotheses[1].experiment,  # the exact-closure candidate
            evidence,
            sources,
        )


def test_vss_feedback_ack_round_trip() -> None:
    campaign = build_vss_campaign()
    evidence = build_vss_evidence()
    sources = build_vss_sources()
    matrix = build_vss_matrix()
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
        evidence=("fixture wiring reproduces the closed benchmark",),
        recommended_actions=("execute the frontier campaign in VSS4-03",),
    )
    feedback = create_hypothesis_feedback(matrix, outcome, diagnosis)
    assert feedback.matrix_id == MATRIX_ID

    executed_spec = next(
        c.experiment
        for c in matrix.hypotheses
        if c.experiment.experiment_id == executed
    )
    successor = matrix.model_copy(
        update={
            "matrix_id": "verified-scope-solver-m2",
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
            "recommended_experiment_id": "m2-vss402-exact-closure",
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
    with pytest.raises(ValueError, match="acknowledge"):
        validate_hypothesis_matrix(
            campaign,
            successor.model_copy(update={"feedback_ids": ()}),
            evidence,
            sources,
            feedback=(feedback,),
            previous_matrix=matrix,
        )
