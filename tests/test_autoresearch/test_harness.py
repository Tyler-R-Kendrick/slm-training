from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.engine import (
    compile_commands,
    create_hypothesis_feedback,
    diagnose_outcome,
    execute_commands,
    validate_experiment,
    validate_hypothesis_matrix,
)
from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignDeviationV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.evidence import collect_evidence
from slm_training.autoresearch.literature import HuggingFacePapersClient
from slm_training.levers import MAX_RUN_MINUTES, MAX_RUN_SECONDS
from slm_training.autoresearch.providers import (
    OpenAIHypothesizer,
    OpenAIProposalCompiler,
    OpenAIResearchProvider,
)
from slm_training.autoresearch.researchers import IsolatedResearcher, ResearcherSpec
from slm_training.autoresearch.persistence import sync_campaign
from slm_training.autoresearch.rl_gate import (
    assert_rl_ready,
    assess_rl_readiness,
    write_rl_readiness,
)
from slm_training.autoresearch.schemas import (
    AutotrainActionV1,
    AutotrainCycleHandoffV1,
    CampaignBudget,
    CampaignSpec,
    CategoricalNoveltyAudit,
    Diagnosis,
    EvidenceUse,
    EvidenceItem,
    EvidenceSnapshot,
    ExperimentKnobs,
    ExperimentOutcome,
    ExperimentSpec,
    HarnessSignalV1,
    HypothesisCandidate,
    HypothesisFeedback,
    HypothesisMatrix,
    NextRunPriorityV1,
    OpenDeepResearchConfig,
    OpenResearcherConfig,
    OptimumFeedbackV1,
    ResearchSource,
)
from slm_training.autoresearch.storage import (
    CampaignStore,
    pending_autotrain_actions,
    render_loop_result_matrix,
)


def campaign() -> CampaignSpec:
    return CampaignSpec(
        campaign_id="test-campaign",
        objective="Improve honest held-out structural similarity.",
        primary_metric="held_out.structural_similarity",
        researcher_mode="fixture",
    )


def experiment_campaign(**overrides) -> ExperimentCampaignV1:
    payload = {
        "campaign_id": "test-campaign",
        "experiment_id": "hyp-0",
        "hypothesis": "More supervised steps improve held-out structure.",
        "decision": "Promote only after the declared endpoint clears its gate.",
        "endpoints": (
            CampaignEndpointV1(
                endpoint_id="primary",
                metric="binder_reference_f1",
                role="primary",
                direction="increase",
                minimum_effect=0.01,
            ),
        ),
        "arms": (
            CampaignArmV1(
                arm_id="control",
                role="control",
                config_sha256="a" * 64,
            ),
            CampaignArmV1(
                arm_id="candidate",
                role="candidate",
                config_sha256="b" * 64,
            ),
        ),
        "seeds": (7, 11),
        "budget": CampaignBudget(
            max_experiments=1,
            max_wall_minutes=MAX_RUN_MINUTES,
        ),
        "stopping_rules": ("Stop after both declared seeds finish.",),
        "controls": (
            CampaignControlV1(
                control_id="matched-control",
                description="Matched baseline with the same evaluation recipe.",
                kind="negative",
            ),
        ),
        "negative_controls": ("matched-control",),
        "multiplicity_families": (
            MultiplicityFamilyV1(
                family_id="primary-family",
                hypothesis_ids=("primary",),
                alpha=0.05,
            ),
        ),
        "promotion_gates": (
            CampaignGateV1(
                gate_id="promote-primary",
                endpoint_id="primary",
                operator="ge",
                threshold=0.01,
            ),
        ),
        "rollback_gates": (
            CampaignGateV1(
                gate_id="rollback-primary",
                endpoint_id="primary",
                operator="lt",
                threshold=0,
            ),
        ),
        "artifact_requirements": (ArtifactRequirementV1(kind="version_stamp"),),
        "claim_class": "diagnostic",
        "source_commit": "c" * 40,
        "source_dirty": False,
        "author": "test",
    }
    payload.update(overrides)
    return ExperimentCampaignV1(**payload)


def experiment(**overrides) -> ExperimentSpec:
    payload = {
        "experiment_id": "exp-1",
        "campaign_id": "test-campaign",
        "hypothesis": "More supervised steps improve held-out structure.",
        "rationale": "Prior run evidence shows a stable validation decline.",
        "expected_effect": "Positive held-out structural delta.",
        "falsification_criteria": ("No improvement against the matched control.",),
        "stop_conditions": ("Stop at 300 steps.",),
        "citations": ("fixture://prior-run",),
        "knobs": ExperimentKnobs(steps=300),
    }
    payload.update(overrides)
    return ExperimentSpec(**payload)


def evidence() -> EvidenceSnapshot:
    return EvidenceSnapshot(
        snapshot_id="evidence-test",
        roots=("outputs",),
        items=(
            EvidenceItem(
                path="fixture://prior-run",
                kind="prior_run",
                sha256="a" * 64,
                size_bytes=10,
            ),
        ),
    )


def source() -> ResearchSource:
    return ResearchSource(
        source_id="prior",
        kind="prior_run",
        title="Prior run",
        uri="fixture://prior-run",
    )


def matrix_evidence() -> EvidenceSnapshot:
    return EvidenceSnapshot(
        snapshot_id="evidence-matrix",
        roots=("outputs",),
        items=(
            EvidenceItem(
                path="docs/design/research-lineage.md",
                kind="repo_lineage",
                sha256="a" * 64,
                size_bytes=10,
            ),
            EvidenceItem(
                path="outputs/runs/prior/run_insights.json",
                kind="run_insight",
                sha256="b" * 64,
                size_bytes=10,
            ),
            EvidenceItem(
                path="outputs/runs/prior/scoreboard.json",
                kind="evaluation",
                sha256="c" * 64,
                size_bytes=10,
            ),
        ),
    )


def hypothesis_matrix(
    count: int = 5,
    *,
    matrix_id: str = "matrix-1",
    predecessor_matrix_id: str | None = None,
    feedback_ids: tuple[str, ...] = (),
    offset: int = 0,
    diagnosis_lanes: tuple[str, ...] = (),
    campaign_id: str = "test-campaign",
    next_run_priorities: tuple[NextRunPriorityV1, ...] = (),
) -> HypothesisMatrix:
    citations = (
        "docs/design/research-lineage.md",
        "outputs/runs/prior/run_insights.json",
        "outputs/runs/prior/scoreboard.json",
    )
    candidates = []
    for index in range(count):
        number = index + offset
        candidates.append(
            HypothesisCandidate(
                experiment=experiment(
                    experiment_id=f"hyp-{number}",
                    campaign_id=campaign_id,
                    hypothesis=f"Distinct grounded hypothesis number {number} improves structure.",
                    citations=citations,
                    knobs=ExperimentKnobs(steps=100 + number),
                ),
                evidence_uses=(
                    EvidenceUse(
                        role="research",
                        citation=citations[0],
                        contribution="Defines relevant prior methods.",
                    ),
                    EvidenceUse(
                        role="prior_trace",
                        citation=citations[1],
                        contribution="Identifies the observed failure mode.",
                    ),
                    EvidenceUse(
                        role="prior_result",
                        citation=citations[2],
                        contribution="Supplies the matched baseline result.",
                    ),
                ),
                novelty=CategoricalNoveltyAudit(
                    transition_kind=(
                        "regime_transition_candidate"
                        if index == 0
                        else "fixed_regime_search"
                    ),
                    old_schema_elements=("training recipe",),
                    proposed_schema_elements=(
                        "new evidence relation" if index == 0 else "training recipe",
                    ),
                    transported_elements=("prior scoreboard",),
                    transport_analysis=(
                        "Declared residual is not generated by the baseline recipe.",
                    ),
                    residual_elements=(f"candidate mechanism {number}",),
                    preservation_checks=("rerun the matched control",),
                    stress_tests=("evaluate every honest suite",),
                    worthiness_criteria=("positive primary delta without regression",),
                ),
                diagnosis_lane=(
                    diagnosis_lanes[index] if index < len(diagnosis_lanes) else None
                ),
            )
        )
    return HypothesisMatrix(
        matrix_id=matrix_id,
        campaign_id=campaign_id,
        evidence_snapshot_id="evidence-matrix",
        hypotheses=tuple(candidates),
        recommended_experiment_id=f"hyp-{offset}",
        selection_rationale="Highest-information safe candidate in this matrix.",
        predecessor_matrix_id=predecessor_matrix_id,
        feedback_ids=feedback_ids,
        next_run_priorities=next_run_priorities,
    )


def next_priority(
    *,
    experiment_id: str = "hyp-0",
    evidence_ids: tuple[str, ...] = ("fixture://prior-run",),
    area: str = "model",
    authority: str = "speculative",
    disposition: str = "experiment_next",
    rank: int = 1,
) -> NextRunPriorityV1:
    return NextRunPriorityV1(
        rank=rank,
        area=area,
        hypothesis=f"Test the {area} lane as the next bounded intervention.",
        evidence_ids=evidence_ids,
        confidence=0.5,
        expected_information_gain="Distinguishes the leading failure mechanisms.",
        authority=authority,
        disposition=disposition,
        proposed_experiment_id=(
            experiment_id if disposition == "experiment_next" else None
        ),
    )


def test_out_of_band_feedback_requires_complete_diagnosis_matrix() -> None:
    first = hypothesis_matrix()
    optimum = OptimumFeedbackV1(
        policy="block_promotion_and_diagnose",
        breaches=(
            {
                "metric_id": "tokens_per_step",
                "authority": "assumption_backed",
                "relation": "above",
            },
        ),
        diagnosis_lanes=(
            "measurement_control",
            "training_method",
            "architecture",
            "lean_model",
            "assumptions",
        ),
        metric_evidence_sha256="a" * 64,
        metric_certificate_sha256="b" * 64,
    )
    feedback = HypothesisFeedback(
        feedback_id="feedback-aaaaaaaaaaaaaaaa",
        campaign_id="test-campaign",
        matrix_id=first.matrix_id,
        experiment_id="hyp-0",
        hypothesis=first.hypotheses[0].experiment.hypothesis,
        knob_signature='{"steps": 100}',
        outcome_status="completed",
        diagnosis_target="none",
        diagnosis_evidence=("Observed metric exceeded its calculated band.",),
        recommended_actions=("Run the five-lane diagnosis matrix.",),
        optimum_feedback=optimum,
    )
    incomplete = hypothesis_matrix(
        matrix_id="matrix-2",
        predecessor_matrix_id=first.matrix_id,
        feedback_ids=(feedback.feedback_id,),
        offset=10,
    )

    with pytest.raises(ValueError, match="diagnosis lane"):
        validate_hypothesis_matrix(
            campaign(),
            incomplete,
            matrix_evidence(),
            [source()],
            feedback=(feedback,),
            previous_matrix=first,
        )

    complete = hypothesis_matrix(
        matrix_id="matrix-2",
        predecessor_matrix_id=first.matrix_id,
        feedback_ids=(feedback.feedback_id,),
        offset=10,
        diagnosis_lanes=optimum.diagnosis_lanes,
    )
    validate_hypothesis_matrix(
        campaign(),
        complete,
        matrix_evidence(),
        [source()],
        feedback=(feedback,),
        previous_matrix=first,
    )


def passing_evaluation() -> dict:
    return {
        "evaluation_snapshot": {
            "metadata": {
                "kind": "frozen_production_evaluation",
                "suite_sizes": {"rico_held": 1500},
                "human_feedback_holdout_n": 10,
            }
        },
        "suites": {
            "smoke": {
                "n": 32,
                "parse_rate": 1,
                "structural_similarity": 1,
                "component_type_recall": 1,
                "fallback_count": 0,
                "placeholder_fidelity": 1,
                "reward_score": 1,
            },
            "held_out": {
                "n": 32,
                "parse_rate": 1,
                "structural_similarity": 1,
                "component_type_recall": 1,
                "fallback_count": 0,
                "placeholder_fidelity": 1,
            },
            "adversarial": {
                "n": 32,
                "parse_rate": 1,
                "structural_similarity": 1,
                "component_type_recall": 1,
                "fallback_count": 0,
            },
            "ood": {
                "n": 32,
                "parse_rate": 1,
                "structural_similarity": 1,
                "component_type_recall": 1,
                "fallback_count": 0,
            },
            "rico_held": {
                "n": 1500,
                "parse_rate": 1,
                "structural_similarity": 1,
                "component_type_recall": 1,
                "fallback_count": 0,
            },
        },
        "evals": {
            "criteria": {"pass": True, "total": 1},
            "runner": {"name": "AgentV", "execution_errors": 0},
        },
        "gates": {
            "authority": "AgentEvals assertions",
            "pass": True,
        },
        "reward_samples": [0.1, 0.4, 0.8],
    }


def test_strict_schema_and_allowlist() -> None:
    with pytest.raises(ValidationError):
        ExperimentKnobs.model_validate({"steps": 10, "shell": "rm -rf /"})
    with pytest.raises(ValidationError, match="derive_from"):
        ExperimentKnobs(data_source="existing")
    restricted = campaign().model_copy(update={"allowed_knobs": frozenset({"lr"})})
    with pytest.raises(ValueError, match="forbidden"):
        validate_experiment(restricted, experiment(), evidence(), [source()])


def test_hypothesis_matrix_requires_five_distinct_grounded_candidates() -> None:
    with pytest.raises(ValidationError, match="at least 5 items"):
        hypothesis_matrix(4)
    matrix = hypothesis_matrix()
    validate_hypothesis_matrix(campaign(), matrix, matrix_evidence(), [])
    with pytest.raises(ValueError, match="previously run"):
        validate_hypothesis_matrix(
            campaign(),
            matrix,
            matrix_evidence(),
            [],
            prior_experiments=(matrix.hypotheses[0].experiment,),
        )
    duplicate = matrix.hypotheses[0].model_copy(
        update={
            "experiment": matrix.hypotheses[0].experiment.model_copy(
                update={"experiment_id": "different-id"}
            )
        }
    )
    with pytest.raises(ValidationError, match="knob signatures"):
        HypothesisMatrix(
            matrix_id="duplicate",
            campaign_id="test-campaign",
            evidence_snapshot_id="evidence-matrix",
            hypotheses=(*matrix.hypotheses[:4], duplicate),
            recommended_experiment_id="hyp-0",
            selection_rationale="Duplicate fixture should fail before selection.",
        )


def test_feedback_requires_lineage_and_informs_next_matrix() -> None:
    first = hypothesis_matrix()
    outcome = ExperimentOutcome(
        experiment_id="hyp-0",
        campaign_id="test-campaign",
        status="completed",
        metrics={"held_out.structural_similarity": 0.2},
    )
    diagnosis = Diagnosis(
        experiment_id="hyp-0",
        target="model",
        confidence=0.8,
        evidence=("Held-out structure remained below the gate.",),
        recommended_actions=("Test a matched model repair.",),
    )
    feedback = create_hypothesis_feedback(first, outcome, diagnosis)
    second = hypothesis_matrix(
        matrix_id="matrix-2",
        predecessor_matrix_id="matrix-1",
        feedback_ids=(feedback.feedback_id,),
    )
    validate_hypothesis_matrix(
        campaign(),
        second,
        matrix_evidence(),
        [],
        feedback=(feedback,),
        previous_matrix=first,
    )
    with pytest.raises(ValueError, match="acknowledge"):
        validate_hypothesis_matrix(
            campaign(),
            hypothesis_matrix(matrix_id="matrix-3"),
            matrix_evidence(),
            [],
            feedback=(feedback,),
            previous_matrix=first,
        )
    foreign = feedback.model_copy(update={"campaign_id": "other-campaign"})
    with pytest.raises(ValueError, match="does not belong"):
        validate_hypothesis_matrix(
            campaign(),
            second,
            matrix_evidence(),
            [],
            feedback=(foreign,),
            previous_matrix=first,
        )


def test_successor_rejects_campaign_wide_experiment_id_reuse() -> None:
    with pytest.raises(ValueError, match="reuses campaign experiment ids"):
        validate_hypothesis_matrix(
            campaign(),
            hypothesis_matrix(matrix_id="matrix-2", offset=10),
            matrix_evidence(),
            [],
            prior_experiment_ids=frozenset({"hyp-10"}),
        )


def test_predecessor_requires_feedback_acknowledgment() -> None:
    with pytest.raises(ValidationError, match="must acknowledge feedback_ids"):
        hypothesis_matrix(matrix_id="matrix-2", predecessor_matrix_id="matrix-1")


def test_candidate_floor_is_independent_of_execution_budget() -> None:
    spec = CampaignSpec(
        campaign_id="focused-budget",
        objective="Choose one experiment from a broad hypothesis matrix.",
        primary_metric="score",
        min_hypotheses=6,
        budget={"max_experiments": 1},
    )
    assert spec.min_hypotheses == 6
    assert spec.budget.max_experiments == 1


def test_campaign_store_is_content_addressed_and_chained(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    spec = experiment()
    first = store.write_artifact("experiments", spec)
    second = store.write_artifact("experiments", spec)
    assert first == second
    event = store.append_event("experiment_proposed", artifact_sha256=first.stem)
    lines = [
        json.loads(line)
        for line in (store.root / "events.jsonl").read_text().splitlines()
    ]
    assert lines[-1]["event_id"] == event["event_id"]
    assert lines[-1]["previous_event_sha256"] == lines[-2]["event_id"]
    assert (store.root / "checksums.jsonl").is_file()
    assert (store.root / "results.tsv").is_file()


def test_campaign_store_locks_manifest_idempotently(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    manifest = experiment_campaign()

    first = store.lock_experiment_campaign(manifest)
    second = store.lock_experiment_campaign(manifest)

    assert first == second
    assert second.manifest_sha256 == campaign_manifest_sha256(manifest)
    lock_events = [
        event
        for event in store.verify_event_chain()
        if event["event_type"] == "experiment_campaign_locked"
    ]
    assert len(lock_events) == 1


def test_campaign_store_rejects_different_relock_and_artifact_mutation(
    tmp_path: Path,
) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    manifest = experiment_campaign()
    store.lock_experiment_campaign(manifest)

    with pytest.raises(FileExistsError, match="different content"):
        store.lock_experiment_campaign(
            manifest.model_copy(
                update={"hypothesis": "A different preregistered hypothesis."}
            )
        )

    lock_event = next(
        event
        for event in store.verify_event_chain()
        if event["event_type"] == "experiment_campaign_locked"
    )
    lock_path = (
        store.root
        / "artifacts"
        / "experiment_campaigns"
        / f"{lock_event['artifact_sha256']}.json"
    )
    payload = json.loads(lock_path.read_text())
    payload["manifest"]["hypothesis"] = "A post-result mutation is invalid."
    lock_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="content digest mismatch"):
        store.load_experiment_campaign(manifest.experiment_id)


def test_campaign_store_requires_lock_before_experiment_start(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    store.append_event(
        "experiment_started",
        experiment_id="hyp-0",
        status="running",
    )

    with pytest.raises(RuntimeError, match="after experiment outcome access"):
        store.lock_experiment_campaign(experiment_campaign())


def test_campaign_store_rejects_lock_after_finish_without_start(
    tmp_path: Path,
) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    store.append_event(
        "experiment_finished",
        experiment_id="hyp-0",
        status="failed",
    )
    with pytest.raises(RuntimeError, match="after experiment outcome access"):
        store.lock_experiment_campaign(experiment_campaign())


def test_campaign_store_serializes_conflicting_locks(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    first = experiment_campaign()
    second = first.model_copy(
        update={"hypothesis": "A conflicting preregistered hypothesis is rejected."}
    )

    def attempt(manifest: ExperimentCampaignV1) -> str:
        try:
            return store.lock_experiment_campaign(manifest).manifest_sha256
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, (first, second)))
    assert results.count("rejected") == 1
    assert (
        len(
            [
                event
                for event in store.verify_event_chain()
                if event["event_type"] == "experiment_campaign_locked"
            ]
        )
        == 1
    )


def test_campaign_store_detects_event_chain_tampering(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    store.lock_experiment_campaign(experiment_campaign())
    events_path = store.root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[-1]["status"] = "edited"
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    with pytest.raises(RuntimeError, match="event_id digest mismatch"):
        store.verify_event_chain()


def test_campaign_deviation_is_exploratory_and_does_not_replace_lock(
    tmp_path: Path,
) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    manifest = experiment_campaign()
    lock = store.lock_experiment_campaign(manifest)
    deviation = CampaignDeviationV1(
        campaign_id="test-campaign",
        experiment_id="hyp-0",
        manifest_sha256=lock.manifest_sha256,
        changed_field="budget.max_experiments",
        old_value=1,
        new_value=2,
        reason="Preserve the unexpected follow-up as exploratory evidence.",
        author="test",
        outcome_accessed=True,
    )

    deviation_path = store.append_campaign_deviation(deviation)

    assert deviation_path.is_file()
    assert store.load_experiment_campaign("hyp-0") == lock
    event = store.verify_event_chain()[-1]
    assert event["event_type"] == "campaign_deviation_appended"
    assert event["status"] == "exploratory"


def test_run_requires_exact_member_of_latest_hypothesis_matrix(tmp_path: Path) -> None:
    from scripts.autoresearch import _require_hypothesis_matrix

    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    matrix = hypothesis_matrix()
    path = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event(
        "hypothesis_matrix_formed", artifact_sha256=path.stem, status="planned"
    )
    assert (
        _require_hypothesis_matrix(
            store, campaign(), matrix.hypotheses[0].experiment
        ).matrix_id
        == "matrix-1"
    )
    with pytest.raises(ValueError, match="exact member"):
        _require_hypothesis_matrix(store, campaign(), experiment())


def test_run_defaults_to_matrix_recommendation(tmp_path: Path) -> None:
    from scripts.autoresearch import cmd_run

    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    matrix = hypothesis_matrix()
    path = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event(
        "hypothesis_matrix_formed", artifact_sha256=path.stem, status="planned"
    )
    assert (
        cmd_run(
            SimpleNamespace(
                campaign_id="test-campaign",
                root=tmp_path,
                experiment=None,
                execute=False,
                trackio=False,
            )
        )
        == 0
    )
    events = json.loads((store.root / "events.jsonl").read_text().splitlines()[-1])
    assert events["event_type"] == "execution_planned"
    assert events["experiment_id"] == matrix.recommended_experiment_id


def test_run_execution_requires_lock_and_dry_plan_binds_manifest(
    tmp_path: Path,
) -> None:
    from scripts.autoresearch import cmd_run

    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    matrix = hypothesis_matrix()
    matrix_path = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event(
        "hypothesis_matrix_formed",
        artifact_sha256=matrix_path.stem,
        status="planned",
    )
    with pytest.raises(ValueError, match="requires a preregistered"):
        cmd_run(
            SimpleNamespace(
                campaign_id="test-campaign",
                root=tmp_path,
                experiment=None,
                campaign_manifest=None,
                execute=True,
                trackio=False,
            )
        )

    manifest = experiment_campaign()
    manifest_path = tmp_path / "campaign-manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    assert (
        cmd_run(
            SimpleNamespace(
                campaign_id="test-campaign",
                root=tmp_path,
                experiment=None,
                campaign_manifest=manifest_path,
                execute=False,
                trackio=False,
            )
        )
        == 0
    )
    plan_event = store.verify_event_chain()[-1]
    plan_path = (
        store.root
        / "artifacts"
        / "execution_plans"
        / f"{plan_event['artifact_sha256']}.json"
    )
    plan = json.loads(plan_path.read_text())
    assert plan["campaign_manifest_sha256"] == campaign_manifest_sha256(manifest)


def test_execution_budget_counts_started_experiments_not_matrix_rows(
    tmp_path: Path,
) -> None:
    from scripts.autoresearch import cmd_run

    spec = campaign()
    spec = spec.model_copy(
        update={"budget": spec.budget.model_copy(update={"max_experiments": 1})}
    )
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(spec)
    matrix = hypothesis_matrix()
    path = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event(
        "hypothesis_matrix_formed", artifact_sha256=path.stem, status="planned"
    )
    store.append_event(
        "experiment_started", experiment_id="already-ran", status="running"
    )
    store.lock_experiment_campaign(experiment_campaign())
    with pytest.raises(ValueError, match="execution budget exhausted"):
        cmd_run(
            SimpleNamespace(
                campaign_id="test-campaign",
                root=tmp_path,
                experiment=None,
                execute=True,
                trackio=False,
            )
        )


def test_terminal_outcome_persists_hypothesizer_feedback(tmp_path: Path) -> None:
    from scripts.autoresearch import _record_hypothesis_feedback

    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    matrix = hypothesis_matrix()
    outcome = ExperimentOutcome(
        experiment_id="hyp-0",
        campaign_id="test-campaign",
        status="failed",
        error="held-out regression",
    )
    diagnosis = diagnose_outcome(outcome)
    path = _record_hypothesis_feedback(store, matrix, outcome, diagnosis)
    feedback = json.loads(path.read_text())
    assert feedback["matrix_id"] == matrix.matrix_id
    assert feedback["hypothesis"] == matrix.hypotheses[0].experiment.hypothesis
    assert path.parent.name == "hypothesizer_feedback"


def test_outcome_matrix_resolution_uses_recorded_run_provenance(tmp_path: Path) -> None:
    from scripts.autoresearch import _matrix_for_outcome

    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    first = hypothesis_matrix()
    first_path = store.write_artifact("hypothesis_matrices", first)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=first_path.stem)
    outcome = ExperimentOutcome(
        experiment_id="hyp-0", campaign_id="test-campaign", status="completed"
    )
    outcome_path = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_started",
        experiment_id="hyp-0",
        detail={"hypothesis_matrix_id": first.matrix_id},
    )
    store.append_event(
        "experiment_finished",
        experiment_id="hyp-0",
        artifact_sha256=outcome_path.stem,
    )
    second = hypothesis_matrix(matrix_id="matrix-2", offset=10)
    second_path = store.write_artifact("hypothesis_matrices", second)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=second_path.stem)

    assert _matrix_for_outcome(store, outcome) == first


def test_agent_hypothesizer_persists_matrix_and_formation_event(
    tmp_path: Path,
) -> None:
    from scripts.autoresearch import cmd_hypothesize

    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    store.write_artifact("evidence", matrix_evidence())
    sources = store.write_artifact("research_sources", {"sources": []})
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(hypothesis_matrix().model_dump_json(), encoding="utf-8")
    assert (
        cmd_hypothesize(
            SimpleNamespace(
                campaign_id="test-campaign",
                root=tmp_path,
                evidence=None,
                sources=sources,
                provider="agent",
                matrix=matrix_path,
                model="unused",
                memo=None,
            )
        )
        == 0
    )
    events = [
        json.loads(line)
        for line in (store.root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(row["event_type"] == "experiment_proposed" for row in events) == 5
    assert any(row["event_type"] == "hypothesis_matrix_formed" for row in events)


def test_hypothesizer_forms_feedback_linked_successor_matrix(tmp_path: Path) -> None:
    from scripts.autoresearch import _record_hypothesis_feedback, cmd_hypothesize

    store = CampaignStore("test-campaign", tmp_path)
    base_campaign = campaign()
    store.initialize(
        base_campaign.model_copy(
            update={
                "budget": base_campaign.budget.model_copy(update={"max_experiments": 5})
            }
        )
    )
    store.write_artifact("evidence", matrix_evidence())
    sources = store.write_artifact("research_sources", {"sources": []})
    first = hypothesis_matrix()
    first_path = store.write_artifact("hypothesis_matrices", first)
    store.append_event(
        "hypothesis_matrix_formed", artifact_sha256=first_path.stem, status="planned"
    )
    for candidate in first.hypotheses:
        store.write_artifact("experiments", candidate.experiment)
    outcome = ExperimentOutcome(
        experiment_id="hyp-0",
        campaign_id="test-campaign",
        status="failed",
        error="held-out regression",
    )
    store.append_event("experiment_finished", experiment_id="hyp-0", status="failed")
    feedback_path = _record_hypothesis_feedback(
        store, first, outcome, diagnose_outcome(outcome)
    )
    feedback_id = json.loads(feedback_path.read_text())["feedback_id"]
    second = hypothesis_matrix(
        matrix_id="matrix-2",
        predecessor_matrix_id="matrix-1",
        feedback_ids=(feedback_id,),
        offset=10,
    )
    matrix_path = tmp_path / "matrix-2.json"
    matrix_path.write_text(second.model_dump_json(), encoding="utf-8")
    assert (
        cmd_hypothesize(
            SimpleNamespace(
                campaign_id="test-campaign",
                root=tmp_path,
                evidence=None,
                sources=sources,
                provider="agent",
                matrix=matrix_path,
                model="unused",
                memo=None,
            )
        )
        == 0
    )
    formed = [
        row
        for row in (
            json.loads(line)
            for line in store.root.joinpath("events.jsonl").read_text().splitlines()
        )
        if row["event_type"] == "hypothesis_matrix_formed"
    ]
    assert len(formed) == 2
    assert formed[-1]["detail"]["recommended_experiment_id"] == "hyp-10"


def test_evidence_normalizes_feedback_telemetry_and_lineage(tmp_path: Path) -> None:
    (tmp_path / "docs/design").mkdir(parents=True)
    (tmp_path / "docs/design/research-lineage.md").write_text("# lineage\nPrior result")
    outputs = tmp_path / "outputs/run-1"
    outputs.mkdir(parents=True)
    (outputs / "train_telemetry.json").write_text(json.dumps({"loss": 1.2}))
    (outputs / "human_feedback.jsonl").write_text('{"reward":0.5}\n')
    feedback_dir = (
        tmp_path / "outputs/autoresearch/prior/artifacts/hypothesizer_feedback"
    )
    feedback_dir.mkdir(parents=True)
    feedback_dir.joinpath("feedback.json").write_text(
        HypothesisFeedback(
            feedback_id="feedback-aaaaaaaaaaaaaaaa",
            campaign_id="prior",
            matrix_id="matrix-1",
            experiment_id="exp-1",
            hypothesis="A failed learning-rate hypothesis informs the next matrix.",
            knob_signature='{"lr": 0.1}',
            outcome_status="failed",
            diagnosis_target="model",
            diagnosis_evidence=("Held-out score regressed.",),
            recommended_actions=("Lower the learning rate.",),
        ).model_dump_json(),
        encoding="utf-8",
    )
    snapshot = collect_evidence(["outputs"], repo_root=tmp_path)
    assert snapshot.source_counts["repo_lineage"] == 1
    assert snapshot.source_counts["telemetry"] == 1
    assert snapshot.source_counts["feedback"] == 2
    feedback = next(
        item for item in snapshot.items if "hypothesizer_feedback" in item.path
    )
    assert "Lower the learning rate" in feedback.summary
    assert snapshot.snapshot_id.startswith("evidence-")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTTP:
    def get(self, url, **kwargs):
        return FakeResponse(
            [
                {
                    "paper": {
                        "id": "2601.00001",
                        "title": "A paper",
                        "summary": "Useful",
                    },
                    "numUpvotes": 5,
                }
            ]
        )


def test_hf_daily_papers_client_preserves_sources() -> None:
    client = HuggingFacePapersClient(client=FakeHTTP())
    rows = client.daily(days=1, limit_per_day=2)
    assert rows[0].kind == "hf_daily_paper"
    assert rows[0].uri.endswith("2601.00001")


def test_research_always_captures_categorical_discovery_source() -> None:
    from scripts.autoresearch import _research_sources

    rows = _research_sources(SimpleNamespace(source_manifest=(), offline=True))
    assert [row.uri for row in rows] == ["https://arxiv.org/abs/2606.01444"]


def test_init_explicit_evidence_roots_replace_legacy_default(tmp_path: Path) -> None:
    from scripts.autoresearch import build_parser

    parser = build_parser()
    explicit = tmp_path / "prior-campaign"
    args = parser.parse_args(
        [
            "--root",
            str(tmp_path / "store"),
            "init",
            "--campaign-id",
            "explicit-roots",
            "--objective",
            "Use only bounded predecessor evidence.",
            "--primary-metric",
            "score",
            "--evidence-root",
            str(explicit),
        ]
    )
    assert args.func(args) == 0
    campaign_spec = CampaignStore("explicit-roots", tmp_path / "store").load_campaign()
    assert campaign_spec.evidence_roots == (str(explicit),)

    legacy = parser.parse_args(
        [
            "--root",
            str(tmp_path / "store"),
            "init",
            "--campaign-id",
            "legacy-root",
            "--objective",
            "Retain compatibility when no explicit root is supplied.",
            "--primary-metric",
            "score",
        ]
    )
    assert legacy.func(legacy) == 0
    legacy_spec = CampaignStore("legacy-root", tmp_path / "store").load_campaign()
    assert legacy_spec.evidence_roots == ("outputs",)


def test_ack_action_receipt_closes_predecessor_prerequisite(tmp_path: Path) -> None:
    from scripts.autoresearch import build_parser

    root = tmp_path / "autoresearch"
    campaign_id = "cycle-1"
    handoff = AutotrainCycleHandoffV1(
        loop_id="loop-1",
        campaign_id=campaign_id,
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="rejected",
        ship_state="blocked",
        primary_metric="smoke.parse_rate",
        actions=(
            AutotrainActionV1(
                kind="document",
                owner="documenting-experiment-results",
                reason="Persist the negative result.",
                evidence_ids=(f"campaign:{campaign_id}",),
            ),
            AutotrainActionV1(
                kind="next_experiment",
                owner="autotrain",
                reason="Continue the bounded loop.",
                evidence_ids=(f"campaign:{campaign_id}",),
            ),
        ),
    )
    handoff_path = root / campaign_id / "cycle_handoff.json"
    handoff_path.parent.mkdir(parents=True)
    handoff_path.write_text(handoff.model_dump_json(indent=2) + "\n")
    assert [action.kind for _, action in pending_autotrain_actions(root, handoff)] == [
        "document"
    ]

    invalid_args = build_parser().parse_args(
        [
            "--root",
            str(root),
            "ack-action",
            "--loop-id",
            "loop-1",
            "--campaign-id",
            campaign_id,
            "--action-index",
            "0",
            "--evidence",
            "done",
        ]
    )
    with pytest.raises(ValueError, match="existing durable path or commit"):
        invalid_args.func(invalid_args)

    args = build_parser().parse_args(
        [
            "--root",
            str(root),
            "ack-action",
            "--loop-id",
            "loop-1",
            "--campaign-id",
            campaign_id,
            "--action-index",
            "0",
            "--evidence",
            "docs/design/autoresearch-autotraining.md",
        ]
    )
    assert args.func(args) == 0
    assert pending_autotrain_actions(root, handoff) == ()

    stopped_handoff = handoff.model_copy(
        update={
            "actions": (
                AutotrainActionV1(
                    kind="stop_campaign",
                    owner="autotrain",
                    reason="A theorem disproved the campaign assumption.",
                    evidence_ids=(f"campaign:{campaign_id}",),
                ),
            )
        }
    )
    assert [
        action.kind for _, action in pending_autotrain_actions(root, stopped_handoff)
    ] == ["stop_campaign"]


def test_dynamic_symbol_source_manifest_is_complete() -> None:
    from scripts.autoresearch import _load_sources

    path = Path("src/slm_training/resources/autoresearch/dynamic-symbol-sources.json")
    rows = _load_sources(path)
    assert len(rows) == 50
    assert len({row.uri for row in rows}) == 50
    assert sum(row.uri.startswith("https://arxiv.org/abs/") for row in rows) == 49


def test_scope_diffusion_source_manifest_is_complete() -> None:
    from scripts.autoresearch import _load_sources

    path = Path("src/slm_training/resources/autoresearch/scope-diffusion-sources.json")
    manifest = json.loads(path.read_text())
    rows = _load_sources(path)
    assert "6a583787-8e9c-83ea-94e2-c36b0f4d093e" in manifest["source_scope"]
    assert len(rows) == 19
    assert len({row.uri for row in rows}) == 19
    assert all(row.uri.startswith("https://arxiv.org/abs/") for row in rows)
    assert all(row.metadata.get("scope_diff_takeaway") for row in rows)
    # "Faithful (mechanism)" is the Kapur tree-diffusion status after D3
    # (SLM-31) reproduced the paper's mechanism (models/tree_edit_diffusion.py).
    assert {row.metadata.get("implementation_status") for row in rows} == {
        "Adapted",
        "Adjacent",
        "Faithful (mechanism)",
    }


def test_dsl_program_source_manifest_is_complete() -> None:
    from scripts.autoresearch import _load_sources

    path = Path("src/slm_training/resources/autoresearch/dsl-program-sources.json")
    rows = _load_sources(path)
    assert len(rows) == 24
    assert len({row.uri for row in rows}) == 24
    assert sum(row.uri.startswith("https://arxiv.org/abs/") for row in rows) == 21
    assert all(row.metadata.get("category") == "dsl_program" for row in rows)
    assert all(row.metadata.get("implementation_status") == "Adjacent" for row in rows)
    assert all(row.metadata.get("limitations") for row in rows)
    # No overlap with the earlier committed manifests (dedupe honesty).
    for other in ("dynamic-symbol-sources.json", "scope-diffusion-sources.json"):
        other_rows = _load_sources(
            Path("src/slm_training/resources/autoresearch") / other
        )
        assert not {row.uri for row in rows} & {row.uri for row in other_rows}


def test_local_decision_source_manifest_is_complete() -> None:
    from scripts.autoresearch import _load_sources

    allowed_status = {"Faithful", "Adapted", "Surrogate", "Adjacent"}
    required_metadata = (
        "category",
        "local_decision_takeaway",
        "repo_relevance",
        "implementation_status",
        "limitations",
    )
    path = Path("src/slm_training/resources/autoresearch/local-decision-sources.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = _load_sources(path)
    papers = [row for row in rows if row.uri.startswith("https://arxiv.org/abs/")]

    assert "6a593158-85c4-83ea-80b1-b6fb893b26bc" in manifest["source_scope"]
    # LDI0-01 (SLM-114) expanded the inventory from 33 rows / 25 papers to 42 / 34.
    assert len(rows) == 42
    assert len(papers) == 34
    # Unique source IDs and canonical URIs: no paper duplicated under an alternate URL.
    assert len({row.source_id for row in rows}) == len(rows)
    assert len({row.uri for row in rows}) == len(rows)
    assert all(row.source_id for row in rows)
    # Required metadata is present on every row.
    for row in rows:
        for field in required_metadata:
            assert row.metadata.get(field), (row.source_id, field)
    # Implementation-status vocabulary is constrained to the research-lineage set.
    assert {row.metadata.get("implementation_status") for row in rows} <= allowed_status
    # The nine LDI0-01 additions are present with verified arXiv IDs.
    added = {
        "arxiv-1810.04650",
        "arxiv-2001.06782",
        "arxiv-2106.09685",
        "arxiv-2109.05093",
        "arxiv-2303.10512",
        "arxiv-2402.03300",
        "arxiv-2405.21047",
        "arxiv-2407.01082",
        "arxiv-2603.00025",
    }
    assert added <= {row.source_id for row in rows}


class FakeResponses:
    def create(self, **kwargs):
        assert kwargs["store"] is False
        return SimpleNamespace(
            id="resp-discovery",
            model="gpt-test",
            output_text="memo",
            usage={"input_tokens": 10},
            to_dict=lambda: {
                "sources": [{"url": "https://example.com/paper", "title": "Paper"}]
            },
        )

    def parse(self, **kwargs):
        assert kwargs["store"] is False
        return SimpleNamespace(
            id="resp-structured",
            model="gpt-test",
            usage={"output_tokens": 20},
            output_parsed=experiment(citations=("fixture://prior-run",)),
        )


class FakeHypothesisResponses:
    def create(self, **kwargs):
        assert kwargs["store"] is False
        assert kwargs["tools"] == [{"type": "web_search"}]
        return SimpleNamespace(
            id="resp-hypothesis-discovery",
            output_text="Cited discovery memo",
            to_dict=lambda: {
                "sources": [
                    {"url": "https://example.com/new-paper", "title": "New paper"}
                ]
            },
        )

    def parse(self, **kwargs):
        assert kwargs["store"] is False
        assert kwargs["text_format"] is HypothesisMatrix
        assert "arXiv:2606.01444" in kwargs["input"]
        assert "recurrence.layerscale_stability" in kwargs["input"]
        return SimpleNamespace(
            id="resp-hypotheses",
            model="gpt-test",
            usage={"output_tokens": 50},
            output_parsed=hypothesis_matrix(),
        )


def test_openai_provider_is_two_pass_and_persists_usage() -> None:
    provider = OpenAIResearchProvider(
        model="gpt-test", client=SimpleNamespace(responses=FakeResponses())
    )
    result = provider.propose(campaign(), evidence(), [source()])
    assert result.experiment.experiment_id == "exp-1"
    assert result.telemetry["store"] is False
    assert result.telemetry["discovery_response_id"] == "resp-discovery"
    assert any(item.kind == "web" for item in result.sources)


def test_openai_compiler_uses_persisted_memo_without_discovery() -> None:
    responses = FakeResponses()
    compiler = OpenAIProposalCompiler(
        model="gpt-test", client=SimpleNamespace(responses=responses)
    )
    result = compiler.propose(campaign(), evidence(), [source()], "cited memo")
    assert result.experiment.experiment_id == "exp-1"
    assert result.research_memo == "cited memo"
    assert result.telemetry["provider"] == "openai_proposal_compiler"


def test_openai_hypothesizer_forms_matrix_from_persisted_memo() -> None:
    harness = OpenAIHypothesizer(
        model="gpt-test", client=SimpleNamespace(responses=FakeHypothesisResponses())
    )
    result = harness.propose(campaign(), matrix_evidence(), [], "cited research memo")
    assert len(result.matrix.hypotheses) == 5
    assert result.research_memo == "cited research memo"
    assert result.telemetry["provider"] == "openai_hypothesizer"


def test_openai_hypothesizer_discovers_memo_when_none_is_persisted() -> None:
    harness = OpenAIHypothesizer(
        model="gpt-test", client=SimpleNamespace(responses=FakeHypothesisResponses())
    )
    result = harness.propose(campaign(), matrix_evidence(), [], "")
    assert result.research_memo == "Cited discovery memo"
    assert result.telemetry["discovery_response_id"] == "resp-hypothesis-discovery"
    assert any(
        source.uri == "https://example.com/new-paper" for source in result.sources
    )


def test_hypothesizer_benchmark_scores_feedback_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.autoresearch import hypothesizer_eval

    cases = [
        {
            "case_id": "feedback-case",
            "campaign_id": "test-campaign",
            "evidence_snapshot_id": "evidence-matrix",
            "criteria": "Form a grounded feedback-linked repair matrix.",
            "evidence": [
                {"uri": "docs/design/research-lineage.md", "role": "research"},
                {
                    "uri": "outputs/runs/prior/run_insights.json",
                    "role": "prior_trace",
                },
                {
                    "uri": "outputs/runs/prior/scoreboard.json",
                    "role": "prior_result",
                },
            ],
            "required_roles": ["research", "prior_trace", "prior_result"],
            "expected_knobs": ["steps"],
            "feedback_ids": ["feedback-aaaaaaaaaaaaaaaa"],
            "predecessor_matrix_id": "matrix-before-feedback",
        }
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    matrix = hypothesis_matrix(
        predecessor_matrix_id="matrix-before-feedback",
        feedback_ids=("feedback-aaaaaaaaaaaaaaaa",),
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {"case_id": "feedback-case", "matrix": matrix.model_dump(mode="json")}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hypothesizer_eval,
        "publish_agentv_evaluation",
        lambda *args, **kwargs: {"passed": True},
    )
    report = hypothesizer_eval.evaluate_hypothesizer(
        cases_path,
        predictions_path,
        run_dir=tmp_path / "run",
        hypothesizer_id="fixture-v1",
    )
    assert report.passed
    assert report.feedback_lineage_rate == 1.0
    assert report.promotable
    assert report.promotion_policy == "automated_frozen_meta_gate_v1"


def test_hypothesizer_benchmark_threshold_allows_partial_case_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.autoresearch import hypothesizer_eval

    cases = []
    predictions = []
    for index in range(2):
        cases.append(
            {
                "case_id": f"case-{index}",
                "campaign_id": "test-campaign",
                "evidence_snapshot_id": "evidence-matrix",
                "criteria": "Form a grounded candidate matrix.",
                "evidence": [
                    {"uri": "docs/design/research-lineage.md", "role": "research"},
                    {
                        "uri": "outputs/runs/prior/run_insights.json",
                        "role": "prior_trace",
                    },
                    {
                        "uri": "outputs/runs/prior/scoreboard.json",
                        "role": "prior_result",
                    },
                ],
                "required_roles": ["research", "prior_trace", "prior_result"],
                "expected_knobs": ["steps" if index == 0 else "lr"],
                "feedback_ids": [],
                "predecessor_matrix_id": None,
            }
        )
        predictions.append(
            {
                "case_id": f"case-{index}",
                "matrix": hypothesis_matrix().model_dump(mode="json"),
            }
        )
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    monkeypatch.setattr(
        hypothesizer_eval,
        "publish_agentv_evaluation",
        lambda *args, **kwargs: {"passed": True},
    )

    report = hypothesizer_eval.evaluate_hypothesizer(
        cases_path,
        predictions_path,
        run_dir=tmp_path / "run",
        hypothesizer_id="fixture-v1",
        pass_threshold=0.5,
    )

    assert report.actionable_rate == 0.5
    assert report.passed


def test_frozen_hypothesizer_cases_cover_initial_and_feedback_loops() -> None:
    rows = json.loads(
        Path(
            "src/slm_training/resources/autoresearch/hypothesizer_cases.json"
        ).read_text(encoding="utf-8")
    )
    assert {row["case_id"] for row in rows} == {
        "initial-evidence-matrix",
        "feedback-repair-matrix",
    }
    assert rows[0]["feedback_ids"] == []
    assert rows[1]["feedback_ids"] == ["feedback-aaaaaaaaaaaaaaaa"]


def _commit_fixture_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Fixture"], check=True
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_open_deep_research_runs_in_pinned_isolated_worker(tmp_path: Path) -> None:
    checkout = tmp_path / "open-deep-research"
    package = checkout / "src/open_deep_research"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "deep_researcher.py").write_text(
        "class Graph:\n"
        "    async def ainvoke(self, request, config):\n"
        "        return {'final_report': 'Memo https://example.com/paper', "
        "'request': request, 'config': config}\n"
        "deep_researcher = Graph()\n"
    )
    revision = _commit_fixture_repo(checkout)
    spec = ResearcherSpec(
        "open-deep-research",
        "https://example.com/open-deep-research",
        revision,
        OpenDeepResearchConfig,
    )
    researcher = IsolatedResearcher(
        spec,
        checkout=checkout,
        python=sys.executable,
        worker=Path(__file__).resolve().parents[2] / "scripts/researcher_worker.py",
        timeout_seconds=10,
    )
    result = researcher.run(campaign(), evidence(), [source()])
    assert result.status == "completed"
    assert result.upstream_revision == revision
    assert any(item.uri == "https://example.com/paper" for item in result.sources)
    assert "final_report" in result.trace


def test_open_researcher_runs_in_pinned_isolated_worker(tmp_path: Path) -> None:
    checkout = tmp_path / "open-researcher"
    (checkout / "utils").mkdir(parents=True)
    (checkout / "utils/__init__.py").write_text("")
    (checkout / "utils/openai_generator.py").write_text(
        "class OpenAIAsyncGenerator:\n"
        "    def __init__(self, **kwargs): self.kwargs = kwargs\n"
    )
    (checkout / "deploy_agent.py").write_text(
        "class BrowserPool:\n"
        "    def __init__(self, **kwargs): self.kwargs = kwargs\n"
        "async def run_one(**kwargs):\n"
        "    return [{'role': 'tool', 'content': 'https://example.com/source'}, "
        "{'role': 'assistant', 'content': 'Final cited memo'}]\n"
    )
    revision = _commit_fixture_repo(checkout)
    spec = ResearcherSpec(
        "open-researcher",
        "https://example.com/open-researcher",
        revision,
        OpenResearcherConfig,
    )
    researcher = IsolatedResearcher(
        spec,
        checkout=checkout,
        python=sys.executable,
        worker=Path(__file__).resolve().parents[2] / "scripts/researcher_worker.py",
        config={"base_url": "http://127.0.0.1:8001/v1"},
        timeout_seconds=10,
    )
    result = researcher.run(campaign(), evidence(), [source()])
    assert result.status == "completed"
    assert result.memo == "Final cited memo"
    assert any(item.uri == "https://example.com/source" for item in result.sources)


def test_researcher_fails_closed_on_revision_drift(tmp_path: Path) -> None:
    checkout = tmp_path / "researcher"
    checkout.mkdir()
    (checkout / "README.md").write_text("fixture")
    _commit_fixture_repo(checkout)
    researcher = IsolatedResearcher(
        ResearcherSpec(
            "open-deep-research",
            "https://example.com/researcher",
            "0" * 40,
            OpenDeepResearchConfig,
        ),
        checkout=checkout,
        python=sys.executable,
        worker=Path(__file__).resolve().parents[2] / "scripts/researcher_worker.py",
    )
    result = researcher.run(campaign(), evidence(), [source()])
    assert result.status == "failed"
    assert "revision mismatch" in str(result.error)


def test_researcher_config_and_timeout_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="api_key"):
        OpenDeepResearchConfig.model_validate({"api_key": "must-stay-in-env"})

    checkout = tmp_path / "researcher"
    checkout.mkdir()
    (checkout / "README.md").write_text("fixture")
    revision = _commit_fixture_repo(checkout)
    with pytest.raises(ValueError, match=rf"\(0, {MAX_RUN_SECONDS}\]"):
        IsolatedResearcher(
            ResearcherSpec(
                "open-deep-research",
                "https://example.com/researcher",
                revision,
                OpenDeepResearchConfig,
            ),
            checkout=checkout,
            python=sys.executable,
            worker=tmp_path / "unused.py",
            timeout_seconds=MAX_RUN_SECONDS + 0.01,
        )
    worker = tmp_path / "sleeping_worker.py"
    worker.write_text("import time\ntime.sleep(1)\n")
    researcher = IsolatedResearcher(
        ResearcherSpec(
            "open-deep-research",
            "https://example.com/researcher",
            revision,
            OpenDeepResearchConfig,
        ),
        checkout=checkout,
        python=sys.executable,
        worker=worker,
        timeout_seconds=0.01,
    )
    result = researcher.run(campaign(), evidence(), [source()])
    assert result.status == "failed"
    assert "timed out" in str(result.error)

    empty_worker = tmp_path / "empty_worker.py"
    empty_worker.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "output.write_text(json.dumps({'memo': '', 'trace': {}, 'telemetry': {}}))\n"
    )
    empty = IsolatedResearcher(
        researcher.spec,
        checkout=checkout,
        python=sys.executable,
        worker=empty_worker,
        timeout_seconds=1,
    ).run(campaign(), evidence(), [source()])
    assert empty.status == "failed"
    assert "empty memo" in str(empty.error)


def test_compile_is_typed_and_diagnosis_routes_bad_data() -> None:
    spec = experiment(
        knobs=ExperimentKnobs(
            data_source="existing",
            derive_from="outputs/data/train/old/records.jsonl",
            min_quality_score=0.7,
            steps=20,
        )
    )
    validate_experiment(campaign(), spec, evidence(), [source()])
    commands = compile_commands(campaign(), spec)
    assert commands[0][:4] == [
        sys.executable,
        "-m",
        "scripts.build_train_data",
        "--source",
    ]
    assert all(isinstance(command, list) for command in commands)
    diagnosis = diagnose_outcome(
        ExperimentOutcome(
            experiment_id="exp-1",
            campaign_id="test-campaign",
            status="completed",
            data_metrics={"valid_rate": 0.7},
        )
    )
    assert diagnosis.target == "data"
    assert "immutable data snapshot" in diagnosis.recommended_actions[0]


def test_campaign_loop_lineage_is_strict() -> None:
    first = CampaignSpec(
        campaign_id="cycle-1",
        objective="Continue a bounded evidence-driven campaign.",
        primary_metric="score",
        loop_id="continuous-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
    )
    assert first.predecessor_campaign_id is None
    with pytest.raises(ValidationError, match="later loop cycles require"):
        CampaignSpec(
            campaign_id="cycle-2",
            objective="Continue a bounded evidence-driven campaign.",
            primary_metric="score",
            loop_id="continuous-1",
            cycle_index=2,
            upstream_commit="a" * 40,
            integration_commit="b" * 40,
        )


def test_continuous_matrix_requires_ranked_priorities() -> None:
    loop_campaign = CampaignSpec(
        campaign_id="cycle-1",
        objective="Continue a bounded evidence-driven campaign.",
        primary_metric="score",
        loop_id="continuous-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
    )
    without_priorities = hypothesis_matrix(campaign_id="cycle-1")
    with pytest.raises(ValueError, match="next-run priorities"):
        validate_hypothesis_matrix(
            loop_campaign, without_priorities, matrix_evidence(), []
        )

    with_priorities = hypothesis_matrix(
        campaign_id="cycle-1",
        next_run_priorities=(next_priority(experiment_id="hyp-0"),),
    )
    validate_hypothesis_matrix(loop_campaign, with_priorities, matrix_evidence(), [])


def test_continuous_commit_validation_requires_latest_integrated_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import autoresearch

    upstream = "a" * 40
    integration = "b" * 40

    def fake_git(*args: str, check: bool = True):
        if args[:2] == ("rev-parse", "--verify"):
            ref = args[2]
            value = (
                upstream
                if ref in {f"{upstream}^{{commit}}", "origin/main^{commit}"}
                else integration
            )
            return subprocess.CompletedProcess(args, 0, value + "\n", "")
        if args[:2] == ("merge-base", "--is-ancestor"):
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(autoresearch, "_git", fake_git)
    autoresearch._validate_continuous_commits(upstream, integration)

    def stale_git(*args: str, check: bool = True):
        result = fake_git(*args, check=check)
        if args == ("rev-parse", "--verify", "origin/main^{commit}"):
            return subprocess.CompletedProcess(args, 0, "c" * 40 + "\n", "")
        return result

    monkeypatch.setattr(autoresearch, "_git", stale_git)
    with pytest.raises(ValueError, match="stale"):
        autoresearch._validate_continuous_commits(upstream, integration)


def test_continuous_run_binds_manifest_to_integration_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import autoresearch

    loop_campaign = CampaignSpec(
        campaign_id="cycle-1",
        objective="Run from the exact fetched and integrated source revision.",
        primary_metric="score",
        loop_id="loop-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
    )
    store = CampaignStore("cycle-1", tmp_path)
    store.initialize(loop_campaign)
    matrix = hypothesis_matrix(
        campaign_id="cycle-1",
        next_run_priorities=(next_priority(experiment_id="hyp-0"),),
    )
    matrix_path = store.write_artifact("hypothesis_matrices", matrix)
    store.append_event("hypothesis_matrix_formed", artifact_sha256=matrix_path.stem)
    manifest = experiment_campaign(
        campaign_id="cycle-1",
        experiment_id="hyp-0",
        source_commit="c" * 40,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        autoresearch, "_validate_continuous_commits", lambda *_args: None
    )

    with pytest.raises(ValueError, match="integration_commit"):
        autoresearch.cmd_run(
            SimpleNamespace(
                campaign_id="cycle-1",
                root=tmp_path,
                experiment=None,
                campaign_manifest=manifest_path,
                execute=False,
                trackio=False,
            )
        )


def test_continuous_successor_accepts_cross_campaign_feedback(
    tmp_path: Path,
) -> None:
    from scripts.autoresearch import cmd_hypothesize

    first_campaign = CampaignSpec(
        campaign_id="cycle-1",
        objective="Run the first bounded evidence-driven campaign.",
        primary_metric="score",
        loop_id="loop-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
    )
    first_store = CampaignStore("cycle-1", tmp_path)
    first_store.initialize(first_campaign)
    first = hypothesis_matrix(campaign_id="cycle-1")
    first_path = first_store.write_artifact("hypothesis_matrices", first)
    first_store.append_event(
        "hypothesis_matrix_formed", artifact_sha256=first_path.stem
    )
    feedback = HypothesisFeedback(
        feedback_id="feedback-aaaaaaaaaaaaaaaa",
        campaign_id="cycle-1",
        matrix_id=first.matrix_id,
        experiment_id="hyp-0",
        hypothesis=first.hypotheses[0].experiment.hypothesis,
        knob_signature='{"steps": 100}',
        outcome_status="completed",
        diagnosis_target="model",
        diagnosis_evidence=("The matched run remained below its gate.",),
        recommended_actions=("Test a bounded model repair.",),
    )
    feedback_path = first_store.write_artifact("hypothesizer_feedback", feedback)
    first_store.append_event(
        "hypothesizer_feedback_recorded",
        experiment_id="hyp-0",
        artifact_sha256=feedback_path.stem,
    )

    second_campaign = CampaignSpec(
        campaign_id="cycle-2",
        objective="Continue from the predecessor campaign feedback.",
        primary_metric="score",
        loop_id="loop-1",
        cycle_index=2,
        predecessor_campaign_id="cycle-1",
        upstream_commit="c" * 40,
        integration_commit="d" * 40,
    )
    second_store = CampaignStore("cycle-2", tmp_path)
    second_store.initialize(second_campaign)
    second_store.write_artifact("evidence", matrix_evidence())
    sources = second_store.write_artifact("research_sources", {"sources": []})
    successor = hypothesis_matrix(
        matrix_id="matrix-2",
        campaign_id="cycle-2",
        predecessor_matrix_id=first.matrix_id,
        feedback_ids=(feedback.feedback_id,),
        offset=10,
        next_run_priorities=(
            next_priority(
                experiment_id="hyp-10",
                evidence_ids=(feedback.feedback_id,),
            ),
        ),
    )
    matrix_path = tmp_path / "successor.json"
    matrix_path.write_text(successor.model_dump_json(), encoding="utf-8")

    assert (
        cmd_hypothesize(
            SimpleNamespace(
                campaign_id="cycle-2",
                root=tmp_path,
                evidence=None,
                sources=sources,
                provider="agent",
                matrix=matrix_path,
                model="unused",
                memo=None,
            )
        )
        == 0
    )
    formed = [
        event
        for event in second_store.verify_event_chain()
        if event["event_type"] == "hypothesis_matrix_formed"
    ]
    assert len(formed) == 1


def test_continuous_lean_miss_requires_five_priority_lanes() -> None:
    first = hypothesis_matrix(campaign_id="cycle-1")
    optimum = OptimumFeedbackV1(
        policy="block_promotion_and_diagnose",
        breaches=(
            {
                "metric_id": "latency",
                "authority": "assumption_backed",
                "relation": "above",
            },
        ),
        diagnosis_lanes=(
            "measurement_control",
            "training_method",
            "architecture",
            "lean_model",
            "assumptions",
        ),
        metric_evidence_sha256="a" * 64,
        metric_certificate_sha256="b" * 64,
    )
    feedback = HypothesisFeedback(
        feedback_id="feedback-aaaaaaaaaaaaaaaa",
        campaign_id="cycle-1",
        matrix_id=first.matrix_id,
        experiment_id="hyp-0",
        hypothesis=first.hypotheses[0].experiment.hypothesis,
        knob_signature='{"steps": 100}',
        outcome_status="completed",
        diagnosis_target="none",
        diagnosis_evidence=("Latency exceeded its assumption-backed band.",),
        recommended_actions=("Run all five controlled diagnosis lanes.",),
        optimum_feedback=optimum,
    )
    loop_campaign = CampaignSpec(
        campaign_id="cycle-2",
        objective="Diagnose every lane implicated by the certified band miss.",
        primary_metric="score",
        loop_id="loop-1",
        cycle_index=2,
        predecessor_campaign_id="cycle-1",
        upstream_commit="c" * 40,
        integration_commit="d" * 40,
    )
    priorities = tuple(
        next_priority(
            experiment_id="hyp-10",
            evidence_ids=(feedback.feedback_id,),
            area=lane,
            authority="lean_assumption",
            disposition="experiment_next" if rank == 1 else "block_promotion",
            rank=rank,
        )
        for rank, lane in enumerate(optimum.diagnosis_lanes, start=1)
    )
    complete = hypothesis_matrix(
        matrix_id="matrix-2",
        campaign_id="cycle-2",
        predecessor_matrix_id=first.matrix_id,
        feedback_ids=(feedback.feedback_id,),
        offset=10,
        diagnosis_lanes=optimum.diagnosis_lanes,
        next_run_priorities=priorities,
    )
    validate_hypothesis_matrix(
        loop_campaign,
        complete,
        matrix_evidence(),
        [],
        feedback=(feedback,),
        previous_matrix=first,
    )

    missing_lane = complete.model_copy(update={"next_run_priorities": priorities[:-1]})
    with pytest.raises(ValueError, match="Lean-assumption priority"):
        validate_hypothesis_matrix(
            loop_campaign,
            missing_lane,
            matrix_evidence(),
            [],
            feedback=(feedback,),
            previous_matrix=first,
        )


def test_continuous_harness_feedback_requires_repair_priority() -> None:
    first = hypothesis_matrix(campaign_id="cycle-1")
    feedback = HypothesisFeedback(
        feedback_id="feedback-aaaaaaaaaaaaaaaa",
        campaign_id="cycle-1",
        matrix_id=first.matrix_id,
        experiment_id="hyp-0",
        hypothesis=first.hypotheses[0].experiment.hypothesis,
        knob_signature='{"steps": 100}',
        outcome_status="failed",
        diagnosis_target="harness",
        harness_family="model_build",
        diagnosis_evidence=("Frozen replay reproduced the evaluator failure.",),
        recommended_actions=("Repair the canonical evaluator and replay.",),
    )
    loop_campaign = CampaignSpec(
        campaign_id="cycle-2",
        objective="Repair the reproduced canonical harness failure.",
        primary_metric="score",
        loop_id="loop-1",
        cycle_index=2,
        predecessor_campaign_id="cycle-1",
        upstream_commit="c" * 40,
        integration_commit="d" * 40,
    )
    wrong = hypothesis_matrix(
        matrix_id="matrix-2",
        campaign_id="cycle-2",
        predecessor_matrix_id=first.matrix_id,
        feedback_ids=(feedback.feedback_id,),
        offset=10,
        next_run_priorities=(
            next_priority(
                experiment_id="hyp-10",
                evidence_ids=(feedback.feedback_id,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="repair priority"):
        validate_hypothesis_matrix(
            loop_campaign,
            wrong,
            matrix_evidence(),
            [],
            feedback=(feedback,),
            previous_matrix=first,
        )

    repair = wrong.model_copy(
        update={
            "next_run_priorities": (
                next_priority(
                    evidence_ids=(feedback.feedback_id,),
                    area="model_build",
                    authority="reproduced_harness_signal",
                    disposition="repair_before_run",
                ),
            )
        }
    )
    validate_hypothesis_matrix(
        loop_campaign,
        repair,
        matrix_evidence(),
        [],
        feedback=(feedback,),
        previous_matrix=first,
    )


def test_frozen_harness_signal_routes_one_canonical_owner() -> None:
    diagnosis = diagnose_outcome(
        ExperimentOutcome(
            experiment_id="exp-1",
            campaign_id="test-campaign",
            status="failed",
            error="evaluation crashed",
            harness_signals=(
                HarnessSignalV1(
                    family="model_build",
                    code="fixture-replay-crash",
                    evidence_uri="outputs/repro/frozen.json",
                    reproduced_on_frozen_input=True,
                    primary=True,
                ),
            ),
        )
    )
    assert diagnosis.target == "harness"
    assert diagnosis.harness_family == "model_build"
    assert "identical frozen model/data arm" in diagnosis.recommended_actions[1]

    unconfirmed = diagnose_outcome(
        ExperimentOutcome(
            experiment_id="exp-2",
            campaign_id="test-campaign",
            status="failed",
            harness_signals=(
                HarnessSignalV1(
                    family="model_build",
                    code="suspected-crash",
                    evidence_uri="outputs/repro/unconfirmed.json",
                    reproduced_on_frozen_input=False,
                ),
            ),
        )
    )
    assert unconfirmed.target == "infrastructure"


def test_loop_result_matrix_is_derived_from_verified_campaign_chain(
    tmp_path: Path,
) -> None:
    loop_campaign = CampaignSpec(
        campaign_id="cycle-1",
        objective="Improve one declared metric with bounded evidence.",
        primary_metric="held_out.structural_similarity",
        loop_id="loop-1",
        cycle_index=1,
        upstream_commit="c" * 40,
        integration_commit="d" * 40,
    )
    store = CampaignStore(loop_campaign.campaign_id, tmp_path)
    store.initialize(loop_campaign)
    manifest = experiment_campaign(
        campaign_id="cycle-1",
        experiment_id="hyp-0",
        source_commit="d" * 40,
    )
    store.lock_experiment_campaign(manifest)
    outcome = ExperimentOutcome(
        experiment_id="hyp-0",
        campaign_id="cycle-1",
        status="completed",
        metrics={
            "held_out.structural_similarity": 0.75,
            "trainable_params": 1234,
            "ship_gates_pass": 1,
        },
    )
    outcome_path = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_finished",
        experiment_id="hyp-0",
        status="completed",
        artifact_sha256=outcome_path.stem,
        detail={"campaign_manifest_sha256": campaign_manifest_sha256(manifest)},
    )
    diagnosis = Diagnosis(
        experiment_id="hyp-0",
        target="none",
        confidence=0.6,
        evidence=("no defect",),
        recommended_actions=("retain evidence",),
    )
    diagnosis_path = store.write_artifact("diagnoses", diagnosis)
    store.append_event(
        "outcome_diagnosed",
        experiment_id="hyp-0",
        status="none",
        artifact_sha256=diagnosis_path.stem,
    )

    matrix = render_loop_result_matrix(tmp_path, "loop-1")
    assert "Liveness" in matrix
    assert (
        "| 1 | cccccccc | dddddddd | hyp-0 | 1234 | 0.75 | — | — | pass | complete | none | fixture | — | blocked | completed |"
    ) in matrix
    assert len(matrix.encode("utf-8")) < 64 * 1024


def test_loop_result_matrix_marks_timeout_measurement_incomplete(
    tmp_path: Path,
) -> None:
    campaign = CampaignSpec(
        campaign_id="cycle-1",
        objective="Expose timeout incompleteness without blaming the model.",
        primary_metric="smoke.parse_rate",
        loop_id="loop-1",
        cycle_index=1,
        upstream_commit="c" * 40,
        integration_commit="d" * 40,
    )
    store = CampaignStore(campaign.campaign_id, tmp_path)
    store.initialize(campaign)
    outcome = ExperimentOutcome(
        experiment_id="hyp-0",
        campaign_id=campaign.campaign_id,
        status="completed",
        metrics={"smoke.completed_document_n": 0, "smoke.decode_timeout_count": 3},
    )
    artifact = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_finished",
        experiment_id="hyp-0",
        status="completed",
        artifact_sha256=artifact.stem,
    )
    diagnosis = Diagnosis(
        experiment_id="hyp-0",
        target="infrastructure",
        confidence=0.95,
        evidence=("measurement incomplete",),
        recommended_actions=("retry frozen arm",),
    )
    artifact = store.write_artifact("diagnoses", diagnosis)
    store.append_event(
        "outcome_diagnosed",
        experiment_id="hyp-0",
        status="infrastructure",
        artifact_sha256=artifact.stem,
    )

    rendered = render_loop_result_matrix(tmp_path, "loop-1")
    assert "| incomplete | infrastructure |" in rendered


def test_loop_result_matrix_marks_partial_measurement_incomplete(
    tmp_path: Path,
) -> None:
    campaign = CampaignSpec(
        campaign_id="cycle-partial",
        objective="Expose partial measurements without blaming the model.",
        primary_metric="smoke.parse_rate",
        loop_id="loop-1",
        cycle_index=1,
        upstream_commit="c" * 40,
        integration_commit="d" * 40,
    )
    store = CampaignStore(campaign.campaign_id, tmp_path)
    store.initialize(campaign)
    outcome = ExperimentOutcome(
        experiment_id="hyp-partial",
        campaign_id=campaign.campaign_id,
        status="completed",
        metrics={
            "smoke.n": 3,
            "smoke.completed_document_n": 1,
            "smoke.incomplete_document_n": 2,
        },
    )
    artifact = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_finished",
        experiment_id=outcome.experiment_id,
        status=outcome.status,
        artifact_sha256=artifact.stem,
    )

    assert "incomplete" in render_loop_result_matrix(tmp_path, "loop-1")


def test_loop_tables_surface_lean_authority_and_stop_priority(tmp_path: Path) -> None:
    loop_campaign = CampaignSpec(
        campaign_id="cycle-1",
        objective="Classify a preregistered theorem-backed metric band.",
        primary_metric="score",
        loop_id="loop-1",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
    )
    store = CampaignStore("cycle-1", tmp_path)
    store.initialize(loop_campaign)
    outcome = ExperimentOutcome(
        experiment_id="hyp-0",
        campaign_id="cycle-1",
        status="completed",
        metrics={"score": 1.0},
    )
    outcome_path = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_finished",
        experiment_id="hyp-0",
        artifact_sha256=outcome_path.stem,
    )
    optimum = OptimumFeedbackV1(
        policy="stop",
        breaches=(
            {
                "metric_id": "invalid_grammar_count",
                "authority": "theorem",
                "relation": "above",
            },
        ),
        diagnosis_lanes=(
            "measurement_control",
            "training_method",
            "architecture",
            "lean_model",
            "assumptions",
        ),
        metric_evidence_sha256="c" * 64,
        metric_certificate_sha256="d" * 64,
    )
    feedback = HypothesisFeedback(
        feedback_id="feedback-aaaaaaaaaaaaaaaa",
        campaign_id="cycle-1",
        matrix_id="matrix-1",
        experiment_id="hyp-0",
        hypothesis="The exact grammar invariant remains inside its proved band.",
        knob_signature='{"steps": 100}',
        outcome_status="completed",
        diagnosis_target="none",
        diagnosis_evidence=("Lean replay found a theorem-backed contradiction.",),
        recommended_actions=("Stop and repair the measurement or formal model.",),
        optimum_feedback=optimum,
    )
    feedback_path = store.write_artifact("hypothesizer_feedback", feedback)
    store.append_event(
        "hypothesizer_feedback_recorded",
        experiment_id="hyp-0",
        artifact_sha256=feedback_path.stem,
    )

    rendered = render_loop_result_matrix(tmp_path, "loop-1")
    assert "Run Results" in rendered
    assert "Diagnostic Signals" in rendered
    assert "Speculative Next-Run Priorities" in rendered
    assert "invalid_grammar_count:above[theorem]" in rendered
    assert "e:cccccccc c:dddddddd" in rendered
    assert "lean_theorem" in rendered
    assert "stop_campaign" in rendered


def test_loop_result_matrix_rejects_broken_predecessor_chain(tmp_path: Path) -> None:
    CampaignStore("cycle-1", tmp_path).initialize(
        CampaignSpec(
            campaign_id="cycle-1",
            objective="Run the first bounded continuous campaign.",
            primary_metric="score",
            loop_id="loop-1",
            cycle_index=1,
            upstream_commit="a" * 40,
            integration_commit="b" * 40,
        )
    )
    CampaignStore("cycle-2", tmp_path).initialize(
        CampaignSpec(
            campaign_id="cycle-2",
            objective="Run the next bounded continuous campaign.",
            primary_metric="score",
            loop_id="loop-1",
            cycle_index=2,
            predecessor_campaign_id="wrong-cycle",
            upstream_commit="c" * 40,
            integration_commit="d" * 40,
        )
    )
    with pytest.raises(RuntimeError, match="predecessor chain"):
        render_loop_result_matrix(tmp_path, "loop-1")


def test_compile_resolves_canonical_published_train_version() -> None:
    spec = experiment(
        knobs=ExperimentKnobs(
            train_version="e218_schema_normalized_judge_v5",
            eval_version="remediated",
            allow_unconstrained_fallback=False,
            steps=32,
            output_tokenizer="lexer",
            compiler_alignment_loss_weight=1.0,
            compiler_alignment_margin=1.0,
            compiler_alignment_stratified=True,
            compiler_alignment_semantic_exhaustive=True,
            component_inventory_loss_weight=1.0,
            component_inventory_decode_weight=0.75,
            component_plan_loss_weight=1.25,
            component_plan_decode_weight=0.5,
            component_edge_loss_weight=1.5,
            component_edge_alignment_loss_weight=1.75,
            component_edge_decode_weight=0.25,
            binder_component_plan_loss_weight=1.1,
            binder_component_plan_decode_weight=0.2,
            binder_topology_loss_weight=1.3,
            binder_topology_decode_weight=0.4,
            binder_arity_loss_weight=1.2,
            binder_arity_decode_weight=0.3,
            compiler_decode_mode="tree",
            compiler_search_mode="ptrm",
            compiler_search_trigger="stagnation",
            compiler_search_width=4,
            compiler_search_noise=1.0,
            compiler_search_stagnation_patience=2,
            compiler_search_backtrack_limit=8,
            schema_in_context=True,
            slot_contract_in_context=True,
            design_md_context=False,
            local_files_only=True,
            sync_checkpoints=False,
            mixture_sampling_policy="capacity_aware",
        )
    )

    commands = compile_commands(campaign(), spec)

    assert "--train-version" in commands[0]
    assert "e218_schema_normalized_judge_v5" in commands[0]
    assert "--train-dir" not in commands[0]
    assert "--train-version" in commands[-1]
    assert commands[-1][commands[-1].index("--test-dir") + 1] == "remediated"
    assert commands[0][commands[0].index("--output-tokenizer") + 1] == "lexer"
    assert "--compiler-alignment-stratified" in commands[0]
    assert commands[0][commands[0].index("--compiler-alignment-margin") + 1] == "1.0"
    assert "--compiler-alignment-semantic-exhaustive" in commands[0]
    assert (
        commands[0][commands[0].index("--component-inventory-loss-weight") + 1] == "1.0"
    )
    assert (
        commands[0][commands[0].index("--component-inventory-decode-weight") + 1]
        == "0.75"
    )
    assert commands[0][commands[0].index("--component-plan-loss-weight") + 1] == "1.25"
    assert commands[0][commands[0].index("--component-plan-decode-weight") + 1] == "0.5"
    assert commands[0][commands[0].index("--component-edge-loss-weight") + 1] == "1.5"
    assert (
        commands[0][commands[0].index("--component-edge-alignment-loss-weight") + 1]
        == "1.75"
    )
    assert (
        commands[0][commands[0].index("--component-edge-decode-weight") + 1] == "0.25"
    )
    assert (
        commands[0][commands[0].index("--binder-component-plan-loss-weight") + 1]
        == "1.1"
    )
    assert (
        commands[0][commands[0].index("--binder-component-plan-decode-weight") + 1]
        == "0.2"
    )
    assert commands[0][commands[0].index("--binder-topology-loss-weight") + 1] == "1.3"
    assert (
        commands[0][commands[0].index("--binder-topology-decode-weight") + 1] == "0.4"
    )
    assert commands[0][commands[0].index("--binder-arity-loss-weight") + 1] == "1.2"
    assert commands[0][commands[0].index("--binder-arity-decode-weight") + 1] == "0.3"
    assert "--schema-in-context" in commands[0]
    assert "--slot-contract-in-context" in commands[0]
    assert "--no-design-md-context" in commands[0]
    assert "--local-files-only" in commands[0]
    assert "--no-sync-checkpoints" in commands[0]
    assert (
        commands[0][commands[0].index("--mixture-sampling-policy") + 1]
        == "capacity_aware"
    )
    assert commands[-1][commands[-1].index("--compiler-decode-mode") + 1] == "tree"
    assert commands[-1][commands[-1].index("--compiler-search-mode") + 1] == "ptrm"
    assert commands[-1][commands[-1].index("--compiler-search-width") + 1] == "4"
    assert "--grammar-ltr-primary" in commands[-1]
    assert "--no-unconstrained-fallback" in commands[-1]
    assert "--honest-slot-contract" in commands[-1]
    assert "--slot-contract-constrained-decode" in commands[-1]


def test_execute_records_process_launch_failure() -> None:
    outcome = execute_commands(experiment(), [["/__missing_executable__"]])

    assert outcome.status == "failed"
    assert "stage could not start" in str(outcome.error)
    assert outcome.stage_telemetry[0]["launch_error"]


def test_experiment_wall_budget_reads_history_but_execution_is_capped() -> None:
    assert CampaignBudget().max_wall_minutes == float(MAX_RUN_MINUTES)
    assert CampaignBudget(max_wall_minutes=0.25).max_wall_minutes == 0.25
    historical = CampaignSpec(
        campaign_id="historical-long-wall",
        objective="Read a legacy campaign artifact.",
        primary_metric="score",
        budget=CampaignBudget(max_wall_minutes=12),
    )
    from scripts.autoresearch import _bounded_campaign_seconds

    assert _bounded_campaign_seconds(historical) == float(MAX_RUN_MINUTES * 60)


def test_execute_shares_one_wall_budget_across_stages(monkeypatch) -> None:
    import slm_training.autoresearch.engine as engine

    time_reads = iter((10.0, 11.0, 13.5))
    timeouts: list[float] = []

    monkeypatch.setattr(engine.time, "monotonic", lambda: next(time_reads))

    def complete(command, **kwargs):
        timeouts.append(
            kwargs["interrupt_after_seconds"] + kwargs["kill_grace_seconds"]
        )
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(engine, "run_bounded_process", complete)
    outcome = execute_commands(
        experiment(),
        [["stage-one"], ["stage-two"]],
        timeout_seconds=5.0,
    )

    assert outcome.status == "completed"
    assert timeouts == [4.0, 1.5]
    assert outcome.wall_time_budget_seconds == 5.0


def test_execute_rejects_partial_train_before_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    import slm_training.autoresearch.engine as engine

    calls: list[list[str]] = []

    def complete(command, **kwargs):
        del kwargs
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "steps": 7,
                    "stopped_on": "wall_time_budget",
                    "checkpoint": "unused.pt",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(engine, "run_bounded_process", complete)
    outcome = execute_commands(
        experiment(),
        [
            ["python", "-m", "scripts.train_model", "--steps", "20"],
            ["python", "-m", "scripts.evaluate_model", "--ship-gates"],
        ],
        cwd=tmp_path,
    )

    assert outcome.status == "stopped"
    assert outcome.error == "training stopped_on=wall_time_budget before declared steps"
    assert len(calls) == 1
    assert outcome.stage_telemetry[0]["measurement_complete"] is False


def test_execute_rejects_invalid_declared_training_steps(
    tmp_path: Path, monkeypatch
) -> None:
    import slm_training.autoresearch.engine as engine

    monkeypatch.setattr(
        engine,
        "run_bounded_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "steps": 20,
                    "stopped_on": "steps",
                    "checkpoint": "unused.pt",
                }
            ),
            stderr="",
        ),
    )
    outcome = execute_commands(
        experiment(),
        [["python", "-m", "scripts.train_model", "--steps", "many"]],
        cwd=tmp_path,
    )

    assert outcome.status == "stopped"
    assert outcome.error == "training declared invalid --steps value 'many'"


def test_execute_keeps_typed_ship_gate_rejection_as_completed_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    import slm_training.autoresearch.engine as engine

    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    replies = iter(
        (
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "steps": 20,
                        "stopped_on": "steps",
                        "checkpoint": str(checkpoint),
                    }
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                [],
                8,
                stdout=json.dumps(
                    {
                        "suites": {"smoke": {"n": 3, "parse_rate": 1.0}},
                        "evals": {"runner": {"name": "AgentV", "execution_errors": 0}},
                        "gates": {"pass": False},
                    }
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(
        engine, "run_bounded_process", lambda *args, **kwargs: next(replies)
    )

    outcome = execute_commands(
        experiment(),
        [
            ["python", "-m", "scripts.train_model", "--steps", "20"],
            ["python", "-m", "scripts.evaluate_model", "--ship-gates"],
        ],
        cwd=tmp_path,
    )

    assert outcome.status == "completed"
    assert outcome.metrics["gates.pass"] == 0.0
    assert outcome.stage_telemetry[-1]["expected_gate_rejection"] is True


def test_execute_rejects_partial_ship_gate_scoreboard(
    tmp_path: Path, monkeypatch
) -> None:
    import slm_training.autoresearch.engine as engine

    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    replies = iter(
        (
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "steps": 20,
                        "stopped_on": "steps",
                        "checkpoint": str(checkpoint),
                    }
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                [],
                8,
                stdout=json.dumps(
                    {
                        "suites": {
                            "smoke": {
                                "n": 3,
                                "completed_document_n": 1,
                                "incomplete_document_n": 2,
                                "decode_timeout_count": 2,
                            }
                        },
                        "evals": {"runner": {"name": "AgentV", "execution_errors": 0}},
                        "gates": {"pass": False},
                    }
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(
        engine, "run_bounded_process", lambda *args, **kwargs: next(replies)
    )

    outcome = execute_commands(
        experiment(),
        [
            ["python", "-m", "scripts.train_model", "--steps", "20"],
            ["python", "-m", "scripts.evaluate_model", "--ship-gates"],
        ],
        cwd=tmp_path,
    )

    assert outcome.status == "failed"
    assert outcome.stage_telemetry[-1]["expected_gate_rejection"] is False


def test_decode_timeout_scoreboard_routes_to_infrastructure() -> None:
    diagnosis = diagnose_outcome(
        ExperimentOutcome(
            experiment_id="c1707-control",
            campaign_id="continuous-loop-c1707",
            status="completed",
            metrics={
                "suites.smoke.n": 3,
                "suites.smoke.completed_document_n": 0,
                "suites.smoke.decode_timeout_count": 3,
                "gates.pass": 0,
            },
        )
    )

    assert diagnosis.target == "infrastructure"
    assert "model attribution is unavailable" in diagnosis.evidence[0]


def test_partial_decode_scoreboard_routes_to_infrastructure() -> None:
    diagnosis = diagnose_outcome(
        ExperimentOutcome(
            experiment_id="partial-candidate",
            campaign_id="partial-cycle",
            status="completed",
            metrics={
                "suites.smoke.n": 3,
                "suites.smoke.completed_document_n": 1,
                "suites.smoke.incomplete_document_n": 2,
                "gates.pass": 0,
            },
        )
    )

    assert diagnosis.target == "infrastructure"


def test_train_version_and_data_build_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="train_version or data_source"):
        ExperimentKnobs(
            train_version="published", data_source="existing", derive_from="old.jsonl"
        )


def test_compile_grammar_topology_campaign_uses_typed_knobs() -> None:
    grammar_campaign = campaign().model_copy(update={"track": "grammar_diffusion"})
    spec = experiment(
        knobs=ExperimentKnobs(
            steps=20,
            context_backend="scratch",
            topology_actions=True,
            topology_critic_decode=False,
            topology_max_nodes=128,
            topology_max_active=24,
            topology_accept_threshold=0.4,
        )
    )
    commands = compile_commands(grammar_campaign, spec)
    train = next(command for command in commands if "scripts.train_model" in command)
    evaluate = next(
        command for command in commands if "scripts.evaluate_model" in command
    )
    assert train[train.index("--model") + 1] == "grammar_diffusion"
    assert "--topology-actions" in train
    assert "--no-topology-critic-decode" in train
    assert train[train.index("--topology-max-nodes") + 1] == "128"
    assert train[train.index("--topology-max-active") + 1] == "24"
    assert train[train.index("--topology-accept-threshold") + 1] == "0.4"
    assert evaluate[evaluate.index("--model") + 1] == "grammar_diffusion"


def test_compile_scope_campaign_builds_contract_data_and_flags() -> None:
    grammar_campaign = campaign().model_copy(update={"track": "grammar_diffusion"})
    spec = experiment(
        knobs=ExperimentKnobs(
            data_source="programspec",
            scope_contracts=True,
            scope_independent_noise=True,
            scope_local_oracle=True,
            scope_contract_negatives=True,
        )
    )
    commands = compile_commands(grammar_campaign, spec)
    build = next(
        command for command in commands if "scripts.build_train_data" in command
    )
    train = next(command for command in commands if "scripts.train_model" in command)
    assert "--scope-derivatives" in build
    assert "--scope-contracts" in train
    assert "--scope-independent-noise" in train
    assert "--scope-local-oracle" in train
    assert "--scope-contract-negatives" in train


def test_compile_dynamic_symbol_campaign_uses_typed_flags() -> None:
    spec = experiment(
        knobs=ExperimentKnobs(
            runtime_symbol_features="role_gated",
            symbol_slot_augmentation=True,
            semantic_candidate_masks=True,
            constraint_graph_mode="hybrid",
            grammar_equivalence_cache=True,
            compact_active_canvas=False,
        )
    )
    train = next(
        command
        for command in compile_commands(campaign(), spec)
        if "scripts.train_model" in command
    )
    assert train[train.index("--output-tokenizer") + 1] == "lexer"
    assert train[train.index("--runtime-symbol-features") + 1] == "role_gated"
    assert train[train.index("--constraint-graph-mode") + 1] == "hybrid"
    assert "--symbol-slot-augmentation" in train
    assert "--semantic-candidate-masks" in train
    assert "--grammar-equivalence-cache" in train
    assert "--no-compact-active-canvas" in train


def test_rl_readiness_is_fail_closed() -> None:
    evaluation = passing_evaluation()
    report = assess_rl_readiness(evaluation)
    assert report.approved
    assert_rl_ready(report, evaluation=evaluation)
    with pytest.raises(ValueError, match="not bound"):
        assert_rl_ready(report)
    failed = assess_rl_readiness({"suites": {}, "reward_samples": [1, 1]})
    assert not failed.approved
    with pytest.raises(ValueError, match="RL is locked"):
        assert_rl_ready(failed)
    with pytest.raises(ValueError, match="provide an approved"):
        assert_rl_ready(None)


def test_rl_readiness_report_uses_portable_evaluation_reference(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "evidence" / "evaluation.json"
    evaluation.parent.mkdir()
    evaluation.write_text(json.dumps(passing_evaluation()), encoding="utf-8")
    report = assess_rl_readiness(evaluation)
    report_path = write_rl_readiness(tmp_path / "reports" / "ready.json", report)
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    assert not Path(stored["evaluation_uri"]).is_absolute()
    assert assert_rl_ready(report_path).approved


def test_remote_sync_is_explicit_and_non_destructive(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    plan = sync_campaign(tmp_path, "test-campaign")
    assert plan["push"] is False
    assert plan["command"][:3] == ["hf", "buckets", "sync"]
    assert "--no-delete" in plan["command"]
    assert str(plan["remote_uri"]).endswith("/autoresearch/test-campaign")


def test_compile_action_alias_knobs() -> None:
    spec = experiment(
        knobs=ExperimentKnobs(
            action_embedding_init="alias_aware_description",
            action_embedding_train="frozen",
            action_alias_mode="fixed",
            action_alias_manifest="/tmp/alias_map.json",
            action_description_name_mode="alias_aware_description",
        )
    )
    train = next(
        command
        for command in compile_commands(campaign(), spec)
        if "scripts.train_model" in command
    )
    assert (
        train[train.index("--action-embedding-init") + 1] == "alias_aware_description"
    )
    assert train[train.index("--action-embedding-train") + 1] == "frozen"
    assert train[train.index("--action-alias-mode") + 1] == "fixed"
    assert train[train.index("--action-alias-manifest") + 1] == "/tmp/alias_map.json"
    assert (
        train[train.index("--action-description-name-mode") + 1]
        == "alias_aware_description"
    )


def test_default_eval_version_resolves_published_smoke_suite() -> None:
    from slm_training.autoresearch.engine import default_eval_version
    from slm_training.data.store import DataStore

    name = default_eval_version()
    root = DataStore().resolve_path("eval", name)
    assert (root / "suites" / "smoke" / "records.jsonl").is_file()


def test_compile_commands_default_eval_is_not_missing_v1() -> None:
    from slm_training.autoresearch.engine import compile_commands
    from slm_training.autoresearch.schemas import (
        CampaignSpec,
        ExperimentKnobs,
        ExperimentSpec,
    )

    campaign = CampaignSpec(
        campaign_id="eval-default",
        objective="Ensure continuous eval defaults resolve.",
        primary_metric="smoke.parse_rate",
    )
    experiment = ExperimentSpec(
        experiment_id="eval-default-1",
        campaign_id="eval-default",
        hypothesis="Default eval points at a published smoke suite.",
        rationale="Continuous loops must not die on missing v1.",
        expected_effect="evaluate_model receives a resolvable test-dir.",
        falsification_criteria=("Default still points at missing v1.",),
        stop_conditions=("Stop after plan compile.",),
        citations=("docs/design/research-lineage.md",),
        knobs=ExperimentKnobs(train_version="wf_smoke_v2", steps=2, seed=0),
    )
    commands = compile_commands(campaign, experiment)
    evaluate = next(cmd for cmd in commands if "scripts.evaluate_model" in cmd)
    test_dir = evaluate[evaluate.index("--test-dir") + 1]
    assert test_dir != "v1"
