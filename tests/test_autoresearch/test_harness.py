from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.engine import (
    compile_commands,
    create_hypothesis_feedback,
    diagnose_outcome,
    validate_experiment,
    validate_hypothesis_matrix,
)
from slm_training.autoresearch.evidence import collect_evidence
from slm_training.autoresearch.literature import HuggingFacePapersClient
from slm_training.autoresearch.providers import (
    OpenAIHypothesizer,
    OpenAIProposalCompiler,
    OpenAIResearchProvider,
)
from slm_training.autoresearch.researchers import IsolatedResearcher, ResearcherSpec
from slm_training.autoresearch.persistence import sync_campaign
from slm_training.autoresearch.rl_gate import assert_rl_ready, assess_rl_readiness
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    CategoricalNoveltyAudit,
    Diagnosis,
    EvidenceUse,
    EvidenceItem,
    EvidenceSnapshot,
    ExperimentKnobs,
    ExperimentOutcome,
    ExperimentSpec,
    HypothesisCandidate,
    HypothesisFeedback,
    HypothesisMatrix,
    OpenDeepResearchConfig,
    OpenResearcherConfig,
    ResearchSource,
)
from slm_training.autoresearch.storage import CampaignStore


def campaign() -> CampaignSpec:
    return CampaignSpec(
        campaign_id="test-campaign",
        objective="Improve honest held-out structural similarity.",
        primary_metric="held_out.structural_similarity",
        researcher_mode="fixture",
    )


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
            )
        )
    return HypothesisMatrix(
        matrix_id=matrix_id,
        campaign_id="test-campaign",
        evidence_snapshot_id="evidence-matrix",
        hypotheses=tuple(candidates),
        recommended_experiment_id=f"hyp-{offset}",
        selection_rationale="Highest-information safe candidate in this matrix.",
        predecessor_matrix_id=predecessor_matrix_id,
        feedback_ids=feedback_ids,
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
                "n": 10,
                "parse_rate": 1,
                "structural_similarity": 1,
                "placeholder_fidelity": 1,
                "reward_score": 1,
            },
            "held_out": {
                "n": 10,
                "parse_rate": 1,
                "structural_similarity": 1,
                "placeholder_fidelity": 1,
            },
            "adversarial": {"n": 10, "parse_rate": 1, "structural_similarity": 1},
            "ood": {"n": 10, "parse_rate": 1, "structural_similarity": 1},
            "rico_held": {"n": 1500, "parse_rate": 1, "structural_similarity": 1},
        },
        "agentv": {"passed": True},
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
        for line in (store.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
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
                "budget": base_campaign.budget.model_copy(
                    update={"max_experiments": 5}
                )
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
        for row in (json.loads(line) for line in store.root.joinpath("events.jsonl").read_text().splitlines())
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
        tmp_path
        / "outputs/autoresearch/prior/artifacts/hypothesizer_feedback"
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
    rows = _load_sources(path)
    assert len(rows) == 19
    assert len({row.uri for row in rows}) == 19
    assert all(row.uri.startswith("https://arxiv.org/abs/") for row in rows)
    assert all(row.metadata.get("scope_diff_takeaway") for row in rows)
    assert {row.metadata.get("implementation_status") for row in rows} == {
        "Adapted",
        "Adjacent",
    }


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
    assert any(source.uri == "https://example.com/new-paper" for source in result.sources)


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
        json.dumps({"case_id": "feedback-case", "matrix": matrix.model_dump(mode="json")})
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
    assert not report.promotable


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
                    {"uri": "outputs/runs/prior/run_insights.json", "role": "prior_trace"},
                    {"uri": "outputs/runs/prior/scoreboard.json", "role": "prior_result"},
                ],
                "required_roles": ["research", "prior_trace", "prior_result"],
                "expected_knobs": ["steps" if index == 0 else "lr"],
                "feedback_ids": [],
                "predecessor_matrix_id": None,
            }
        )
        predictions.append(
            {"case_id": f"case-{index}", "matrix": hypothesis_matrix().model_dump(mode="json")}
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
    assert commands[0][:4] == ["python", "-m", "scripts.build_train_data", "--source"]
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


def test_compile_resolves_canonical_published_train_version() -> None:
    spec = experiment(
        knobs=ExperimentKnobs(
            train_version="e218_schema_normalized_judge_v5",
            steps=32,
            output_tokenizer="lexer",
            compiler_alignment_loss_weight=1.0,
            compiler_alignment_stratified=True,
            compiler_decode_mode="tree",
            schema_in_context=True,
            slot_contract_in_context=True,
            design_md_context=False,
        )
    )

    commands = compile_commands(campaign(), spec)

    assert "--train-version" in commands[0]
    assert "e218_schema_normalized_judge_v5" in commands[0]
    assert "--train-dir" not in commands[0]
    assert "--train-version" in commands[-1]
    assert commands[0][commands[0].index("--output-tokenizer") + 1] == "lexer"
    assert "--compiler-alignment-stratified" in commands[0]
    assert "--schema-in-context" in commands[0]
    assert "--slot-contract-in-context" in commands[0]
    assert "--no-design-md-context" in commands[0]
    assert commands[-1][commands[-1].index("--compiler-decode-mode") + 1] == "tree"


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
    report = assess_rl_readiness(passing_evaluation())
    assert report.approved
    assert_rl_ready(report)
    failed = assess_rl_readiness({"suites": {}, "reward_samples": [1, 1]})
    assert not failed.approved
    with pytest.raises(ValueError, match="RL is locked"):
        assert_rl_ready(failed)
    with pytest.raises(ValueError, match="provide an approved"):
        assert_rl_ready(None)


def test_remote_sync_is_explicit_and_non_destructive(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    plan = sync_campaign(tmp_path, "test-campaign")
    assert plan["push"] is False
    assert plan["command"][:3] == ["hf", "buckets", "sync"]
    assert "--no-delete" in plan["command"]
    assert str(plan["remote_uri"]).endswith("/autoresearch/test-campaign")
