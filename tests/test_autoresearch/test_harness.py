from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.engine import (
    compile_commands,
    diagnose_outcome,
    validate_experiment,
)
from slm_training.autoresearch.evidence import collect_evidence
from slm_training.autoresearch.literature import HuggingFacePapersClient
from slm_training.autoresearch.providers import OpenAIResearchProvider
from slm_training.autoresearch.persistence import sync_campaign
from slm_training.autoresearch.rl_gate import assert_rl_ready, assess_rl_readiness
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    EvidenceItem,
    EvidenceSnapshot,
    ExperimentKnobs,
    ExperimentOutcome,
    ExperimentSpec,
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
            "held_out": {"n": 10, "parse_rate": 1, "structural_similarity": 1, "placeholder_fidelity": 1},
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


def test_campaign_store_is_content_addressed_and_chained(tmp_path: Path) -> None:
    store = CampaignStore("test-campaign", tmp_path)
    store.initialize(campaign())
    spec = experiment()
    first = store.write_artifact("experiments", spec)
    second = store.write_artifact("experiments", spec)
    assert first == second
    event = store.append_event("experiment_proposed", artifact_sha256=first.stem)
    lines = [json.loads(line) for line in (store.root / "events.jsonl").read_text().splitlines()]
    assert lines[-1]["event_id"] == event["event_id"]
    assert lines[-1]["previous_event_sha256"] == lines[-2]["event_id"]
    assert (store.root / "checksums.jsonl").is_file()
    assert (store.root / "results.tsv").is_file()


def test_evidence_normalizes_feedback_telemetry_and_lineage(tmp_path: Path) -> None:
    (tmp_path / "docs/design").mkdir(parents=True)
    (tmp_path / "docs/design/research-lineage.md").write_text("# lineage\nPrior result")
    outputs = tmp_path / "outputs/run-1"
    outputs.mkdir(parents=True)
    (outputs / "train_telemetry.json").write_text(json.dumps({"loss": 1.2}))
    (outputs / "human_feedback.jsonl").write_text('{"reward":0.5}\n')
    snapshot = collect_evidence(["outputs"], repo_root=tmp_path)
    assert snapshot.source_counts["repo_lineage"] == 1
    assert snapshot.source_counts["telemetry"] == 1
    assert snapshot.source_counts["feedback"] == 1
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
                    "paper": {"id": "2601.00001", "title": "A paper", "summary": "Useful"},
                    "numUpvotes": 5,
                }
            ]
        )


def test_hf_daily_papers_client_preserves_sources() -> None:
    client = HuggingFacePapersClient(client=FakeHTTP())
    rows = client.daily(days=1, limit_per_day=2)
    assert rows[0].kind == "hf_daily_paper"
    assert rows[0].uri.endswith("2601.00001")


class FakeResponses:
    def create(self, **kwargs):
        assert kwargs["store"] is False
        return SimpleNamespace(
            id="resp-discovery",
            model="gpt-test",
            output_text="memo",
            usage={"input_tokens": 10},
            to_dict=lambda: {"sources": [{"url": "https://example.com/paper", "title": "Paper"}]},
        )

    def parse(self, **kwargs):
        assert kwargs["store"] is False
        return SimpleNamespace(
            id="resp-structured",
            model="gpt-test",
            usage={"output_tokens": 20},
            output_parsed=experiment(citations=("fixture://prior-run",)),
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


def test_compile_is_typed_and_diagnosis_routes_bad_data() -> None:
    spec = experiment(
        knobs=ExperimentKnobs(
            data_source="existing",
            derive_from="outputs/train_data/old/records.jsonl",
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
