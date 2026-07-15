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
    diagnose_outcome,
    validate_experiment,
)
from slm_training.autoresearch.evidence import collect_evidence
from slm_training.autoresearch.literature import HuggingFacePapersClient
from slm_training.autoresearch.providers import (
    OpenAIProposalCompiler,
    OpenAIResearchProvider,
)
from slm_training.autoresearch.researchers import IsolatedResearcher, ResearcherSpec
from slm_training.autoresearch.persistence import sync_campaign
from slm_training.autoresearch.rl_gate import assert_rl_ready, assess_rl_readiness
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    EvidenceItem,
    EvidenceSnapshot,
    ExperimentKnobs,
    ExperimentOutcome,
    ExperimentSpec,
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
