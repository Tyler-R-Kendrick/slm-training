#!/usr/bin/env python3
"""Evidence-grounded autonomous research and training campaign harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from slm_training.autoresearch.engine import (
    compile_commands,
    diagnose_outcome,
    execute_commands,
    validate_experiment,
)
from slm_training.autoresearch.evidence import collect_evidence
from slm_training.autoresearch.literature import HuggingFacePapersClient
from slm_training.autoresearch.persistence import sync_campaign
from slm_training.autoresearch.providers import (
    AgentProposalProvider,
    FixtureResearchProvider,
    OpenAIResearchProvider,
)
from slm_training.autoresearch.researcher_eval import evaluate_researcher
from slm_training.autoresearch.rl_gate import assess_rl_readiness, write_rl_readiness
from slm_training.autoresearch.schemas import (
    CampaignBudget,
    CampaignSpec,
    EvidenceSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    ResearchSource,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.autoresearch.telemetry import TrackioSink
from slm_training.data.mixture import MixtureManifest, write_mixture_manifest

ROOT = Path(__file__).resolve().parents[1]


def _store(args: argparse.Namespace) -> CampaignStore:
    return CampaignStore(args.campaign_id, args.root)


def _artifact(store: CampaignStore, kind: str, path: Path | None):
    if path:
        return path
    files = list((store.root / "artifacts" / kind).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no {kind} artifact for {store.campaign_id}")
    return max(files, key=lambda item: item.stat().st_mtime_ns)


def cmd_init(args: argparse.Namespace) -> int:
    campaign = CampaignSpec(
        campaign_id=args.campaign_id,
        objective=args.objective,
        primary_metric=args.primary_metric,
        track=args.track,
        researcher_mode=args.researcher_mode,
        evidence_roots=tuple(str(path) for path in args.evidence_root),
        budget=CampaignBudget(
            max_experiments=args.max_experiments,
            max_gpu_hours=args.max_gpu_hours,
            max_wall_minutes=args.max_wall_minutes,
        ),
        notes=args.notes,
    )
    path = _store(args).initialize(campaign)
    print(json.dumps({"campaign": str(path), "spec": campaign.model_dump(mode="json")}, indent=2))
    return 0


def _capture(store: CampaignStore, campaign: CampaignSpec) -> tuple[EvidenceSnapshot, Path]:
    evidence = collect_evidence(campaign.evidence_roots, repo_root=ROOT)
    path = store.write_artifact("evidence", evidence)
    store.append_event("evidence_captured", artifact_sha256=path.stem, detail=evidence.source_counts)
    return evidence, path


def _research_sources(args: argparse.Namespace) -> list[ResearchSource]:
    if args.offline:
        return []
    client = HuggingFacePapersClient(token=os.environ.get("HF_TOKEN"))
    sources: list[ResearchSource] = []
    for query in args.paper_query:
        sources.extend(client.search(query, limit=args.hf_limit))
    sources.extend(client.daily(days=args.hf_days, limit_per_day=args.hf_limit))
    return list({source.uri: source for source in sources}.values())


def cmd_research(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    evidence, evidence_path = _capture(store, campaign)
    sources = _research_sources(args)
    source_path = store.write_artifact(
        "research_sources", {"sources": [item.model_dump(mode="json") for item in sources]}
    )
    store.append_event(
        "literature_captured",
        artifact_sha256=source_path.stem,
        detail={"sources": len(sources), "evidence_snapshot_id": evidence.snapshot_id},
    )
    print(json.dumps({"evidence": str(evidence_path), "sources": str(source_path)}, indent=2))
    return 0


def _load_sources(path: Path | None) -> list[ResearchSource]:
    if not path:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ResearchSource.model_validate(item) for item in payload.get("sources", [])]


def cmd_propose(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    proposed = set()
    events_path = store.root / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event_type") == "experiment_proposed":
                proposed.add(str(row.get("experiment_id")))
    if len(proposed) >= campaign.budget.max_experiments:
        raise ValueError(
            f"campaign experiment budget exhausted: {len(proposed)}/"
            f"{campaign.budget.max_experiments}"
        )
    evidence_path = _artifact(store, "evidence", args.evidence)
    evidence = EvidenceSnapshot.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    source_path = args.sources
    if source_path is None:
        candidates = list((store.root / "artifacts" / "research_sources").glob("*.json"))
        source_path = max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None
    sources = _load_sources(source_path)
    if args.provider == "agent":
        if not args.proposal:
            raise ValueError("--provider agent requires --proposal")
        provider = AgentProposalProvider(args.proposal)
    elif args.provider == "fixture":
        if not args.proposal:
            raise ValueError("--provider fixture requires --proposal")
        fixture = ExperimentSpec.model_validate_json(args.proposal.read_text(encoding="utf-8"))
        provider = FixtureResearchProvider(fixture)
    else:
        provider = OpenAIResearchProvider(model=args.model)
    result = provider.propose(campaign, evidence, sources)
    validate_experiment(campaign, result.experiment, evidence, list(result.sources))
    experiment_path = store.write_artifact("experiments", result.experiment)
    source_out = store.write_artifact(
        "experiment_sources",
        {"sources": [item.model_dump(mode="json") for item in result.sources]},
    )
    telemetry_path = store.write_artifact("researcher_telemetry", result.telemetry)
    if result.research_memo:
        store.write_artifact("research_memos", {"text": result.research_memo})
    store.append_event(
        "experiment_proposed",
        experiment_id=result.experiment.experiment_id,
        status="planned",
        artifact_sha256=experiment_path.stem,
        detail={"sources_sha": source_out.stem, "telemetry_sha": telemetry_path.stem},
    )
    print(json.dumps({"experiment": str(experiment_path)}, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    experiment = ExperimentSpec.model_validate_json(args.experiment.read_text(encoding="utf-8"))
    evidence_path = _artifact(store, "evidence", args.evidence)
    evidence = EvidenceSnapshot.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    sources = _load_sources(args.sources)
    validate_experiment(campaign, experiment, evidence, sources)
    print(json.dumps({"valid": True, "experiment_id": experiment.experiment_id}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    experiment = ExperimentSpec.model_validate_json(args.experiment.read_text(encoding="utf-8"))
    commands = compile_commands(campaign, experiment, output_root=args.root)
    plan_path = store.write_artifact("execution_plans", {"commands": commands, "execute": args.execute})
    store.append_event(
        "execution_planned",
        experiment_id=experiment.experiment_id,
        status="planned",
        artifact_sha256=plan_path.stem,
    )
    if not args.execute:
        print(json.dumps({"execute": False, "commands": commands}, indent=2))
        return 0
    outcome = execute_commands(
        experiment,
        commands,
        cwd=ROOT,
        timeout_seconds=float(campaign.budget.max_wall_minutes * 60),
    )
    outcome_path = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_finished",
        experiment_id=experiment.experiment_id,
        status=outcome.status,
        artifact_sha256=outcome_path.stem,
        detail={"exit_code": outcome.exit_code},
    )
    diagnosis = diagnose_outcome(outcome)
    diagnosis_path = store.write_artifact("diagnoses", diagnosis)
    store.append_event(
        "outcome_diagnosed",
        experiment_id=experiment.experiment_id,
        status=diagnosis.target,
        artifact_sha256=diagnosis_path.stem,
    )
    if args.trackio:
        try:
            TrackioSink(
                project="openui-autoresearch",
                run=f"{campaign.campaign_id}-{experiment.experiment_id}",
            ).log(
                {
                    "completed": float(outcome.status == "completed"),
                    "exit_code": float(outcome.exit_code or 0),
                }
            )
        except Exception as exc:  # noqa: BLE001
            store.append_event(
                "trackio_mirror_failed",
                experiment_id=experiment.experiment_id,
                status="failed",
                detail={"error": str(exc)},
            )
    print(outcome.model_dump_json(indent=2))
    return 0 if outcome.status == "completed" else 2


def cmd_diagnose(args: argparse.Namespace) -> int:
    store = _store(args)
    outcome = ExperimentOutcome.model_validate_json(args.outcome.read_text(encoding="utf-8"))
    diagnosis = diagnose_outcome(outcome)
    path = store.write_artifact("diagnoses", diagnosis)
    store.append_event(
        "outcome_diagnosed",
        experiment_id=outcome.experiment_id,
        status=diagnosis.target,
        artifact_sha256=path.stem,
    )
    print(diagnosis.model_dump_json(indent=2))
    return 0


def cmd_validate_rl(args: argparse.Namespace) -> int:
    report = assess_rl_readiness(args.evaluation)
    write_rl_readiness(args.output, report, overwrite=args.overwrite)
    print(report.model_dump_json(indent=2))
    return 0 if report.approved else 2


def cmd_evaluate_researcher(args: argparse.Namespace) -> int:
    report = evaluate_researcher(
        args.cases,
        args.predictions,
        run_dir=args.run_dir,
        researcher_id=args.researcher_id,
        pass_threshold=args.pass_threshold,
        human_approved=args.human_approve,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0 if report.passed else 2


def cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(_store(args).status(), indent=2))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    result = sync_campaign(args.root, args.campaign_id, push=args.push)
    store = _store(args)
    artifact = store.write_artifact("sync", result)
    store.append_event(
        "campaign_synced" if args.push else "campaign_sync_planned",
        artifact_sha256=artifact.stem,
        detail={"remote_uri": result["remote_uri"]},
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_materialize_mixture(args: argparse.Namespace) -> int:
    weights = json.loads(args.weights_json)
    manifest = MixtureManifest(
        mixture_id=args.mixture_id,
        weights={str(key): float(value) for key, value in weights.items()},
        notes="Typed autoresearch experiment mixture.",
    )
    path = write_mixture_manifest(args.output, manifest)
    print(json.dumps({"mixture": str(path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/autoresearch"))
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--objective", required=True)
    init.add_argument("--primary-metric", required=True)
    init.add_argument("--track", choices=("twotower", "causal_lm"), default="twotower")
    init.add_argument("--researcher-mode", choices=("agent", "openai", "fixture"), default="agent")
    init.add_argument("--evidence-root", type=Path, action="append", default=[Path("outputs")])
    init.add_argument("--max-experiments", type=int, default=12)
    init.add_argument("--max-gpu-hours", type=float, default=0)
    init.add_argument("--max-wall-minutes", type=int, default=240)
    init.add_argument("--notes", default="")
    init.set_defaults(func=cmd_init)

    research = sub.add_parser("research")
    research.add_argument("--campaign-id", required=True)
    research.add_argument("--offline", action="store_true")
    research.add_argument("--hf-days", type=int, default=7)
    research.add_argument("--hf-limit", type=int, default=20)
    research.add_argument(
        "--paper-query",
        action="append",
        default=["structured generation constrained decoding synthetic data small language models"],
    )
    research.set_defaults(func=cmd_research)

    propose = sub.add_parser("propose")
    propose.add_argument("--campaign-id", required=True)
    propose.add_argument("--provider", choices=("agent", "openai", "fixture"), required=True)
    propose.add_argument("--proposal", type=Path)
    propose.add_argument("--evidence", type=Path)
    propose.add_argument("--sources", type=Path)
    propose.add_argument("--model", default="gpt-5.6-sol")
    propose.set_defaults(func=cmd_propose)

    validate = sub.add_parser("validate")
    validate.add_argument("--campaign-id", required=True)
    validate.add_argument("--experiment", type=Path, required=True)
    validate.add_argument("--evidence", type=Path)
    validate.add_argument("--sources", type=Path)
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run")
    run.add_argument("--campaign-id", required=True)
    run.add_argument("--experiment", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--trackio", action="store_true")
    run.set_defaults(func=cmd_run)

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--campaign-id", required=True)
    diagnose.add_argument("--outcome", type=Path, required=True)
    diagnose.set_defaults(func=cmd_diagnose)

    readiness = sub.add_parser("validate-rl")
    readiness.add_argument("--evaluation", type=Path, required=True)
    readiness.add_argument("--output", type=Path, required=True)
    readiness.add_argument("--overwrite", action="store_true")
    readiness.set_defaults(func=cmd_validate_rl)

    benchmark = sub.add_parser("evaluate-researcher")
    benchmark.add_argument("--cases", type=Path, default=Path("src/slm_training/resources/autoresearch/researcher_cases.json"))
    benchmark.add_argument("--predictions", type=Path, required=True)
    benchmark.add_argument("--run-dir", type=Path, default=Path("outputs/autoresearch/researcher_eval"))
    benchmark.add_argument("--researcher-id", required=True)
    benchmark.add_argument("--pass-threshold", type=float, default=0.8)
    benchmark.add_argument("--human-approve", action="store_true")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.set_defaults(func=cmd_evaluate_researcher)

    status = sub.add_parser("status")
    status.add_argument("--campaign-id", required=True)
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync")
    sync.add_argument("--campaign-id", required=True)
    sync.add_argument("--push", action="store_true")
    sync.set_defaults(func=cmd_sync)

    mixture = sub.add_parser("materialize-mixture")
    mixture.add_argument("--output", type=Path, required=True)
    mixture.add_argument("--mixture-id", required=True)
    mixture.add_argument("--weights-json", required=True)
    mixture.set_defaults(func=cmd_materialize_mixture)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
