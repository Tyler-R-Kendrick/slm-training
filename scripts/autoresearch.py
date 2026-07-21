#!/usr/bin/env python3
"""Evidence-grounded autonomous research and training campaign harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from slm_training.levers import MAX_RUN_MINUTES
from slm_training.autoresearch.engine import (
    compile_commands,
    create_hypothesis_feedback,
    diagnose_outcome,
    execute_commands,
    validate_experiment,
    validate_hypothesis_matrix,
)
from slm_training.autoresearch.evidence import collect_evidence
from slm_training.autoresearch.hypothesizer_eval import evaluate_hypothesizer
from slm_training.autoresearch.literature import (
    HuggingFacePapersClient,
    categorical_discovery_source,
)
from slm_training.autoresearch.persistence import sync_campaign
from slm_training.autoresearch.providers import (
    AgentHypothesisProvider,
    AgentProposalProvider,
    FixtureResearchProvider,
    OpenAIProposalCompiler,
    OpenAIHypothesizer,
    OpenAIResearchProvider,
)
from slm_training.autoresearch.researchers import RESEARCHERS, get_researcher
from slm_training.autoresearch.researcher_eval import evaluate_researcher
from slm_training.autoresearch.rl_gate import assess_rl_readiness, write_rl_readiness
from slm_training.autoresearch.schemas import (
    CampaignBudget,
    CampaignSpec,
    Diagnosis,
    EvidenceSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    HypothesisFeedback,
    HypothesisMatrix,
    ResearcherRun,
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
        min_hypotheses=args.min_hypotheses,
        evidence_roots=tuple(str(path) for path in args.evidence_root),
        budget=CampaignBudget(
            max_experiments=args.max_experiments,
            max_gpu_hours=args.max_gpu_hours,
            max_wall_minutes=args.max_wall_minutes,
        ),
        notes=args.notes,
    )
    path = _store(args).initialize(campaign)
    print(
        json.dumps(
            {"campaign": str(path), "spec": campaign.model_dump(mode="json")}, indent=2
        )
    )
    return 0


def _capture(
    store: CampaignStore, campaign: CampaignSpec
) -> tuple[EvidenceSnapshot, Path]:
    evidence = collect_evidence(campaign.evidence_roots, repo_root=ROOT)
    path = store.write_artifact("evidence", evidence)
    store.append_event(
        "evidence_captured", artifact_sha256=path.stem, detail=evidence.source_counts
    )
    return evidence, path


def _research_sources(args: argparse.Namespace) -> list[ResearchSource]:
    sources: list[ResearchSource] = [categorical_discovery_source()]
    for manifest in args.source_manifest or ():
        sources.extend(_load_sources(manifest))
    if not args.offline:
        client = HuggingFacePapersClient(token=os.environ.get("HF_TOKEN"))
        for query in args.paper_query:
            sources.extend(client.search(query, limit=args.hf_limit))
        sources.extend(client.daily(days=args.hf_days, limit_per_day=args.hf_limit))
    return list({source.uri: source for source in sources}.values())


def cmd_research(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    evidence, evidence_path = _capture(store, campaign)
    sources = _research_sources(args)
    researcher_path = None
    researcher_id = args.researcher
    if researcher_id is None and campaign.researcher_mode in RESEARCHERS:
        researcher_id = campaign.researcher_mode
    if researcher_id:
        if args.offline:
            raise ValueError("--offline cannot be combined with an external researcher")
        if not args.researcher_checkout or not args.researcher_python:
            raise ValueError(
                "external researcher requires --researcher-checkout and --researcher-python"
            )
        config = (
            json.loads(args.researcher_config.read_text(encoding="utf-8"))
            if args.researcher_config
            else {}
        )
        researcher = get_researcher(
            researcher_id,
            checkout=args.researcher_checkout,
            python=args.researcher_python,
            worker=ROOT / "scripts" / "researcher_worker.py",
            config=config,
            timeout_seconds=(
                args.researcher_timeout
                if args.researcher_timeout is not None
                else float(campaign.budget.max_wall_minutes * 60)
            ),
        )
        run = researcher.run(campaign, evidence, sources)
        researcher_path = store.write_artifact("researcher_runs", run)
        store.append_event(
            "researcher_completed"
            if run.status == "completed"
            else "researcher_failed",
            status=run.status,
            artifact_sha256=researcher_path.stem,
            detail={
                "researcher_id": run.researcher_id,
                "upstream_revision": run.upstream_revision,
                "error": run.error,
            },
        )
        if run.status == "failed":
            raise RuntimeError(run.error)
        sources = list(run.sources)
    source_path = store.write_artifact(
        "research_sources",
        {"sources": [item.model_dump(mode="json") for item in sources]},
    )
    store.append_event(
        "literature_captured",
        artifact_sha256=source_path.stem,
        detail={"sources": len(sources), "evidence_snapshot_id": evidence.snapshot_id},
    )
    print(
        json.dumps(
            {
                "evidence": str(evidence_path),
                "sources": str(source_path),
                "researcher_run": str(researcher_path) if researcher_path else None,
            },
            indent=2,
        )
    )
    return 0


def _load_sources(path: Path | None) -> list[ResearchSource]:
    if not path:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    retrieved_at = payload.get("retrieved_at")
    return [
        ResearchSource.model_validate(
            {**({"retrieved_at": retrieved_at} if retrieved_at else {}), **item}
        )
        for item in payload.get("sources", [])
    ]


def _load_memo(
    store: CampaignStore, path: Path | None, *, required: bool = True
) -> str:
    if path:
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(payload, dict):
            raise ValueError("memo JSON must be an object with memo or text")
        return str(payload.get("memo") or payload.get("text") or "")
    candidates = sorted(
        (store.root / "artifacts" / "researcher_runs").glob("*.json"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        run = ResearcherRun.model_validate_json(candidate.read_text(encoding="utf-8"))
        if run.status == "completed":
            return run.memo
    if required:
        raise FileNotFoundError(f"no completed researcher run for {store.campaign_id}")
    return ""


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
    evidence = EvidenceSnapshot.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    source_path = args.sources
    if source_path is None:
        candidates = list(
            (store.root / "artifacts" / "research_sources").glob("*.json")
        )
        source_path = (
            max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None
        )
    sources = _load_sources(source_path)
    if args.provider and args.compiler:
        raise ValueError("choose --provider (legacy) or --compiler, not both")
    mode = args.compiler or args.provider
    if not mode:
        raise ValueError("propose requires --compiler or legacy --provider")
    if mode == "agent":
        if not args.proposal:
            raise ValueError("--provider agent requires --proposal")
        provider = AgentProposalProvider(args.proposal)
        result = provider.propose(campaign, evidence, sources)
    elif mode == "fixture":
        if not args.proposal:
            raise ValueError("--provider fixture requires --proposal")
        fixture = ExperimentSpec.model_validate_json(
            args.proposal.read_text(encoding="utf-8")
        )
        provider = FixtureResearchProvider(fixture)
        result = provider.propose(campaign, evidence, sources)
    elif args.compiler == "openai":
        provider = OpenAIProposalCompiler(model=args.model)
        result = provider.propose(
            campaign,
            evidence,
            sources,
            _load_memo(store, args.memo),
        )
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


def cmd_hypothesize(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    evidence_path = _artifact(store, "evidence", args.evidence)
    evidence = EvidenceSnapshot.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    source_path = args.sources
    if source_path is None:
        candidates = list(
            (store.root / "artifacts" / "research_sources").glob("*.json")
        )
        source_path = (
            max(candidates, key=lambda path: path.stat().st_mtime_ns)
            if candidates
            else None
        )
    sources = _load_sources(source_path)
    previous_matrix = _latest_formed_matrix(store, required=False)
    feedback = _hypothesis_feedback(store, previous_matrix)
    if previous_matrix is not None and not feedback:
        raise ValueError(
            "latest hypothesis matrix has no terminal feedback; run a matrix member "
            "before forming its successor"
        )
    if args.provider == "agent":
        if not args.matrix:
            raise ValueError("--provider agent requires --matrix")
        result = AgentHypothesisProvider(args.matrix).propose(
            campaign, evidence, sources, feedback
        )
    else:
        result = OpenAIHypothesizer(model=args.model).propose(
            campaign,
            evidence,
            sources,
            _load_memo(store, args.memo, required=False),
            feedback,
        )
    validate_hypothesis_matrix(
        campaign,
        result.matrix,
        evidence,
        list(result.sources),
        prior_experiments=_finished_experiments(store),
        prior_experiment_ids=frozenset(
            candidate.experiment.experiment_id
            for formed in _formed_matrices(store)
            for candidate in formed.hypotheses
        ),
        feedback=feedback,
        previous_matrix=previous_matrix,
    )

    proposed = {
        str(row.get("experiment_id"))
        for row in _events(store)
        if row.get("event_type") == "experiment_proposed"
    }
    matrix_path = store.write_artifact("hypothesis_matrices", result.matrix)
    source_out = store.write_artifact(
        "hypothesis_sources",
        {"sources": [item.model_dump(mode="json") for item in result.sources]},
    )
    telemetry_path = store.write_artifact("hypothesizer_telemetry", result.telemetry)
    if result.research_memo:
        store.write_artifact("research_memos", {"text": result.research_memo})
    store.append_event(
        "hypothesis_matrix_formed",
        status="planned",
        artifact_sha256=matrix_path.stem,
        detail={
            "count": len(result.matrix.hypotheses),
            "recommended_experiment_id": result.matrix.recommended_experiment_id,
            "sources_sha": source_out.stem,
            "telemetry_sha": telemetry_path.stem,
        },
    )
    experiment_paths = []
    for candidate in result.matrix.hypotheses:
        experiment_path = store.write_artifact("experiments", candidate.experiment)
        experiment_paths.append(str(experiment_path))
        if candidate.experiment.experiment_id not in proposed:
            store.append_event(
                "experiment_proposed",
                experiment_id=candidate.experiment.experiment_id,
                status="planned",
                artifact_sha256=experiment_path.stem,
                detail={"hypothesis_matrix_sha": matrix_path.stem},
            )
    print(
        json.dumps(
            {
                "hypothesis_matrix": str(matrix_path),
                "recommended_experiment_id": result.matrix.recommended_experiment_id,
                "experiments": experiment_paths,
            },
            indent=2,
        )
    )
    return 0


def _events(store: CampaignStore) -> list[dict]:
    path = store.root / "events.jsonl"
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )


def _finished_experiments(store: CampaignStore) -> tuple[ExperimentSpec, ...]:
    finished = {
        str(row.get("experiment_id"))
        for row in _events(store)
        if row.get("event_type") == "experiment_finished"
    }
    found = {}
    for path in (store.root / "artifacts" / "experiments").glob("*.json"):
        experiment = ExperimentSpec.model_validate_json(path.read_text(encoding="utf-8"))
        if experiment.experiment_id in finished:
            found[experiment.experiment_id] = experiment
    return tuple(found.values())


def _latest_formed_matrix(
    store: CampaignStore, *, required: bool = True
) -> HypothesisMatrix | None:
    formed = [
        row
        for row in _events(store)
        if row.get("event_type") == "hypothesis_matrix_formed"
    ]
    if not formed:
        if required:
            raise FileNotFoundError(
                f"no formed hypothesis matrix for {store.campaign_id}"
            )
        return None
    digest = str(formed[-1]["artifact_sha256"])
    path = store.root / "artifacts" / "hypothesis_matrices" / f"{digest}.json"
    return HypothesisMatrix.model_validate_json(path.read_text(encoding="utf-8"))


def _formed_matrices(store: CampaignStore) -> tuple[HypothesisMatrix, ...]:
    matrices = []
    for row in _events(store):
        if row.get("event_type") != "hypothesis_matrix_formed":
            continue
        digest = str(row["artifact_sha256"])
        path = store.root / "artifacts" / "hypothesis_matrices" / f"{digest}.json"
        matrices.append(
            HypothesisMatrix.model_validate_json(path.read_text(encoding="utf-8"))
        )
    return tuple(matrices)


def _recorded_outcome_matches(
    store: CampaignStore, event: dict, outcome: ExperimentOutcome
) -> bool:
    digest = event.get("artifact_sha256")
    if not digest:
        return False
    path = store.root / "artifacts" / "outcomes" / f"{digest}.json"
    return path.exists() and ExperimentOutcome.model_validate_json(
        path.read_text(encoding="utf-8")
    ) == outcome


def _matrix_for_outcome(
    store: CampaignStore, outcome: ExperimentOutcome
) -> HypothesisMatrix | None:
    """Resolve matrix lineage only from the outcome's recorded run provenance."""
    events = _events(store)
    finished_index = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index].get("event_type") == "experiment_finished"
            and events[index].get("experiment_id") == outcome.experiment_id
            and _recorded_outcome_matches(store, events[index], outcome)
        ),
        None,
    )
    if finished_index is None:
        return None
    matrix_id = next(
        (
            str(row.get("detail", {}).get("hypothesis_matrix_id"))
            for row in reversed(events[:finished_index])
            if row.get("event_type") == "experiment_started"
            and row.get("experiment_id") == outcome.experiment_id
            and row.get("detail", {}).get("hypothesis_matrix_id")
        ),
        None,
    )
    if matrix_id is None:
        return None
    return next(
        (
            matrix
            for matrix in reversed(_formed_matrices(store))
            if matrix.matrix_id == matrix_id
        ),
        None,
    )


def _hypothesis_feedback(
    store: CampaignStore, previous_matrix: HypothesisMatrix | None
) -> tuple[HypothesisFeedback, ...]:
    if previous_matrix is None:
        return ()
    rows = []
    for path in sorted(
        (store.root / "artifacts" / "hypothesizer_feedback").glob("*.json")
    ):
        item = HypothesisFeedback.model_validate_json(path.read_text(encoding="utf-8"))
        if item.matrix_id == previous_matrix.matrix_id:
            rows.append(item)
    return tuple(rows)


def _require_hypothesis_matrix(
    store: CampaignStore, campaign: CampaignSpec, experiment: ExperimentSpec
) -> HypothesisMatrix:
    matrix = _latest_formed_matrix(store)
    assert matrix is not None
    if matrix.campaign_id != campaign.campaign_id:
        raise ValueError("latest hypothesis matrix belongs to a different campaign")
    if len(matrix.hypotheses) < campaign.min_hypotheses:
        raise ValueError(
            f"run requires at least {campaign.min_hypotheses} formed hypotheses"
        )
    matches = [
        item.experiment
        for item in matrix.hypotheses
        if item.experiment.experiment_id == experiment.experiment_id
    ]
    if not matches or matches[0] != experiment:
        raise ValueError("experiment is not an exact member of the latest hypothesis matrix")
    return matrix


def cmd_validate(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    experiment = ExperimentSpec.model_validate_json(
        args.experiment.read_text(encoding="utf-8")
    )
    evidence_path = _artifact(store, "evidence", args.evidence)
    evidence = EvidenceSnapshot.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    sources = _load_sources(args.sources)
    validate_experiment(campaign, experiment, evidence, sources)
    print(
        json.dumps({"valid": True, "experiment_id": experiment.experiment_id}, indent=2)
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store = _store(args)
    campaign = store.load_campaign()
    matrix = _latest_formed_matrix(store)
    assert matrix is not None
    if args.experiment:
        experiment = ExperimentSpec.model_validate_json(
            args.experiment.read_text(encoding="utf-8")
        )
    else:
        experiment = next(
            item.experiment
            for item in matrix.hypotheses
            if item.experiment.experiment_id == matrix.recommended_experiment_id
        )
    _require_hypothesis_matrix(store, campaign, experiment)
    started = {
        str(row.get("experiment_id"))
        for row in _events(store)
        if row.get("event_type") == "experiment_started"
    }
    if (
        args.execute
        and experiment.experiment_id not in started
        and len(started) >= campaign.budget.max_experiments
    ):
        raise ValueError(
            f"campaign execution budget exhausted: {len(started)}/"
            f"{campaign.budget.max_experiments}"
        )
    commands = compile_commands(campaign, experiment, output_root=args.root)
    plan_path = store.write_artifact(
        "execution_plans", {"commands": commands, "execute": args.execute}
    )
    store.append_event(
        "execution_planned",
        experiment_id=experiment.experiment_id,
        status="planned",
        artifact_sha256=plan_path.stem,
    )
    if not args.execute:
        print(json.dumps({"execute": False, "commands": commands}, indent=2))
        return 0
    if experiment.experiment_id not in started:
        store.append_event(
            "experiment_started",
            experiment_id=experiment.experiment_id,
            status="running",
            detail={"hypothesis_matrix_id": matrix.matrix_id},
        )
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
    _record_hypothesis_feedback(store, matrix, outcome, diagnosis)
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
    outcome = ExperimentOutcome.model_validate_json(
        args.outcome.read_text(encoding="utf-8")
    )
    diagnosis = diagnose_outcome(outcome)
    path = store.write_artifact("diagnoses", diagnosis)
    store.append_event(
        "outcome_diagnosed",
        experiment_id=outcome.experiment_id,
        status=diagnosis.target,
        artifact_sha256=path.stem,
    )
    matrix = _matrix_for_outcome(store, outcome)
    if matrix is not None:
        _record_hypothesis_feedback(store, matrix, outcome, diagnosis)
    print(diagnosis.model_dump_json(indent=2))
    return 0


def _record_hypothesis_feedback(
    store: CampaignStore,
    matrix: HypothesisMatrix,
    outcome: ExperimentOutcome,
    diagnosis: Diagnosis,
) -> Path:
    feedback = create_hypothesis_feedback(matrix, outcome, diagnosis)
    path = store.write_artifact("hypothesizer_feedback", feedback)
    already_recorded = any(
        row.get("event_type") == "hypothesizer_feedback_recorded"
        and row.get("artifact_sha256") == path.stem
        for row in _events(store)
    )
    if not already_recorded:
        store.append_event(
            "hypothesizer_feedback_recorded",
            experiment_id=outcome.experiment_id,
            status=feedback.diagnosis_target,
            artifact_sha256=path.stem,
            detail={"feedback_id": feedback.feedback_id, "matrix_id": matrix.matrix_id},
        )
    return path


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


def cmd_evaluate_hypothesizer(args: argparse.Namespace) -> int:
    report = evaluate_hypothesizer(
        args.cases,
        args.predictions,
        run_dir=args.run_dir,
        hypothesizer_id=args.hypothesizer_id,
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


def cmd_remine(args: argparse.Namespace) -> int:
    """Run the LDI3-04 immutable remine → intervene → regenerate campaign."""
    # Lazy import so importing autoresearch.py does not pull torch-heavy paths.
    from slm_training.harnesses.preference.remine_campaign import (
        RemineCampaignConfig,
        describe_campaign,
        run_campaign,
    )

    _smoke_config = {
        "campaign_id": "ldi-remine-smoke",
        "created_at": "2026-07-18T00:00:00Z",
        "base_checkpoint_sha": "fixture-parent",
        "tokenizer_sha": "fixture-tokenizer",
        "prompt_group_ids": ["group_a", "group_b"],
        "suite_mix": ["grammar", "schema", "dataflow"],
        "decode_config_hash": "fixture-decode-v1",
        "seeds": [0, 1],
        "verifier_bundle_hash": "fixture-verifier-v1",
        "adapter_spec": {"method": "twotower_adapter", "rank": 4},
        "max_iterations": 2,
        "min_new_evidence": 1,
        "notes": "wiring-only fixture smoke",
    }

    data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else _smoke_config
    config = RemineCampaignConfig.from_mapping(data)

    if args.describe or not args.smoke:
        print(json.dumps(describe_campaign(config), indent=2, sort_keys=True))
        return 0

    result = run_campaign(config, root=args.root)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/autoresearch"))
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--objective", required=True)
    init.add_argument("--primary-metric", required=True)
    init.add_argument(
        "--track",
        choices=("twotower", "grammar_diffusion", "causal_lm"),
        default="twotower",
    )
    init.add_argument("--researcher-mode", default="agent")
    init.add_argument("--min-hypotheses", type=int, default=5)
    init.add_argument(
        "--evidence-root", type=Path, action="append", default=[Path("outputs")]
    )
    init.add_argument("--max-experiments", type=int, default=12)
    init.add_argument("--max-gpu-hours", type=float, default=0)
    init.add_argument(
        "--max-wall-minutes",
        type=float,
        default=float(MAX_RUN_MINUTES),
        help=(
            "Cumulative per-experiment wall budget "
            f"(default and maximum: {MAX_RUN_MINUTES} minutes)."
        ),
    )
    init.add_argument("--notes", default="")
    init.set_defaults(func=cmd_init)

    research = sub.add_parser("research")
    research.add_argument("--campaign-id", required=True)
    research.add_argument("--offline", action="store_true")
    research.add_argument("--hf-days", type=int, default=7)
    research.add_argument("--hf-limit", type=int, default=20)
    research.add_argument(
        "--source-manifest",
        type=Path,
        action="append",
        help="Committed ResearchSource JSON manifest; repeatable and offline-safe.",
    )
    research.add_argument("--researcher", choices=tuple(RESEARCHERS))
    research.add_argument("--researcher-checkout", type=Path)
    research.add_argument("--researcher-python", type=Path)
    research.add_argument("--researcher-config", type=Path)
    research.add_argument("--researcher-timeout", type=float)
    research.add_argument(
        "--paper-query",
        action="append",
        default=[
            "structured generation constrained decoding synthetic data small language models"
        ],
    )
    research.set_defaults(func=cmd_research)

    propose = sub.add_parser("propose")
    propose.add_argument("--campaign-id", required=True)
    propose.add_argument("--provider", choices=("agent", "openai", "fixture"))
    propose.add_argument("--compiler", choices=("openai",))
    propose.add_argument("--proposal", type=Path)
    propose.add_argument("--memo", type=Path)
    propose.add_argument("--evidence", type=Path)
    propose.add_argument("--sources", type=Path)
    propose.add_argument("--model", default="gpt-5.6-sol")
    propose.set_defaults(func=cmd_propose)

    hypothesize = sub.add_parser("hypothesize")
    hypothesize.add_argument("--campaign-id", required=True)
    hypothesize.add_argument("--provider", choices=("agent", "openai"), default="openai")
    hypothesize.add_argument("--matrix", type=Path)
    hypothesize.add_argument("--memo", type=Path)
    hypothesize.add_argument("--evidence", type=Path)
    hypothesize.add_argument("--sources", type=Path)
    hypothesize.add_argument("--model", default="gpt-5.6-sol")
    hypothesize.set_defaults(func=cmd_hypothesize)

    validate = sub.add_parser("validate")
    validate.add_argument("--campaign-id", required=True)
    validate.add_argument("--experiment", type=Path, required=True)
    validate.add_argument("--evidence", type=Path)
    validate.add_argument("--sources", type=Path)
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run")
    run.add_argument("--campaign-id", required=True)
    run.add_argument(
        "--experiment",
        type=Path,
        help="Exact matrix member; defaults to the matrix recommendation.",
    )
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
    benchmark.add_argument(
        "--cases",
        type=Path,
        default=Path("src/slm_training/resources/autoresearch/researcher_cases.json"),
    )
    benchmark.add_argument("--predictions", type=Path, required=True)
    benchmark.add_argument(
        "--run-dir", type=Path, default=Path("outputs/autoresearch/researcher_eval")
    )
    benchmark.add_argument("--researcher-id", required=True)
    benchmark.add_argument("--pass-threshold", type=float, default=0.8)
    benchmark.add_argument("--human-approve", action="store_true")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.set_defaults(func=cmd_evaluate_researcher)

    hypothesis_benchmark = sub.add_parser("evaluate-hypothesizer")
    hypothesis_benchmark.add_argument(
        "--cases",
        type=Path,
        default=Path(
            "src/slm_training/resources/autoresearch/hypothesizer_cases.json"
        ),
    )
    hypothesis_benchmark.add_argument("--predictions", type=Path, required=True)
    hypothesis_benchmark.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/autoresearch/hypothesizer_eval"),
    )
    hypothesis_benchmark.add_argument("--hypothesizer-id", required=True)
    hypothesis_benchmark.add_argument("--pass-threshold", type=float, default=0.8)
    hypothesis_benchmark.add_argument("--human-approve", action="store_true")
    hypothesis_benchmark.add_argument("--output", type=Path, required=True)
    hypothesis_benchmark.set_defaults(func=cmd_evaluate_hypothesizer)

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

    remine = sub.add_parser("remine")
    remine.add_argument("--campaign-id", default="ldi-remine-smoke")
    remine.add_argument("--config", type=Path, help="Path to RemineCampaignConfig JSON")
    remine.add_argument(
        "--describe",
        action="store_true",
        help="Print the default campaign description and exit",
    )
    remine.add_argument(
        "--smoke",
        action="store_true",
        help="Run the built-in fixture smoke (default if no --config)",
    )
    remine.set_defaults(func=cmd_remine)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
