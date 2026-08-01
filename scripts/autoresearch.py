#!/usr/bin/env python3
"""Evidence-grounded autonomous research and training campaign harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
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
from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.formal import (
    bind_preflight,
    run_formal_preflight,
    validate_formal_preflights,
)
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
from slm_training.autoresearch.rl_gate import (
    assert_rl_ready,
    assess_rl_readiness,
    write_rl_readiness,
)
from slm_training.autoresearch.schemas import (
    AutotrainActionReceiptV1,
    AutotrainCycleHandoffV1,
    CampaignBudget,
    CampaignSpec,
    Diagnosis,
    EvidenceSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    HypothesisFeedback,
    HypothesisMatrix,
    OptimumFeedbackV1,
    ResearcherRun,
    ResearchSource,
)
from slm_training.autoresearch.storage import (
    CampaignStore,
    append_autotrain_action_receipt,
    autotrain_action_sha256,
    loop_result_rows,
    render_loop_result_matrix,
)
from slm_training.autoresearch.telemetry import TrackioSink
from slm_training.data.mixture import MixtureManifest, write_mixture_manifest
from slm_training.harnesses.experiments.verified_metrics import (
    optimum_feedback,
    sha256_file,
    verify_metric_certificate,
)

ROOT = Path(__file__).resolve().parents[1]


def _bounded_campaign_seconds(campaign: CampaignSpec) -> float:
    """Honor old campaign records without exceeding the current run cap."""
    return min(
        float(campaign.budget.max_wall_minutes * 60),
        float(MAX_RUN_MINUTES * 60),
    )


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _validate_continuous_commits(upstream: str, integration: str) -> None:
    resolved_upstream = _git(
        "rev-parse", "--verify", f"{upstream}^{{commit}}"
    ).stdout.strip()
    resolved_integration = _git(
        "rev-parse", "--verify", f"{integration}^{{commit}}"
    ).stdout.strip()
    current_upstream = _git(
        "rev-parse", "--verify", "origin/main^{commit}"
    ).stdout.strip()
    current_head = _git("rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if resolved_upstream != current_upstream:
        raise ValueError("upstream_commit is stale; fetch origin/main before the cycle")
    if resolved_integration != current_head:
        raise ValueError("integration_commit must be the current checked-out HEAD")
    if _git(
        "merge-base",
        "--is-ancestor",
        resolved_upstream,
        resolved_integration,
        check=False,
    ).returncode:
        raise ValueError("integration_commit does not contain upstream_commit")
    if _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise ValueError("continuous cycle requires a clean tracked worktree")


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
    if args.max_wall_minutes > MAX_RUN_MINUTES:
        raise ValueError(f"--max-wall-minutes cannot exceed {MAX_RUN_MINUTES}")
    if args.loop_id is not None:
        if args.upstream_commit is None or args.integration_commit is None:
            raise ValueError(
                "continuous init requires --upstream-commit and --integration-commit"
            )
        _validate_continuous_commits(args.upstream_commit, args.integration_commit)
    evidence_roots = args.evidence_root or [Path("outputs")]
    campaign = CampaignSpec(
        campaign_id=args.campaign_id,
        objective=args.objective,
        primary_metric=args.primary_metric,
        track=args.track,
        researcher_mode=args.researcher_mode,
        min_hypotheses=args.min_hypotheses,
        evidence_roots=tuple(str(path) for path in evidence_roots),
        budget=CampaignBudget(
            max_experiments=args.max_experiments,
            max_gpu_hours=args.max_gpu_hours,
            max_wall_minutes=args.max_wall_minutes,
        ),
        loop_id=args.loop_id,
        cycle_index=args.cycle_index,
        predecessor_campaign_id=args.predecessor_campaign_id,
        upstream_commit=args.upstream_commit,
        integration_commit=args.integration_commit,
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
                else _bounded_campaign_seconds(campaign)
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
    lineage_stores = _lineage_stores(store, campaign)
    previous_store = store
    previous_matrix = _latest_formed_matrix(store, required=False)
    if previous_matrix is None and len(lineage_stores) > 1:
        previous_store = lineage_stores[1]
        previous_matrix = _latest_formed_matrix(previous_store, required=False)
    feedback = _hypothesis_feedback(previous_store, previous_matrix)
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
    from slm_training.autoresearch.climb_policy import (
        assert_recipe_null_regime_pressure,
        infer_metric_direction,
        load_climb_policy,
        load_loop_exhausted_ledger,
        loop_data_eval_identity,
        train_eval_from_knobs,
    )
    from slm_training.autoresearch.hillclimb import load_exhausted_ledger

    policy = load_climb_policy()
    claim_class = str(policy.defaults.get("exploratory_claim_class") or "fixture")
    direction = infer_metric_direction(campaign.primary_metric)
    proposed_knobs = [
        candidate.experiment.knobs.model_dump(exclude_none=True, mode="json")
        for candidate in result.matrix.hypotheses
    ]
    train_version, eval_version = train_eval_from_knobs(proposed_knobs)
    # Prefer loop-scoped ledger when loop lineage exists; else campaign root.
    loop_id = campaign.loop_id
    if loop_id:
        exhausted_ledger = load_loop_exhausted_ledger(
            store.root.parent, loop_id, policy
        )
        exhausted_identity = loop_data_eval_identity(
            policy,
            claim_class=claim_class,
            train_version=train_version,
            eval_version=eval_version,
            primary_metric=campaign.primary_metric,
            direction=direction,
            data_manifest_sha256=evidence.snapshot_id,
        )
    else:
        exhausted_ledger = load_exhausted_ledger(store.root)
        exhausted_identity = loop_data_eval_identity(
            policy,
            claim_class=claim_class,
            train_version=train_version,
            eval_version=eval_version,
            primary_metric=campaign.primary_metric,
            direction=direction,
            data_manifest_sha256=evidence.snapshot_id,
        )
    has_regime = any(
        item.novelty.transition_kind == "regime_transition_candidate"
        for item in result.matrix.hypotheses
    )
    assert_recipe_null_regime_pressure(
        policy,
        ledger=exhausted_ledger,
        data_eval_identity=exhausted_identity,
        claim_class=claim_class,
        matrix_has_regime_transition=has_regime,
        proposed_knobs=proposed_knobs,
    )
    validate_hypothesis_matrix(
        campaign,
        result.matrix,
        evidence,
        list(result.sources),
        prior_experiments=tuple(
            experiment
            for lineage_store in lineage_stores
            for experiment in _finished_experiments(lineage_store)
        ),
        prior_experiment_ids=frozenset(
            candidate.experiment.experiment_id
            for lineage_store in lineage_stores
            for formed in _formed_matrices(lineage_store)
            for candidate in formed.hypotheses
        ),
        feedback=feedback,
        previous_matrix=previous_matrix,
        exhausted_knob_ledger=exhausted_ledger,
        exhausted_data_eval_identity=exhausted_identity,
        exhausted_claim_class=claim_class,
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


def _lineage_stores(
    store: CampaignStore, campaign: CampaignSpec
) -> tuple[CampaignStore, ...]:
    stores = [store]
    current = campaign
    seen = {current.campaign_id}
    while current.predecessor_campaign_id is not None:
        predecessor = CampaignStore(current.predecessor_campaign_id, store.root.parent)
        predecessor_campaign = predecessor.load_campaign()
        if predecessor_campaign.campaign_id in seen:
            raise RuntimeError("continuous campaign predecessor cycle detected")
        if (
            predecessor_campaign.loop_id != campaign.loop_id
            or predecessor_campaign.cycle_index != (current.cycle_index or 0) - 1
        ):
            raise RuntimeError("continuous campaign predecessor lineage is invalid")
        seen.add(predecessor_campaign.campaign_id)
        stores.append(predecessor)
        current = predecessor_campaign
    return tuple(stores)


def _finished_experiments(store: CampaignStore) -> tuple[ExperimentSpec, ...]:
    finished = {
        str(row.get("experiment_id"))
        for row in _events(store)
        if row.get("event_type") == "experiment_finished"
    }
    found = {}
    for path in (store.root / "artifacts" / "experiments").glob("*.json"):
        experiment = ExperimentSpec.model_validate_json(
            path.read_text(encoding="utf-8")
        )
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
    return (
        path.exists()
        and ExperimentOutcome.model_validate_json(path.read_text(encoding="utf-8"))
        == outcome
    )


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
        raise ValueError(
            "experiment is not an exact member of the latest hypothesis matrix"
        )
    return matrix


def _verified_optimum_feedback(
    args: argparse.Namespace,
    *,
    campaign_manifest_sha256: str | None,
    metric_expectations_sha256: str | None,
) -> OptimumFeedbackV1 | None:
    evidence = getattr(args, "metric_evidence", None)
    certificate = getattr(args, "metric_certificate", None)
    if evidence is None and certificate is None:
        return None
    if evidence is None or certificate is None:
        raise ValueError(
            "optimum feedback requires both --metric-evidence and --metric-certificate"
        )
    if campaign_manifest_sha256 is None or metric_expectations_sha256 is None:
        raise ValueError(
            "optimum feedback requires a locked campaign manifest with "
            "metric_expectations_sha256"
        )
    verified = verify_metric_certificate(
        evidence_path=evidence,
        certificate_path=certificate,
        expected_campaign_manifest_sha256=campaign_manifest_sha256,
        expected_metric_expectations_sha256=metric_expectations_sha256,
        checker=getattr(args, "leverproof_bin", None),
        require_band_certificate=False,
    )
    return OptimumFeedbackV1(
        **optimum_feedback(verified),
        metric_evidence_sha256=sha256_file(evidence),
        metric_certificate_sha256=sha256_file(certificate),
    )


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
    manifest_path = getattr(args, "campaign_manifest", None)
    if manifest_path is not None:
        manifest = ExperimentCampaignV1.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.experiment_id != experiment.experiment_id:
            raise ValueError("campaign manifest belongs to a different experiment")
        if campaign.loop_id is not None:
            assert campaign.upstream_commit is not None
            assert campaign.integration_commit is not None
            _validate_continuous_commits(
                campaign.upstream_commit, campaign.integration_commit
            )
            if manifest.source_commit != campaign.integration_commit:
                raise ValueError(
                    "continuous manifest source_commit must equal integration_commit"
                )
            if manifest.source_dirty:
                raise ValueError("continuous manifest source must be clean")
        validate_formal_preflights(store.root, experiment, manifest)
        lock = store.lock_experiment_campaign(manifest)
    else:
        try:
            lock = store.load_experiment_campaign(experiment.experiment_id)
        except FileNotFoundError:
            lock = None
    if args.execute and lock is None:
        raise ValueError("execution requires a preregistered --campaign-manifest lock")
    if lock is not None and manifest_path is None:
        validate_formal_preflights(store.root, experiment, lock.manifest)
    if lock is not None and lock.manifest.requires_rl:
        if not experiment.requires_rl or not experiment.rl_readiness_report:
            raise ValueError("RL campaign requires experiment readiness evidence")
        readiness_path = Path(experiment.rl_readiness_report)
        resolved = assert_rl_ready(readiness_path)
        report_sha = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
        if report_sha != lock.manifest.rl_readiness_report_sha256:
            raise ValueError("RL readiness report digest does not match campaign lock")
        if resolved.evaluation_sha256 != lock.manifest.rl_evaluation_sha256:
            raise ValueError("RL evaluation digest does not match campaign lock")
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
        "execution_plans",
        {
            "commands": commands,
            "execute": args.execute,
            "campaign_manifest_sha256": (
                lock.manifest_sha256 if lock is not None else None
            ),
        },
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
            detail={
                "hypothesis_matrix_id": matrix.matrix_id,
                "campaign_manifest_sha256": lock.manifest_sha256,
            },
        )
    outcome = execute_commands(
        experiment,
        commands,
        cwd=ROOT,
        timeout_seconds=_bounded_campaign_seconds(campaign),
        campaign_manifest_sha256=lock.manifest_sha256,
    )
    outcome_path = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_finished",
        experiment_id=experiment.experiment_id,
        status=outcome.status,
        artifact_sha256=outcome_path.stem,
        detail={
            "exit_code": outcome.exit_code,
            "campaign_manifest_sha256": lock.manifest_sha256,
        },
    )
    diagnosis = diagnose_outcome(outcome)
    diagnosis_path = store.write_artifact("diagnoses", diagnosis)
    store.append_event(
        "outcome_diagnosed",
        experiment_id=experiment.experiment_id,
        status=diagnosis.target,
        artifact_sha256=diagnosis_path.stem,
    )
    optimum = _verified_optimum_feedback(
        args,
        campaign_manifest_sha256=lock.manifest_sha256,
        metric_expectations_sha256=lock.manifest.metric_expectations_sha256,
    )
    feedback_path = _record_hypothesis_feedback(
        store, matrix, outcome, diagnosis, optimum
    )
    if optimum is not None:
        store.append_event(
            "optimum_feedback_recorded",
            experiment_id=experiment.experiment_id,
            status=optimum.policy,
            artifact_sha256=feedback_path.stem,
            detail={
                "breach_count": len(optimum.breaches),
                "diagnosis_lanes": list(optimum.diagnosis_lanes),
            },
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
    if optimum is not None and optimum.policy == "stop":
        return 2
    return 0 if outcome.status == "completed" else 2


def cmd_formalize(args: argparse.Namespace) -> int:
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
    if not experiment.formal_claims:
        raise ValueError("experiment declares no formal claims")
    rows: list[dict[str, object]] = []
    required_blocked = False
    deadline = time.monotonic() + _bounded_campaign_seconds(campaign)
    for claim in experiment.formal_claims:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("formal preflight campaign budget exhausted")
        preflight, obligation = run_formal_preflight(
            campaign.campaign_id,
            experiment,
            claim,
            timeout_seconds=remaining,
        )
        path = store.write_artifact("formal_preflights", preflight)
        obligation = bind_preflight(obligation, path.stem)
        event_type = (
            "formal_hypothesis_refuted"
            if preflight.status == "refuted"
            else "formal_preflight_completed"
        )
        store.append_event(
            event_type,
            experiment_id=experiment.experiment_id,
            status=preflight.status,
            artifact_sha256=path.stem,
            detail={
                "obligation_id": obligation.obligation_id,
                "template_id": obligation.template_id,
                "policy": obligation.policy,
            },
        )
        required_blocked |= (
            obligation.policy == "required" and preflight.status != "proved"
        )
        rows.append(
            {
                "formal_obligation": obligation.model_dump(mode="json"),
                "preflight": preflight.model_dump(mode="json"),
            }
        )
    print(json.dumps({"results": rows}, indent=2))
    return 2 if required_blocked else 0


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


def _baseline_primary_from_store(
    store: CampaignStore,
    *,
    primary_metric: str,
    exclude_experiment_id: str = "",
) -> float | None:
    """Best-effort matched-control primary from prior completed outcomes.

    Prefers experiment ids that look like controls/baselines; otherwise uses
    metrics that already name a baseline. Absolute candidate scores alone are
    never treated as the baseline of themselves.
    """
    import math

    from slm_training.autoresearch.hillclimb import resolve_baseline_primary

    outcomes_dir = store.root / "artifacts" / "outcomes"
    if not outcomes_dir.is_dir():
        return None
    control_values: list[float] = []
    embedded: list[float] = []
    for path in sorted(outcomes_dir.glob("*.json")):
        try:
            prior = ExperimentOutcome.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            continue
        if prior.experiment_id == exclude_experiment_id:
            continue
        if prior.status != "completed":
            continue
        from_metrics = resolve_baseline_primary(
            prior.metrics, primary_metric=primary_metric
        )
        if from_metrics is not None:
            embedded.append(from_metrics)
        primary_value = prior.metrics.get(primary_metric)
        if primary_value is None:
            continue
        try:
            number = float(primary_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        exp_id = prior.experiment_id.lower()
        if any(token in exp_id for token in ("control", "baseline", "matched")):
            control_values.append(number)
    if control_values:
        return control_values[-1]
    if embedded:
        return embedded[-1]
    return None


def _record_hypothesis_feedback(
    store: CampaignStore,
    matrix: HypothesisMatrix,
    outcome: ExperimentOutcome,
    diagnosis: Diagnosis,
    optimum: OptimumFeedbackV1 | None = None,
) -> Path:
    feedback = create_hypothesis_feedback(
        matrix, outcome, diagnosis, optimum_feedback=optimum
    )
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
    # After a measured null, mark the knob signature exhausted at loop scope
    # (policy identity fields) so successor matrices cannot replay it.
    try:
        campaign = store.load_campaign()
        primary_metric = campaign.primary_metric
        evidence_snapshot_id = matrix.evidence_snapshot_id
        loop_id = campaign.loop_id
        minimum_effect = 0.0
    except Exception:  # noqa: BLE001 — never block feedback on ledger errors
        primary_metric = "primary"
        evidence_snapshot_id = matrix.evidence_snapshot_id
        loop_id = None
        minimum_effect = 0.0
    from slm_training.autoresearch.climb_policy import (
        is_recipe_tweak_knobs,
        load_climb_policy,
        load_loop_exhausted_ledger,
        loop_data_eval_identity,
        save_loop_exhausted_ledger,
    )
    from slm_training.autoresearch.hillclimb import (
        infer_metric_direction,
        is_measured_null_feedback,
        load_exhausted_ledger,
        record_null_from_knob_signature_json,
        resolve_baseline_primary,
        resolve_primary_effect,
        save_exhausted_ledger,
    )

    policy = load_climb_policy()
    claim_class = str(policy.defaults.get("exploratory_claim_class") or "fixture")
    baseline = resolve_baseline_primary(feedback.metrics, primary_metric=primary_metric)
    if baseline is None:
        baseline = _baseline_primary_from_store(
            store,
            primary_metric=primary_metric,
            exclude_experiment_id=outcome.experiment_id,
        )
    direction = infer_metric_direction(primary_metric)
    seed_values = None
    raw_seeds = feedback.metrics.get("primary_endpoint_seed_values")
    if isinstance(raw_seeds, (list, tuple)) and len(raw_seeds) >= 2:
        try:
            seed_values = tuple(float(v) for v in raw_seeds)
        except (TypeError, ValueError):
            seed_values = None

    null = is_measured_null_feedback(
        outcome_status=feedback.outcome_status,
        metrics=feedback.metrics,
        primary_metric=primary_metric,
        direction=direction,
        minimum_effect=minimum_effect,
        baseline_primary=baseline,
        primary_seed_values=seed_values,
    )
    if not null and diagnosis.target == "researcher":
        improvement, source = resolve_primary_effect(
            feedback.metrics,
            primary_metric=primary_metric,
            direction=direction,
            baseline_primary=baseline,
        )
        if source == "explicit_delta" and improvement is not None and improvement <= 0:
            null = True
    if null:
        knobs_raw: dict = {}
        try:
            parsed = json.loads(feedback.knob_signature)
            if isinstance(parsed, dict):
                knobs_raw = parsed
        except json.JSONDecodeError:
            knobs_raw = {}
        train_v = (
            str(knobs_raw["train_version"]) if knobs_raw.get("train_version") else None
        )
        eval_v = (
            str(knobs_raw["eval_version"]) if knobs_raw.get("eval_version") else None
        )
        identity = loop_data_eval_identity(
            policy,
            claim_class=claim_class,
            train_version=train_v,
            eval_version=eval_v,
            primary_metric=primary_metric,
            direction=direction,
            data_manifest_sha256=evidence_snapshot_id,
        )
        recipe = is_recipe_tweak_knobs(policy, knobs_raw)
        reason = "recipe_null" if recipe else "primary_lcb_within_noise"
        note = "recipe_family" if recipe else ""
        if loop_id:
            ledger = load_loop_exhausted_ledger(store.root.parent, loop_id, policy)
        else:
            ledger = load_exhausted_ledger(store.root)
        entry = record_null_from_knob_signature_json(
            ledger,
            knob_signature_json=feedback.knob_signature,
            data_eval_identity=identity,
            claim_class=claim_class,
            reason=reason,
            note=note,
        )
        if entry is not None:
            if loop_id:
                save_loop_exhausted_ledger(ledger, store.root.parent, loop_id, policy)
            else:
                save_exhausted_ledger(ledger, store.root)
            store.append_event(
                "exhausted_knob_recorded",
                experiment_id=outcome.experiment_id,
                status="null",
                detail={
                    "knob_signature_sha256": entry.knob_signature_sha256,
                    "data_eval_identity": identity,
                    "claim_class": claim_class,
                    "baseline_primary": baseline,
                    "reason": reason,
                    "loop_id": loop_id,
                    "policy_sha256": policy.sha256,
                },
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
    if args.loop_id:
        limit = None if args.all else args.last
        rows = loop_result_rows(args.root, args.loop_id, last=limit)
        if args.matrix:
            print(render_loop_result_matrix(args.root, args.loop_id, last=limit))
        else:
            campaigns = sorted(
                [
                    CampaignSpec.model_validate_json(path.read_text(encoding="utf-8"))
                    for path in args.root.glob("*/campaign.json")
                ],
                key=lambda item: (item.cycle_index or 0, item.campaign_id),
            )
            campaigns = [item for item in campaigns if item.loop_id == args.loop_id]
            print(
                json.dumps(
                    {
                        "loop_id": args.loop_id,
                        "active": True,
                        "campaign_count": len(campaigns),
                        "campaign_ids": [item.campaign_id for item in campaigns],
                        "result_count": len(rows),
                    },
                    indent=2,
                )
            )
    else:
        if args.matrix:
            raise ValueError("--matrix requires --loop-id")
        if args.all or args.last != 5:
            raise ValueError("--last/--all require --loop-id")
        print(json.dumps(_store(args).status(), indent=2))
    return 0


def cmd_ack_action(args: argparse.Namespace) -> int:
    handoff_path = args.root / args.campaign_id / "cycle_handoff.json"
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != args.loop_id:
        raise ValueError("handoff loop_id does not match --loop-id")
    if args.action_index >= len(handoff.actions):
        raise ValueError("--action-index is outside the handoff action list")
    action = handoff.actions[args.action_index]
    receipt = AutotrainActionReceiptV1(
        loop_id=args.loop_id,
        campaign_id=args.campaign_id,
        action_index=args.action_index,
        action_sha256=autotrain_action_sha256(action),
        action_kind=action.kind,
        status=args.status,
        evidence_uris=tuple(args.evidence),
    )
    path = append_autotrain_action_receipt(args.root, receipt)
    print(
        json.dumps({"receipt": str(path), **receipt.model_dump(mode="json")}, indent=2)
    )
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


def cmd_export_openfeature(args: argparse.Namespace) -> int:
    """Export a campaign's hypothesis matrix as a flagd flag definition."""
    from slm_training.autoresearch.openfeature import export_flagd_flags

    store = _store(args)
    if args.matrix:
        matrix = HypothesisMatrix.model_validate_json(
            args.matrix.read_text(encoding="utf-8")
        )
    else:
        matrix = _latest_formed_matrix(store)
        assert matrix is not None
    if matrix.campaign_id != store.campaign_id:
        raise ValueError("hypothesis matrix belongs to a different campaign")
    document = export_flagd_flags(
        (item.experiment for item in matrix.hypotheses),
        flag_set_id=matrix.campaign_id,
    )
    artifact = store.write_artifact("openfeature", document)
    store.append_event(
        "openfeature_exported",
        artifact_sha256=artifact.stem,
        detail={"matrix_id": matrix.matrix_id, "flags": len(document["flags"])},
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(document, indent=2, sort_keys=True))
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

    data = (
        json.loads(args.config.read_text(encoding="utf-8"))
        if args.config
        else _smoke_config
    )
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
    init.add_argument("--evidence-root", type=Path, action="append")
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
    init.add_argument("--loop-id")
    init.add_argument("--cycle-index", type=int)
    init.add_argument("--predecessor-campaign-id")
    init.add_argument("--upstream-commit")
    init.add_argument("--integration-commit")
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
    hypothesize.add_argument(
        "--provider", choices=("agent", "openai"), default="openai"
    )
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

    formalize = sub.add_parser("formalize")
    formalize.add_argument("--campaign-id", required=True)
    formalize.add_argument(
        "--experiment",
        type=Path,
        help="Exact matrix member; defaults to the matrix recommendation.",
    )
    formalize.set_defaults(func=cmd_formalize)

    run = sub.add_parser("run")
    run.add_argument("--campaign-id", required=True)
    run.add_argument(
        "--experiment",
        type=Path,
        help="Exact matrix member; defaults to the matrix recommendation.",
    )
    run.add_argument("--execute", action="store_true")
    run.add_argument(
        "--campaign-manifest",
        type=Path,
        help="ExperimentCampaignV1 JSON locked before --execute.",
    )
    run.add_argument("--trackio", action="store_true")
    run.add_argument("--metric-evidence", type=Path)
    run.add_argument("--metric-certificate", type=Path)
    run.add_argument("--leverproof-bin", type=Path)
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
    benchmark.add_argument(
        "--human-approve",
        action="store_true",
        help="Deprecated compatibility metadata; frozen meta-gates promote automatically.",
    )
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.set_defaults(func=cmd_evaluate_researcher)

    hypothesis_benchmark = sub.add_parser("evaluate-hypothesizer")
    hypothesis_benchmark.add_argument(
        "--cases",
        type=Path,
        default=Path("src/slm_training/resources/autoresearch/hypothesizer_cases.json"),
    )
    hypothesis_benchmark.add_argument("--predictions", type=Path, required=True)
    hypothesis_benchmark.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/autoresearch/hypothesizer_eval"),
    )
    hypothesis_benchmark.add_argument("--hypothesizer-id", required=True)
    hypothesis_benchmark.add_argument("--pass-threshold", type=float, default=0.8)
    hypothesis_benchmark.add_argument(
        "--human-approve",
        action="store_true",
        help="Deprecated compatibility metadata; frozen meta-gates promote automatically.",
    )
    hypothesis_benchmark.add_argument("--output", type=Path, required=True)
    hypothesis_benchmark.set_defaults(func=cmd_evaluate_hypothesizer)

    status = sub.add_parser("status")
    status_target = status.add_mutually_exclusive_group(required=True)
    status_target.add_argument("--campaign-id")
    status_target.add_argument("--loop-id")
    status.add_argument("--matrix", action="store_true")
    status_history = status.add_mutually_exclusive_group()
    status_history.add_argument("--last", type=int, default=5)
    status_history.add_argument("--all", action="store_true")
    status.set_defaults(func=cmd_status)

    acknowledge = sub.add_parser("ack-action")
    acknowledge.add_argument("--loop-id", required=True)
    acknowledge.add_argument("--campaign-id", required=True)
    acknowledge.add_argument("--action-index", type=int, required=True)
    acknowledge.add_argument(
        "--status", choices=("completed", "blocked"), default="completed"
    )
    acknowledge.add_argument("--evidence", action="append", required=True)
    acknowledge.set_defaults(func=cmd_ack_action)

    sync = sub.add_parser("sync")
    sync.add_argument("--campaign-id", required=True)
    sync.add_argument("--push", action="store_true")
    sync.set_defaults(func=cmd_sync)

    mixture = sub.add_parser("materialize-mixture")
    mixture.add_argument("--output", type=Path, required=True)
    mixture.add_argument("--mixture-id", required=True)
    mixture.add_argument("--weights-json", required=True)
    mixture.set_defaults(func=cmd_materialize_mixture)

    openfeature = sub.add_parser("export-openfeature")
    openfeature.add_argument("--campaign-id", required=True)
    openfeature.add_argument(
        "--matrix",
        type=Path,
        help="Hypothesis matrix JSON (defaults to the latest formed matrix)",
    )
    openfeature.add_argument(
        "--output", type=Path, help="Optional file destination for flags JSON"
    )
    openfeature.set_defaults(func=cmd_export_openfeature)

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
