"""Experiment validation, bounded command compilation, and failure diagnosis."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from slm_training.autoresearch.rl_gate import assert_rl_ready
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    Diagnosis,
    EvidenceSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    HypothesisFeedback,
    HypothesisMatrix,
    ResearchSource,
    utc_now,
)


RESEARCH_SOURCE_KINDS = {
    "repo_lineage",
    "hf_daily_paper",
    "hf_paper_search",
    "web",
    "researcher",
}
TRACE_EVIDENCE_KINDS = {"run_insight", "telemetry", "agentv", "feedback"}
RESULT_EVIDENCE_KINDS = {
    "prior_run",
    "evaluation",
    "prior_campaign",
    "data_snapshot",
}


def validate_experiment(
    campaign: CampaignSpec,
    experiment: ExperimentSpec,
    evidence: EvidenceSnapshot,
    sources: list[ResearchSource],
) -> None:
    if experiment.campaign_id != campaign.campaign_id:
        raise ValueError("experiment campaign_id does not match campaign")
    changed = set(experiment.knobs.model_dump(exclude_none=True))
    forbidden = changed - set(campaign.allowed_knobs)
    if forbidden:
        raise ValueError(f"experiment changes forbidden knobs: {sorted(forbidden)}")
    known_citations = {source.uri for source in sources}
    known_citations.update(item.path for item in evidence.items)
    if not set(experiment.citations) & known_citations:
        raise ValueError(
            "experiment is ungrounded: no citation matches captured evidence"
        )
    if experiment.requires_rl:
        assert_rl_ready(experiment.rl_readiness_report)


def validate_hypothesis_matrix(
    campaign: CampaignSpec,
    matrix: HypothesisMatrix,
    evidence: EvidenceSnapshot,
    sources: list[ResearchSource],
    prior_experiments: tuple[ExperimentSpec, ...] = (),
    prior_experiment_ids: frozenset[str] = frozenset(),
    feedback: tuple[HypothesisFeedback, ...] = (),
    previous_matrix: HypothesisMatrix | None = None,
) -> None:
    if matrix.campaign_id != campaign.campaign_id:
        raise ValueError("hypothesis matrix campaign_id does not match campaign")
    if matrix.evidence_snapshot_id != evidence.snapshot_id:
        raise ValueError("hypothesis matrix does not match the captured evidence")
    if len(matrix.hypotheses) < campaign.min_hypotheses:
        raise ValueError(
            f"hypothesis matrix requires at least {campaign.min_hypotheses} candidates"
        )
    reused_ids = sorted(
        candidate.experiment.experiment_id
        for candidate in matrix.hypotheses
        if candidate.experiment.experiment_id in prior_experiment_ids
    )
    if reused_ids:
        raise ValueError(
            f"hypothesis matrix reuses campaign experiment ids: {reused_ids}"
        )
    expected_feedback = {item.feedback_id for item in feedback}
    if set(matrix.feedback_ids) != expected_feedback:
        raise ValueError("hypothesis matrix must acknowledge all supplied feedback")
    if feedback:
        if previous_matrix is None:
            raise ValueError("feedback-informed matrix requires its predecessor")
        if any(
            item.campaign_id != campaign.campaign_id
            or item.matrix_id != previous_matrix.matrix_id
            for item in feedback
        ):
            raise ValueError("feedback does not belong to the predecessor matrix")
        if matrix.predecessor_matrix_id != previous_matrix.matrix_id:
            raise ValueError(
                "feedback-informed matrix must identify its predecessor matrix"
            )
    prior_signatures = {
        json.dumps(
            experiment.knobs.model_dump(exclude_none=True, mode="json"),
            sort_keys=True,
        )
        for experiment in prior_experiments
    }
    repeated = [
        candidate.experiment.experiment_id
        for candidate in matrix.hypotheses
        if json.dumps(
            candidate.experiment.knobs.model_dump(exclude_none=True, mode="json"),
            sort_keys=True,
        )
        in prior_signatures
    ]
    if repeated:
        raise ValueError(
            f"hypothesis matrix repeats previously run knob signatures: {repeated}"
        )

    citation_roles: dict[str, set[str]] = {}
    for source in sources:
        roles = citation_roles.setdefault(source.uri, set())
        if source.kind in RESEARCH_SOURCE_KINDS:
            roles.add("research")
        if source.kind in {"telemetry", "feedback"}:
            roles.add("prior_trace")
        if source.kind in {"prior_run", "data_snapshot"}:
            roles.add("prior_result")
    for item in evidence.items:
        roles = citation_roles.setdefault(item.path, set())
        if item.kind == "repo_lineage":
            roles.add("research")
        if item.kind in TRACE_EVIDENCE_KINDS:
            roles.add("prior_trace")
        if item.kind in RESULT_EVIDENCE_KINDS:
            roles.add("prior_result")

    used_roles: set[str] = set()
    for candidate in matrix.hypotheses:
        validate_experiment(campaign, candidate.experiment, evidence, sources)
        for use in candidate.evidence_uses:
            if use.role not in citation_roles.get(use.citation, set()):
                raise ValueError(
                    f"{use.citation} is not captured {use.role} evidence"
                )
            used_roles.add(use.role)
    available_roles = set().union(*citation_roles.values()) if citation_roles else set()
    missing_roles = available_roles - used_roles
    if missing_roles:
        raise ValueError(
            f"hypothesis matrix does not use available evidence roles: {sorted(missing_roles)}"
        )
    if not any(
        item.novelty.transition_kind == "regime_transition_candidate"
        for item in matrix.hypotheses
    ):
        raise ValueError(
            "hypothesis matrix requires at least one regime-transition candidate"
        )


def create_hypothesis_feedback(
    matrix: HypothesisMatrix,
    outcome: ExperimentOutcome,
    diagnosis: Diagnosis,
) -> HypothesisFeedback:
    """Turn one completed matrix experiment into bounded hypothesizer feedback."""
    candidates = {
        item.experiment.experiment_id: item for item in matrix.hypotheses
    }
    candidate = candidates.get(outcome.experiment_id)
    if candidate is None:
        raise ValueError("outcome experiment is not a matrix member")
    if outcome.campaign_id != matrix.campaign_id:
        raise ValueError("outcome campaign does not match hypothesis matrix")
    if diagnosis.experiment_id != outcome.experiment_id:
        raise ValueError("diagnosis experiment does not match outcome")
    if outcome.status not in {"completed", "failed", "stopped"}:
        raise ValueError("hypothesizer feedback requires a terminal outcome")
    signature = json.dumps(
        candidate.experiment.knobs.model_dump(exclude_none=True, mode="json"),
        sort_keys=True,
    )
    identity = {
        "matrix_id": matrix.matrix_id,
        "experiment_id": outcome.experiment_id,
        "outcome_status": outcome.status,
        "metrics": outcome.metrics,
        "data_metrics": outcome.data_metrics,
        "diagnosis_target": diagnosis.target,
        "diagnosis_evidence": diagnosis.evidence,
        "recommended_actions": diagnosis.recommended_actions,
    }
    feedback_id = "feedback-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return HypothesisFeedback(
        feedback_id=feedback_id,
        campaign_id=matrix.campaign_id,
        matrix_id=matrix.matrix_id,
        experiment_id=outcome.experiment_id,
        hypothesis=candidate.experiment.hypothesis,
        knob_signature=signature,
        outcome_status=outcome.status,
        metrics=outcome.metrics,
        data_metrics=outcome.data_metrics,
        diagnosis_target=diagnosis.target,
        diagnosis_evidence=diagnosis.evidence,
        recommended_actions=diagnosis.recommended_actions,
        created_at=outcome.finished_at or outcome.started_at or matrix.created_at,
    )


def compile_commands(
    campaign: CampaignSpec,
    experiment: ExperimentSpec,
    *,
    output_root: Path | str = Path("outputs/autoresearch"),
) -> list[list[str]]:
    """Compile typed knobs only; no researcher-authored shell is accepted."""
    knobs = experiment.knobs
    root = Path(output_root) / campaign.campaign_id / "runs" / experiment.experiment_id
    version = f"autoresearch-{campaign.campaign_id}-{experiment.experiment_id}"
    train_dir = Path("outputs/data/train") / version
    commands: list[list[str]] = []
    if knobs.data_source:
        build = [
            "python",
            "-m",
            "scripts.build_train_data",
            "--source",
            knobs.data_source,
            "--version",
            version,
            "--immutable",
        ]
        if knobs.derive_from:
            build.extend(["--derive-from", knobs.derive_from])
        if knobs.synthesizer:
            build.extend(["--synthesizer", knobs.synthesizer])
        if knobs.max_records_per_parent:
            build.extend(
                ["--max-records-per-parent", str(knobs.max_records_per_parent)]
            )
        if knobs.min_quality_score is not None:
            build.extend(["--min-quality-score", str(knobs.min_quality_score)])
        if any(
            value is not None
            for value in (
                knobs.scope_contracts,
                knobs.scope_independent_noise,
                knobs.scope_local_oracle,
                knobs.scope_contract_negatives,
            )
        ):
            build.append("--scope-derivatives")
        commands.append(build)
    elif not knobs.train_version:
        train_dir = Path("outputs/data/train/v1")
    mixture_path = root / "mixture.json"
    if knobs.mixture_weights:
        commands.append(
            [
                "python",
                "-m",
                "scripts.autoresearch",
                "materialize-mixture",
                "--output",
                str(mixture_path),
                "--mixture-id",
                f"{campaign.campaign_id}-{experiment.experiment_id}",
                "--weights-json",
                json.dumps(knobs.mixture_weights, sort_keys=True),
            ]
        )
    if campaign.track not in {"twotower", "grammar_diffusion"}:
        raise ValueError(
            "embedded execution supports twotower and grammar_diffusion; "
            "causal_lm proposals must use the agent-driven model_cycle lineage workflow"
        )
    train = [
        "python",
        "-m",
        "scripts.train_model",
        "--run-root",
        str(root.parent),
        "--run-id",
        root.name,
        "--steps",
        str(knobs.steps or 200),
        "--batch-size",
        str(knobs.batch_size or 4),
        "--lr",
        str(knobs.lr or 3e-4),
        "--seed",
        str(knobs.seed or 0),
        "--context-backend",
        knobs.context_backend or "hf",
        "--device",
        "cpu" if campaign.budget.max_gpu_hours == 0 else "auto",
    ]
    if knobs.train_version:
        train.extend(["--train-version", knobs.train_version])
    else:
        train.extend(["--train-dir", str(train_dir)])
    if campaign.track == "grammar_diffusion":
        train.extend(["--model", "grammar_diffusion"])
        boolean_knobs = {
            "topology_actions": "topology-actions",
            "topology_structural_embeddings": "topology-structural-embeddings",
            "topology_heterogeneous_noise": "topology-heterogeneous-noise",
            "topology_critic_decode": "topology-critic-decode",
            "topology_bounded_buffer": "topology-bounded-buffer",
            "scope_contracts": "scope-contracts",
            "scope_independent_noise": "scope-independent-noise",
            "scope_local_oracle": "scope-local-oracle",
            "scope_contract_negatives": "scope-contract-negatives",
        }
        for field, flag in boolean_knobs.items():
            value = getattr(knobs, field)
            if value is not None:
                train.append(f"--{flag}" if value else f"--no-{flag}")
        for field, flag in {
            "topology_max_nodes": "topology-max-nodes",
            "topology_max_active": "topology-max-active",
            "topology_max_arity": "topology-max-arity",
            "topology_max_depth": "topology-max-depth",
            "topology_max_phases": "topology-max-phases",
            "topology_global_sync_interval": "topology-global-sync-interval",
            "topology_accept_threshold": "topology-accept-threshold",
            "topology_contract_threshold": "topology-contract-threshold",
        }.items():
            value = getattr(knobs, field)
            if value is not None:
                train.extend([f"--{flag}", str(value)])
    if campaign.track == "twotower":
        symbol_fields = {
            "runtime_symbol_features",
            "symbol_slot_augmentation",
            "semantic_candidate_masks",
            "constraint_graph_mode",
            "grammar_completion_bounds",
            "grammar_equivalence_cache",
            "grammar_active_symbol_bitsets",
            "compact_active_canvas",
        }
        if any(getattr(knobs, field) is not None for field in symbol_fields):
            train.extend(["--output-tokenizer", "lexer"])
        for field, flag in {
            "runtime_symbol_features": "runtime-symbol-features",
            "constraint_graph_mode": "constraint-graph-mode",
        }.items():
            value = getattr(knobs, field)
            if value is not None:
                train.extend([f"--{flag}", str(value)])
        for field, flag in {
            "symbol_slot_augmentation": "symbol-slot-augmentation",
            "semantic_candidate_masks": "semantic-candidate-masks",
            "grammar_completion_bounds": "grammar-completion-bounds",
            "grammar_equivalence_cache": "grammar-equivalence-cache",
            "grammar_active_symbol_bitsets": "grammar-active-symbol-bitsets",
            "compact_active_canvas": "compact-active-canvas",
        }.items():
            value = getattr(knobs, field)
            if value is not None:
                train.append(f"--{flag}" if value else f"--no-{flag}")
    if knobs.mixture_weights:
        train.extend(["--mixture-manifest", str(mixture_path)])
    commands.append(train)
    evaluate = [
        "python",
        "-m",
        "scripts.evaluate_model",
        "--test-dir",
        "outputs/data/eval/v1",
        "--run-root",
        str(root.parent),
        "--run-id",
        root.name,
        "--ship-gates",
    ]
    if knobs.train_version:
        evaluate.extend(["--train-version", knobs.train_version])
    else:
        evaluate.extend(["--train-dir", str(train_dir)])
    commands.append(evaluate)
    if campaign.track == "grammar_diffusion":
        commands[-1].extend(["--model", "grammar_diffusion"])
    return commands


def execute_commands(
    experiment: ExperimentSpec,
    commands: list[list[str]],
    *,
    cwd: Path | str = Path("."),
    timeout_seconds: float | None = None,
) -> ExperimentOutcome:
    started = utc_now()
    combined: list[str] = []
    stages: list[dict[str, object]] = []
    metrics: dict[str, float] = {}
    data_metrics: dict[str, float] = {}
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stages.append(
                {
                    "command": command,
                    "timed_out": True,
                    "stdout": str(exc.stdout or "")[-8000:],
                    "stderr": str(exc.stderr or "")[-8000:],
                }
            )
            return ExperimentOutcome(
                experiment_id=experiment.experiment_id,
                campaign_id=experiment.campaign_id,
                status="stopped",
                metrics=metrics,
                data_metrics=data_metrics,
                command=tuple(" ".join(item) for item in commands),
                error=f"stage exceeded wall-time limit: {' '.join(command)}",
                stage_telemetry=tuple(stages),
                started_at=started,
                finished_at=utc_now(),
            )
        combined.extend([completed.stdout, completed.stderr])
        parsed = _parse_json_output(completed.stdout)
        flattened = _numeric_metrics(parsed) if parsed is not None else {}
        if "scripts.build_train_data" in command:
            data_metrics.update(flattened)
        elif "scripts.evaluate_model" in command:
            metrics.update(flattened)
        stages.append(
            {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout[-8000:],
                "stderr": completed.stderr[-8000:],
                "parsed_output": parsed,
            }
        )
        if completed.returncode:
            return ExperimentOutcome(
                experiment_id=experiment.experiment_id,
                campaign_id=experiment.campaign_id,
                status="failed",
                command=tuple(" ".join(command) for command in commands),
                exit_code=completed.returncode,
                error="\n".join(combined)[-8000:],
                metrics=metrics,
                data_metrics=data_metrics,
                stage_telemetry=tuple(stages),
                started_at=started,
                finished_at=utc_now(),
            )
    return ExperimentOutcome(
        experiment_id=experiment.experiment_id,
        campaign_id=experiment.campaign_id,
        status="completed",
        metrics=metrics,
        data_metrics=data_metrics,
        command=tuple(" ".join(command) for command in commands),
        exit_code=0,
        stage_telemetry=tuple(stages),
        started_at=started,
        finished_at=utc_now(),
    )


def _parse_json_output(stdout: str) -> object | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _numeric_metrics(value: object, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}".strip(".")
            if isinstance(child, bool):
                result[name] = float(child)
            elif isinstance(child, (int, float)):
                result[name] = float(child)
            elif len(result) < 300:
                result.update(_numeric_metrics(child, name))
    return result


def diagnose_outcome(outcome: ExperimentOutcome) -> Diagnosis:
    data = outcome.data_metrics
    metrics = outcome.metrics
    evidence: list[str] = []
    actions: list[str] = []
    if outcome.status in {"failed", "stopped"} and not metrics and not data:
        target = "infrastructure"
        evidence.append(outcome.error or "experiment process failed")
        actions.append("repair the failed harness stage and rerun the identical spec")
        confidence = 0.9
    elif (
        _metric(data, "leakage_count", default=0) > 0
        or _metric(data, "valid_rate", default=1) < 0.98
        or _metric(data, "quality_score", "mean_quality_score", default=1) < 0.55
        or _metric(data, "error_count", default=0)
        > max(1, 0.02 * _metric(data, "record_count", default=1))
    ):
        target = "data"
        evidence.append(f"data metrics={json.dumps(data, sort_keys=True)}")
        actions.extend(
            (
                "derive a new immutable data snapshot from the failing snapshot",
                "adjust synthesis filters or family mixture, then rerun matched controls",
            )
        )
        confidence = 0.9
    elif (
        _has_metric(metrics, "primary_delta")
        and _metric(metrics, "primary_delta", default=0) <= 0
    ):
        target = "researcher"
        evidence.append("valid experiment did not improve the declared primary metric")
        actions.extend(
            (
                "feed this outcome into the next evidence snapshot",
                "revise the hypothesis policy on the frozen researcher benchmark",
            )
        )
        confidence = 0.7
    elif metrics and not bool(
        _metric(metrics, "ship_gates_pass", "gates.pass", "pass", default=0)
    ):
        target = "model"
        evidence.append("experiment improved locally but still fails honest ship gates")
        actions.append("continue SFT or architecture experiments; keep RL locked")
        confidence = 0.75
    else:
        target = "none"
        evidence.append("no actionable defect detected")
        actions.append("retain as evidence and compare against matched controls")
        confidence = 0.6
    return Diagnosis(
        experiment_id=outcome.experiment_id,
        target=target,
        confidence=confidence,
        evidence=tuple(evidence),
        recommended_actions=tuple(actions),
    )


def _metric(metrics: dict[str, float], *names: str, default: float) -> float:
    for key, value in metrics.items():
        if key in names or any(key.endswith(f".{name}") for name in names):
            return float(value)
    return default


def _has_metric(metrics: dict[str, float], name: str) -> bool:
    return any(key == name or key.endswith(f".{name}") for key in metrics)
