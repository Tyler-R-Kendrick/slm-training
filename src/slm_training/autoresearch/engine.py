"""Experiment validation, bounded command compilation, and failure diagnosis."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping

from slm_training.autoresearch.rl_gate import assert_rl_ready
from slm_training.autoresearch.schemas import (
    CampaignSpec,
    Diagnosis,
    EvidenceSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    HypothesisFeedback,
    HypothesisMatrix,
    OptimumFeedbackV1,
    ResearchSource,
    evaluation_completeness_failures,
    evaluation_measurement_incomplete,
    utc_now,
)
from slm_training.harness_core.bounded_process import (
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import (
    DEFAULT_TRAIN_DATA_DIR,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    MAX_RUN_SECONDS,
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

_DEFAULT_EVAL_VERSION_CANDIDATES = (
    "e938_role_safe_all_targets_v2",
    "e842_harness_owned_slots_v1",
    "e827_target_slots_only_v4",
    "e763_symbol_only_eval_r2_20260722",
)


def default_eval_version() -> str:
    """Return a published eval snapshot that has a smoke suite on disk."""
    from slm_training.data.store import DataStore

    store = DataStore()
    for name in _DEFAULT_EVAL_VERSION_CANDIDATES:
        root = store.resolve_path("eval", name)
        if (root / "suites" / "smoke" / "records.jsonl").is_file():
            return name
    eval_root = store.resolve_path("eval", "e938_role_safe_all_targets_v2").parent
    if eval_root.is_dir():
        for child in sorted(eval_root.iterdir()):
            if (
                child.is_dir()
                and (child / "suites" / "smoke" / "records.jsonl").is_file()
            ):
                return child.name
    raise FileNotFoundError(
        "no published eval snapshot with a smoke suite is available; "
        "set ExperimentKnobs.eval_version explicitly"
    )


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


def normalized_knobs_sha256(experiment: ExperimentSpec) -> str:
    knobs = experiment.knobs.model_dump(exclude_none=True, mode="json")
    return hashlib.sha256(json.dumps(knobs, sort_keys=True).encode("utf-8")).hexdigest()


def validate_hypothesis_matrix(
    campaign: CampaignSpec,
    matrix: HypothesisMatrix,
    evidence: EvidenceSnapshot,
    sources: list[ResearchSource],
    prior_experiments: tuple[ExperimentSpec, ...] = (),
    prior_experiment_ids: frozenset[str] = frozenset(),
    feedback: tuple[HypothesisFeedback, ...] = (),
    previous_matrix: HypothesisMatrix | None = None,
    *,
    exhausted_knob_ledger: object | None = None,
    exhausted_data_eval_identity: str | None = None,
    exhausted_claim_class: str = "fixture",
    authorized_replay_configs: Mapping[str, str] | None = None,
) -> None:
    if matrix.campaign_id != campaign.campaign_id:
        raise ValueError("hypothesis matrix campaign_id does not match campaign")
    if matrix.evidence_snapshot_id != evidence.snapshot_id:
        raise ValueError("hypothesis matrix does not match the captured evidence")
    if len(matrix.hypotheses) < campaign.min_hypotheses:
        raise ValueError(
            f"hypothesis matrix requires at least {campaign.min_hypotheses} candidates"
        )
    if campaign.loop_id is not None and not matrix.next_run_priorities:
        raise ValueError("continuous hypothesis matrix requires next-run priorities")
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
            item.campaign_id != previous_matrix.campaign_id
            or item.matrix_id != previous_matrix.matrix_id
            for item in feedback
        ):
            raise ValueError("feedback does not belong to the predecessor matrix")
        if matrix.predecessor_matrix_id != previous_matrix.matrix_id:
            raise ValueError(
                "feedback-informed matrix must identify its predecessor matrix"
            )
        required_lanes = {
            lane
            for item in feedback
            if item.optimum_feedback is not None
            for lane in item.optimum_feedback.diagnosis_lanes
        }
        if any(
            item.optimum_feedback is not None and item.optimum_feedback.policy == "stop"
            for item in feedback
        ):
            raise ValueError("theorem-backed metric contradiction stops this campaign")
        if required_lanes:
            supplied_lanes = {
                item.diagnosis_lane
                for item in matrix.hypotheses
                if item.diagnosis_lane is not None
            }
            if supplied_lanes != required_lanes:
                raise ValueError(
                    "out-of-band feedback requires one hypothesis for every "
                    f"diagnosis lane: {sorted(required_lanes)}"
                )
            if campaign.loop_id is not None:
                priority_lanes = {
                    item.area
                    for item in matrix.next_run_priorities
                    if item.authority == "lean_assumption"
                }
                if priority_lanes != required_lanes:
                    raise ValueError(
                        "out-of-band feedback requires one Lean-assumption priority "
                        f"for every diagnosis lane: {sorted(required_lanes)}"
                    )
        if campaign.loop_id is not None:
            priority_evidence = {
                evidence_id
                for priority in matrix.next_run_priorities
                for evidence_id in priority.evidence_ids
            }
            if not expected_feedback <= priority_evidence:
                raise ValueError(
                    "continuous priorities must cite every supplied feedback id"
                )
            for item in feedback:
                if item.diagnosis_target != "harness":
                    continue
                if not any(
                    priority.area == item.harness_family
                    and priority.authority == "reproduced_harness_signal"
                    and priority.disposition == "repair_before_run"
                    for priority in matrix.next_run_priorities
                ):
                    raise ValueError(
                        "reproduced harness feedback requires a repair priority for "
                        f"{item.harness_family}"
                    )
    prior_signatures = {
        json.dumps(
            experiment.knobs.model_dump(exclude_none=True, mode="json"),
            sort_keys=True,
        )
        for experiment in prior_experiments
    }
    replay_configs = authorized_replay_configs or {}
    repeated = []
    for candidate in matrix.hypotheses:
        signature = json.dumps(
            candidate.experiment.knobs.model_dump(exclude_none=True, mode="json"),
            sort_keys=True,
        )
        if signature not in prior_signatures:
            continue
        config_sha256 = normalized_knobs_sha256(candidate.experiment)
        if replay_configs.get(candidate.experiment.experiment_id) != config_sha256:
            repeated.append(candidate.experiment.experiment_id)
    if repeated:
        raise ValueError(
            f"hypothesis matrix repeats previously run knob signatures: {repeated}"
        )
    # Optional exhausted-knob ledger (hill-climb governance): same knob signature
    # + claim_class + data/eval identity after a measured null is rejected.
    if exhausted_knob_ledger is not None and exhausted_data_eval_identity:
        from slm_training.autoresearch.hillclimb import (
            HillClimbError,
            assert_matrix_knobs_not_exhausted,
            knob_signature_sha256,
        )

        try:
            assert_matrix_knobs_not_exhausted(
                knob_signatures=[
                    knob_signature_sha256(
                        candidate.experiment.knobs.model_dump(
                            exclude_none=True, mode="json"
                        )
                    )
                    for candidate in matrix.hypotheses
                    if replay_configs.get(candidate.experiment.experiment_id)
                    != normalized_knobs_sha256(candidate.experiment)
                ],
                ledger=exhausted_knob_ledger,  # type: ignore[arg-type]
                data_eval_identity=exhausted_data_eval_identity,
                claim_class=exhausted_claim_class,
            )
        except HillClimbError as exc:
            raise ValueError(str(exc)) from exc

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
                raise ValueError(f"{use.citation} is not captured {use.role} evidence")
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
    optimum_feedback: OptimumFeedbackV1 | None = None,
) -> HypothesisFeedback:
    """Turn one completed matrix experiment into bounded hypothesizer feedback."""
    candidates = {item.experiment.experiment_id: item for item in matrix.hypotheses}
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
        "harness_family": diagnosis.harness_family,
        "diagnosis_evidence": diagnosis.evidence,
        "recommended_actions": diagnosis.recommended_actions,
        "optimum_feedback": (
            optimum_feedback.model_dump(mode="json")
            if optimum_feedback is not None
            else None
        ),
    }
    feedback_id = (
        "feedback-"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    )
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
        harness_family=diagnosis.harness_family,
        diagnosis_evidence=diagnosis.evidence,
        recommended_actions=diagnosis.recommended_actions,
        optimum_feedback=optimum_feedback,
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
            sys.executable,
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
        train_dir = DEFAULT_TRAIN_DATA_DIR
    mixture_path = root / "mixture.json"
    if knobs.mixture_weights:
        commands.append(
            [
                sys.executable,
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
        sys.executable,
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
    if knobs.local_files_only:
        train.append("--local-files-only")
    if knobs.sync_checkpoints is not None:
        train.append(
            "--sync-checkpoints" if knobs.sync_checkpoints else "--no-sync-checkpoints"
        )
    if (knobs.context_backend or "hf") == "scratch" and knobs.sync_checkpoints is False:
        # One-shot local campaign arms have no resume consumer. Keep last.pt,
        # but do not spend the wall budget serializing optimizer/RNG state.
        train.append("--no-full-state-checkpoint")
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
        if knobs.output_tokenizer:
            train.extend(["--output-tokenizer", knobs.output_tokenizer])
        elif any(getattr(knobs, field) is not None for field in symbol_fields):
            train.extend(["--output-tokenizer", "lexer"])
        if knobs.compiler_decode_mode:
            train.extend(["--compiler-decode-mode", knobs.compiler_decode_mode])
            if knobs.compiler_decode_mode != "off":
                train.append("--grammar-ltr-primary")
        # DSL diffusion program levers (G1): typed knob -> bounded flag only.
        if knobs.mask_pattern:
            train.extend(["--mask-pattern", knobs.mask_pattern])
        if knobs.bind_encoding:
            train.extend(["--bind-encoding", knobs.bind_encoding])
        if knobs.denoiser_backend:
            train.extend(["--denoiser-backend", knobs.denoiser_backend])
        if knobs.decode_min_content is not None:
            train.extend(["--decode-min-content", str(knobs.decode_min_content)])
        if knobs.asap_decode:
            train.append("--asap-decode")
        if knobs.compiler_alignment_loss_weight is not None:
            train.extend(
                [
                    "--compiler-alignment-loss-weight",
                    str(knobs.compiler_alignment_loss_weight),
                ]
            )
        if knobs.compiler_alignment_margin is not None:
            train.extend(
                ["--compiler-alignment-margin", str(knobs.compiler_alignment_margin)]
            )
        if knobs.compiler_alignment_stratified:
            train.append("--compiler-alignment-stratified")
        if knobs.compiler_alignment_semantic_exhaustive:
            train.append("--compiler-alignment-semantic-exhaustive")
        if knobs.component_inventory_loss_weight is not None:
            train.extend(
                [
                    "--component-inventory-loss-weight",
                    str(knobs.component_inventory_loss_weight),
                ]
            )
        if knobs.structural_aux_head_profile is not None:
            train.extend(
                [
                    "--structural-aux-head-profile",
                    knobs.structural_aux_head_profile,
                ]
            )
        if knobs.component_inventory_decode_weight is not None:
            train.extend(
                [
                    "--component-inventory-decode-weight",
                    str(knobs.component_inventory_decode_weight),
                ]
            )
        if knobs.component_plan_loss_weight is not None:
            train.extend(
                ["--component-plan-loss-weight", str(knobs.component_plan_loss_weight)]
            )
        if knobs.component_plan_decode_weight is not None:
            train.extend(
                [
                    "--component-plan-decode-weight",
                    str(knobs.component_plan_decode_weight),
                ]
            )
        if knobs.component_edge_loss_weight is not None:
            train.extend(
                ["--component-edge-loss-weight", str(knobs.component_edge_loss_weight)]
            )
        if knobs.component_edge_alignment_loss_weight is not None:
            train.extend(
                [
                    "--component-edge-alignment-loss-weight",
                    str(knobs.component_edge_alignment_loss_weight),
                ]
            )
        if knobs.component_edge_decode_weight is not None:
            train.extend(
                [
                    "--component-edge-decode-weight",
                    str(knobs.component_edge_decode_weight),
                ]
            )
        if knobs.binder_component_plan_loss_weight is not None:
            train.extend(
                [
                    "--binder-component-plan-loss-weight",
                    str(knobs.binder_component_plan_loss_weight),
                ]
            )
        if knobs.binder_component_plan_decode_weight is not None:
            train.extend(
                [
                    "--binder-component-plan-decode-weight",
                    str(knobs.binder_component_plan_decode_weight),
                ]
            )
        if knobs.binder_topology_loss_weight is not None:
            train.extend(
                [
                    "--binder-topology-loss-weight",
                    str(knobs.binder_topology_loss_weight),
                ]
            )
        if knobs.binder_topology_decode_weight is not None:
            train.extend(
                [
                    "--binder-topology-decode-weight",
                    str(knobs.binder_topology_decode_weight),
                ]
            )
        if knobs.binder_arity_loss_weight is not None:
            train.extend(
                ["--binder-arity-loss-weight", str(knobs.binder_arity_loss_weight)]
            )
        if knobs.binder_arity_decode_weight is not None:
            train.extend(
                ["--binder-arity-decode-weight", str(knobs.binder_arity_decode_weight)]
            )
        if knobs.schema_in_context:
            train.append("--schema-in-context")
        if knobs.slot_contract_in_context:
            train.append("--slot-contract-in-context")
        if knobs.design_md_context is False:
            train.append("--no-design-md-context")
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
        for field, flag in {
            "action_embedding_init": "action-embedding-init",
            "action_embedding_train": "action-embedding-train",
            "action_alias_mode": "action-alias-mode",
            "action_description_name_mode": "action-description-name-mode",
        }.items():
            value = getattr(knobs, field)
            if value is not None:
                train.extend([f"--{flag}", str(value)])
        if knobs.action_alias_manifest is not None:
            train.extend(["--action-alias-manifest", str(knobs.action_alias_manifest)])
    if knobs.mixture_weights:
        train.extend(["--mixture-manifest", str(mixture_path)])
    if knobs.mixture_sampling_policy:
        train.extend(["--mixture-sampling-policy", knobs.mixture_sampling_policy])
    if knobs.mixture_exposure_target_profile:
        train.extend(
            ["--mixture-exposure-target-profile", knobs.mixture_exposure_target_profile]
        )
    if knobs.mixture_total_decision_budget is not None:
        train.extend(
            [
                "--mixture-total-decision-budget",
                str(knobs.mixture_total_decision_budget),
            ]
        )
    if knobs.mixture_per_root_cap is not None:
        train.extend(["--mixture-per-root-cap", str(knobs.mixture_per_root_cap)])
    if knobs.mixture_per_template_cap is not None:
        train.extend(
            ["--mixture-per-template-cap", str(knobs.mixture_per_template_cap)]
        )
    if knobs.mixture_max_importance_weight is not None:
        train.extend(
            [
                "--mixture-max-importance-weight",
                str(knobs.mixture_max_importance_weight),
            ]
        )
    commands.append(train)
    evaluate = [
        sys.executable,
        "-m",
        "scripts.evaluate_model",
        "--test-dir",
        knobs.eval_version or default_eval_version(),
        "--run-root",
        str(root.parent),
        "--run-id",
        root.name,
        "--ship-gates",
        "--honest-slot-contract",
        "--slot-contract-constrained-decode",
    ]
    if knobs.train_version:
        evaluate.extend(["--train-version", knobs.train_version])
    else:
        evaluate.extend(["--train-dir", str(train_dir)])
    if knobs.eval_suites:
        evaluate.extend(["--suites", str(knobs.eval_suites)])
    if knobs.decode_timeout_seconds is not None:
        evaluate.extend(["--decode-timeout-seconds", str(knobs.decode_timeout_seconds)])
    commands.append(evaluate)
    if campaign.track == "grammar_diffusion":
        commands[-1].extend(["--model", "grammar_diffusion"])
    elif campaign.track == "twotower":
        if knobs.local_files_only:
            evaluate.append("--local-files-only")
        if knobs.output_tokenizer:
            evaluate.extend(["--output-tokenizer", knobs.output_tokenizer])
        if knobs.compiler_decode_mode:
            evaluate.extend(["--compiler-decode-mode", knobs.compiler_decode_mode])
            if knobs.compiler_decode_mode != "off":
                evaluate.append("--grammar-ltr-primary")
        for name in (
            "compiler_search_mode",
            "compiler_search_trigger",
            "compiler_search_width",
            "compiler_search_noise",
            "compiler_search_stagnation_patience",
            "compiler_search_backtrack_limit",
        ):
            value = getattr(knobs, name)
            if value is not None:
                evaluate.extend([f"--{name.replace('_', '-')}", str(value)])
        if knobs.allow_unconstrained_fallback is False:
            evaluate.append("--no-unconstrained-fallback")
        if knobs.schema_in_context:
            evaluate.append("--schema-in-context")
        if knobs.slot_contract_in_context:
            evaluate.append("--slot-contract-in-context")
        if knobs.design_md_context is False:
            evaluate.append("--no-design-md-context")
    return commands


def execute_commands(
    experiment: ExperimentSpec,
    commands: list[list[str]],
    *,
    cwd: Path | str = Path("."),
    timeout_seconds: float | None = None,
    campaign_manifest_sha256: str | None = None,
) -> ExperimentOutcome:
    started = utc_now()
    deadline = (
        time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    )
    combined: list[str] = []
    stages: list[dict[str, object]] = []
    metrics: dict[str, float] = {}
    data_metrics: dict[str, float] = {}
    for command in commands:
        remaining_seconds = (
            max(0.0, deadline - time.monotonic()) if deadline is not None else None
        )
        if remaining_seconds == 0:
            return ExperimentOutcome(
                experiment_id=experiment.experiment_id,
                campaign_id=experiment.campaign_id,
                status="stopped",
                metrics=metrics,
                data_metrics=data_metrics,
                command=tuple(" ".join(item) for item in commands),
                error="experiment exceeded cumulative wall-time budget",
                wall_time_budget_seconds=timeout_seconds,
                stage_telemetry=tuple(stages),
                started_at=started,
                finished_at=utc_now(),
                campaign_manifest_sha256=campaign_manifest_sha256,
            )
        stage_total = min(
            float(MAX_RUN_SECONDS),
            remaining_seconds
            if remaining_seconds is not None
            else float(MAX_RUN_SECONDS),
        )
        grace = min(float(KILL_GRACE_SECONDS), stage_total * 0.1)
        interrupt_after = min(
            float(INTERRUPT_AFTER_SECONDS), max(0.001, stage_total - grace)
        )
        stage_artifact = _stage_artifact_path(command, cwd=cwd)
        artifact_revision_before = _artifact_revision(stage_artifact)
        progress_artifact = _stage_progress_artifact_path(command, cwd=cwd)
        progress_revision_before = _artifact_revision(progress_artifact)
        completed = run_bounded_process(
            command,
            cwd=cwd,
            interrupt_after_seconds=interrupt_after,
            kill_grace_seconds=grace,
        )
        if getattr(completed, "timed_out", False):
            stage: dict[str, object] = {
                "command": command,
                "timed_out": True,
                "interrupted": getattr(completed, "interrupted", False),
                "killed": getattr(completed, "killed", False),
                "process_outcome": completed.outcome.value,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "measurement_complete": False,
            }
            partial = _read_changed_json(progress_artifact, progress_revision_before)
            run_id = _command_value(command, "--run-id")
            if (
                isinstance(partial, dict)
                and progress_artifact is not None
                and partial.get("schema_version") == "DecodeProgressV1"
                and partial.get("run_id") == run_id
                and partial.get("measurement_complete") is False
                and partial.get("scoreable") is False
            ):
                stage["partial_output"] = partial
                stage["partial_output_source"] = str(progress_artifact)
            stages.append(stage)
            return ExperimentOutcome(
                experiment_id=experiment.experiment_id,
                campaign_id=experiment.campaign_id,
                status="stopped",
                metrics=metrics,
                data_metrics=data_metrics,
                command=tuple(" ".join(item) for item in commands),
                error=f"stage exceeded wall-time limit: {' '.join(command)}",
                wall_time_budget_seconds=timeout_seconds,
                stage_telemetry=tuple(stages),
                started_at=started,
                finished_at=utc_now(),
                campaign_manifest_sha256=campaign_manifest_sha256,
            )
        if (
            getattr(completed, "outcome", ProcessOutcome.COMPLETED)
            is ProcessOutcome.LAUNCH_FAILED
        ):
            stages.append(
                {
                    "command": command,
                    "launch_error": completed.launch_error,
                    "stdout": "",
                    "stderr": "",
                }
            )
            return ExperimentOutcome(
                experiment_id=experiment.experiment_id,
                campaign_id=experiment.campaign_id,
                status="failed",
                metrics=metrics,
                data_metrics=data_metrics,
                command=tuple(" ".join(item) for item in commands),
                error=(
                    f"stage could not start: {' '.join(command)}: "
                    f"{completed.launch_error}"
                ),
                wall_time_budget_seconds=timeout_seconds,
                stage_telemetry=tuple(stages),
                started_at=started,
                finished_at=utc_now(),
                campaign_manifest_sha256=campaign_manifest_sha256,
            )
        combined.extend([completed.stdout, completed.stderr])
        parsed, parsed_source = _resolve_stage_output(
            command,
            completed.stdout,
            cwd=cwd,
            artifact_revision_before=artifact_revision_before,
        )
        flattened = _numeric_metrics(parsed) if parsed is not None else {}
        if "scripts.build_train_data" in command:
            data_metrics.update(flattened)
        elif "scripts.train_model" in command and completed.returncode == 0:
            incomplete_reason = _incomplete_train_reason(command, parsed, cwd=cwd)
            if incomplete_reason is not None:
                stages.append(
                    {
                        "command": command,
                        "exit_code": completed.returncode,
                        "stdout": completed.stdout[-8000:],
                        "stderr": completed.stderr[-8000:],
                        "parsed_output": parsed,
                        "parsed_output_source": parsed_source,
                        "measurement_complete": False,
                    }
                )
                return ExperimentOutcome(
                    experiment_id=experiment.experiment_id,
                    campaign_id=experiment.campaign_id,
                    status="stopped",
                    command=tuple(" ".join(item) for item in commands),
                    exit_code=completed.returncode,
                    error=incomplete_reason,
                    metrics=metrics,
                    data_metrics=data_metrics,
                    wall_time_budget_seconds=timeout_seconds,
                    stage_telemetry=tuple(stages),
                    started_at=started,
                    finished_at=utc_now(),
                    campaign_manifest_sha256=campaign_manifest_sha256,
                )
            trainable_params = flattened.get("track.trainable_params")
            if trainable_params is not None:
                metrics["trainable_params"] = trainable_params
        elif "scripts.evaluate_model" in command:
            metrics.update(flattened)
        expected_gate_rejection = _expected_gate_rejection(
            command, completed.returncode, parsed, artifact_root=Path(cwd)
        )
        stages.append(
            {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout[-8000:],
                "stderr": completed.stderr[-8000:],
                "parsed_output": parsed,
                "parsed_output_source": parsed_source,
                "expected_gate_rejection": expected_gate_rejection,
            }
        )
        if completed.returncode and not expected_gate_rejection:
            return ExperimentOutcome(
                experiment_id=experiment.experiment_id,
                campaign_id=experiment.campaign_id,
                status="failed",
                command=tuple(" ".join(command) for command in commands),
                exit_code=completed.returncode,
                error="\n".join(combined)[-8000:],
                metrics=metrics,
                data_metrics=data_metrics,
                wall_time_budget_seconds=timeout_seconds,
                stage_telemetry=tuple(stages),
                started_at=started,
                finished_at=utc_now(),
                campaign_manifest_sha256=campaign_manifest_sha256,
            )
    return ExperimentOutcome(
        experiment_id=experiment.experiment_id,
        campaign_id=experiment.campaign_id,
        status="completed",
        metrics=metrics,
        data_metrics=data_metrics,
        command=tuple(" ".join(command) for command in commands),
        exit_code=0,
        wall_time_budget_seconds=timeout_seconds,
        stage_telemetry=tuple(stages),
        started_at=started,
        finished_at=utc_now(),
        campaign_manifest_sha256=campaign_manifest_sha256,
    )


def _command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _stage_artifact_path(command: list[str], *, cwd: Path | str) -> Path | None:
    artifact_name = None
    if "scripts.train_model" in command:
        artifact_name = "train_summary.json"
    elif "scripts.evaluate_model" in command:
        artifact_name = "scoreboard.json"
    run_root = _command_value(command, "--run-root")
    run_id = _command_value(command, "--run-id")
    if artifact_name is None or run_root is None or run_id is None:
        return None
    root = Path(run_root)
    if not root.is_absolute():
        root = Path(cwd) / root
    return root / run_id / artifact_name


def _stage_progress_artifact_path(
    command: list[str], *, cwd: Path | str
) -> Path | None:
    if "scripts.evaluate_model" not in command:
        return None
    run_root = _command_value(command, "--run-root")
    run_id = _command_value(command, "--run-id")
    if run_root is None or run_id is None:
        return None
    root = Path(run_root)
    if not root.is_absolute():
        root = Path(cwd) / root
    return root / run_id / "decode_progress.json"


def _read_changed_json(
    path: Path | None, revision_before: tuple[int, int] | None
) -> object | None:
    revision_after = _artifact_revision(path)
    if path is None or revision_after is None or revision_after == revision_before:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _artifact_revision(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _resolve_stage_output(
    command: list[str],
    stdout: str,
    *,
    cwd: Path | str,
    artifact_revision_before: tuple[int, int] | None,
) -> tuple[object | None, str | None]:
    """Resolve typed JSON from stdout or a current-stage canonical artifact."""

    parsed = _parse_json_output(stdout)
    if parsed is not None:
        return parsed, "stdout"
    path = _stage_artifact_path(command, cwd=cwd)
    revision_after = _artifact_revision(path)
    if (
        path is None
        or revision_after is None
        or revision_after == artifact_revision_before
    ):
        return None, None
    try:
        parsed = _parse_json_output(path.read_text(encoding="utf-8"))
    except OSError:
        return None, None
    return parsed, str(path) if parsed is not None else None


def _incomplete_train_reason(
    command: list[str], parsed: object | None, *, cwd: Path | str
) -> str | None:
    """Return why a nominally successful train is not decision-bearing."""

    if not isinstance(parsed, dict):
        return "training produced no typed summary"
    stopped_on = str(parsed.get("stopped_on") or "")
    requested_raw = _command_value(command, "--steps")
    try:
        requested = int(requested_raw) if requested_raw is not None else None
    except ValueError:
        return f"training declared invalid --steps value {requested_raw!r}"
    completed = parsed.get("steps")
    if stopped_on != "steps":
        return f"training stopped_on={stopped_on or 'missing'} before declared steps"
    if requested is not None and completed != requested:
        return f"training completed {completed!r}/{requested} declared steps"
    checkpoint = parsed.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        return "training summary omitted checkpoint"
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(cwd) / checkpoint_path
    if not checkpoint_path.is_file():
        return f"training checkpoint is missing: {checkpoint}"
    return None


def _expected_gate_rejection(
    command: list[str],
    returncode: int,
    parsed: object | None,
    *,
    artifact_root: Path | None = None,
) -> bool:
    """Exit 8 is evidence when the typed AgentV/gate scoreboard is complete."""

    if (
        returncode != 8
        or "scripts.evaluate_model" not in command
        or "--ship-gates" not in command
        or not isinstance(parsed, dict)
    ):
        return False
    gates = parsed.get("gates")
    suites = parsed.get("suites")
    if evaluation_completeness_failures(
        parsed,
        require_agent_bindings=True,
        artifact_root=artifact_root,
    ):
        return False
    return bool(
        isinstance(gates, dict)
        and gates.get("pass") is False
        and isinstance(suites, dict)
        and suites
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
    reproduced_signals = [
        signal
        for signal in outcome.harness_signals
        if signal.reproduced_on_frozen_input
    ]
    harness_family = None
    if reproduced_signals:
        signal = next(
            (item for item in reproduced_signals if item.primary),
            reproduced_signals[0],
        )
        target = "harness"
        harness_family = signal.family
        evidence.append(f"{signal.code}: {signal.evidence_uri}")
        actions.extend(
            (
                f"route to improve-openui-harnesses/{signal.family}",
                "repair the canonical owner and replay the identical frozen model/data arm",
            )
        )
        confidence = 0.95
    elif evaluation_measurement_incomplete(metrics):
        target = "infrastructure"
        evidence.append(
            "evaluation measurement is incomplete; model attribution is unavailable"
        )
        actions.append("repair runtime capacity and replay the identical frozen arm")
        confidence = 0.95
    elif outcome.status == "stopped" or (
        outcome.status == "failed" and not metrics and not data
    ):
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
        evidence.append(
            "experiment completed locally but still fails honest ship gates"
        )
        actions.append(
            "compare against the matched control; continue model experiments; keep RL locked"
        )
        confidence = 0.75
    else:
        target = "none"
        evidence.append("no actionable defect detected")
        actions.append("retain as evidence and compare against matched controls")
        confidence = 0.6
    return Diagnosis(
        experiment_id=outcome.experiment_id,
        target=target,
        harness_family=harness_family,
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
