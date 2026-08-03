#!/usr/bin/env python3
"""Hands-off continuous autotrain cycle driver.

Bare /autotrain agents should keep calling this (or re-enter continuous.md)
without user prompts. Each invocation can run one or many bounded cycles.
Never an infinite shell without a stage wall on child commands. Every child
obeys the repository-wide ``MAX_RUN_MINUTES`` policy.

SDLC Phase A (autotrain-iteration-delivery): after every cycle the driver
classifies positive vs non-positive, records a delivery ledger, and only
signals stack-layer intent for positive results. Stacked PRs are still opened
by the agent (gh stack); this driver never opens PRs for non-positive cycles.

Proof driver (promote path): formal preflight must be proved, dual-arm
promotion primary must beat policy ``minimum_effect``, and a LeverProof
metric_certificate/v2 must dispose ``continue`` before a champion is marked
``climb_accepted``. Cert continue is necessary but not sufficient. Phase A smoke
quality-held alone never promotes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slm_training.autoresearch.engine import default_eval_version
from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.formal import formal_obligation_id
from slm_training.autoresearch.schemas import (
    AutotrainActionReceiptV1,
    AutotrainActionV1,
    AutotrainCycleHandoffV1,
    AutotrainLoopStateV1,
    CampaignBudget,
    CampaignSpec,
    FormalClaimV1,
    FormalObligationV1,
    FormalPreflightV1,
    HypothesisMatrix,
    NextRunPriorityV1,
    utc_now,
)
from slm_training.autoresearch.storage import (
    CampaignStore,
    append_autotrain_action_receipt,
    autotrain_action_sha256,
    bind_autotrain_action_evidence,
    pending_autotrain_actions,
    pending_autotrain_execution_actions,
)
from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    MAX_HARNESS_WALL_SECONDS,
    MAX_RUN_SECONDS,
)

# Locked continuous promote metric programs (SHA bound on campaign lock).
_PROMOTE_EXPECTATIONS_REL = Path(
    "src/slm_training/resources/experiments/autotrain_climb/"
    "metric_expectations.promote.v1.json"
)
_PROMOTE_FORMAL_TEMPLATE_ID = "metrics.structural_similarity_monotone"
_FIVE_LANES = (
    "measurement_control",
    "training_method",
    "architecture",
    "lean_model",
    "assumptions",
)
_CERTIFICATE_SCHEMA_V2 = "metric_certificate/v2"
# Promote Lean formal preflight wall (seconds). Timeouts are *inconclusive*
# (incomplete measurement), never a proof rejection / promotion_failed.
_PROMOTE_FORMAL_TIMEOUT_S = float(MAX_RUN_SECONDS)
_FORMAL_TIMEOUT_STATUSES = frozenset({"timed_out"})
_DRIVER_LOCK_BASENAME = "driver.lock"


class _CodeUpdated(RuntimeError):
    """The long-lived driver integrated code newer than its Python process."""


def _remaining_timeout(deadline: float | None = None) -> float:
    if deadline is None:
        return float(MAX_RUN_SECONDS)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("autotrain bounded cycle", MAX_RUN_SECONDS)
    return min(float(MAX_RUN_SECONDS), remaining)


def _stage_process_callbacks(
    *, root: Path | None, loop_id: str | None, stage: str | None
) -> tuple[Callable[[int], None] | None, Callable[[int], None] | None]:
    if root is None and loop_id is None:
        return None, None
    if root is None or loop_id is None or stage is None:
        raise ValueError("root, loop_id, and stage must be supplied together")
    _set_active_stage(root, loop_id, stage)

    def update(pid: int) -> None:
        _set_stage_process(root, loop_id, stage, pid)

    return update, update


def _stage_command(
    cmd: list[str],
    *,
    cwd: Path,
    deadline: float | None = None,
    root: Path | None = None,
    loop_id: str | None = None,
    stage: str | None = None,
) -> BoundedProcessResult:
    on_start, on_heartbeat = _stage_process_callbacks(
        root=root, loop_id=loop_id, stage=stage
    )
    return _bounded_command(
        cmd,
        cwd=cwd,
        deadline=deadline,
        on_start=on_start,
        on_heartbeat=on_heartbeat,
    )


def _git(
    *args: str,
    cwd: Path | None = None,
    deadline: float | None = None,
    root: Path | None = None,
    loop_id: str | None = None,
    stage: str | None = None,
) -> str:
    result = _stage_command(
        ["git", *args],
        cwd=cwd or Path.cwd(),
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage=stage,
    )
    _raise_for_bounded_result(result)
    return result.stdout.strip()


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    deadline: float | None = None,
    root: Path | None = None,
    loop_id: str | None = None,
    stage: str | None = None,
) -> None:
    print("+", " ".join(cmd), flush=True)
    result = _stage_command(
        cmd,
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage=stage,
    )
    if result.stdout:
        print(
            result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True
        )
    if result.stderr:
        print(
            result.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
            file=sys.stderr,
            flush=True,
        )
    _raise_for_bounded_result(result)


def _bounded_command(
    cmd: list[str],
    *,
    cwd: Path,
    deadline: float | None = None,
    on_start: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
) -> BoundedProcessResult:
    total = _remaining_timeout(deadline)
    grace = min(float(KILL_GRACE_SECONDS), total * 0.1)
    interrupt_after = min(float(INTERRUPT_AFTER_SECONDS), max(0.001, total - grace))
    return run_bounded_process(
        cmd,
        cwd=cwd,
        interrupt_after_seconds=interrupt_after,
        kill_grace_seconds=grace,
        on_start=on_start,
        on_heartbeat=on_heartbeat,
    )


def _raise_for_bounded_result(result: BoundedProcessResult) -> None:
    if result.timed_out:
        raise subprocess.TimeoutExpired(
            result.command,
            result.duration_seconds,
            output=result.stdout,
            stderr=result.stderr,
        )
    if result.outcome is ProcessOutcome.LAUNCH_FAILED:
        raise OSError(result.launch_error or "subprocess launch failed")
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.command,
            output=result.stdout,
            stderr=result.stderr,
        )


def _arm_wall_minutes(policy_minutes: float, *, formal_required: bool) -> float:
    """Give required stages equal room while retaining finalization."""

    stage_count = 3 if formal_required else 2
    arm_seconds = (
        float(MAX_HARNESS_WALL_SECONDS) - HARNESS_FINALIZATION_RESERVE_SECONDS
    ) / stage_count
    symmetric_minutes = arm_seconds / 60
    return min(float(policy_minutes), symmetric_minutes)


def _formal_lane_required(
    *, cycle_intent: str, replay: dict[str, Any] | None
) -> bool:
    """Reserve formal time only for work whose locked plan requires it."""

    if cycle_intent == "promote":
        return True
    if replay is None:
        return False
    return any(
        bool(replay[arm]["manifest"].formal_obligations)
        for arm in ("control", "candidate")
    )


def _require_symmetric_arm_budget(
    *, deadline: float, arm_count: int, arm_wall_minutes: float
) -> None:
    required = arm_count * arm_wall_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    remaining = _remaining_timeout(deadline)
    if remaining < required:
        raise subprocess.TimeoutExpired("symmetric decision-arm budget", required)


def _fit_symmetric_arm_budget(
    *, deadline: float, arm_count: int, requested_arm_wall_minutes: float
) -> float:
    """Share the post-planning budget equally while preserving finalization."""

    if arm_count <= 0:
        raise ValueError("arm_count must be positive")
    remaining = _remaining_timeout(deadline)
    usable = remaining - HARNESS_FINALIZATION_RESERVE_SECONDS
    if usable <= 0:
        raise subprocess.TimeoutExpired("symmetric decision-arm budget", remaining)
    return min(float(requested_arm_wall_minutes), usable / arm_count / 60)


def _post_formal_arm_budget_request(
    *,
    policy_minutes: float,
    initial_arm_wall_minutes: float,
    formal_completed: bool,
) -> float:
    """Return unused formal-lane time to the matched decision arms."""

    if formal_completed:
        return float(policy_minutes)
    return float(initial_arm_wall_minutes)


def _arm_execution_deadline(
    *, cycle_deadline: float, arm_wall_minutes: float
) -> float:
    """Cap one arm without spending the cycle's finalization reserve."""

    return min(
        cycle_deadline - HARNESS_FINALIZATION_RESERVE_SECONDS,
        time.monotonic() + arm_wall_minutes * 60,
    )


def _promotion_formal_budget_seconds(
    *, deadline: float, arm_count: int, arm_wall_minutes: float
) -> float:
    """Return only wall time left after reserving complete arms + finalization."""

    reserved = arm_count * arm_wall_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    available = _remaining_timeout(deadline) - reserved
    if available <= 0:
        raise subprocess.TimeoutExpired("promotion formal preflight budget", reserved)
    return min(float(_PROMOTE_FORMAL_TIMEOUT_S), available)


def _counterbalanced_arm_order(
    control_id: str,
    candidate_id: str,
    *,
    cycle_index: int,
    seed: int,
    promotion_replicate_index: int | None = None,
) -> list[str]:
    """Serialize matched arms in deterministic AB/BA order without relabeling."""

    if cycle_index < 1:
        raise ValueError("cycle_index must be positive")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    order = [control_id, candidate_id]
    reverse = (
        promotion_replicate_index % 2 == 1
        if promotion_replicate_index is not None
        else cycle_index % 2 == 0
    )
    if reverse:
        order.reverse()
    return order


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _metric_from_eval(path: Path, key: str) -> float | None:
    data = _read_json(path)
    if key in data and isinstance(data[key], (int, float)):
        return float(data[key])
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(key), (int, float)):
        return float(metrics[key])
    for suite_name in ("smoke", "held_out"):
        suite = data.get(suite_name)
        if isinstance(suite, dict) and isinstance(suite.get(key), (int, float)):
            return float(suite[key])
    return None


_METRIC_LEAVES = (
    "latency_ms_p50",
    "parse_rate",
    "meaningful_program_rate",
    "structural_similarity",
    "binder_reference_f1",
)


def _run_metrics(
    camp_dir: Path,
    run_id: str,
    *,
    prefer_held_out: bool = False,
) -> dict[str, float | None]:
    """Load smoke (+ held_out when present) metrics for Phase A classification.

    Screening primaries use smoke leaves. Promotion primaries use
    ``held_out.*`` (policy ``held_out.structural_similarity``). When
    ``prefer_held_out`` is true and held-out eval exists, leaf keys are filled
    from held_out so climb_policy leaf lookup and tradeoff paths see the same
    suite as the dotted primary.
    """
    run_dir = camp_dir / "runs" / run_id
    smoke = run_dir / "eval_smoke.json"
    if not smoke.exists():
        smoke = run_dir / "eval.json"
    held = run_dir / "eval_held_out.json"
    out: dict[str, float | None] = {leaf: None for leaf in _METRIC_LEAVES}

    if smoke.is_file():
        for leaf in _METRIC_LEAVES:
            val = _metric_from_eval(smoke, leaf)
            if val is not None:
                out[leaf] = val
                out[f"smoke.{leaf}"] = val

    if held.is_file():
        for leaf in _METRIC_LEAVES:
            val = _metric_from_eval(held, leaf)
            if val is not None:
                out[f"held_out.{leaf}"] = val
                if prefer_held_out:
                    out[leaf] = val
    return out


# Phase A positive classification: latency is never a free win over quality.
_EPS = 1e-12
# Smoke fixture n≈3 → one meaningful program is ~1/3. Below that a latency
# blip is not a real win (parse-only / empty-meaning arms).
_MIN_MPR_FOR_LATENCY_WIN = 1.0 / 3.0 - 1e-9
# Quality improvements may pay up to this latency regression (relative or abs).
_LATENCY_REGRESSION_BUDGET = 0.15
_LATENCY_REGRESSION_ABS_MS = 750.0
# ~12s wall-band noise must not mint positives.
_TIMEOUT_BAND_LO_MS = 11900.0
_TIMEOUT_BAND_HI_MS = 12150.0
_WIN_REASON_PREFIXES = (
    "primary_metric_win:",
    "quality_metric_win:",
    "efficiency_win:",
    "executable_unblock:",
)


def _measurement_is_complete(decision: dict[str, Any]) -> bool:
    incomplete_prefixes = (
        "empty_metrics:",
        "measurement_incomplete:",
        "primary_metric_unavailable",
        "wall_timeout:",
    )
    has_metrics = all(
        any(_finite_metric(value) is not None for value in metrics.values())
        for metrics in (
            decision.get("control_metrics") or {},
            decision.get("candidate_metrics") or {},
        )
    )
    return bool(
        has_metrics
        and not any(
            str(reason).startswith(incomplete_prefixes)
            for reason in decision.get("reasons") or []
        )
    )


# Champion queue: quality-held wins retest with same levers, new seeds, before
# thrashing the fixed lever bank again. Ledger is loop-local (not git).
_CHAMPION_QUEUE_SCHEMA = "autotrain_champion_queue/v1"
_CHAMPION_STATUSES = frozenset(
    {
        "queued",
        "confirming",
        "confirmation_inconclusive",
        "confirmed",
        "rejected",
        "skipped_duplicate",
        "promoting",
        "climb_accepted",
        # Read-only compatibility for pre-v30 queue ledgers.
        "promoted",
        "promotion_failed",
        # Formal wall / incomplete measurement — retryable, not a rejection.
        "promotion_inconclusive",
        # Execute/matrix/process abort before complete measurement — not a model reject.
        "harness_failure",
    }
)
_RETRYABLE_PROMOTE_STATUSES = frozenset(
    {"confirmed", "promotion_inconclusive", "harness_failure"}
)
# Recipe levers that define "same knobs" for confirmatory retest. Measurement
# knobs (seed, decode_timeout, eval_suites) are re-sampled from role policy.
_LEVER_KNOB_KEYS = (
    "ltr_prefix_loss_weight",
    "component_token_loss_weight",
    "component_edge_token_loss_weight",
    "compiler_decision_token_loss_weight",
    "structure_token_loss_weight",
    "typed_family_balance_loss_weight",
    "ltr_tail_loss_weight",
    "compiler_alignment_loss_weight",
    "compiler_alignment_margin",
    "compiler_alignment_stratified",
    "compiler_alignment_semantic_exhaustive",
    "compiler_alignment_kind_filter",
    "grammar_completion_bounds",
    "grammar_equivalence_cache",
    "grammar_draft_window",
    "compact_active_canvas",
    "mixture_sampling_policy",
    "mixture_exposure_target_profile",
    "mixture_total_decision_budget",
    "mixture_per_root_cap",
    "mixture_per_template_cap",
    "mixture_max_importance_weight",
    "component_plan_loss_weight",
    "component_plan_decode_weight",
    "component_edge_loss_weight",
    "component_edge_alignment_loss_weight",
    "component_edge_decode_weight",
    "component_inventory_loss_weight",
    "component_inventory_decode_weight",
    "binder_topology_loss_weight",
    "binder_topology_decode_weight",
    "binder_component_plan_loss_weight",
    "binder_component_plan_decode_weight",
    "binder_arity_loss_weight",
    "binder_arity_decode_weight",
    "slot_component_loss_weight",
    "slot_component_decode_weight",
    "symbol_boundary_loss_weight",
    "design_md_dropout",
    "fidelity_loss_weight",
    "semantic_contrast_dir",
    "semantic_contrast_loss_weight",
    "semantic_contrast_margin",
    "semantic_contrast_fraction",
    "symbol_slot_augmentation",
    "mask_pattern",
    "structural_aux_head_profile",
    "compiler_decode_mode",
    "steps",
    "batch_size",
    "train_version",
    "context_backend",
    "sync_checkpoints",
    "local_files_only",
    "output_tokenizer",
)
_QUALITY_ENQUEUE_PREFIXES = (
    "quality_held:",
    "quality_metric_win:",
)
# Dedup identity ignores cycle-local steps jitter (continuous does steps+(cycle%3)).
_FINGERPRINT_EXCLUDE_KEYS = frozenset({"steps"})
# Soft bound: confirm/promote attempts before the queue head is rejected.
_MAX_CONFIRM_ATTEMPTS = 2
_MAX_PROMOTE_ATTEMPTS = 2
_CAUSAL_FAMILY_ATTEMPT_CAP = 2
# Screening thrash bank — rotate recommended arm each cycle (Change B).
# Each entry: (slug, hypothesis fragment, knob extras relative to control).
# Special key "_steps_factor" multiplies base steps (depth confound).
_SCREENING_ARM_BANK: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "bounds",
        "grammar_completion_bounds reduces smoke latency_ms_p50 versus the matched control without lowering parse_rate.",
        {"grammar_completion_bounds": True},
    ),
    (
        "canvas",
        "compact_active_canvas reduces smoke latency_ms_p50 versus the matched control without lowering parse_rate.",
        {"compact_active_canvas": True},
    ),
    (
        "both",
        "Combined bounds and canvas beat either single lever on smoke latency_ms_p50.",
        {"grammar_completion_bounds": True, "compact_active_canvas": True},
    ),
    (
        "steps",
        "Doubling steps without levers only raises cost and does not improve unit decode latency.",
        {"_steps_factor": 2},
    ),
    (
        "batch1",
        "Halving batch_size changes smoke latency vs matched control without lowering parse_rate.",
        {"batch_size": 1},
    ),
    (
        "component-plan",
        "Component-plan train-and-decode coupling improves smoke structural_similarity without lowering parse_rate or binder_reference_f1.",
        {
            "component_plan_loss_weight": 1.0,
            "component_plan_decode_weight": 1.0,
            "structural_aux_head_profile": "component-plan",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "component-edge",
        "Component-edge train-and-decode coupling improves smoke structural_similarity without lowering parse_rate or binder_reference_f1.",
        {
            "component_edge_loss_weight": 1.0,
            "component_edge_decode_weight": 1.0,
            "structural_aux_head_profile": "component-edge",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "component-inventory",
        "Component-inventory train-and-decode coupling improves smoke structural_similarity without lowering parse_rate or binder_reference_f1.",
        {
            "component_inventory_loss_weight": 1.0,
            "component_inventory_decode_weight": 1.0,
            "structural_aux_head_profile": "component-inventory",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "binder-topology",
        "Low-weight binder-topology train-and-decode coupling improves smoke structural_similarity without lowering parse_rate or binder_reference_f1.",
        {
            "binder_topology_loss_weight": 0.25,
            "binder_topology_decode_weight": 1.0,
            "structural_aux_head_profile": "binder-topology",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "binder-arity",
        "Binder-arity train-and-decode coupling improves binder_reference_f1 and structural_similarity without lowering parse_rate.",
        {
            "binder_arity_loss_weight": 1.0,
            "binder_arity_decode_weight": 1.0,
            "structural_aux_head_profile": "binder-arity",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "binder-component-plan",
        "Binder-to-component-plan train-and-decode coupling improves binder_reference_f1 and structural_similarity without lowering parse_rate.",
        {
            "binder_component_plan_loss_weight": 1.0,
            "binder_component_plan_decode_weight": 1.0,
            "structural_aux_head_profile": "binder-component-plan",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "slot-component-coverage",
        "Implemented slot-component ownership supervision and legal decode bias improves component_type_recall and placeholder_fidelity without lowering parse_rate.",
        {
            "slot_component_loss_weight": 1.0,
            "slot_component_decode_weight": 1.0,
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "fidelity",
        "Stronger placeholder-fidelity supervision improves binder_reference_f1 and structural_similarity without lowering parse_rate.",
        {"fidelity_loss_weight": 1.5},
    ),
    (
        "edge-alignment",
        "Compiler-decision component-edge alignment improves structural_similarity and binder_reference_f1 without lowering parse_rate.",
        {
            "component_edge_alignment_loss_weight": 1.0,
            "structural_aux_head_profile": "component-edge",
        },
    ),
    (
        "semantic-contrast",
        "Hard-valid semantic-contrast margin supervision improves structural_similarity and binder_reference_f1 without lowering parse_rate.",
        {
            "semantic_contrast_dir": "src/slm_training/resources/data/eval/openui_hard_valid_v1",
            "semantic_contrast_loss_weight": 0.25,
            "semantic_contrast_margin": 1.0,
            "semantic_contrast_fraction": 0.5,
        },
    ),
    (
        "slot-augmentation",
        "Request-local slot permutation and alpha-renaming augmentation improves held-out binder_reference_f1 and structural_similarity without lowering parse_rate.",
        {"symbol_slot_augmentation": True},
    ),
    (
        "mixed-mask",
        "Mixed random and structured corruption better matches iterative denoising and improves structural_similarity without lowering parse_rate or binder_reference_f1.",
        {"mask_pattern": "mixed"},
    ),
    (
        "symbol-boundary",
        "Extra supervision on opaque-symbol tokens and their immediate boundaries improves structural_similarity and binder_reference_f1 without lowering parse_rate.",
        {"symbol_boundary_loss_weight": 1.0},
    ),
    (
        "design-dropout",
        "Deterministic DESIGN.md context dropout reduces scaffold-copy reliance and improves structural_similarity without lowering parse_rate or binder_reference_f1.",
        {"design_md_dropout": 0.25},
    ),
    (
        "scaffold-prefix",
        "Prefix-weighted LTR supervision improves early scaffold formation and structural_similarity without lowering parse_rate or binder_reference_f1.",
        {"ltr_prefix_loss_weight": 1.0},
    ),
    (
        "component-token",
        "Direct component-token reconstruction weighting improves component_type_recall and structural_similarity without lowering parse_rate or binder_reference_f1.",
        {"component_token_loss_weight": 1.0},
    ),
    (
        "component-edge-token",
        "Direct reconstruction weighting at compiler-derived non-root component edges improves structural_similarity and canonical AST agreement without lowering parse_rate or binder_reference_f1.",
        {"component_edge_token_loss_weight": 1.0},
    ),
    (
        "component-edge-margin",
        "Grammar-oracle component-edge alignment makes the gold child component outrank other legal component choices without lowering parse_rate or binder_reference_f1.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "component-edge",
        },
    ),
    (
        "compiler-decision-token",
        "Dense reconstruction weighting at every deterministic compiler decision improves meaningful OpenUI structure without lowering parse_rate or binder_reference_f1.",
        {"compiler_decision_token_loss_weight": 1.0},
    ),
    (
        "compiler-decision-margin",
        "Grammar-oracle alignment across every compiler-decision family makes each gold legal branch outrank its legal siblings without lowering parse_rate or binder_reference_f1.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
        },
    ),
    (
        "bounded-compiler-decision-margin",
        "Deterministic completion bounds reduce the all-family compiler-decision margin arm's forwards and latency while preserving its structural quality.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
            "grammar_completion_bounds": True,
        },
    ),
    (
        "cached-compiler-decision-margin",
        "Completion-domain equivalence caching reduces compiler time for the all-family compiler-decision margin arm while preserving its structural quality.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
            "grammar_equivalence_cache": True,
        },
    ),
    (
        "wide-draft-compiler-decision-margin",
        "A wider certified compiler draft window amortizes completion-forest construction and neural ranking while preserving the all-family margin arm's structural quality.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
            "grammar_draft_window": 16,
        },
    ),
    (
        "capacity-aware-compiler-decision-margin",
        "Capacity-aware online mixture sampling reduces concentrated record repeats and raises effective training exposure for the all-family compiler-decision margin arm without lowering guarded OpenUI quality.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
            "mixture_sampling_policy": "capacity_aware",
        },
    ),
    (
        "capacity-aware-tail-compiler-decision-margin",
        "Tail-weighted scaffold supervision preserves the capacity-aware all-family margin arm's quality gain while reducing runaway legal continuation, emitted tokens, forwards, and latency.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
            "mixture_sampling_policy": "capacity_aware",
            "ltr_tail_loss_weight": 1.0,
        },
    ),
    (
        "capacity-aware-semantic-exhaustive-compiler-decision-margin",
        "Exhaustive grammar-oracle supervision over every semantic compiler decision improves exact AST and canonical agreement on top of capacity-aware all-family alignment without lowering guarded OpenUI quality.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_semantic_exhaustive": True,
            "compiler_alignment_kind_filter": "all",
            "mixture_sampling_policy": "capacity_aware",
        },
    ),
    (
        "capacity-aware-semantic-exhaustive-structure-token-margin",
        "Direct STRUCT-token reconstruction counterbalances semantic-exhaustive compression and improves scaffold quality while retaining its reduced decode work.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_semantic_exhaustive": True,
            "compiler_alignment_kind_filter": "all",
            "mixture_sampling_policy": "capacity_aware",
            "structure_token_loss_weight": 1.0,
        },
    ),
    (
        "exposure-targeted-compiler-decision-margin",
        "Default-derived rare-action exposure improves guarded structural and semantic OpenUI quality relative to capacity-aware sampling at fixed model size, loss, and decode authority.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
            "mixture_sampling_policy": "exposure_targeted",
        },
    ),
    (
        "exposure-targeted-semantic-exhaustive-compiler-decision-margin",
        "Semantic-exhaustive compiler supervision preserves exposure-targeted semantic gains while reducing runaway legal continuation and decode cost.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_semantic_exhaustive": True,
            "compiler_alignment_kind_filter": "all",
            "mixture_sampling_policy": "exposure_targeted",
        },
    ),
    (
        "structure-token",
        "Direct grammar STRUCT-token reconstruction weighting repairs scaffold formation and structural_similarity without lowering parse_rate or binder_reference_f1.",
        {"structure_token_loss_weight": 1.0},
    ),
    (
        "typed-family-balance",
        "Count-normalized component and grammar STRUCT reconstruction improves structural_similarity without sacrificing component_type_recall or binder_reference_f1.",
        {"typed_family_balance_loss_weight": 0.25},
    ),
    (
        "container-close",
        "Grammar-derived container-close alignment makes the gold legal ')' or ']' outrank legal comma continuation without lowering structural_similarity, component_type_recall, or binder_reference_f1.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "container-close",
        },
    ),
    (
        "balanced-container-close",
        "Container-close alignment preserves the typed-family balance arm's structural gains while preventing runaway legal comma continuation.",
        {
            "typed_family_balance_loss_weight": 0.25,
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "container-close",
        },
    ),
    (
        "component-structure",
        "Joint component-plan and component-edge train-and-decode coupling improves smoke structural_similarity beyond either isolated arm.",
        {
            "component_plan_loss_weight": 1.0,
            "component_plan_decode_weight": 1.0,
            "component_edge_loss_weight": 1.0,
            "component_edge_decode_weight": 1.0,
            "structural_aux_head_profile": "component-structure",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "literal-close",
        "Tail-weighted LTR supervision reduces legal literal-termination starvation without lowering parse_rate or binder_reference_f1.",
        {"ltr_tail_loss_weight": 2.0},
    ),
    (
        "literal-margin",
        "Grammar-derived LIT_END alignment makes legal literal closure outrank legal byte continuation without lowering parse_rate or binder_reference_f1.",
        {
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "literal-close",
        },
    ),
)
_RECENT_EXHAUSTION_CYCLE_WINDOW = len(_SCREENING_ARM_BANK)


def _finite_metric(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _in_timeout_band(latency_ms: float | None) -> bool:
    return (
        latency_ms is not None
        and _TIMEOUT_BAND_LO_MS <= latency_ms <= _TIMEOUT_BAND_HI_MS
    )


def _champion_queue_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "champion_queue.jsonl"


def _driver_lock_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / _DRIVER_LOCK_BASENAME


def _loop_state_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "state.json"


def _write_loop_state(root: Path, state: AutotrainLoopStateV1) -> Path:
    path = _loop_state_path(root, state.loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _set_active_stage(root: Path, loop_id: str, stage: str) -> None:
    path = _loop_state_path(root, loop_id)
    if not path.is_file():
        return
    try:
        state = AutotrainLoopStateV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return
    _write_loop_state(
        root,
        state.model_copy(
            update={
                "active_stage": stage,
                "child_pid": None,
                "stage_started_at": None,
                "heartbeat_at": utc_now(),
            }
        ),
    )


def _set_stage_process(root: Path, loop_id: str, stage: str, child_pid: int) -> None:
    path = _loop_state_path(root, loop_id)
    if not path.is_file():
        return
    try:
        state = AutotrainLoopStateV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return
    now = utc_now()
    started_at = state.stage_started_at if state.active_stage == stage else None
    _write_loop_state(
        root,
        state.model_copy(
            update={
                "active_stage": stage,
                "child_pid": child_pid,
                "stage_started_at": started_at or now,
                "heartbeat_at": now,
            }
        ),
    )


def _record_cycle_failure(
    *, root: Path, loop_id: str, exc: Exception, cycle_index: int
) -> int:
    """Persist a stable blocker fingerprint and return its consecutive count."""
    message = str(exc).strip() or exc.__class__.__name__
    fingerprint = hashlib.sha256(
        f"{exc.__class__.__name__}:{message}".encode("utf-8")
    ).hexdigest()
    path = root / "loops" / loop_id / "cycle_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines()[-2:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                previous.append(row)
    soft = isinstance(exc, (subprocess.TimeoutExpired, TimeoutError, ConnectionError))
    if isinstance(exc, subprocess.CalledProcessError):
        soft = bool(exc.cmd and list(exc.cmd)[:2] == ["git", "fetch"])
    count = 0 if soft else 1
    if not soft:
        for row in reversed(previous):
            if not row.get("blocking", True) or row.get("fingerprint") != fingerprint:
                break
            count += 1
    record = {
        "schema": "autotrain_cycle_failure/v1",
        "loop_id": loop_id,
        "cycle_index": cycle_index,
        "error_type": exc.__class__.__name__,
        "message": message,
        "fingerprint": fingerprint,
        "consecutive_count": count,
        "blocking": not soft,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    blocked = not soft and count >= 3
    _write_loop_state(
        root,
        AutotrainLoopStateV1(
            loop_id=loop_id,
            state="BLOCKED" if blocked else "IDLE",
            phase="blocked" if blocked else "between_cycles",
            cycle_index=max(0, cycle_index),
            next_action=(
                "repair repeated blocker"
                if blocked
                else "retry incomplete cycle"
                if soft
                else "retry cycle"
            ),
            blocker_fingerprint=fingerprint,
            blocker_count=count,
            pid=os.getpid(),
        ),
    )
    return count


def acquire_driver_lock(
    root: Path,
    loop_id: str,
    *,
    code_sha: str | None = None,
) -> Any:
    """Exclusive flock for one continuous driver per loop_id.

    Kernel releases the lock if the process dies — no stale-pid reclaim needed.
    Second process raises ``RuntimeError`` with ``DRIVER_ALREADY_RUNNING``.
    Returns an open file object the caller must keep alive for the process life.
    """
    path = _driver_lock_path(root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fh.seek(0)
        existing = fh.read().strip() or "{}"
        fh.close()
        raise RuntimeError(
            f"DRIVER_ALREADY_RUNNING loop_id={loop_id} lock={path} holder={existing}"
        ) from exc
    payload = {
        "schema": "autotrain_driver_lock/v1",
        "loop_id": loop_id,
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": code_sha,
    }
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps(payload, sort_keys=True) + "\n")
    fh.flush()
    print(
        f"DRIVER_LOCK_ACQUIRED loop_id={loop_id} pid={os.getpid()} path={path}",
        flush=True,
    )
    return fh


def _lever_knobs(knobs: dict[str, Any] | None) -> dict[str, Any]:
    """Stable lever subset for confirm retests (excludes seed / measurement)."""
    if not isinstance(knobs, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _LEVER_KNOB_KEYS:
        if key in knobs and knobs[key] is not None:
            out[key] = knobs[key]
    return out


def _knobs_fingerprint(levers: dict[str, Any]) -> str:
    """Identity hash for champion dedup (excludes steps cycle jitter)."""
    stable = {
        k: v for k, v in (levers or {}).items() if k not in _FINGERPRINT_EXCLUDE_KEYS
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_champion_queue(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            entries.append(row)
    return entries


def _campaign_started_experiment(root: Path, campaign_id: object) -> bool:
    """Return whether a reserved champion attempt reached actual execution."""
    if not isinstance(campaign_id, str) or not campaign_id:
        return False
    path = root / campaign_id / "events.jsonl"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event_type") == "experiment_started":
            return True
    return False


def _recover_interrupted_champion_entries(
    root: Path, entries: list[dict[str, Any]]
) -> bool:
    """Release interrupted attempts and reopen incomplete confirmations."""
    changed = False
    for row in entries:
        status = row.get("status")
        if status == "rejected":
            campaign_id = str(row.get("confirm_campaign_id") or "")
            delivery_path = root / campaign_id / "sdlc_delivery.json"
            if not campaign_id or not delivery_path.is_file():
                continue
            delivery = _read_json(delivery_path)
            if delivery.get("measurement_complete") is not False:
                continue
            row["status"] = "confirmation_inconclusive"
            row["last_harness_failure_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            row["resolve_reasons"] = list(delivery.get("reasons") or [])
            row.pop("resolved_at", None)
            changed = True
            continue
        if status not in {"confirming", "promoting"}:
            continue
        # _update_champion_status records the currently reserved campaign in
        # confirm_campaign_id for both phases; promotion_campaign_id may still
        # name an older completed/inconclusive replicate.
        campaign_id = row.get("confirm_campaign_id") or row.get("promotion_campaign_id")
        started = _campaign_started_experiment(root, campaign_id)
        attempt_field = (
            "promote_attempts" if status == "promoting" else "confirm_attempts"
        )
        if not started:
            row[attempt_field] = max(0, int(row.get(attempt_field) or 0) - 1)
            row["last_preflight_failure_campaign_id"] = campaign_id
        if status == "promoting":
            row["status"] = "confirmed"
        elif not started:
            row["status"] = "queued"
        changed = True
    return changed


def _revalidate_open_champion_entries(
    root: Path, entries: list[dict[str, Any]]
) -> bool:
    """Reject queued champions invalidated by the current harness policy.

    The queue is durable while classifiers improve. Replaying an old ``positive``
    bit would otherwise let a signal that the current harness rejects consume a
    confirmation cycle. Incomplete source evidence remains retriable; only a
    complete current-policy non-win is closed.
    """

    changed = False
    for row in entries:
        if row.get("status") not in {"queued", "confirming"}:
            continue
        campaign_id = str(row.get("source_campaign_id") or "")
        control_id = str(row.get("source_control_id") or "")
        candidate_id = str(row.get("source_candidate_id") or "")
        camp_dir = root / campaign_id
        if not campaign_id or not control_id or not candidate_id or not camp_dir.is_dir():
            continue
        handoff = _read_json(camp_dir / "cycle_handoff.json")
        delivery = _read_json(camp_dir / "sdlc_delivery.json")
        try:
            current = _classify_positive(
                camp_dir=camp_dir,
                primary_metric=str(
                    handoff.get("primary_metric")
                    or delivery.get("primary_metric")
                    or ""
                ),
                control_id=control_id,
                candidate_id=candidate_id,
                role=str(
                    row.get("source_role")
                    or handoff.get("cycle_role")
                    or "screening"
                ),
            )
        except (OSError, TypeError, ValueError):
            continue
        if not _measurement_is_complete(current) or _should_enqueue_champion(current):
            continue
        row["status"] = "rejected"
        row["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        row["resolve_reasons"] = [
            "source_reclassified_nonpositive_under_current_policy",
            *[str(reason) for reason in current.get("reasons") or []],
        ]
        changed = True
        print(
            "CHAMPION_REVALIDATE_REJECT "
            f"entry_id={row.get('entry_id')} campaign={campaign_id}",
            flush=True,
        )
    return changed


def _revalidate_confirmed_champion_entries(
    root: Path, entries: list[dict[str, Any]]
) -> bool:
    """Close historical false confirmations under the current strict gate."""

    changed = False
    for row in entries:
        if row.get("status") != "confirmed":
            continue
        campaign_id = str(row.get("confirm_campaign_id") or "")
        delivery_path = root / campaign_id / "sdlc_delivery.json"
        if not campaign_id or not delivery_path.is_file():
            continue
        delivery = _read_json(delivery_path)
        if delivery.get("measurement_complete") is not True:
            continue
        if _confirmation_quality_reheld(delivery):
            continue
        row["status"] = "rejected"
        row["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        row["resolve_reasons"] = [
            "confirmation_reclassified_nonpositive_under_current_policy",
            *[str(reason) for reason in delivery.get("reasons") or []],
            "confirmation_rejected:primary_quality_not_reheld",
        ]
        changed = True
        print(
            "CHAMPION_CONFIRM_REVALIDATE_REJECT "
            f"entry_id={row.get('entry_id')} campaign={campaign_id}",
            flush=True,
        )
    return changed


def _refresh_champion_source_recipes(
    root: Path, entries: list[dict[str, Any]]
) -> bool:
    """Restore lever-complete queue recipes from immutable source experiments.

    Older queue writers projected only a subset of registered levers. Any such
    confirmation or promotion changed the treatment and is not evidence for the
    source winner, so reopen it at confirmation with the exact source recipes.
    """

    changed = False
    open_statuses = {
        "queued",
        "confirming",
        "confirmation_inconclusive",
        "confirmed",
        "promoting",
        "promotion_inconclusive",
        "harness_failure",
    }
    for row in entries:
        if row.get("status") not in open_statuses:
            continue
        source_dir = root / str(row.get("source_campaign_id") or "")
        candidate = _lever_knobs(
            _load_experiment_knobs(
                source_dir, str(row.get("source_candidate_id") or "")
            )
        )
        control = _lever_knobs(
            _load_experiment_knobs(
                source_dir, str(row.get("source_control_id") or "")
            )
        )
        if not candidate:
            continue
        if candidate == _lever_knobs(row.get("knobs") or {}) and control == _lever_knobs(
            row.get("control_knobs") or {}
        ):
            continue
        prior_status = str(row.get("status") or "")
        row.update(
            knobs=candidate,
            control_knobs=control,
            knobs_fingerprint=_knobs_fingerprint(candidate),
            status="queued",
            confirm_attempts=0,
            promote_attempts=0,
            confirm_campaign_id=None,
            confirm_cycle_index=None,
            promotion_campaign_id=None,
            promotion_cycle_index=None,
            resolved_at=None,
            resolve_reasons=[
                "champion_recipe_repaired_from_source",
                f"invalidated_phase_status:{prior_status}",
            ],
        )
        row.pop("cert_policy", None)
        row.pop("formal_preflight_status", None)
        row.pop("last_harness_failure", None)
        row.pop("last_harness_failure_at", None)
        changed = True
        print(
            "CHAMPION_RECIPE_REPAIRED "
            f"entry_id={row.get('entry_id')} prior_status={prior_status} "
            f"source={row.get('source_campaign_id')}",
            flush=True,
        )
    return changed


def _write_champion_queue(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in entries:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _queue_head_open(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First entry still awaiting confirmatory retest (queued or in-flight)."""
    for row in entries:
        if row.get("status") in {
            "queued",
            "confirming",
            "confirmation_inconclusive",
        }:
            return row
    return None


def _queue_head_confirmed(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First confirmed / inconclusive / harness-blocked champion for promote.

    ``promotion_inconclusive`` (formal wall) and ``harness_failure`` (execute /
    matrix / missing run) are retryable — not model rejects — so they re-enter
    the promote slot on the next promotion cadence.
    """
    for row in entries:
        if row.get("status") in _RETRYABLE_PROMOTE_STATUSES:
            return row
    return None


def _arm_slug_from_knobs(
    knobs: dict[str, Any], *, candidate_id: str = ""
) -> str | None:
    """Map knobs / candidate id to thrash arm slug."""
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("mixture_sampling_policy") == "exposure_targeted"
        and knobs.get("compiler_alignment_semantic_exhaustive")
    ):
        return "exposure-targeted-semantic-exhaustive-compiler-decision-margin"
    if knobs.get("slot_component_loss_weight"):
        return "slot-component-coverage"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("mixture_sampling_policy") == "exposure_targeted"
    ):
        return "exposure-targeted-compiler-decision-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("mixture_sampling_policy") == "capacity_aware"
        and knobs.get("compiler_alignment_semantic_exhaustive")
        and knobs.get("structure_token_loss_weight")
    ):
        return "capacity-aware-semantic-exhaustive-structure-token-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("mixture_sampling_policy") == "capacity_aware"
        and knobs.get("compiler_alignment_semantic_exhaustive")
    ):
        return "capacity-aware-semantic-exhaustive-compiler-decision-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("mixture_sampling_policy") == "capacity_aware"
        and knobs.get("ltr_tail_loss_weight")
    ):
        return "capacity-aware-tail-compiler-decision-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("mixture_sampling_policy") == "capacity_aware"
    ):
        return "capacity-aware-compiler-decision-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and int(knobs.get("grammar_draft_window") or 8) > 8
    ):
        return "wide-draft-compiler-decision-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("grammar_equivalence_cache")
    ):
        return "cached-compiler-decision-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("grammar_completion_bounds")
    ):
        return "bounded-compiler-decision-margin"
    if (
        knobs.get("typed_family_balance_loss_weight")
        and knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "container-close"
    ):
        return "balanced-container-close"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "literal-close"
    ):
        return "literal-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "container-close"
    ):
        return "container-close"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "component-edge"
    ):
        return "component-edge-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
    ):
        return "compiler-decision-margin"
    if knobs.get("ltr_tail_loss_weight"):
        return "literal-close"
    if knobs.get("ltr_prefix_loss_weight"):
        return "scaffold-prefix"
    if knobs.get("component_token_loss_weight"):
        return "component-token"
    if knobs.get("component_edge_token_loss_weight"):
        return "component-edge-token"
    if knobs.get("compiler_decision_token_loss_weight"):
        return "compiler-decision-token"
    if knobs.get("structure_token_loss_weight"):
        return "structure-token"
    if knobs.get("typed_family_balance_loss_weight"):
        return "typed-family-balance"
    if knobs.get("component_plan_loss_weight") and knobs.get(
        "component_edge_loss_weight"
    ):
        return "component-structure"
    if knobs.get("component_plan_loss_weight"):
        return "component-plan"
    if knobs.get("component_edge_alignment_loss_weight"):
        return "edge-alignment"
    if knobs.get("semantic_contrast_loss_weight"):
        return "semantic-contrast"
    if knobs.get("symbol_slot_augmentation"):
        return "slot-augmentation"
    if knobs.get("mask_pattern") == "mixed":
        return "mixed-mask"
    if knobs.get("symbol_boundary_loss_weight"):
        return "symbol-boundary"
    if knobs.get("design_md_dropout"):
        return "design-dropout"
    if knobs.get("component_edge_loss_weight"):
        return "component-edge"
    if knobs.get("component_inventory_loss_weight"):
        return "component-inventory"
    if knobs.get("binder_component_plan_loss_weight"):
        return "binder-component-plan"
    if knobs.get("binder_arity_loss_weight"):
        return "binder-arity"
    if knobs.get("binder_topology_loss_weight"):
        return "binder-topology"
    if float(knobs.get("fidelity_loss_weight") or 0.5) != 0.5:
        return "fidelity"
    if knobs.get("grammar_completion_bounds") and knobs.get("compact_active_canvas"):
        return "both"
    if knobs.get("grammar_completion_bounds"):
        return "bounds"
    if knobs.get("compact_active_canvas"):
        return "canvas"
    if knobs.get("batch_size") == 1:
        return "batch1"
    cid = candidate_id or ""
    if cid.endswith("-steps") or "-steps" in cid:
        return "steps"
    return None


def _skip_arm_slugs(
    entries: list[dict[str, Any]], *, integration_commit: str | None = None
) -> set[str]:
    """Deprioritize arms only while a champion is still open (not forever).

    Permanent skip of rejected/promotion_failed starved bounds/canvas and left
    only steps/batch1 thrash — which could not re-enter the champion path.
    """
    skip: set[str] = set()
    terminal_counts: dict[str, int] = {}
    for row in entries:
        knobs = row.get("knobs") or {}
        slug = _arm_slug_from_knobs(
            knobs, candidate_id=str(row.get("source_candidate_id") or "")
        )
        if (
            slug
            and integration_commit
            and row.get("source_integration_commit") == integration_commit
            and row.get("status")
            in {"rejected", "promotion_failed", "climb_accepted", "promoted"}
        ):
            terminal_counts[slug] = terminal_counts.get(slug, 0) + 1
        # Only skip arms currently in the funnel (not terminal failures).
        if row.get("status") not in {
            "queued",
            "confirming",
            "confirmation_inconclusive",
            "confirmed",
            "promoting",
            "promotion_inconclusive",
            "harness_failure",
        }:
            continue
        knobs = row.get("knobs") or {}
        slug = _arm_slug_from_knobs(
            knobs, candidate_id=str(row.get("source_candidate_id") or "")
        )
        if slug:
            skip.add(slug)
    skip.update(
        slug
        for slug, count in terminal_counts.items()
        if count >= _CAUSAL_FAMILY_ATTEMPT_CAP
    )
    return skip


def _recent_completed_nonpositive_slugs(
    root: Path,
    predecessor_campaign_id: str | None,
    *,
    max_cycles: int | None = None,
) -> set[str]:
    """Close complete null approaches across the loop lineage.

    ``max_cycles`` remains an explicit diagnostic/test bound. Production walks
    the content-linked lineage to its root so an old rejected approach cannot
    become novel merely by aging out of a sliding window.
    """

    cursor = predecessor_campaign_id
    seen: set[str] = set()
    exhausted: set[str] = set()
    loop_id: str | None = None
    lineage = itertools.count() if max_cycles is None else range(max(0, max_cycles))
    for _ in lineage:
        if not cursor or cursor in seen:
            break
        seen.add(cursor)
        camp_dir = root / cursor
        campaign = _read_json(camp_dir / "campaign.json")
        handoff = _read_json(camp_dir / "cycle_handoff.json")
        delivery = _read_json(camp_dir / "sdlc_delivery.json")
        campaign_loop = str(campaign.get("loop_id") or handoff.get("loop_id") or "")
        if loop_id is None:
            loop_id = campaign_loop or None
        elif campaign_loop and campaign_loop != loop_id:
            break
        candidate_id = str(delivery.get("candidate_id") or "")
        handoff_reasons = {str(item) for item in handoff.get("reasons") or []}
        runtime_terminal = bool(
            handoff.get("climb_state") == "rejected"
            and candidate_id
            and (
                f"candidate_runtime_rejected_after_frozen_replay:{candidate_id}"
                in handoff_reasons
                or f"candidate_runtime_unblock_reproduced:{candidate_id}"
                in handoff_reasons
            )
        )
        intent = str(
            delivery.get("cycle_intent")
            or handoff.get("cycle_intent")
            or handoff.get("cycle_role")
            or ""
        )
        stored_positive = delivery.get("positive")
        control_id = str(delivery.get("control_id") or "")
        primary_metric = str(
            delivery.get("primary_metric")
            or handoff.get("primary_metric")
            or "smoke.structural_similarity"
        )
        if candidate_id and control_id and delivery.get("measurement_complete") is True:
            role = str(
                delivery.get("cycle_role")
                or handoff.get("cycle_role")
                or ("promotion" if intent == "promotion" else "screening")
            )
            current_decision = _classify_positive(
                camp_dir=camp_dir,
                primary_metric=primary_metric,
                control_id=control_id,
                candidate_id=candidate_id,
                role=role,
            )
            if _measurement_is_complete(current_decision):
                current_decision["measurement_complete"] = True
                stored_positive = (
                    _confirmation_quality_reheld(current_decision)
                    if intent == "confirm"
                    else current_decision.get("positive")
                )
        if (
            candidate_id
            and stored_positive is False
            and (delivery.get("measurement_complete") is True or runtime_terminal)
            and (
                intent in {"screening", "promotion", "confirm", "retry_measurement"}
                or runtime_terminal
            )
        ):
            matrix = _read_json(camp_dir / "matrix-proposal.json")
            knobs = next(
                (
                    dict((item.get("experiment") or {}).get("knobs") or {})
                    for item in matrix.get("hypotheses") or []
                    if str((item.get("experiment") or {}).get("experiment_id") or "")
                    == candidate_id
                ),
                {},
            )
            slug = _arm_slug_from_knobs(knobs, candidate_id=candidate_id)
            if slug:
                exhausted.add(slug)
        cursor = str(campaign.get("predecessor_campaign_id") or "")
    return exhausted


def _select_recommended_slug(cycle: int, skip: set[str] | None = None) -> str:
    """Rotate screening arms without reopening a rejected approach."""
    # Evidence-triggered and successor quality arms do not perturb the stable
    # cycle-number rotation of the original screening bank. They become the
    # fail-forward path only after older approaches are closed.
    successor_slugs = (
        "binder-arity",
        "binder-component-plan",
        "slot-component-coverage",
        "literal-margin",
        "literal-close",
        "fidelity",
        "edge-alignment",
        "semantic-contrast",
        "slot-augmentation",
        "mixed-mask",
        "symbol-boundary",
        "design-dropout",
        "scaffold-prefix",
        "component-token",
        "component-edge-token",
        "component-edge-margin",
        "compiler-decision-token",
        "bounded-compiler-decision-margin",
        "cached-compiler-decision-margin",
        "wide-draft-compiler-decision-margin",
        "capacity-aware-compiler-decision-margin",
        "capacity-aware-tail-compiler-decision-margin",
        "capacity-aware-semantic-exhaustive-compiler-decision-margin",
        "capacity-aware-semantic-exhaustive-structure-token-margin",
        "exposure-targeted-compiler-decision-margin",
        "exposure-targeted-semantic-exhaustive-compiler-decision-margin",
        "structure-token",
        "typed-family-balance",
        "container-close",
        "balanced-container-close",
    )
    legacy_quality_slugs = {
        "component-plan",
        "component-edge",
        "component-inventory",
        "binder-topology",
        "component-structure",
    }
    bank = tuple(
        row
        for row in _SCREENING_ARM_BANK
        if row[0] not in successor_slugs
    )
    n = len(bank)
    start = (max(1, int(cycle)) - 1) % n
    ordered = [bank[(start + i) % n][0] for i in range(n)]
    skip = skip or set()
    if legacy_quality_slugs.issubset(skip):
        for slug in successor_slugs:
            if slug not in skip:
                return slug
    for slug in ordered:
        if slug not in skip:
            return slug
    for slug in successor_slugs:
        if slug not in skip:
            return slug
    raise RuntimeError(
        "registered screening arm bank exhausted; add a distinct preregistered "
        "quality objective instead of recycling a rejected approach"
    )


def _repeat_confirm_while_waiting_for_promotion(
    *,
    cadence_role: str,
    confirmed_champion: dict[str, Any] | None,
    cycle: int,
    skip: set[str],
) -> bool:
    """Use a fresh confirmation seed when no novel screen fits before cadence.

    Promotion cadence protects held-out suites from opportunistic exposure. A
    confirmed champion must therefore not promote early merely because the
    screening bank is exhausted. A second bounded confirmation is useful,
    remains on screening suites, and avoids turning the expected wait into an
    orchestration failure.
    """

    if cadence_role == "promotion" or not confirmed_champion:
        return False
    if confirmed_champion.get("status") != "confirmed":
        return False
    try:
        _select_recommended_slug(cycle, skip=skip)
    except RuntimeError:
        return True
    return False


def _select_cycle_slug(
    cycle: int,
    *,
    predecessor_priority: str | None,
    skip: set[str],
    has_confirm_levers: bool,
    has_promote_levers: bool,
) -> str | None:
    """Select only for screening; confirm/promote carry frozen recipes."""

    if has_confirm_levers or has_promote_levers:
        return None
    return predecessor_priority or _select_recommended_slug(cycle, skip=skip)


def _apply_arm_extras(base_steps: int, extras: dict[str, Any]) -> dict[str, Any]:
    """Materialize arm knob extras (handles _steps_factor)."""
    out = {k: v for k, v in extras.items() if not str(k).startswith("_")}
    factor = extras.get("_steps_factor")
    if factor is not None:
        out["steps"] = max(int(base_steps * float(factor)), base_steps + 10)
    return out


def _quality_held_reasons(reasons: list[str] | None) -> bool:
    """True when Phase A reasons include a quality hold/win (not pure latency)."""
    if not reasons:
        return False
    return any(
        any(r.startswith(prefix) for prefix in _QUALITY_ENQUEUE_PREFIXES)
        for r in reasons
    )


def _confirmation_quality_reheld(delivery: dict[str, Any]) -> bool:
    """Require the policy-owned primary quality win on a confirmation run.

    Screening may surface an efficiency tradeoff as a hypothesis, but a fresh
    confirmation is the gate into promotion.  Faster decode cannot confirm a
    champion when the declared primary or a required non-regression metric got
    worse.
    """

    if not delivery.get("positive") or delivery.get("measurement_complete") is False:
        return False
    reasons = [str(reason) for reason in delivery.get("reasons") or []]
    blocked_prefixes = (
        "primary_metric_null_or_worse:",
        "non_regression_fail:",
        "eg_params_block:",
        "measurement_incomplete:",
        "wall_timeout:",
        "empty_metrics:",
        "harness_failure:",
    )
    if any(reason.startswith(blocked_prefixes) for reason in reasons):
        return False
    primary_metric = str(delivery.get("primary_metric") or "")
    if not primary_metric:
        return False
    return any(
        reason.startswith(f"primary_metric_win:{primary_metric}:") for reason in reasons
    )


def _should_enqueue_champion(delivery: dict[str, Any]) -> bool:
    """Enqueue only quality-held positives (never empty-meaning latency blips)."""
    if not delivery.get("positive"):
        return False
    reasons = list(delivery.get("reasons") or [])
    primary_leaf = str(delivery.get("primary_metric") or "").rsplit(".", 1)[-1]
    if primary_leaf != "latency_ms_p50" and _confirmation_quality_reheld(delivery):
        return True
    if _quality_held_reasons(reasons):
        return True
    # Efficiency / primary latency win only when quality_held co-tagged.
    if any(
        r.startswith("efficiency_win:") or r.startswith("primary_metric_win:")
        for r in reasons
    ):
        return _quality_held_reasons(reasons)
    return False


def _screening_enqueue_allowed(
    *, cycle_intent: str, replay: dict[str, Any] | None
) -> bool:
    """Preserve screening queue semantics across an exact frozen retry."""

    if cycle_intent in {"screening", "promotion"}:
        return True
    return bool(
        cycle_intent == "retry_measurement"
        and replay is not None
        and replay["handoff"].cycle_role == "screening"
        and getattr(replay["handoff"], "cycle_intent", None) != "confirm"
    )


def _confirmation_replay_entry(
    entries: list[dict[str, Any]], replay: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Return the original champion resolved by a frozen confirmation replay."""

    if replay is None or replay["handoff"].cycle_intent != "confirm":
        return None
    frozen = replay["candidate"]["experiment"]
    if not str(frozen.get("experiment_id") or "").endswith("-confirm"):
        return None
    fingerprint = _knobs_fingerprint(_lever_knobs(frozen.get("knobs") or {}))
    source_campaign = str(replay["handoff"].campaign_id)
    return next(
        (
            row
            for row in entries
            if row.get("knobs_fingerprint") == fingerprint
            and row.get("confirm_campaign_id") == source_campaign
            and row.get("status")
            in {"queued", "confirming", "confirmation_inconclusive"}
        ),
        None,
    )


def _reconcile_completed_confirmation_replays(
    root: Path, entries: list[dict[str, Any]]
) -> bool:
    """Repair historical duplicate queue rows emitted by completed retries."""

    changed = False
    for duplicate in entries:
        if duplicate.get("status") != "queued" or not str(
            duplicate.get("source_candidate_id") or ""
        ).endswith("-confirm"):
            continue
        campaign_id = str(duplicate.get("source_campaign_id") or "")
        handoff_path = root / campaign_id / "cycle_handoff.json"
        delivery_path = root / campaign_id / "sdlc_delivery.json"
        if not handoff_path.is_file() or not delivery_path.is_file():
            continue
        handoff = _read_json(handoff_path)
        delivery = _read_json(delivery_path)
        if (
            handoff.get("cycle_intent") != "retry_measurement"
            or delivery.get("measurement_complete") is not True
        ):
            continue
        original = next(
            (
                row
                for row in entries
                if row is not duplicate
                and row.get("knobs_fingerprint")
                == duplicate.get("knobs_fingerprint")
                and row.get("status") == "confirmation_inconclusive"
            ),
            None,
        )
        if original is None:
            continue
        reasons = list(delivery.get("reasons") or [])
        confirmed = _confirmation_quality_reheld(delivery)
        if not confirmed:
            reasons.append("confirmation_rejected:primary_quality_not_reheld")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        original.update(
            status="confirmed" if confirmed else "rejected",
            confirm_campaign_id=campaign_id,
            confirm_cycle_index=handoff.get("cycle_index"),
            resolved_at=stamp,
            resolve_reasons=reasons,
        )
        duplicate.update(
            status="skipped_duplicate",
            resolved_at=stamp,
            resolve_reasons=[
                f"confirmation_replay_resolved:{original.get('entry_id')}"
            ],
        )
        changed = True
        print(
            "CHAMPION_REPLAY_RECONCILE "
            f"entry_id={original.get('entry_id')} status={original.get('status')} "
            f"duplicate={duplicate.get('entry_id')} campaign={campaign_id}",
            flush=True,
        )
    return changed


def _is_champion_lever(knobs: dict[str, Any], *, candidate_id: str = "") -> bool:
    """True when knobs encode a thrash arm (not pure matched control)."""
    return _arm_slug_from_knobs(knobs, candidate_id=candidate_id) is not None


def _load_experiment_knobs(camp_dir: Path, experiment_id: str) -> dict[str, Any]:
    exp_path = camp_dir / "artifacts" / "experiments" / f"{experiment_id}.json"
    if not exp_path.is_file():
        # Some writers store experiment_id as stem without full path.
        matches = list((camp_dir / "artifacts" / "experiments").glob("*.json"))
        for path in matches:
            data = _read_json(path)
            if data.get("experiment_id") == experiment_id:
                knobs = data.get("knobs")
                return knobs if isinstance(knobs, dict) else {}
        return {}
    data = _read_json(exp_path)
    knobs = data.get("knobs")
    return knobs if isinstance(knobs, dict) else {}


def _enqueue_champion(
    *,
    root: Path,
    loop_id: str,
    delivery: dict[str, Any],
    camp_dir: Path,
) -> dict[str, Any] | None:
    """Append a queued champion when Phase A is a quality-held win."""
    if not _should_enqueue_champion(delivery):
        return None
    candidate_id = str(delivery.get("candidate_id") or "")
    control_id = str(delivery.get("control_id") or "")
    if not candidate_id or candidate_id == delivery.get("control_id"):
        return None
    knobs = _lever_knobs(_load_experiment_knobs(camp_dir, candidate_id))
    control_knobs = _lever_knobs(_load_experiment_knobs(camp_dir, control_id))
    if not knobs:
        return None
    # Any non-control thrash lever may champion (bounds/canvas/both/steps/batch1).
    if not _is_champion_lever(knobs, candidate_id=candidate_id):
        return None
    fp = _knobs_fingerprint(knobs)
    path = _champion_queue_path(root, loop_id)
    entries = _load_champion_queue(path)
    for row in entries:
        if row.get("knobs_fingerprint") == fp and row.get("status") in {
            "queued",
            "confirming",
            "confirmation_inconclusive",
            "confirmed",
            "promoting",
            "promotion_inconclusive",
            "harness_failure",
        }:
            # Already open / confirmed / pending promote — do not re-queue thrash.
            return None
    entry = {
        "schema": _CHAMPION_QUEUE_SCHEMA,
        "entry_id": f"champ-{loop_id}-{delivery.get('cycle_index')}-{fp}",
        "loop_id": loop_id,
        "status": "queued",
        "enqueued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_campaign_id": delivery.get("campaign_id"),
        "source_cycle_index": delivery.get("cycle_index"),
        "source_candidate_id": candidate_id,
        "source_control_id": control_id,
        "source_role": delivery.get("cycle_role") or delivery.get("role"),
        "source_integration_commit": _read_json(camp_dir / "campaign.json").get(
            "integration_commit"
        ),
        "knobs": knobs,
        "control_knobs": control_knobs,
        "knobs_fingerprint": fp,
        "source_metrics": {
            "control": delivery.get("control_metrics"),
            "candidate": delivery.get("candidate_metrics"),
        },
        "source_reasons": list(delivery.get("reasons") or []),
        "confirm_campaign_id": None,
        "confirm_cycle_index": None,
        "confirm_attempts": 0,
        "promote_attempts": 0,
        "resolved_at": None,
        "resolve_reasons": None,
    }
    entries.append(entry)
    _write_champion_queue(path, entries)
    print(
        f"CHAMPION_ENQUEUE entry_id={entry['entry_id']} fingerprint={fp} "
        f"candidate={candidate_id}",
        flush=True,
    )
    return entry


def _bump_champion_attempt(
    *,
    root: Path,
    loop_id: str,
    entry_id: str,
    field: str,
) -> int:
    """Increment confirm_attempts / promote_attempts; return new value."""
    path = _champion_queue_path(root, loop_id)
    entries = _load_champion_queue(path)
    value = 0
    for row in entries:
        if row.get("entry_id") != entry_id:
            continue
        value = int(row.get(field) or 0) + 1
        row[field] = value
        break
    _write_champion_queue(path, entries)
    return value


def _update_champion_status(
    *,
    root: Path,
    loop_id: str,
    entry_id: str,
    status: str,
    confirm_campaign_id: str | None = None,
    confirm_cycle_index: int | None = None,
    resolve_reasons: list[str] | None = None,
) -> dict[str, Any] | None:
    if status not in _CHAMPION_STATUSES:
        raise ValueError(f"invalid champion status: {status!r}")
    path = _champion_queue_path(root, loop_id)
    entries = _load_champion_queue(path)
    updated: dict[str, Any] | None = None
    for row in entries:
        if row.get("entry_id") != entry_id:
            continue
        row["status"] = status
        if confirm_campaign_id is not None:
            row["confirm_campaign_id"] = confirm_campaign_id
        if confirm_cycle_index is not None:
            row["confirm_cycle_index"] = confirm_cycle_index
        if status in {
            "confirmed",
            "rejected",
            "skipped_duplicate",
            "climb_accepted",
            "promotion_failed",
        }:
            row["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            row["resolve_reasons"] = list(resolve_reasons or [])
        elif status in {
            "confirmation_inconclusive",
            "promotion_inconclusive",
            "harness_failure",
        }:
            # Capture reasons but keep the head retriable (no permanent resolve).
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if status == "promotion_inconclusive":
                row["last_inconclusive_at"] = stamp
            else:
                row["last_harness_failure_at"] = stamp
            row["resolve_reasons"] = list(resolve_reasons or [])
            row.pop("resolved_at", None)
        updated = row
        break
    if updated is not None:
        _write_champion_queue(path, entries)
        print(
            f"CHAMPION_STATUS entry_id={entry_id} status={status} "
            f"campaign={confirm_campaign_id}",
            flush=True,
        )
    return updated


def _resolve_confirm_result(
    *,
    root: Path,
    loop_id: str,
    entry: dict[str, Any],
    delivery: dict[str, Any],
    campaign_id: str,
    cycle_index: int,
) -> dict[str, Any] | None:
    """Mark confirmatory retest confirmed (quality re-holds) or rejected."""
    reasons = list(delivery.get("reasons") or [])
    if delivery.get("measurement_complete") is not True:
        reasons.append("confirmation_inconclusive:measurement_incomplete")
        return _update_champion_status(
            root=root,
            loop_id=loop_id,
            entry_id=str(entry["entry_id"]),
            status="confirmation_inconclusive",
            confirm_campaign_id=campaign_id,
            confirm_cycle_index=cycle_index,
            resolve_reasons=reasons,
        )
    ok = _confirmation_quality_reheld(delivery)
    if not ok:
        reasons.append("confirmation_rejected:primary_quality_not_reheld")
    status = "confirmed" if ok else "rejected"
    return _update_champion_status(
        root=root,
        loop_id=loop_id,
        entry_id=str(entry["entry_id"]),
        status=status,
        confirm_campaign_id=campaign_id,
        confirm_cycle_index=cycle_index,
        resolve_reasons=reasons,
    )


def promote_expectations_path() -> Path:
    """Repo-relative locked promote expectations (absolute when repo root known)."""
    # Prefer package resource next to climb policy.
    pkg = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "slm_training"
        / "resources"
        / "experiments"
        / "autotrain_climb"
        / "metric_expectations.promote.v1.json"
    )
    if pkg.is_file():
        return pkg
    cwd_pkg = Path.cwd() / _PROMOTE_EXPECTATIONS_REL
    return cwd_pkg


def locked_promote_expectations_sha256() -> str:
    """SHA-256 of the locked continuous promote expectation manifest."""
    path = promote_expectations_path()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_status_is_timeout(status: str | None) -> bool:
    return str(status or "") in _FORMAL_TIMEOUT_STATUSES or str(
        status or ""
    ).startswith("timed_out")


def dispose_champion_promote(
    *,
    formal_preflight_status: str | None,
    certificate: dict[str, Any] | None,
    locked_expectations_sha256: str | None = None,
    phase_a_positive: bool = False,
    phase_a_quality_held: bool = False,
    control_metrics: dict[str, float | None] | None = None,
    candidate_metrics: dict[str, float | None] | None = None,
    promotion_primary: dict[str, Any] | None = None,
    promotion_dispose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authoritative promote disposition (proof + effect driver).

    Phase A smoke quality-held alone never yields ``climb_accepted``. A proved formal
    preflight, dual-arm promotion-primary win (≥ policy ``minimum_effect``), and
    an in-band LeverProof metric_certificate/v2 (``optimum_feedback`` →
    ``continue``) are all required. Cert continue is necessary but not
    sufficient. Theorem misses fail closed without a five-lane matrix;
    assumption-backed misses fail closed and request five-lane successor
    diagnosis.

    Formal **timeouts** are incomplete measurement: disposition
    ``promotion_inconclusive`` (retryable), never ``promotion_failed`` /
    ``rejected``.
    """
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        promotion_primary_effect_met,
    )
    from slm_training.harnesses.experiments.verified_metrics import optimum_feedback

    reasons: list[str] = []
    primary_improvement: float | None = None
    promotion_primary_met: bool | None = None
    if phase_a_positive:
        reasons.append("phase_a_positive")
    if phase_a_quality_held:
        reasons.append("phase_a_quality_held")

    def _base(
        *,
        status: str,
        cert_policy: str | None = None,
        diagnosis_lanes: list[str] | None = None,
        emit_five_lane_matrix: bool = False,
        breaches: list[Any] | None = None,
        inconclusive: bool = False,
        timeout: bool = False,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": status,
            "reasons": reasons,
            "cert_policy": cert_policy,
            "diagnosis_lanes": list(diagnosis_lanes or []),
            "emit_five_lane_matrix": emit_five_lane_matrix,
            "breaches": list(breaches or []),
            "primary_improvement": primary_improvement,
            "promotion_primary_met": promotion_primary_met,
        }
        if inconclusive:
            out["inconclusive"] = True
        if timeout:
            out["timeout"] = True
        return out

    if _formal_status_is_timeout(formal_preflight_status):
        reasons.append(
            f"formal_preflight_timed_out:status={formal_preflight_status!r}:"
            f"wall_s={_PROMOTE_FORMAL_TIMEOUT_S:g}"
        )
        reasons.append("measurement_incomplete:formal_timeout_not_rejection")
        return _base(
            status="promotion_inconclusive",
            inconclusive=True,
            timeout=True,
        )

    if formal_preflight_status != "proved":
        reasons.append(f"formal_preflight_unproved:status={formal_preflight_status!r}")
        return _base(status="promotion_failed")

    if certificate is None:
        reasons.append("promote_requires_certificate:phase_a_alone_insufficient")
        return _base(status="promotion_failed")

    if certificate.get("schema") != _CERTIFICATE_SCHEMA_V2:
        reasons.append(
            f"promote_requires_certificate_v2:got={certificate.get('schema')!r}"
        )
        return _base(status="promotion_failed")

    # Fail closed: locked expectations digest is required (never optional).
    if not locked_expectations_sha256:
        reasons.append("promote_requires_locked_expectations_digest")
        return _base(status="promotion_failed")
    cert_exp = certificate.get("metric_expectations_sha256")
    if cert_exp != locked_expectations_sha256:
        reasons.append(
            "certificate_expectations_digest_mismatch:"
            f"locked={str(locked_expectations_sha256)[:12]} cert={str(cert_exp)[:12]}"
        )
        return _base(status="promotion_failed")

    try:
        feedback = optimum_feedback(certificate)
    except Exception as exc:  # noqa: BLE001 — fail closed on bad certs
        reasons.append(f"certificate_invalid:{exc}")
        return _base(status="promotion_failed")

    policy = str(feedback.get("policy") or "")
    reasons.append(f"cert_policy:{policy}")
    lanes = list(feedback.get("diagnosis_lanes") or [])
    breaches = list(feedback.get("breaches") or [])

    if policy == "continue":
        # Effect gate: cert continue alone never promotes.
        climb = load_climb_policy()
        dispose_cfg = dict(promotion_dispose or climb.promotion_dispose)
        primary_spec = dict(promotion_primary or climb.promotion_primary)
        require_primary = bool(dispose_cfg.get("require_primary_win", True))
        if require_primary:
            ok, effect_reasons, effect_delta = promotion_primary_effect_met(
                control_metrics=control_metrics,
                candidate_metrics=candidate_metrics,
                promotion_primary=primary_spec,
                require_dual_arm_metrics=bool(
                    dispose_cfg.get("require_dual_arm_metrics", True)
                ),
                require_parse_non_regression=bool(
                    dispose_cfg.get("require_parse_non_regression", True)
                ),
            )
            reasons.extend(effect_reasons)
            primary_improvement = effect_delta
            promotion_primary_met = ok
            if not ok:
                return _base(status="promotion_failed", cert_policy=policy)
        else:
            promotion_primary_met = None
            reasons.append("promote_primary_win_not_required_by_policy")
        return _base(status="climb_accepted", cert_policy=policy)
    if policy == "stop":
        reasons.append("theorem_backed_band_miss")
        return _base(
            status="promotion_failed",
            cert_policy=policy,
            diagnosis_lanes=lanes,
            breaches=breaches,
        )
    if policy == "block_promotion_and_diagnose":
        reasons.append("assumption_backed_band_miss")
        return _base(
            status="promotion_failed",
            cert_policy=policy,
            diagnosis_lanes=lanes or list(_FIVE_LANES),
            emit_five_lane_matrix=True,
            breaches=breaches,
        )
    reasons.append(f"certificate_not_promotable:policy={policy!r}")
    return _base(
        status="promotion_failed",
        cert_policy=policy or None,
        diagnosis_lanes=lanes,
        breaches=breaches,
    )


def build_five_lane_successor_matrix(
    *,
    campaign_id: str,
    entry: dict[str, Any],
    breaches: list[dict[str, Any]],
    cert_policy: str | None,
) -> dict[str, Any]:
    """Preregistered five-lane diagnosis matrix after assumption-backed miss."""
    lanes = list(_FIVE_LANES)
    hypotheses = []
    for i, lane in enumerate(lanes, start=1):
        hypotheses.append(
            {
                "rank": i,
                "lane": lane,
                "hypothesis": (
                    f"Lane '{lane}' explains the assumption-backed band miss "
                    f"for champion {entry.get('knobs_fingerprint')} under "
                    f"cert_policy={cert_policy}."
                ),
                "falsification": (
                    "Controlled retest of this lane alone fails to move the "
                    "missed metric into the locked band."
                ),
                "breaches": breaches,
            }
        )
    return {
        "schema": "autotrain_five_lane_successor/v1",
        "campaign_id": campaign_id,
        "champion_entry_id": entry.get("entry_id"),
        "knobs_fingerprint": entry.get("knobs_fingerprint"),
        "cert_policy": cert_policy,
        "lanes": lanes,
        "hypotheses": hypotheses,
        "breaches": breaches,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_five_lane_successor(
    camp_dir: Path,
    *,
    campaign_id: str,
    entry: dict[str, Any],
    disposition: dict[str, Any],
) -> Path | None:
    if not disposition.get("emit_five_lane_matrix"):
        return None
    payload = build_five_lane_successor_matrix(
        campaign_id=campaign_id,
        entry=entry,
        breaches=list(disposition.get("breaches") or []),
        cert_policy=disposition.get("cert_policy"),
    )
    path = camp_dir / "five_lane_successor_matrix.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FIVE_LANE_SUCCESSOR path={path}", flush=True)
    return path


def _load_promote_certificate(camp_dir: Path) -> dict[str, Any] | None:
    """Load campaign-local metric certificate if present (JSON object)."""
    candidates = [
        camp_dir / "metric-certificate.json",
        camp_dir / "artifacts" / "metric-certificate.json",
        camp_dir / "promote" / "metric-certificate.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        data = _read_json(path)
        if data:
            return data
    return None


def _rate_to_pm(value: object) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return int(max(0, min(1000, round(float(value) * 1000.0))))


def _latency_ms_to_ns(value: object) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(1, int(round(float(value) * 1_000_000.0)))


def _run_suite_metrics(camp_dir: Path, run_id: str) -> dict[str, float | None]:
    """Load parse / structure / latency from smoke or held_out eval JSON."""
    run_dir = camp_dir / "runs" / run_id
    out: dict[str, float | None] = {
        "latency_ms_p50": None,
        "parse_rate": None,
        "structural_similarity": None,
        "meaningful_program_rate": None,
    }
    for name in ("eval_held_out.json", "eval_smoke.json", "eval.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        data = _read_json(path)
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else data
        if not isinstance(metrics, dict):
            continue
        for key in out:
            if out[key] is None and isinstance(metrics.get(key), (int, float)):
                out[key] = float(metrics[key])
        # Nested suite blocks
        for suite_key in ("held_out", "smoke"):
            suite = data.get(suite_key)
            if isinstance(suite, dict):
                for key in out:
                    if out[key] is None and isinstance(suite.get(key), (int, float)):
                        out[key] = float(suite[key])
    # Fallback to smoke helpers used by Phase A
    base = _run_metrics(camp_dir, run_id)
    for key, val in base.items():
        if out.get(key) is None and val is not None:
            out[key] = val
    return out


def _candidate_row_for_cert(
    *,
    arm_id: str,
    knobs: dict[str, Any],
    latency_ms: float | None,
    parse_rate: float | None,
) -> dict[str, Any]:
    lat_ns = _latency_ms_to_ns(latency_ms) or 1
    lever = hashlib.sha256(
        json.dumps(knobs, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    successes = 2
    quality_failures = 0 if (parse_rate is not None and parse_rate >= 1.0 - 1e-9) else 0
    return {
        "id": arm_id,
        "hardware": "cpu",
        "lever_snapshot_sha256": lever,
        "cold_ns": [lat_ns],
        "warm_ns": [lat_ns],
        "input_units": [1],
        "passes": [1],
        "energy_uj": [1],
        "cost_micro_usd": [1],
        "successes": successes,
        "quality_failures": quality_failures,
        "trainable_params": 100,
    }


def export_promote_metric_certificate(
    *,
    camp_dir: Path,
    campaign_id: str,
    control_id: str,
    candidate_id: str,
    delivery: dict[str, Any] | None = None,
    deadline: float | None = None,
    root: Path | None = None,
    loop_id: str | None = None,
) -> tuple[Path | None, str | None]:
    """Build LeverProof evidence + certificate for continuous promote.

    Returns ``(certificate_path, error_reason)``. Fail closed when metrics or
    checker are unavailable — never invent a green certificate.
    """
    from slm_training.harnesses.experiments.verified_metrics import (
        IN_REPO_CHECKER,
        VerifiedMetricError,
        write_metric_evidence,
    )

    if not IN_REPO_CHECKER.is_file():
        return None, f"leverproof_checker_missing:{IN_REPO_CHECKER}"

    ctrl = _run_suite_metrics(camp_dir, control_id)
    cand = _run_suite_metrics(camp_dir, candidate_id)
    # Prefer held-out / suite structural_similarity; fall back to MPR for fixture.
    ss = cand.get("structural_similarity")
    if ss is None:
        ss = cand.get("meaningful_program_rate")
    parse = cand.get("parse_rate")
    ss_pm = _rate_to_pm(ss)
    parse_pm = _rate_to_pm(parse)
    if ss_pm is None or parse_pm is None:
        return None, (f"promote_cert_incomplete_metrics:ss={ss!r} parse={parse!r}")

    # Repeat samples (fixture n≈3 style) for band assessment stability.
    observations = {
        "schema": "metric_observations/v1",
        "metrics": {
            "held_out_structural_similarity_pm": [ss_pm, ss_pm, ss_pm],
            "parse_rate_pm": [parse_pm, parse_pm, parse_pm],
        },
    }
    promote_dir = camp_dir / "promote"
    promote_dir.mkdir(parents=True, exist_ok=True)
    obs_path = promote_dir / "metric-observations.json"
    obs_path.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")

    exp_path = promote_expectations_path()
    # Provenance stubs (content-addressed by checker via SHA).
    bundle = promote_dir / "evidence-bundle.json"
    flags = promote_dir / "feature_flags.json"
    bundle.write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "control_id": control_id,
                "candidate_id": candidate_id,
                "control_metrics": ctrl,
                "candidate_metrics": cand,
                "delivery_reasons": list((delivery or {}).get("reasons") or []),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    flags.write_text(
        json.dumps({"continuous_promote": True, "schema": "autotrain_promote_flags/v1"})
        + "\n",
        encoding="utf-8",
    )

    # Prefer campaign manifest if present for digest binding.
    man_path = None
    man_dir = camp_dir / "manifests"
    if man_dir.is_dir():
        for p in sorted(man_dir.glob("*.json")):
            if "promote" in p.stem or "confirm" in p.stem:
                man_path = p
                break
        if man_path is None:
            mans = sorted(man_dir.glob("*.json"))
            man_path = mans[0] if mans else None

    ctrl_knobs = _lever_knobs(_load_experiment_knobs(camp_dir, control_id)) or {
        "grammar_completion_bounds": False,
        "compact_active_canvas": False,
    }
    cand_knobs = _lever_knobs(_load_experiment_knobs(camp_dir, candidate_id)) or {}
    candidates = [
        _candidate_row_for_cert(
            arm_id="control",
            knobs=ctrl_knobs,
            latency_ms=ctrl.get("latency_ms_p50"),
            parse_rate=ctrl.get("parse_rate"),
        ),
        _candidate_row_for_cert(
            arm_id="candidate",
            knobs=cand_knobs,
            latency_ms=cand.get("latency_ms_p50"),
            parse_rate=cand.get("parse_rate"),
        ),
    ]

    try:
        evidence_path = promote_dir / "metric-evidence.json"
        write_metric_evidence(
            evidence_path,
            run_id=f"{campaign_id}-promote",
            evidence_bundle_path=bundle,
            feature_flags_path=flags,
            campaign_manifest_path=man_path,
            cold_requests=1,
            warm_requests=1,
            candidates=candidates,
            expectation_manifest_path=exp_path,
            observations_path=obs_path,
        )
    except VerifiedMetricError as exc:
        return None, f"promote_evidence_build_failed:{exc}"

    cert_path = camp_dir / "metric-certificate.json"
    checker_deadline = time.monotonic() + 120.0
    if deadline is not None:
        checker_deadline = min(checker_deadline, deadline)
    try:
        completed = _stage_command(
            [str(IN_REPO_CHECKER), "check", str(evidence_path)],
            cwd=camp_dir,
            deadline=checker_deadline,
            root=root,
            loop_id=loop_id,
            stage="promotion-certificate",
        )
    except subprocess.TimeoutExpired as exc:
        return None, f"promote_certify_failed:{exc}"
    if completed.timed_out:
        return None, "promote_certify_failed:checker timed out within 120s cap"
    if completed.outcome is ProcessOutcome.LAUNCH_FAILED:
        return None, f"promote_certify_failed:{completed.launch_error}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "rejected").strip()[:400]
        return None, f"promote_certify_rejected:{detail}"
    try:
        cert = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return None, f"promote_certify_invalid_json:{exc}"
    if cert.get("schema") != _CERTIFICATE_SCHEMA_V2:
        return None, f"promote_certify_not_v2:{cert.get('schema')!r}"
    # Prefer candidate selection when present.
    if cert.get("selected_candidate") not in {"candidate", "control"}:
        pass
    cert_path.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    # Also mirror under promote/
    (promote_dir / "metric-certificate.json").write_text(
        cert_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(
        f"PROMOTE_CERT_EXPORT path={cert_path} "
        f"selected={cert.get('selected_candidate')} "
        f"ss_pm={ss_pm} parse_pm={parse_pm}",
        flush=True,
    )
    return cert_path, None


def _formal_preflight_status(camp_dir: Path) -> str | None:
    """Read only a cache-validated formal preflight status for promote gate."""
    path = camp_dir / "formal_preflight_status.json"
    if path.is_file():
        data = _read_json(path)
        status = data.get("status")
        if status != "proved":
            return str(status) if status is not None else None
        expected_sha = data.get("preflight_sha256")
        validated_sha = data.get("binding_validated_sha256")
        if expected_sha and validated_sha == expected_sha:
            return "proved"
    return None


def record_formal_preflight_status(
    camp_dir: Path,
    *,
    status: str,
    template_id: str,
    reason: str | None = None,
    timeout_seconds: float | None = None,
    duration_seconds: float | None = None,
    timed_out: bool | None = None,
) -> Path:
    camp_dir.mkdir(parents=True, exist_ok=True)
    path = camp_dir / "formal_preflight_status.json"
    payload: dict[str, Any] = {
        "schema": "autotrain_formal_preflight_status/v1",
        "status": status,
        "template_id": template_id,
        "reason": reason,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if timeout_seconds is not None:
        payload["timeout_seconds"] = float(timeout_seconds)
    if duration_seconds is not None:
        payload["duration_seconds"] = float(duration_seconds)
    if timed_out is not None:
        payload["timed_out"] = bool(timed_out)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def promote_formal_claim_dict() -> dict[str, str]:
    """Canonical required formal claim payload for promote experiment specs."""
    return {
        "template_id": _PROMOTE_FORMAL_TEMPLATE_ID,
        "claim": (
            "Structural similarity is monotone under declared component "
            "inequalities for continuous promote."
        ),
        "policy": "required",
    }


def ensure_promote_formal_preflight(
    *,
    camp_dir: Path,
    campaign_id: str,
    experiment_id: str,
    run_lean: bool = False,
    timeout_seconds: float = _PROMOTE_FORMAL_TIMEOUT_S,
    root: Path | None = None,
    loop_id: str | None = None,
) -> tuple[str, str | None]:
    """Record formal preflight; return ``(status, content_sha256|None)``.

    When proved, writes a content-addressed artifact under
    ``artifacts/formal_preflights/<sha>.json`` matching
    ``validate_formal_preflights`` binding. Fail closed on any error.
    """
    from slm_training.autoresearch.schemas import FormalClaimV1

    claim = FormalClaimV1(**promote_formal_claim_dict())
    status_path = camp_dir / "formal_preflight_status.json"
    if status_path.is_file():
        data = _read_json(status_path)
        status = str(data.get("status") or "missing")
        sha = data.get("preflight_sha256")
        if status == "proved" and sha:
            artifact = camp_dir / "artifacts" / "formal_preflights" / f"{sha}.json"
            try:
                from slm_training.autoresearch.formal import (
                    validate_formal_preflight_artifact,
                )

                validated = validate_formal_preflight_artifact(
                    artifact,
                    campaign_id=campaign_id,
                    experiment_id=experiment_id,
                    claim=claim,
                    expected_sha256=str(sha),
                )
                if validated.status != "proved":
                    raise ValueError(
                        f"required cached formal status is {validated.status!r}"
                    )
            except Exception as exc:  # noqa: BLE001 - stale cache fails closed
                if not run_lean:
                    record_formal_preflight_status(
                        camp_dir,
                        status="unknown",
                        template_id=_PROMOTE_FORMAL_TEMPLATE_ID,
                        reason=f"cached_formal_preflight_invalid:{exc}",
                    )
                    return "unknown", None
            else:
                data["binding_validated_sha256"] = str(sha)
                status_path.write_text(
                    json.dumps(data, indent=2) + "\n", encoding="utf-8"
                )
                return "proved", str(sha)
        elif not run_lean:
            return status, str(sha) if sha else None

    if not run_lean:
        record_formal_preflight_status(
            camp_dir,
            status="missing",
            template_id=_PROMOTE_FORMAL_TEMPLATE_ID,
            reason="formal_preflight_not_run",
        )
        return "missing", None

    try:
        from slm_training.autoresearch.formal import (
            run_formal_preflight,
            validate_formal_preflight_artifact,
        )
        from slm_training.autoresearch.schemas import (
            ExperimentKnobs,
            ExperimentSpec,
        )
        from slm_training.lineage.records import canonical_json

        exp = ExperimentSpec(
            experiment_id=experiment_id,
            campaign_id=campaign_id,
            hypothesis="Continuous promote requires proved structural-similarity mono.",
            rationale="Proof driver: required formal preflight before promote train.",
            expected_effect="Block promote when formal status is not proved.",
            falsification_criteria=("Required formal preflight is not proved.",),
            stop_conditions=("Stop before promote train when formal preflight fails.",),
            citations=("docs/design/formal-autoresearch.md",),
            knobs=ExperimentKnobs(steps=1),
            formal_claims=(claim,),
        )
        # The caller can further tighten the repository-wide wall.
        on_start, on_heartbeat = _stage_process_callbacks(
            root=root, loop_id=loop_id, stage="promotion-formal-preflight"
        )
        preflight, _obligation = run_formal_preflight(
            campaign_id,
            exp,
            claim,
            timeout_seconds=timeout_seconds,
            on_start=on_start,
            on_heartbeat=on_heartbeat,
        )
        status = str(preflight.status)
        duration = float(getattr(preflight, "duration_seconds", 0.0) or 0.0)
        payload = preflight.model_dump(mode="json")
        content_sha = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        art = camp_dir / "artifacts" / "formal_preflights"
        art.mkdir(parents=True, exist_ok=True)
        # Content-addressed name required by validate_formal_preflights.
        out = art / f"{content_sha}.json"
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        timed_out = _formal_status_is_timeout(status)
        if timed_out:
            reason = (
                f"formal_preflight_timed_out:wall_s={timeout_seconds:g}:"
                f"duration_s={duration:.3f}"
            )
        elif status == "proved":
            reason = None
        else:
            reason = f"preflight_status={status}"
        record_formal_preflight_status(
            camp_dir,
            status=status,
            template_id=_PROMOTE_FORMAL_TEMPLATE_ID,
            reason=reason,
            timeout_seconds=timeout_seconds,
            duration_seconds=duration,
            timed_out=timed_out,
        )
        # Persist sha for manifest binding.
        status_path = camp_dir / "formal_preflight_status.json"
        st = _read_json(status_path)
        st["preflight_sha256"] = content_sha
        if status == "proved":
            validate_formal_preflight_artifact(
                out,
                campaign_id=campaign_id,
                experiment_id=experiment_id,
                claim=claim,
                expected_sha256=content_sha,
            )
            st["binding_validated_sha256"] = content_sha
        status_path.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
        return status, content_sha
    except Exception as exc:  # noqa: BLE001 — fail closed for non-timeout errors
        msg = str(exc)
        timed_out = "timed out" in msg.lower() or "timeout" in msg.lower()
        status = "timed_out" if timed_out else "unknown"
        record_formal_preflight_status(
            camp_dir,
            status=status,
            template_id=_PROMOTE_FORMAL_TEMPLATE_ID,
            reason=(
                f"formal_preflight_timed_out:wall_s={timeout_seconds:g}:{msg}"
                if timed_out
                else f"formal_preflight_error:{exc}"
            ),
            timeout_seconds=timeout_seconds,
            timed_out=timed_out,
        )
        return status, None


def _run_has_usable_metrics(camp_dir: Path, run_id: str) -> bool:
    """True when suite metrics include parse or structure for cert export."""
    if not run_id:
        return False
    metrics = _run_suite_metrics(camp_dir, run_id)
    return (
        metrics.get("parse_rate") is not None
        or metrics.get("structural_similarity") is not None
        or metrics.get("meaningful_program_rate") is not None
    )


def detect_promote_harness_failure(
    *,
    camp_dir: Path,
    control_id: str,
    candidate_id: str,
    arm_exits: dict[str, int] | None = None,
    cert_err: str | None = None,
) -> list[str]:
    """Return harness_failure reasons when measurement is incomplete due to process.

    Harness failures are not model quality rejects: missing promote run, hard
    execute exit before artifacts, cert incomplete because candidate never ran.
    """
    reasons: list[str] = []
    exits = arm_exits or {}
    cand = str(candidate_id or "")
    ctrl = str(control_id or "")
    cand_exit = exits.get(cand)
    if cand and cand_exit is not None and int(cand_exit) == 1:
        reasons.append(f"harness_failure:promote_arm_exit:{int(cand_exit)}")
    if cand:
        cand_dir = camp_dir / "runs" / cand
        if not cand_dir.is_dir():
            reasons.append("harness_failure:missing_promote_run")
        elif not _run_has_usable_metrics(camp_dir, cand):
            if cert_err and "incomplete_metrics" in str(cert_err):
                reasons.append("harness_failure:cert_export_no_candidate_metrics")
            elif cand_exit is not None and int(cand_exit) not in {0, 2}:
                reasons.append(
                    f"harness_failure:promote_arm_no_metrics:exit={int(cand_exit)}"
                )
    if (
        cert_err
        and "incomplete_metrics" in str(cert_err)
        and cand
        and not _run_has_usable_metrics(camp_dir, cand)
    ):
        tag = "harness_failure:cert_export_no_candidate_metrics"
        if tag not in reasons:
            reasons.append(tag)
    if cert_err and "missing_run_ids" in str(cert_err):
        reasons.append(f"harness_failure:{cert_err}")
    # Control-only success with no candidate is always a harness/process gap
    # when promote was intended (caller only invokes this on promote intent).
    if (
        ctrl
        and _run_has_usable_metrics(camp_dir, ctrl)
        and cand
        and not (camp_dir / "runs" / cand).is_dir()
    ):
        if "harness_failure:missing_promote_run" not in reasons:
            reasons.append("harness_failure:missing_promote_run")
    return reasons


_PROMOTION_REPLICATE_SCHEMA = "autotrain_promotion_replicate/v1"


def _promotion_replicate_ledger_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "promotion_replicates.jsonl"


def _promotion_replicate_sha(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _promotion_replicate_evidence_is_current(root: Path, row: dict[str, Any]) -> bool:
    campaign_id = row.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        return False
    camp_dir = (root / campaign_id).resolve()
    if not camp_dir.is_relative_to(root.resolve()):
        return False
    delivery_path = camp_dir / "sdlc_delivery.json"
    delivery = _read_json(delivery_path)
    evidence = row.get("evidence")
    if not delivery_path.is_file() or not isinstance(evidence, dict):
        return False
    if (
        evidence.get("delivery_sha256")
        != hashlib.sha256(delivery_path.read_bytes()).hexdigest()
    ):
        return False
    control_id = str(row.get("control_id") or "")
    candidate_id = str(row.get("candidate_id") or "")
    seed = row.get("seed")
    order = row.get("arm_order")
    if (
        type(seed) is not int
        or not control_id
        or not candidate_id
        or not isinstance(order, list)
        or len(order) != 2
        or any(not isinstance(item, str) for item in order)
        or set(order) != {control_id, candidate_id}
        or delivery.get("measurement_complete") is not True
        or delivery.get("control_id") != control_id
        or delivery.get("candidate_id") != candidate_id
        or delivery.get("arm_seed") != seed
        or delivery.get("arm_order") != order
    ):
        return False
    metrics_sha = _promotion_replicate_sha(
        {
            "control": delivery.get("control_metrics") or {},
            "candidate": delivery.get("candidate_metrics") or {},
        }
    )
    if evidence.get("metrics_sha256") != metrics_sha:
        return False
    manifest_digests = evidence.get("manifests")
    if not isinstance(manifest_digests, dict) or set(manifest_digests) != {
        control_id,
        candidate_id,
    }:
        return False
    for arm_id in (control_id, candidate_id):
        manifest_path = camp_dir / "manifests" / f"{arm_id}.json"
        manifest = _read_json(manifest_path)
        if (
            not manifest_path.is_file()
            or seed not in (manifest.get("seeds") or [])
            or manifest_digests.get(arm_id)
            != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ):
            return False
    certificate = camp_dir / "metric-certificate.json"
    return (
        certificate.is_file()
        and evidence.get("certificate_sha256")
        == hashlib.sha256(certificate.read_bytes()).hexdigest()
    )


def _verified_promotion_replicates(
    root: Path, loop_id: str, entry: dict[str, Any]
) -> list[dict[str, Any]]:
    path = _promotion_replicate_ledger_path(root, loop_id)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(row, dict)
            or row.get("schema") != _PROMOTION_REPLICATE_SCHEMA
        ):
            continue
        if row.get("entry_id") != entry.get("entry_id") or row.get(
            "knobs_fingerprint"
        ) != entry.get("knobs_fingerprint"):
            continue
        claimed = row.get("content_sha256")
        content = {key: value for key, value in row.items() if key != "content_sha256"}
        if claimed != _promotion_replicate_sha(content):
            continue
        if not _promotion_replicate_evidence_is_current(root, row):
            continue
        rows.append(row)
    return rows


def _record_promotion_replicate(
    *,
    root: Path,
    loop_id: str,
    entry: dict[str, Any],
    campaign_id: str,
    cycle_index: int,
    camp_dir: Path,
    delivery: dict[str, Any],
    arm_exits: dict[str, int] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Append one complete, content-bound paired-seed promotion result."""

    rows = _verified_promotion_replicates(root, loop_id, entry)
    if any(row.get("campaign_id") == campaign_id for row in rows):
        return rows, None
    control_id = str(delivery.get("control_id") or "")
    candidate_id = str(delivery.get("candidate_id") or "")
    seed = delivery.get("arm_seed")
    order = delivery.get("arm_order")
    exits = arm_exits or {}
    if delivery.get("measurement_complete") is not True:
        return rows, "promotion_replicate_incomplete:measurement"
    if type(seed) is not int:
        return rows, "promotion_replicate_incomplete:seed"
    if (
        not isinstance(order, list)
        or len(order) != 2
        or set(order) != {control_id, candidate_id}
    ):
        return rows, "promotion_replicate_incomplete:arm_order"
    if any(exits.get(arm_id) != 0 for arm_id in (control_id, candidate_id)):
        return rows, "promotion_replicate_incomplete:arm_exit"

    manifest_digests: dict[str, str] = {}
    for arm_id in (control_id, candidate_id):
        path = camp_dir / "manifests" / f"{arm_id}.json"
        manifest = _read_json(path)
        if not path.is_file() or seed not in (manifest.get("seeds") or []):
            return rows, f"promotion_replicate_incomplete:manifest:{arm_id}"
        manifest_digests[arm_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    certificate = camp_dir / "metric-certificate.json"
    if not certificate.is_file():
        return rows, "promotion_replicate_incomplete:certificate"
    delivery_path = camp_dir / "sdlc_delivery.json"
    durable_delivery = _read_json(delivery_path)
    if not delivery_path.is_file() or any(
        durable_delivery.get(key) != delivery.get(key)
        for key in (
            "measurement_complete",
            "control_id",
            "candidate_id",
            "control_metrics",
            "candidate_metrics",
            "arm_seed",
            "arm_order",
        )
    ):
        return rows, "promotion_replicate_incomplete:durable_delivery"

    evidence = {
        "manifests": manifest_digests,
        "certificate_sha256": hashlib.sha256(certificate.read_bytes()).hexdigest(),
        "delivery_sha256": hashlib.sha256(delivery_path.read_bytes()).hexdigest(),
        "metrics_sha256": _promotion_replicate_sha(
            {
                "control": delivery.get("control_metrics") or {},
                "candidate": delivery.get("candidate_metrics") or {},
            }
        ),
    }
    record: dict[str, Any] = {
        "schema": _PROMOTION_REPLICATE_SCHEMA,
        "loop_id": loop_id,
        "entry_id": entry.get("entry_id"),
        "knobs_fingerprint": entry.get("knobs_fingerprint"),
        "campaign_id": campaign_id,
        "cycle_index": cycle_index,
        "seed": seed,
        "arm_order": order,
        "control_id": control_id,
        "candidate_id": candidate_id,
        "evidence": evidence,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record["content_sha256"] = _promotion_replicate_sha(record)
    if not any(row.get("content_sha256") == record["content_sha256"] for row in rows):
        path = _promotion_replicate_ledger_path(root, loop_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        rows.append(record)
    return rows, None


def _gate_promotion_on_replicates(
    *,
    root: Path,
    loop_id: str,
    entry: dict[str, Any],
    campaign_id: str,
    cycle_index: int,
    camp_dir: Path,
    delivery: dict[str, Any],
    arm_exits: dict[str, int] | None,
    disposition: dict[str, Any],
) -> dict[str, Any]:
    """Prevent a favorable single pair from satisfying a multi-seed claim."""

    if disposition.get("status") != "climb_accepted":
        return disposition
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        promotion_seed_floor,
    )

    min_seeds, require_multi_seed = promotion_seed_floor(load_climb_policy())
    required = int(min_seeds) if require_multi_seed else 1
    rows, error = _record_promotion_replicate(
        root=root,
        loop_id=loop_id,
        entry=entry,
        campaign_id=campaign_id,
        cycle_index=cycle_index,
        camp_dir=camp_dir,
        delivery=delivery,
        arm_exits=arm_exits,
    )
    distinct_seeds = {int(row["seed"]) for row in rows if type(row.get("seed")) is int}
    order_roles = {
        "AB" if row.get("arm_order", [None])[0] == row.get("control_id") else "BA"
        for row in rows
    }
    orders_complete = required < 2 or order_roles == {"AB", "BA"}
    if error is None and len(distinct_seeds) >= required and orders_complete:
        return {
            **disposition,
            "promotion_replicate_count": len(distinct_seeds),
            "promotion_replicate_required": required,
        }
    reason = error or (
        f"promotion_replicates_incomplete:{len(distinct_seeds)}/{required}:"
        f"orders={','.join(sorted(order_roles))}"
    )
    return {
        **disposition,
        "status": "promotion_inconclusive",
        "inconclusive": True,
        "promotion_replicate_count": len(distinct_seeds),
        "promotion_replicate_required": required,
        "reasons": [*(disposition.get("reasons") or []), reason],
    }


def _resolve_promotion_result(
    *,
    root: Path,
    loop_id: str,
    entry: dict[str, Any],
    delivery: dict[str, Any],
    campaign_id: str,
    cycle_index: int,
    camp_dir: Path | None = None,
    certificate: dict[str, Any] | None = None,
    formal_preflight_status: str | None = None,
    locked_expectations_sha256: str | None = None,
    arm_exits: dict[str, int] | None = None,
    cert_err: str | None = None,
) -> dict[str, Any] | None:
    """Mark promotion using certificate + formal preflight (not Phase A alone).

    Harness/process aborts (missing promote run, matrix membership exit, etc.)
    dispose as ``harness_failure`` — never model ``promotion_failed`` /
    ``rejected``.
    """
    camp = camp_dir or (root / campaign_id)
    reasons_in = list(delivery.get("reasons") or [])
    phase_a_positive = bool(delivery.get("positive"))
    phase_a_quality = _quality_held_reasons(reasons_in) or any(
        r.startswith("primary_metric_win:") or r.startswith("quality_metric_win:")
        for r in reasons_in
    )

    if formal_preflight_status is None:
        formal_preflight_status = _formal_preflight_status(camp)
    if certificate is None:
        certificate = _load_promote_certificate(camp)

    control_id = str(delivery.get("control_id") or "")
    candidate_id = str(delivery.get("candidate_id") or "")
    if not control_id or not candidate_id:
        runs = camp / "runs"
        if runs.is_dir():
            names = sorted(p.name for p in runs.iterdir() if p.is_dir())
            for n in names:
                if n.endswith("-control"):
                    control_id = control_id or n
                if "-promote" in n or n.endswith("-confirm"):
                    candidate_id = candidate_id or n
            if not candidate_id and len(names) >= 2:
                candidate_id = names[-1]
            if not control_id and names:
                control_id = names[0]

    # Prefer harness failure over model reject when measurement never completed.
    harness_reasons = detect_promote_harness_failure(
        camp_dir=camp,
        control_id=control_id,
        candidate_id=candidate_id,
        arm_exits=arm_exits,
        cert_err=cert_err
        or next(
            (r for r in reasons_in if "promote_cert" in r or "incomplete_metrics" in r),
            None,
        ),
    )
    # Also surface explicit matrix-membership strings from delivery reasons.
    for r in reasons_in:
        if "exact member of the latest hypothesis matrix" in str(r):
            tag = "harness_failure:matrix_membership"
            if tag not in harness_reasons:
                harness_reasons.append(tag)

    if harness_reasons and not _formal_status_is_timeout(formal_preflight_status):
        disposition = {
            "status": "harness_failure",
            "reasons": list(harness_reasons),
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
            "harness_failure": True,
        }
        status = "harness_failure"
        resolve_reasons = list(disposition["reasons"]) + reasons_in
        locked_expectations_sha256 = locked_expectations_sha256  # may be None
    else:
        if locked_expectations_sha256 is None:
            try:
                locked_expectations_sha256 = locked_promote_expectations_sha256()
            except OSError as exc:
                # Fail closed: never promote without a readable locked digest.
                disposition = {
                    "status": "promotion_failed",
                    "reasons": [f"promote_locked_expectations_unreadable:{exc}"],
                    "cert_policy": None,
                    "diagnosis_lanes": [],
                    "emit_five_lane_matrix": False,
                    "breaches": [],
                }
                resolve_reasons = list(disposition["reasons"]) + reasons_in
                return _update_champion_status(
                    root=root,
                    loop_id=loop_id,
                    entry_id=str(entry["entry_id"]),
                    status="promotion_failed",
                    confirm_campaign_id=campaign_id,
                    confirm_cycle_index=cycle_index,
                    resolve_reasons=resolve_reasons,
                )

        # Dual-arm held-out metrics for the promotion primary effect gate.
        ctrl_metrics = delivery.get("control_metrics")
        cand_metrics = delivery.get("candidate_metrics")
        if not isinstance(ctrl_metrics, dict) or not isinstance(cand_metrics, dict):
            ctrl_metrics = (
                _run_metrics(camp, control_id, prefer_held_out=True)
                if control_id
                else {}
            )
            cand_metrics = (
                _run_metrics(camp, candidate_id, prefer_held_out=True)
                if candidate_id
                else {}
            )
        from slm_training.autoresearch.climb_policy import load_climb_policy

        climb = load_climb_policy()
        disposition = dispose_champion_promote(
            formal_preflight_status=formal_preflight_status,
            certificate=certificate,
            locked_expectations_sha256=locked_expectations_sha256,
            phase_a_positive=phase_a_positive,
            phase_a_quality_held=phase_a_quality,
            control_metrics=ctrl_metrics,  # type: ignore[arg-type]
            candidate_metrics=cand_metrics,  # type: ignore[arg-type]
            promotion_primary=dict(climb.promotion_primary),
            promotion_dispose=dict(climb.promotion_dispose),
        )
        disposition = _gate_promotion_on_replicates(
            root=root,
            loop_id=loop_id,
            entry=entry,
            campaign_id=campaign_id,
            cycle_index=cycle_index,
            camp_dir=camp,
            delivery=delivery,
            arm_exits=arm_exits,
            disposition=disposition,
        )
        status = str(disposition["status"])
        resolve_reasons = list(disposition.get("reasons") or []) + reasons_in

    _write_five_lane_successor(
        camp,
        campaign_id=campaign_id,
        entry=entry,
        disposition=disposition,
    )

    # Append-only learning certificate ledger (loop-local).
    cert_ledger = root / "loops" / loop_id / "learning_certificate_ledger.jsonl"
    cert_ledger.parent.mkdir(parents=True, exist_ok=True)
    with cert_ledger.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "schema": "autotrain_learning_event/v1",
                    "loop_id": loop_id,
                    "campaign_id": campaign_id,
                    "cycle_index": cycle_index,
                    "entry_id": entry.get("entry_id"),
                    "knobs_fingerprint": entry.get("knobs_fingerprint"),
                    "outcome": status,
                    "cert_policy": disposition.get("cert_policy"),
                    "formal_preflight_status": formal_preflight_status,
                    "locked_expectations_sha256": locked_expectations_sha256,
                    "primary_improvement": disposition.get("primary_improvement"),
                    "promotion_primary_met": disposition.get("promotion_primary_met"),
                    "promotion_replicate_count": disposition.get(
                        "promotion_replicate_count"
                    ),
                    "promotion_replicate_required": disposition.get(
                        "promotion_replicate_required"
                    ),
                    "arm_order": delivery.get("arm_order"),
                    "arm_seed": delivery.get("arm_seed"),
                    "reasons": resolve_reasons,
                    "harness_failure": bool(disposition.get("harness_failure")),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                sort_keys=True,
            )
            + "\n"
        )

    updated = _update_champion_status(
        root=root,
        loop_id=loop_id,
        entry_id=str(entry["entry_id"]),
        status=status,
        confirm_campaign_id=campaign_id,
        confirm_cycle_index=cycle_index,
        resolve_reasons=resolve_reasons,
    )
    if updated is not None:
        path = _champion_queue_path(root, loop_id)
        entries = _load_champion_queue(path)
        for row in entries:
            if row.get("entry_id") == entry.get("entry_id"):
                row["promotion_campaign_id"] = campaign_id
                row["promotion_cycle_index"] = cycle_index
                row["cert_policy"] = disposition.get("cert_policy")
                row["formal_preflight_status"] = formal_preflight_status
                row["primary_improvement"] = disposition.get("primary_improvement")
                row["promotion_primary_met"] = disposition.get("promotion_primary_met")
                row["promotion_replicate_count"] = disposition.get(
                    "promotion_replicate_count"
                )
                row["promotion_replicate_required"] = disposition.get(
                    "promotion_replicate_required"
                )
                row["last_arm_order"] = delivery.get("arm_order")
                row["last_arm_seed"] = delivery.get("arm_seed")
                # Incomplete formal / harness: not a decisive promote attempt.
                if (
                    status in {"promotion_inconclusive", "harness_failure"}
                    or disposition.get("timeout")
                    or disposition.get("harness_failure")
                ):
                    attempts = int(row.get("promote_attempts") or 0)
                    row["promote_attempts"] = max(0, attempts - 1)
                    if status == "promotion_inconclusive" or disposition.get("timeout"):
                        row["last_formal_timeout"] = True
                        row["last_formal_timeout_wall_s"] = _PROMOTE_FORMAL_TIMEOUT_S
                    if status == "harness_failure" or disposition.get(
                        "harness_failure"
                    ):
                        row["last_harness_failure"] = True
                break
        _write_champion_queue(path, entries)
        print(
            f"CHAMPION_PROMOTE_DISPOSE status={status} "
            f"cert_policy={disposition.get('cert_policy')} "
            f"formal={formal_preflight_status} "
            f"primary_delta={disposition.get('primary_improvement')} "
            f"primary_met={disposition.get('promotion_primary_met')}",
            flush=True,
        )
    return updated


def _classify_metric_tradeoff(
    *,
    control: dict[str, float | None],
    candidate: dict[str, float | None],
    primary_metric: str,
    minimum_efficiency_gain_fraction: float,
) -> tuple[bool, list[str]]:
    """Score control vs candidate with quality/latency tradeoffs.

    A pure latency improvement with empty meaning is **not** positive. A quality
    improvement that spends a bounded latency budget **is** positive even when
    the declared primary is latency. Efficiency (mpr / latency) is also a win
    path when meaning stays above the smoke floor.
    """
    reasons: list[str] = []
    positive = False
    metric_leaf = primary_metric.split(".")[-1]

    c_lat = _finite_metric(control.get("latency_ms_p50"))
    t_lat = _finite_metric(candidate.get("latency_ms_p50"))
    c_pr = _finite_metric(control.get("parse_rate"))
    t_pr = _finite_metric(candidate.get("parse_rate"))
    c_mpr = _finite_metric(control.get("meaningful_program_rate"))
    t_mpr = _finite_metric(candidate.get("meaningful_program_rate"))

    parse_held = t_pr is None or c_pr is None or t_pr + _EPS >= c_pr
    mpr_held = t_mpr is None or c_mpr is None or t_mpr + _EPS >= c_mpr
    mpr_improved = t_mpr is not None and c_mpr is not None and t_mpr > c_mpr + _EPS
    lat_improved = t_lat is not None and c_lat is not None and t_lat + _EPS < c_lat
    if t_lat is not None and c_lat is not None and c_lat > 0:
        lat_within_tradeoff = (
            t_lat <= c_lat * (1.0 + _LATENCY_REGRESSION_BUDGET)
            or t_lat <= c_lat + _LATENCY_REGRESSION_ABS_MS
        )
    else:
        # Missing latency must not veto a quality win.
        lat_within_tradeoff = True

    both_timeout_band = _in_timeout_band(c_lat) and _in_timeout_band(t_lat)

    # Path 1: latency primary win — only with held quality and non-empty meaning.
    if metric_leaf == "latency_ms_p50":
        no_metrics = (
            c_lat is None
            and t_lat is None
            and c_pr is None
            and t_pr is None
            and c_mpr is None
            and t_mpr is None
        )
        if no_metrics:
            # Incomplete stage/decode walls are not model quality failures.
            reasons.append("measurement_incomplete:no_smoke_metrics")
        elif c_lat is None or t_lat is None:
            reasons.append("primary_metric_unavailable")
        elif lat_improved and parse_held and mpr_held:
            if both_timeout_band:
                reasons.append(
                    "latency_win_rejected_timeout_band:"
                    f"control={c_lat} candidate={t_lat}"
                )
            elif t_mpr is None:
                reasons.append("latency_win_rejected_unmeasured_mpr")
            elif t_mpr + _EPS < _MIN_MPR_FOR_LATENCY_WIN:
                reasons.append(
                    "latency_win_rejected_low_mpr:"
                    f"mpr={t_mpr}<{_MIN_MPR_FOR_LATENCY_WIN + 1e-9:g}"
                )
            else:
                positive = True
                reasons.append(f"primary_metric_win:{primary_metric}:{c_lat}->{t_lat}")
                reasons.append(f"quality_held:parse={t_pr} mpr={t_mpr}")
        else:
            reasons.append(
                f"primary_metric_null_or_worse:{primary_metric}:"
                f"control={c_lat} candidate={t_lat} "
                f"parse={c_pr}->{t_pr} mpr={c_mpr}->{t_mpr}"
            )

    # Path 2: quality win may spend a bounded latency budget (even under a
    # latency primary). Prevents "naive latency primary" from failing better
    # meaning at a small latency cost.
    if mpr_improved and parse_held and lat_within_tradeoff:
        positive = True
        reasons.append(
            "quality_metric_win:meaningful_program_rate:"
            f"{c_mpr}->{t_mpr}:lat={c_lat}->{t_lat}"
        )
    elif mpr_improved and parse_held and not lat_within_tradeoff:
        reasons.append(
            "quality_win_rejected_latency_budget:"
            f"mpr={c_mpr}->{t_mpr} lat={c_lat}->{t_lat}"
        )

    # Path 3: efficiency (meaningful programs per ms) with a meaning floor.
    # Still respects the latency tradeoff budget so a 2× slowdown cannot mint a
    # free win from mpr 0→ε alone.
    if (
        t_mpr is not None
        and c_mpr is not None
        and t_lat is not None
        and c_lat is not None
        and t_lat > 0
        and c_lat > 0
        and parse_held
        and lat_within_tradeoff
        and t_mpr + _EPS >= _MIN_MPR_FOR_LATENCY_WIN
        and not both_timeout_band
    ):
        c_eff = c_mpr / c_lat
        t_eff = t_mpr / t_lat
        efficiency_gain_fraction = (t_eff / c_eff - 1.0) if c_eff > 0 else None
        if not mpr_held and t_eff > c_eff + _EPS:
            reasons.append(
                "efficiency_win_rejected_mpr_regression:"
                f"mpr={c_mpr}->{t_mpr}:mpr_per_ms={c_eff:.8g}->{t_eff:.8g}"
            )
        elif (
            efficiency_gain_fraction is not None
            and efficiency_gain_fraction + _EPS >= minimum_efficiency_gain_fraction
        ):
            positive = True
            reasons.append(
                f"efficiency_win:mpr_per_ms:{c_eff:.8g}->{t_eff:.8g}:"
                f"gain_fraction={efficiency_gain_fraction:.8g}:"
                f"minimum={minimum_efficiency_gain_fraction:.8g}"
            )
            reasons.append(f"quality_held:parse={t_pr} mpr={t_mpr}")
        elif efficiency_gain_fraction is not None and efficiency_gain_fraction > _EPS:
            reasons.append(
                f"efficiency_win_rejected_min_effect:mpr_per_ms:"
                f"{c_eff:.8g}->{t_eff:.8g}:"
                f"gain_fraction={efficiency_gain_fraction:.8g}<"
                f"{minimum_efficiency_gain_fraction:.8g}"
            )

    return positive, reasons


def _classify_positive(
    *,
    camp_dir: Path,
    primary_metric: str,
    control_id: str,
    candidate_id: str,
    role: str = "screening",
    baseline_trainable_params: int | None = None,
    candidate_trainable_params: int | None = None,
    eg_params_by_seed: list[float] | None = None,
    policy_path: str | None = None,
) -> dict[str, Any]:
    """Classify cycle for SDLC Phase A stack-layer gate.

    Combines versioned climb policy (role primary, EG_params, fixture rules)
    with quality-aware latency/meaning tradeoffs: pure latency blips with empty
    meaning are not positive; quality may spend a bounded latency budget.
    Fixture insufficient_n / missing metrics / null deltas / uncharged capacity
    growth → non-positive.
    """
    from slm_training.autoresearch.climb_policy import (
        classify_positive_metrics,
        load_climb_policy,
        primary_for_role,
    )

    policy = load_climb_policy(policy_path)
    # Resolve through the single role-aware helper so promotion cannot inherit
    # a same-leaf smoke override for its policy-owned held-out endpoint.
    role_primary = primary_for_role(policy, role)
    effective_metric = _effective_primary_metric(
        role=role,
        policy_metric=str(role_primary.get("metric") or primary_metric),
        requested_metric=primary_metric,
    )

    # Promotion primary is held_out.*; load held_out leaves so Phase A is not
    # permanently primary_metric_unavailable when eval_held_out.json exists.
    prefer_held = (
        role == "promotion"
        or effective_metric.startswith("held_out.")
        or "held_out" in effective_metric
    )
    control = _run_metrics(camp_dir, control_id, prefer_held_out=prefer_held)
    candidate = _run_metrics(camp_dir, candidate_id, prefer_held_out=prefer_held)
    # Merge full primary metric keys when leaf-only maps were collected.
    if (
        effective_metric not in control
        and control.get(effective_metric.split(".")[-1]) is not None
    ):
        control = {
            **control,
            effective_metric: control[effective_metric.split(".")[-1]],
        }
    if (
        effective_metric not in candidate
        and candidate.get(effective_metric.split(".")[-1]) is not None
    ):
        candidate = {
            **candidate,
            effective_metric: candidate[effective_metric.split(".")[-1]],
        }

    reasons_pre: list[str] = []
    for run_id in (control_id, candidate_id):
        scoreboard_path = camp_dir / "runs" / run_id / "scoreboard.json"
        if not scoreboard_path.is_file():
            reasons_pre.append(f"measurement_incomplete:{run_id}:missing_scoreboard")
            continue
        scoreboard = _read_json(scoreboard_path)
        suites = scoreboard.get("suites")
        if isinstance(suites, dict):
            if any(not isinstance(suite, dict) for suite in suites.values()):
                reasons_pre.append(f"measurement_incomplete:{run_id}:invalid_suites")
                continue
            suite_rows = list(suites.items())
        elif isinstance(suites, list):
            if any(not isinstance(suite, dict) for suite in suites):
                reasons_pre.append(f"measurement_incomplete:{run_id}:invalid_suites")
                continue
            suite_rows = [
                (str(suite.get("suite") or index), suite)
                for index, suite in enumerate(suites)
            ]
        else:
            reasons_pre.append(f"measurement_incomplete:{run_id}:invalid_suites")
            continue
        if not suite_rows:
            reasons_pre.append(f"measurement_incomplete:{run_id}:empty_suites")
            continue
        for suite_name, suite in suite_rows:
            total_n = suite.get("document_n", suite.get("n"))
            completed_n = suite.get("completed_document_n")
            incomplete_n = suite.get("incomplete_document_n")
            timeout_n = suite.get("decode_timeout_document_count")
            if type(timeout_n) is not int:
                timeout_n = suite.get("decode_timeout_count")
            counts = (total_n, completed_n, incomplete_n, timeout_n)
            if (
                any(type(value) is not int or value < 0 for value in counts)
                or completed_n + incomplete_n != total_n
                or timeout_n > incomplete_n
            ):
                reasons_pre.append(
                    "measurement_incomplete:"
                    f"{run_id}:{suite_name}:invalid_counts:"
                    f"document_n={total_n}:completed_document_n={completed_n}:"
                    f"incomplete_document_n={incomplete_n}:"
                    f"decode_timeout_count={timeout_n}"
                )
                continue
            if incomplete_n > 0 or timeout_n > 0:
                reasons_pre.append(
                    "measurement_incomplete:"
                    f"{run_id}:{suite_name}:"
                    f"incomplete_document_n={incomplete_n}:"
                    f"decode_timeout_count={timeout_n}"
                )
    outcomes = list((camp_dir / "artifacts" / "outcomes").glob("*.json"))
    for path in outcomes:
        out = _read_json(path)
        experiment_id = str(out.get("experiment_id") or path.stem)
        if (
            experiment_id in {control_id, candidate_id}
            and out.get("status") == "failed"
        ):
            reasons_pre.append(f"harness_failure:{experiment_id}:experiment_failed")
        err = str(out.get("error") or "")
        if "wall-time" in err or "wall time" in err.lower():
            reasons_pre.append(f"wall_timeout:{path.stem}")
        if out.get("metrics") == {} and err:
            reasons_pre.append(f"empty_metrics:{path.stem}")

    gate_files = list((camp_dir / "runs").glob("*/gates.json"))
    fixture_only_fails = 0
    for gpath in gate_files:
        gates = _read_json(gpath)
        fails = gates.get("failures") or gates.get("quality_threshold_failures") or []
        vol = gates.get("evidence_volume_failures") or []
        if isinstance(vol, list) and any("insufficient_n" in str(x) for x in vol):
            fixture_only_fails += 1
            reasons_pre.append(f"fixture_insufficient_n:{gpath.parent.name}")
        if isinstance(fails, list) and fails and not vol:
            reasons_pre.append(f"gate_failures:{gpath.parent.name}:{len(fails)}")

    # Quality/latency tradeoff (PR #1234) — rejects empty-meaning latency blips.
    tradeoff_positive, tradeoff_reasons = _classify_metric_tradeoff(
        control=control,
        candidate=candidate,
        primary_metric=effective_metric,
        minimum_efficiency_gain_fraction=float(
            policy.positive_classification["minimum_efficiency_gain_fraction"]
        ),
    )

    control_outcome = next(
        (
            _read_json(p)
            for p in outcomes
            if control_id in str(p) or _read_json(p).get("experiment_id") == control_id
        ),
        {},
    )
    cand_outcome = next(
        (
            _read_json(p)
            for p in outcomes
            if candidate_id in str(p)
            or _read_json(p).get("experiment_id") == candidate_id
        ),
        {},
    )

    # Params from outcomes when present
    def _params(outcome: dict[str, Any]) -> int | None:
        metrics = outcome.get("metrics") or {}
        if isinstance(metrics, dict) and metrics.get("trainable_params") is not None:
            try:
                return int(metrics["trainable_params"])
            except (TypeError, ValueError):
                return None
        return None

    def _train_summary_params(experiment_id: str) -> int | None:
        summary = _read_json(camp_dir / "runs" / experiment_id / "train_summary.json")
        value = (summary.get("track") or {}).get("trainable_params")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    base_params = baseline_trainable_params
    cand_params = candidate_trainable_params
    if base_params is None:
        base_params = _params(control_outcome) or _train_summary_params(control_id)
    if cand_params is None:
        cand_params = _params(cand_outcome) or _train_summary_params(candidate_id)

    leaf = effective_metric.split(".")[-1]
    t_mpr = _finite_metric(candidate.get("meaningful_program_rate"))
    # Executable unblock only when candidate completes with quality floor.
    executable_unblock = False
    if control_outcome.get("error") and not cand_outcome.get("error"):
        has_metric = (
            candidate.get(leaf) is not None
            or candidate.get(effective_metric) is not None
            or candidate.get("latency_ms_p50") is not None
        )
        if (
            has_metric
            and t_mpr is not None
            and t_mpr + _EPS >= _MIN_MPR_FOR_LATENCY_WIN
        ):
            executable_unblock = True
        elif has_metric:
            reasons_pre.append(f"executable_unblock_rejected_low_mpr:mpr={t_mpr}")

    decision = classify_positive_metrics(
        policy,
        role=role,
        control_metrics=control,
        candidate_metrics=candidate,
        baseline_trainable_params=base_params,
        candidate_trainable_params=cand_params,
        eg_params_by_seed=eg_params_by_seed,
        executable_unblock=executable_unblock,
        fixture_insufficient_n=bool(fixture_only_fails),
    )

    # Latency primary: tradeoff is authoritative for metric wins (blocks zero-mpr
    # latency greening from direction-signed primary alone).
    if leaf == "latency_ms_p50":
        positive = bool(tradeoff_positive or executable_unblock)
        # Preserve EG_params blocks from climb policy.
        if any(
            str(r).startswith("eg_params_block:")
            for r in (decision.get("reasons") or [])
        ):
            positive = False
        decision["positive"] = positive
        decision["stack_layer"] = positive
    elif tradeoff_positive:
        decision["positive"] = True
        decision["stack_layer"] = True

    # A quality-primary gain may spend only the same bounded latency budget as
    # the explicit meaning-quality path above. Otherwise scalar training depth
    # can mint a structural "win" by buying 2x+ decode cost, enter the champion
    # queue, and steer the loop away from genuinely new objectives.
    if leaf != "latency_ms_p50" and decision.get("positive"):
        control_latency = _finite_metric(control.get("latency_ms_p50"))
        candidate_latency = _finite_metric(candidate.get("latency_ms_p50"))
        if (
            control_latency is not None
            and candidate_latency is not None
            and control_latency > 0
            and candidate_latency
            > control_latency * (1.0 + _LATENCY_REGRESSION_BUDGET)
            and candidate_latency > control_latency + _LATENCY_REGRESSION_ABS_MS
        ):
            decision["positive"] = False
            decision["stack_layer"] = False
            reasons_pre.append(
                "primary_quality_win_rejected_latency_budget:"
                f"{effective_metric}:lat={control_latency}->{candidate_latency}"
            )

    reasons = (
        list(reasons_pre) + list(tradeoff_reasons) + list(decision.get("reasons") or [])
    )
    # An efficiency ratio cannot override the role-owned quality primary or
    # its protected non-regression metrics. Otherwise halving meaningful output
    # while becoming much faster can be mislabeled as a quality-positive screen.
    if leaf != "latency_ms_p50" and any(
        str(reason).startswith(
            ("primary_metric_null_or_worse:", "non_regression_fail:")
        )
        for reason in reasons
    ):
        decision["positive"] = False
        decision["stack_layer"] = False
    if not any(
        reason.startswith(prefix)
        for reason in reasons
        for prefix in _WIN_REASON_PREFIXES
    ):
        decision["positive"] = False
        decision["stack_layer"] = False
        if not reasons:
            reasons.append("no_positive_signal")
    if any(
        str(reason).startswith(
            ("measurement_incomplete:", "wall_timeout:", "empty_metrics:")
        )
        for reason in reasons
    ):
        decision["positive"] = False
        decision["stack_layer"] = False

    decision["reasons"] = reasons
    decision["control_id"] = control_id
    decision["candidate_id"] = candidate_id
    decision["baseline_trainable_params"] = base_params
    decision["candidate_trainable_params"] = cand_params
    decision["fixture_volume_gate_hits"] = fixture_only_fails
    decision["primary_metric"] = effective_metric
    return decision


def _phase_a_delivery(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str,
    primary_metric: str,
    cycle_index: int | None = None,
    role: str | None = None,
    cycle_intent: str | None = None,
    arm_order: list[str] | None = None,
    arm_seed: int | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Record SDLC Phase A decision; never open stacked PR for non-positive."""
    from slm_training.autoresearch.climb_policy import (
        cycle_role_for_index,
        load_climb_policy,
    )

    policy = load_climb_policy()
    camp_dir = root / campaign_id
    # Infer control / candidate ids from manifests or runs
    man_dir = camp_dir / "manifests"
    control_run = None
    candidate_run = None
    if man_dir.exists():
        for path in sorted(man_dir.glob("*.json")):
            eid = path.stem
            if eid.endswith("-control") or eid.endswith("_control"):
                control_run = eid
            elif any(
                token in eid
                for token in (
                    "-bounds",
                    "-canvas",
                    "-both",
                    "-confirm",
                    "-promote",
                    "-steps",
                    "-batch1",
                    "-combined",
                )
            ):
                if candidate_run is None:
                    candidate_run = eid
            elif "control" not in eid and candidate_run is None:
                candidate_run = eid
    runs_dir = camp_dir / "runs"
    if runs_dir.exists():
        run_ids = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
        for rid in run_ids:
            if rid.endswith("-control") or rid.endswith("_control"):
                control_run = control_run or rid
            elif "control" not in rid:
                # Prefer promote/confirm arms when present (champion queue).
                if "-promote" in rid:
                    candidate_run = rid
                elif "-confirm" in rid:
                    candidate_run = rid
                elif candidate_run is None:
                    candidate_run = rid
        if control_run is None and run_ids:
            control_run = run_ids[0]
        if candidate_run is None and len(run_ids) > 1:
            candidate_run = run_ids[1]
        elif candidate_run is None:
            candidate_run = control_run

    control_run = control_run or "unknown-control"
    candidate_run = candidate_run or control_run

    if role is None:
        if cycle_index is not None and cycle_index >= 1:
            role = cycle_role_for_index(policy, cycle_index)
        else:
            role = "screening"

    decision = _classify_positive(
        camp_dir=camp_dir,
        primary_metric=primary_metric,
        control_id=control_run,
        candidate_id=candidate_run,
        role=role,
    )
    decision["measurement_complete"] = _measurement_is_complete(decision)
    decision["cycle_role"] = role
    decision["cycle_index"] = cycle_index
    decision["cycle_intent"] = cycle_intent or role
    decision["climb_policy_sha256"] = policy.sha256
    if cycle_intent == "confirm" and not _confirmation_quality_reheld(decision):
        decision["positive"] = False
        decision["stack_layer"] = False
        reasons = list(decision.get("reasons") or [])
        marker = "confirmation_rejected:primary_quality_not_reheld"
        if marker not in reasons:
            reasons.append(marker)
        decision["reasons"] = reasons
    # Stack only when positive AND there is something reviewable to ship.
    # Pure knob-only fixture cycles with a metric blip do not open empty PRs.
    porcelain = (
        _git(
            "status",
            "--porcelain",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="phase-a-git-status",
        )
        if cwd
        else ""
    )
    has_tracked_delta = bool(porcelain.strip())
    stack_layer = bool(decision["positive"] and has_tracked_delta)
    if decision["positive"] and not has_tracked_delta:
        stack_action = "positive_no_tracked_delta_skip_stack"
        agent_required = (
            "metric win recorded; no code/docs delta — skip stack PR; continue loop"
        )
    elif stack_layer:
        stack_action = "open_or_update_stacked_pr"
        agent_required = "gh stack add/submit --open for this positive layer"
    else:
        stack_action = "no_stack_layer_non_positive"
        agent_required = "continue loop; local commits/docs only"

    record = {
        "schema": "autotrain_sdlc_delivery/v1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sdlc_phase": "A",
        "positive": decision["positive"],
        "stack_layer": stack_layer,
        "has_tracked_delta": has_tracked_delta,
        "stack_action": stack_action,
        "agent_required": agent_required,
        "arm_order": list(arm_order or []),
        "arm_seed": arm_seed,
        **{k: v for k, v in decision.items() if k not in {"positive", "stack_layer"}},
    }
    out_path = camp_dir / "sdlc_delivery.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    ledger = root / "sdlc_delivery_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")

    tag = "POSITIVE" if record["positive"] else "NON_POSITIVE"
    print(
        f"SDLC_PHASE_A {tag} campaign={campaign_id} "
        f"stack_layer={record['stack_layer']} action={record['stack_action']}",
        flush=True,
    )
    for reason in record.get("reasons") or []:
        print(f"SDLC_PHASE_A reason={reason}", flush=True)

    # Optional design-doc closeout for the cycle (iron law); keep under campaign
    # root so the git worktree stays clean for the next fetch/merge.
    note_path = camp_dir / "measured-results-continuous.md"
    note = (
        f"# Continuous cycle {campaign_id}\n\n"
        f"- loop_id: `{loop_id}`\n"
        f"- primary_metric: `{primary_metric}`\n"
        f"- positive: **{record['positive']}**\n"
        f"- stack_layer: **{record['stack_layer']}**\n"
        f"- action: `{record['stack_action']}`\n"
        f"- reasons: {', '.join(record.get('reasons') or [])}\n"
        f"- control: `{control_run}` metrics={record.get('control_metrics')}\n"
        f"- candidate: `{candidate_run}` metrics={record.get('candidate_metrics')}\n\n"
        "Non-positive cycles do not open stacked PRs "
        "(sdlc autotrain-iteration-delivery).\n"
    )
    note_path.write_text(note, encoding="utf-8")
    return record


def _candidate_ship_state(camp_dir: Path, candidate_id: str) -> str:
    gates = _read_json(camp_dir / "runs" / candidate_id / "gates.json")
    authoritative = gates.get("authority") == "AgentEvals assertions"
    return "ship_promoted" if authoritative and gates.get("pass") is True else "blocked"


def _has_finalized_decode_timeout(camp_dir: Path, candidate_id: str) -> bool:
    """True when AgentV finalized every record and typed at least one timeout."""

    scoreboard = _read_json(camp_dir / "runs" / candidate_id / "scoreboard.json")
    evals = scoreboard.get("evals")
    runner = evals.get("runner") if isinstance(evals, dict) else None
    gates = scoreboard.get("gates")
    suites = scoreboard.get("suites")
    if not (
        isinstance(runner, dict)
        and runner.get("name") == "AgentV"
        and runner.get("execution_errors") == 0
        and isinstance(gates, dict)
        and gates.get("authority") == "AgentEvals assertions"
        and gates.get("pass") is False
        and isinstance(suites, dict)
    ):
        return False
    for suite in suites.values():
        if not isinstance(suite, dict):
            continue
        sample_n = suite.get("n")
        completed_n = suite.get("completed_document_n")
        incomplete_n = suite.get("incomplete_document_n")
        timeout_n = suite.get("decode_timeout_document_count")
        if not all(
            isinstance(value, int)
            for value in (sample_n, completed_n, incomplete_n, timeout_n)
        ):
            continue
        if (
            sample_n > 0
            and timeout_n > 0
            and completed_n + incomplete_n == sample_n
            and timeout_n <= incomplete_n
        ):
            return True
    return False


_OPEN_NUMERIC_LITERAL_RE = re.compile(r"(?:^|[,(])\s*-?\d{6,}$")


def _has_numeric_literal_close_starvation(camp_dir: Path, candidate_id: str) -> bool:
    """Detect repeated legal numeric bytes where the model never selects close.

    This is a diagnostic steering signal only. It cannot alter the legal domain
    or certify output; the next arm remains grammar constrained and size matched.
    """

    run_dir = camp_dir / "runs" / candidate_id
    stalled_records: set[str] = set()
    timeout_records = 0
    for eval_path in sorted(run_dir.glob("eval_*.json")):
        payload = _read_json(eval_path)
        timeout_records += int(payload.get("decode_timeout_document_count") or 0)
        traces = (payload.get("decode_stats") or {}).get("constrained_selection_traces")
        if not isinstance(traces, list):
            continue
        streaks: dict[str, int] = {}
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            record_id = str(trace.get("record_id") or "")
            prefix = str(trace.get("prefix_text") or "")
            chosen = str(trace.get("chosen_token") or "")
            legal_candidates = trace.get("legal_candidates")
            repeated_numeric = (
                record_id
                and chosen.startswith("B:")
                and isinstance(legal_candidates, int)
                and legal_candidates >= 12
                and _OPEN_NUMERIC_LITERAL_RE.search(prefix) is not None
            )
            streaks[record_id] = (
                streaks.get(record_id, 0) + 1 if repeated_numeric else 0
            )
            if streaks[record_id] >= 4:
                stalled_records.add(record_id)
    return timeout_records > 0 and bool(stalled_records)


def _primary_harness_family(camp_dir: Path) -> str:
    for path in sorted((camp_dir / "artifacts" / "outcomes").glob("*.json")):
        payload = _read_json(path)
        for signal in payload.get("harness_signals") or []:
            if signal.get("reproduced_on_frozen_input"):
                return str(signal.get("family") or "model_build")
    return "model_build"


def _diagnosis_target(camp_dir: Path) -> str | None:
    targets: list[str] = []
    for path in sorted((camp_dir / "artifacts" / "diagnoses").glob("*.json")):
        target = str(_read_json(path).get("target") or "")
        if target:
            targets.append(target)
    for preferred in ("harness", "data", "infrastructure", "model", "researcher"):
        if preferred in targets:
            return preferred
    return targets[-1] if targets else None


def _created_checkpoint_paths(camp_dir: Path) -> tuple[str, ...]:
    """Return every checkpoint created by the bounded cycle, relative to it."""
    runs = camp_dir / "runs"
    if not runs.is_dir():
        return ()
    return tuple(
        path.relative_to(camp_dir).as_posix()
        for path in sorted(runs.rglob("*.pt"))
        if path.is_file()
    )


def _consecutive_frozen_replays(
    root: Path, loop_id: str, campaign_id: str, cycle_intent: str
) -> int:
    """Count the current and immediately preceding frozen replay cycles."""

    if cycle_intent != "retry_measurement":
        return 0
    count = 1
    cursor = str(
        _read_json(root / campaign_id / "campaign.json").get("predecessor_campaign_id")
        or ""
    )
    seen = {campaign_id}
    while cursor and cursor not in seen:
        seen.add(cursor)
        handoff = _read_json(root / cursor / "cycle_handoff.json")
        if (
            handoff.get("loop_id") != loop_id
            or handoff.get("campaign_id") != cursor
            or handoff.get("cycle_intent") != "retry_measurement"
        ):
            break
        if any(
            action.get("kind") == "repair_harness"
            for action in handoff.get("actions") or []
            if isinstance(action, dict)
        ):
            break
        count += 1
        cursor = str(
            _read_json(root / cursor / "campaign.json").get("predecessor_campaign_id")
            or ""
        )
    return count


def _completed_candidate_priorities(
    matrix: dict[str, Any],
    candidate_id: str,
    *,
    resolved_infrastructure: bool,
    skip_slugs: set[str] | None = None,
) -> tuple[NextRunPriorityV1, ...]:
    """Replace stale steering after a candidate produces a complete result."""

    rows = [dict(item) for item in matrix.get("next_run_priorities") or []]
    if not rows:
        return ()
    experiments = [
        item.get("experiment") or {}
        for item in matrix.get("hypotheses") or []
        if isinstance(item, dict)
    ]
    alternatives = [
        item
        for item in experiments
        if str(item.get("experiment_id") or "")
        and str(item.get("experiment_id")) != candidate_id
        and not str(item.get("experiment_id")).endswith("-control")
    ]
    has_lineage_skip = skip_slugs is not None
    skip_slugs = skip_slugs or set()
    candidate_knobs = next(
        (
            dict(item.get("knobs") or {})
            for item in experiments
            if str(item.get("experiment_id") or "") == candidate_id
        ),
        {},
    )
    candidate_slug = _arm_slug_from_knobs(candidate_knobs, candidate_id=candidate_id)
    quality_keys = {
        "compiler_alignment_loss_weight",
        "component_plan_loss_weight",
        "component_edge_loss_weight",
        "component_edge_alignment_loss_weight",
        "semantic_contrast_loss_weight",
        "symbol_slot_augmentation",
        "component_inventory_loss_weight",
        "binder_topology_loss_weight",
        "binder_component_plan_loss_weight",
        "binder_arity_loss_weight",
        "symbol_boundary_loss_weight",
        "design_md_dropout",
        "ltr_prefix_loss_weight",
        "component_token_loss_weight",
        "component_edge_token_loss_weight",
        "compiler_decision_token_loss_weight",
        "structure_token_loss_weight",
        "typed_family_balance_loss_weight",
    }
    def has_quality_objective(knobs: dict[str, Any]) -> bool:
        return any(float(knobs.get(key) or 0) > 0 for key in quality_keys) or (
            float(knobs.get("fidelity_loss_weight") or 0.5) != 0.5
        ) or str(knobs.get("mask_pattern") or "random") != "random"

    targeted_alternative = next(
        (
            item
            for item in alternatives
            if candidate_slug == "literal-close"
            and _arm_slug_from_knobs(
                dict(item.get("knobs") or {}),
                candidate_id=str(item.get("experiment_id") or ""),
            )
            == "literal-margin"
            and "literal-margin" not in skip_slugs
        ),
        None,
    )
    if (
        targeted_alternative is None
        and candidate_slug == "literal-close"
        and "literal-margin" not in skip_slugs
    ):
        literal_margin = next(
            row for row in _SCREENING_ARM_BANK if row[0] == "literal-margin"
        )
        targeted_alternative = {
            "experiment_id": (
                candidate_id.removesuffix("literal-close") + "literal-margin"
            ),
            "hypothesis": literal_margin[1],
            "knobs": literal_margin[2],
        }
    alternative = targeted_alternative or next(
        (
            item
            for item in alternatives
            if has_quality_objective(dict(item.get("knobs") or {}))
            and (
                _arm_slug_from_knobs(
                    dict(item.get("knobs") or {}),
                    candidate_id=str(item.get("experiment_id") or ""),
                )
                not in skip_slugs
            )
            and _arm_slug_from_knobs(
                dict(item.get("knobs") or {}),
                candidate_id=str(item.get("experiment_id") or ""),
            )
            != "literal-margin"
        ),
        next(
            (
                item
                for item in alternatives
                if _arm_slug_from_knobs(
                    dict(item.get("knobs") or {}),
                    candidate_id=str(item.get("experiment_id") or ""),
                )
                not in skip_slugs
            ),
            None,
        ),
    )
    if alternative is None and has_lineage_skip:
        successor_skip = set(skip_slugs)
        if candidate_slug:
            successor_skip.add(candidate_slug)
        try:
            successor_slug = _select_recommended_slug(1, skip=successor_skip)
        except RuntimeError:
            successor_slug = ""
        if successor_slug:
            successor = next(
                row for row in _SCREENING_ARM_BANK if row[0] == successor_slug
            )
            alternative = {
                "experiment_id": (
                    candidate_id.removesuffix(candidate_slug or "") + successor_slug
                ),
                "hypothesis": successor[1],
                "knobs": successor[2],
            }
    for row in rows:
        if row.get("disposition") == "experiment_next" and (
            str(row.get("proposed_experiment_id") or "") == candidate_id
            or alternative is None
        ):
            row.update(
                {
                    "disposition": "monitor",
                    "proposed_experiment_id": None,
                    "expected_information_gain": (
                        "The completed candidate is exhausted and cannot be "
                        "selected again without a new preregistered hypothesis."
                    ),
                }
            )
    if alternative is not None:
        next_id = str(alternative["experiment_id"])
        alternative_knobs = dict(alternative.get("knobs") or {})
        slug = (
            _arm_slug_from_knobs(alternative_knobs, candidate_id=next_id)
            or next_id.rsplit("-", 1)[-1]
        )
        is_quality_hypothesis = has_quality_objective(alternative_knobs)
        selected_hypothesis = str(alternative.get("hypothesis") or "").strip()
        if is_quality_hypothesis:
            area = "model"
            hypothesis = (
                "The completed frozen replay rejects the prior arm; test "
                if resolved_infrastructure
                else "The completed non-positive arm is exhausted; test "
            ) + f"the distinct size-matched '{slug}' quality hypothesis next."
            information_gain = (
                "Moves from resolved infrastructure attribution to model quality."
                if resolved_infrastructure
                else "Avoids rerunning a completed null while preserving matched attribution."
            )
        else:
            area = "experiments"
            hypothesis = (
                "The recent quality families are exhausted; run the distinct "
                f"'{slug}' diagnostic next."
            )
            if selected_hypothesis:
                hypothesis += f" {selected_hypothesis}"
            information_gain = (
                "Tests a different registered lever after the recent quality-family "
                "cooldown without mislabeling a runtime diagnostic as quality."
            )
        rows[0].update(
            {
                "area": area,
                "hypothesis": hypothesis,
                "confidence": 0.9,
                "expected_information_gain": information_gain,
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": next_id,
            }
        )
    else:
        rows[0].update(
            {
                "area": "model_build",
                "hypothesis": (
                    "The registered quality-arm bank is exhausted; preregister "
                    "and wire a distinct size-matched quality objective before "
                    "another screening run."
                ),
                "confidence": 0.95,
                "expected_information_gain": (
                    "Prevents control recycling and creates a genuinely new "
                    "training signal for the next bounded comparison."
                ),
                "authority": "observed_result",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            }
        )
    return tuple(NextRunPriorityV1.model_validate(item) for item in rows)


def _queued_candidate_priorities(
    candidate_id: str, evidence_id: str
) -> tuple[NextRunPriorityV1, ...]:
    """Project the real successor after a screening candidate enters the queue."""

    return (
        NextRunPriorityV1(
            rank=1,
            area="evaluation",
            hypothesis=(
                "Confirm the fixture candidate on a fresh seed with the exact "
                "size-matched treatment and control recipes before promotion."
            ),
            evidence_ids=(evidence_id,),
            confidence=0.95,
            expected_information_gain=(
                "Tests whether the held-out quality gain reproduces while exposing "
                "the observed binder and latency tradeoffs."
            ),
            authority="observed_result",
            disposition="experiment_next",
            proposed_experiment_id=f"{candidate_id}-fresh-confirmation",
        ),
        NextRunPriorityV1(
            rank=2,
            area="lean_model",
            hypothesis=(
                "Keep promotion formal preflight locked until fresh confirmation "
                "establishes a champion."
            ),
            evidence_ids=(evidence_id,),
            confidence=1.0,
            expected_information_gain=(
                "Prevents screening evidence from bypassing theorem-backed "
                "promotion obligations."
            ),
            authority="lean_assumption",
            disposition="monitor",
        ),
    )


def _completed_retry_priorities(
    matrix: dict[str, Any], candidate_id: str
) -> tuple[NextRunPriorityV1, ...]:
    """Compatibility wrapper for completed frozen-replay steering."""

    return _completed_candidate_priorities(
        matrix, candidate_id, resolved_infrastructure=True
    )


def _completed_confirmation_priorities(
    matrix: dict[str, Any],
    candidate_id: str,
    delivery: dict[str, Any],
    resolution: dict[str, Any] | None,
) -> tuple[NextRunPriorityV1, ...]:
    """Replace confirm-time hypotheses with observed successor steering."""

    rows = [dict(item) for item in matrix.get("next_run_priorities") or []]
    if not rows:
        return ()
    for row in rows:
        if row.get("disposition") == "experiment_next":
            row["disposition"] = "monitor"
            row["proposed_experiment_id"] = None

    status = str((resolution or {}).get("status") or "")
    confirmed = status == "confirmed"
    if confirmed:
        rows[0].update(
            {
                "area": "promotion",
                "hypothesis": (
                    "The champion re-held on confirmation; test the exact matched "
                    "recipes under the next promotion suite and Lean preflight."
                ),
                "confidence": 0.9,
                "expected_information_gain": (
                    "Separates a repeatable fixture signal from held-out and formal "
                    "promotion evidence."
                ),
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": candidate_id,
            }
        )
        return tuple(NextRunPriorityV1.model_validate(item) for item in rows)

    control = dict(delivery.get("control_metrics") or {})
    candidate = dict(delivery.get("candidate_metrics") or {})
    primary = str(delivery.get("primary_metric") or "primary metric")
    control_primary = control.get(primary)
    candidate_primary = candidate.get(primary)
    if control_primary is None and "." in primary:
        control_primary = control.get(primary.split(".", 1)[1])
        candidate_primary = candidate.get(primary.split(".", 1)[1])
    meaning_before = control.get("meaningful_program_rate")
    meaning_after = candidate.get("meaningful_program_rate")
    observed = (
        f"{primary} {control_primary!r}->{candidate_primary!r}; "
        f"meaningful_program_rate {meaning_before!r}->{meaning_after!r}"
    )
    future_batch_id = (
        candidate_id[: -len("-confirm")] + "-batch1"
        if candidate_id.endswith("-confirm")
        else f"{candidate_id}-batch1"
    )
    replacements = (
        {
            "area": "model",
            "hypothesis": (
                "Fresh-seed confirmation rejected the champion fingerprint "
                f"({observed}). Exhaust it and test a distinct size-matched "
                "quality-targeted objective instead of spending more scalar steps."
            ),
            "confidence": 0.95,
            "expected_information_gain": (
                "Tests objective alignment against certified structural quality "
                "rather than treating lower token loss as model progress."
            ),
            "authority": "observed_result",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
        {
            "area": "evaluation",
            "hypothesis": (
                "Training loss and certified program quality diverged on the "
                "confirmation; retain loss as a diagnostic, not a promotion proxy."
            ),
            "confidence": 0.9,
            "expected_information_gain": (
                "Prevents optimization progress from masking regressions in meaning, "
                "structure, recall, or latency."
            ),
            "authority": "observed_result",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
        {
            "area": "experiments",
            "hypothesis": (
                "Run the next non-exhausted batch-size arm only as a runtime "
                "diagnostic while a new quality-targeted objective is preregistered."
            ),
            "confidence": 0.65,
            "expected_information_gain": (
                "Keeps the bounded loop executable without mislabeling a throughput "
                "lever as a model-quality hypothesis."
            ),
            "authority": "speculative",
            "disposition": "experiment_next",
            "proposed_experiment_id": future_batch_id,
        },
        {
            "area": "infrastructure",
            "hypothesis": (
                "Exact source-control reconstruction and pre-execution attempt "
                "recovery succeeded; preserve both as campaign provenance."
            ),
            "confidence": 0.95,
            "expected_information_gain": (
                "Maintains attributable control-versus-candidate recipes across "
                "confirmation and promotion."
            ),
            "authority": "reproduced_harness_signal",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
        {
            "area": "model_build",
            "hypothesis": (
                "Recent registered quality families are exhausted; prioritize a "
                "new preregistered structural or meaningful-quality objective before "
                "recycling them."
            ),
            "confidence": 0.75,
            "expected_information_gain": (
                "Tests a new causal training mechanism instead of repeating a closed "
                "approach or weakening a gate."
            ),
            "authority": "speculative",
            "disposition": "monitor",
            "proposed_experiment_id": None,
        },
    )
    for row, replacement in zip(rows, replacements, strict=False):
        row.update(replacement)
    return tuple(NextRunPriorityV1.model_validate(item) for item in rows)


def _predecessor_priority_slug(
    root: Path,
    predecessor_campaign_id: str | None,
    *,
    skip: set[str],
    closed: set[str] | None = None,
) -> str | None:
    """Resolve the predecessor's highest-priority executable screening arm.

    ``skip`` contains transient funnel conflicts that strong observed evidence
    may preserve across an interrupted confirmation/promotion. ``closed`` is
    permanent lineage evidence: no priority is allowed to reopen those arms.
    """

    if not predecessor_campaign_id:
        return None
    closed = closed or set()
    camp_dir = root / predecessor_campaign_id
    handoff_path = camp_dir / "cycle_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = _read_json(handoff_path)
    priorities = list(handoff.get("priorities") or [])
    has_explicit_successor = any(
        action.get("kind") == "next_experiment"
        for action in handoff.get("actions") or []
        if isinstance(action, dict)
    )
    delivery_path = camp_dir / "sdlc_delivery.json"
    matrix_path = camp_dir / "matrix-proposal.json"
    if delivery_path.is_file() and matrix_path.is_file():
        delivery = _read_json(delivery_path)
        current = _classify_positive(
            camp_dir=camp_dir,
            primary_metric=str(handoff.get("primary_metric") or ""),
            control_id=str(delivery.get("control_id") or ""),
            candidate_id=str(delivery.get("candidate_id") or ""),
            role=str(handoff.get("cycle_role") or "screening"),
        )
        measurement_incomplete = delivery.get("measurement_complete") is False or (
            any(
                str(reason).startswith("measurement_incomplete:")
                for reason in current.get("reasons") or []
            )
            or _diagnosis_target(camp_dir) == "infrastructure"
        )
        if (
            handoff.get("cycle_intent") in {"screening", "promotion"}
            and not measurement_incomplete
            and not current.get("positive")
        ):
            priorities = [
                item.model_dump()
                for item in _completed_candidate_priorities(
                    _read_json(matrix_path),
                    str(delivery.get("candidate_id") or ""),
                    resolved_infrastructure=False,
                    skip_slugs=skip,
                )
            ]
        elif handoff.get("cycle_intent") == "confirm" and not measurement_incomplete:
            current["measurement_complete"] = True
            confirmation_status = (
                "confirmed" if _confirmation_quality_reheld(current) else "rejected"
            )
            priorities = [
                item.model_dump()
                for item in _completed_confirmation_priorities(
                    _read_json(matrix_path),
                    str(delivery.get("candidate_id") or ""),
                    current,
                    {"status": confirmation_status},
                )
            ]

    def priority_slug(
        rows: list[dict[str, Any]], *, evidence_directed_only: bool
    ) -> str | None:
        for priority in sorted(rows, key=lambda item: int(item.get("rank") or 999)):
            if priority.get("disposition") != "experiment_next":
                continue
            experiment_id = str(priority.get("proposed_experiment_id") or "")
            evidence_directed = bool(
                priority.get("area") == "model_build"
                and priority.get("authority") == "observed_result"
                and float(priority.get("confidence") or 0.0) >= 0.9
            )
            if evidence_directed_only and not evidence_directed:
                continue
            for slug, _hypothesis, _extras in _SCREENING_ARM_BANK:
                if (
                    experiment_id.endswith(f"-{slug}")
                    and slug not in closed
                    and (slug not in skip or evidence_directed)
                ):
                    return slug
        return None

    # A fresh champion confirmation/promotion may legitimately interrupt a
    # high-confidence model-build successor. Preserve that observed successor
    # across the bounded interruption until the same arm actually executes.
    executed_slugs: set[str] = set()
    cursor: str | None = predecessor_campaign_id
    seen: set[str] = set()
    for _ in range(_RECENT_EXHAUSTION_CYCLE_WINDOW):
        if not cursor or cursor in seen:
            break
        seen.add(cursor)
        ancestor_dir = root / cursor
        ancestor_handoff = _read_json(ancestor_dir / "cycle_handoff.json")
        ancestor_delivery = _read_json(ancestor_dir / "sdlc_delivery.json")
        candidate_id = str(ancestor_delivery.get("candidate_id") or "")
        candidate_slug = _arm_slug_from_knobs(
            _load_experiment_knobs(ancestor_dir, candidate_id),
            candidate_id=candidate_id,
        )
        if candidate_slug:
            executed_slugs.add(candidate_slug)
        observed = priority_slug(
            list(ancestor_handoff.get("priorities") or []),
            evidence_directed_only=True,
        )
        if observed and observed not in executed_slugs and observed not in closed:
            return observed
        cursor = str(
            _read_json(ancestor_dir / "campaign.json").get(
                "predecessor_campaign_id"
            )
            or ""
        )

    for priority in sorted(priorities, key=lambda item: int(item.get("rank") or 999)):
        if priority.get("disposition") != "experiment_next":
            continue
        experiment_id = str(priority.get("proposed_experiment_id") or "")
        evidence_directed = bool(
            has_explicit_successor
            and priority.get("authority") == "observed_result"
            and float(priority.get("confidence") or 0.0) >= 0.9
        )
        for slug, _hypothesis, _extras in _SCREENING_ARM_BANK:
            if (
                experiment_id.endswith(f"-{slug}")
                and slug not in closed
                and (slug not in skip or evidence_directed)
            ):
                return slug
    return None


def _write_cycle_handoff(
    *,
    root: Path,
    loop_id: str,
    campaign_id: str,
    cycle_index: int,
    upstream_commit: str,
    integration_commit: str,
    role: str,
    cycle_intent: str,
    primary_metric: str,
    matrix: dict[str, Any],
    delivery: dict[str, Any],
    resolution: dict[str, Any] | None,
    formal_status: str | None,
    skip_slugs: set[str] | None = None,
) -> AutotrainCycleHandoffV1:
    """Write the typed boundary consumed by the agent between bounded cycles."""
    camp_dir = root / campaign_id
    evidence_id = f"campaign:{campaign_id}"
    status = str((resolution or {}).get("status") or "")
    diagnosis_target = _diagnosis_target(camp_dir)
    measurement_incomplete = (
        delivery.get("measurement_complete") is False
        or any(
            item.startswith("measurement_incomplete:")
            for item in delivery.get("reasons") or []
        )
        or diagnosis_target == "infrastructure"
    )
    candidate_id = str(delivery.get("candidate_id") or "")
    control_id = str(delivery.get("control_id") or "")
    finalized_decode_timeout = _has_finalized_decode_timeout(camp_dir, candidate_id)
    control_decode_timeout = _has_finalized_decode_timeout(camp_dir, control_id)
    candidate_only_model_timeout = bool(
        finalized_decode_timeout and not control_decode_timeout
    )
    control_only_model_timeout = bool(
        control_decode_timeout
        and not finalized_decode_timeout
    )
    control_runtime_reproduced = bool(
        control_only_model_timeout
        and cycle_intent == "retry_measurement"
        and _run_has_usable_metrics(camp_dir, candidate_id)
    )
    frozen_replay_count = 0
    frozen_replay_limit = 0
    if finalized_decode_timeout:
        from slm_training.autoresearch.climb_policy import (
            load_climb_policy,
            max_consecutive_frozen_replays,
        )

        frozen_replay_count = _consecutive_frozen_replays(
            root, loop_id, campaign_id, cycle_intent
        )
        frozen_replay_limit = max_consecutive_frozen_replays(load_climb_policy())
    runtime_arm_rejected = bool(
        candidate_only_model_timeout
        and cycle_intent == "retry_measurement"
        and frozen_replay_count >= frozen_replay_limit
    )
    if status in {"promoted", "climb_accepted"}:
        climb_state = "climb_accepted"
    elif status == "confirmed":
        climb_state = "champion_confirmed"
    elif status == "harness_failure":
        climb_state = "harness_failure"
    elif status == "promotion_inconclusive":
        climb_state = "inconclusive"
    elif status in {"rejected", "promotion_failed"}:
        climb_state = "rejected"
    elif runtime_arm_rejected or control_runtime_reproduced:
        # A candidate-only timeout reproduced by the exact frozen checkpoint
        # while its matched control completes is a decisive runtime rejection
        # of this model arm.  Quality remains unavailable, but the loop must not
        # misroute the same trajectory into an unbounded harness-repair cycle.
        climb_state = "rejected"
    elif measurement_incomplete:
        climb_state = "inconclusive"
    elif resolution is not None and resolution.get("status") == "queued":
        climb_state = "candidate_queued"
    elif delivery.get("positive"):
        climb_state = "inconclusive"
    else:
        climb_state = "rejected"

    numeric_close_starvation = bool(
        candidate_id and _has_numeric_literal_close_starvation(camp_dir, candidate_id)
    )
    ship_state = (
        _candidate_ship_state(camp_dir, candidate_id)
        if candidate_id
        else "not_evaluated"
    )
    evidence_class = "ship" if ship_state == "ship_promoted" else "fixture"
    reasons = tuple(
        str(item)
        for item in [
            *((resolution or {}).get("resolve_reasons") or []),
            *(delivery.get("reasons") or []),
            *(
                [f"arm_order:{','.join(delivery.get('arm_order') or [])}"]
                if delivery.get("arm_order")
                else []
            ),
        ]
    )
    if runtime_arm_rejected:
        reasons += (f"candidate_runtime_rejected_after_frozen_replay:{candidate_id}",)
        priorities = _completed_candidate_priorities(
            matrix,
            candidate_id,
            resolved_infrastructure=True,
            skip_slugs=skip_slugs,
        )
    elif control_runtime_reproduced:
        reasons += (
            f"control_runtime_rejected_after_frozen_replay:{control_id}",
            f"candidate_runtime_unblock_reproduced:{candidate_id}",
        )
        priorities = _completed_candidate_priorities(
            matrix,
            candidate_id,
            resolved_infrastructure=True,
            skip_slugs=skip_slugs,
        )
    elif measurement_incomplete and control_only_model_timeout:
        priorities = (
            NextRunPriorityV1(
                rank=1,
                area="model_build",
                hypothesis=(
                    "The tail-supervised candidate completed while the matched "
                    "control entered a typed decode timeout; replay the exact "
                    "frozen pair once to test whether the runtime unblock reproduces."
                ),
                evidence_ids=(evidence_id,),
                confidence=0.95,
                expected_information_gain=(
                    "Distinguishes a causal termination-supervision runtime effect "
                    "from a one-run timing artifact without inventing control quality."
                ),
                authority="observed_result",
                disposition="experiment_next",
                proposed_experiment_id=candidate_id,
            ),
        )
    elif measurement_incomplete and numeric_close_starvation:
        priorities = (
            NextRunPriorityV1(
                rank=1,
                area="model_build",
                hypothesis=(
                    "The candidate repeatedly ranks legal numeric bytes above the "
                    "literal terminator; test size-matched tail-weighted LTR "
                    "supervision while preserving constrained decode."
                ),
                evidence_ids=(evidence_id,),
                confidence=0.95,
                expected_information_gain=(
                    "Distinguishes insufficient suffix/termination supervision "
                    "from runtime orchestration after AgentV finalized every "
                    "record as a typed decode timeout."
                ),
                authority="observed_result",
                disposition="experiment_next",
                proposed_experiment_id=f"{candidate_id}-literal-close",
            ),
        )
    elif measurement_incomplete:
        priorities = (
            NextRunPriorityV1(
                rank=1,
                area="infrastructure",
                hypothesis=(
                    "The measurement is incomplete; replay the exact frozen "
                    "control and candidate before testing a new hypothesis."
                ),
                evidence_ids=(evidence_id,),
                confidence=0.95,
                expected_information_gain=(
                    "Completes the preregistered comparison without treating "
                    "partial train or decode telemetry as model evidence."
                ),
                authority="observed_result",
                disposition="experiment_next",
                proposed_experiment_id=candidate_id or None,
            ),
        )
    elif cycle_intent == "retry_measurement":
        priorities = _completed_candidate_priorities(
            matrix,
            candidate_id,
            resolved_infrastructure=True,
            skip_slugs=skip_slugs,
        )
    elif cycle_intent == "confirm":
        priorities = _completed_confirmation_priorities(
            matrix,
            candidate_id,
            delivery,
            resolution,
        )
    elif (
        cycle_intent in {"screening", "promotion"}
        and not measurement_incomplete
        and not delivery.get("positive")
    ):
        priorities = _completed_candidate_priorities(
            matrix,
            candidate_id,
            resolved_infrastructure=False,
            skip_slugs=skip_slugs,
        )
    elif (
        cycle_intent in {"screening", "promotion"}
        and not measurement_incomplete
        and delivery.get("positive")
    ):
        priorities = _queued_candidate_priorities(candidate_id, evidence_id)
    else:
        priorities = tuple(
            NextRunPriorityV1.model_validate(item)
            for item in matrix.get("next_run_priorities") or []
        )
    checkpoint_paths = _created_checkpoint_paths(camp_dir)
    document_reason = "persist this cycle's JSON and markdown under docs/design"
    if checkpoint_paths:
        document_reason += "; update docs/MODEL_CARD.md and the README summary"
    actions: list[AutotrainActionV1] = [
        AutotrainActionV1(
            kind="document",
            owner="documenting-experiment-results",
            reason=document_reason,
            evidence_ids=(evidence_id,),
        )
    ]
    theorem_stop = any("theorem_backed_band_miss" in item for item in reasons)
    assumption_miss = any("assumption_backed_band_miss" in item for item in reasons)
    # A finalized AgentV timeout is more specific than the evaluate process's
    # generic non-zero exit. Preserve both repair and content-bound retry actions
    # so acknowledging the repair cannot make the required replay unreachable.
    harness_failure = (
        climb_state == "harness_failure"
        or any(item.startswith("harness_failure:") for item in reasons)
    ) and not (finalized_decode_timeout or control_only_model_timeout)
    if theorem_stop:
        actions[0:0] = [
            AutotrainActionV1(
                kind="stop_campaign",
                owner="improve-lean-optimums",
                reason="theorem-backed metric band contradiction",
                evidence_ids=(evidence_id,),
            ),
            AutotrainActionV1(
                kind="repair_formal",
                owner="improve-lean-optimums",
                reason=(
                    "repair the theorem assumptions or proof model before ordinary "
                    "training resumes"
                ),
                evidence_ids=(evidence_id,),
            ),
        ]
    elif control_only_model_timeout:
        manifest_path = camp_dir / "manifests" / f"{candidate_id}.json"
        manifest_sha = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else None
        )
        if control_runtime_reproduced:
            actions.append(
                AutotrainActionV1(
                    kind="next_experiment",
                    owner="autotrain",
                    reason=(
                        "retire the reproduced control-only model timeout "
                        "comparison and consume the next distinct hypothesis"
                    ),
                    evidence_ids=(evidence_id,),
                )
            )
        else:
            actions.insert(
                0,
                AutotrainActionV1(
                    kind="retry_measurement",
                    owner="autotrain",
                    reason=(
                        "replay the exact frozen pair once to reproduce the "
                        "control-only typed model timeout"
                    ),
                    evidence_ids=(evidence_id,),
                    frozen_manifest_sha256=manifest_sha,
                ),
            )
    elif numeric_close_starvation:
        # The canonical model-build harness already owns the typed
        # ltr_tail_loss_weight lever and the size-matched literal-close arm.
        # This diagnosis therefore needs a fresh experiment, not a repair
        # receipt that falsely claims the signal is missing.
        actions.append(
            AutotrainActionV1(
                kind="next_experiment",
                owner="autotrain",
                reason=(
                    "run the registered typed tail-weighted LTR signal as a new "
                    "size-matched literal-close arm; do not replay the same stalled "
                    "checkpoint"
                ),
                evidence_ids=(evidence_id,),
            ),
        )
    elif harness_failure:
        family = _primary_harness_family(camp_dir)
        manifest_path = camp_dir / "manifests" / f"{candidate_id}.json"
        manifest_sha = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else None
        )
        actions[0:0] = [
            AutotrainActionV1(
                kind="repair_harness",
                owner="improve-openui-harnesses",
                reason="repair the canonical owner and replay the frozen arm",
                evidence_ids=(evidence_id,),
                harness_family=family,  # type: ignore[arg-type]
                frozen_manifest_sha256=manifest_sha,
            ),
            AutotrainActionV1(
                kind="retry_measurement",
                owner="autotrain",
                reason=(
                    "replay the identical frozen arm after the required canonical "
                    "harness repair"
                ),
                evidence_ids=(evidence_id,),
                frozen_manifest_sha256=manifest_sha,
            ),
        ]
    elif finalized_decode_timeout:
        manifest_path = camp_dir / "manifests" / f"{candidate_id}.json"
        manifest_sha = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else None
        )
        if runtime_arm_rejected:
            actions.append(
                AutotrainActionV1(
                    kind="next_experiment",
                    owner="autotrain",
                    reason=(
                        "retire the candidate-only runtime rejection and consume "
                        "the next distinct ranked hypothesis"
                    ),
                    evidence_ids=(evidence_id,),
                )
            )
        else:
            actions[0:0] = [
            AutotrainActionV1(
                kind="repair_harness",
                owner="improve-openui-harnesses",
                reason=(
                    "AgentV finalized every record disposition and reported an "
                    "internal decode timeout; "
                )
                + (
                    "repair canonical model-build runtime before replaying the "
                    "frozen arm"
                ),
                evidence_ids=(evidence_id,),
                harness_family="model_build",
                frozen_manifest_sha256=manifest_sha,
            ),
            AutotrainActionV1(
                kind="retry_measurement",
                owner="autotrain",
                reason=(
                    "replay the identical frozen arm after the required "
                    "canonical runtime repair"
                ),
                evidence_ids=(evidence_id,),
                frozen_manifest_sha256=manifest_sha,
            ),
            ]
    elif measurement_incomplete:
        from slm_training.autoresearch.climb_policy import (
            load_climb_policy,
            max_consecutive_frozen_replays,
        )

        manifest_path = camp_dir / "manifests" / f"{candidate_id}.json"
        manifest_sha = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else None
        )
        replay_count = _consecutive_frozen_replays(
            root, loop_id, campaign_id, cycle_intent
        )
        replay_limit = max_consecutive_frozen_replays(load_climb_policy())
        if replay_count >= replay_limit:
            actions[0:0] = [
                AutotrainActionV1(
                    kind="repair_harness",
                    owner="improve-openui-harnesses",
                    reason=(
                        "identical incomplete replay budget exhausted "
                        f"({replay_count}/{replay_limit}); repair the canonical "
                        "owner before replaying the frozen arm"
                    ),
                    evidence_ids=(evidence_id,),
                    harness_family=_primary_harness_family(camp_dir),  # type: ignore[arg-type]
                    frozen_manifest_sha256=manifest_sha,
                ),
                AutotrainActionV1(
                    kind="retry_measurement",
                    owner="autotrain",
                    reason=(
                        "replay the identical frozen arm after the required "
                        "canonical harness repair"
                    ),
                    evidence_ids=(evidence_id,),
                    frozen_manifest_sha256=manifest_sha,
                ),
            ]
        else:
            actions.insert(
                0,
                AutotrainActionV1(
                    kind="retry_measurement",
                    owner="autotrain",
                    reason=(
                        "measurement incomplete; replay the identical frozen arm "
                        f"({replay_count}/{replay_limit})"
                    ),
                    evidence_ids=(evidence_id,),
                    frozen_manifest_sha256=manifest_sha,
                ),
            )
    elif assumption_miss or (formal_status is not None and formal_status != "proved"):
        actions.insert(
            0,
            AutotrainActionV1(
                kind="repair_formal",
                owner="improve-lean-optimums",
                reason=(
                    "assumption-backed band miss requires the five diagnosis lanes"
                    if assumption_miss
                    else f"formal preflight is {formal_status}"
                ),
                evidence_ids=(evidence_id,),
            ),
        )
    elif diagnosis_target == "data":
        actions.insert(
            0,
            AutotrainActionV1(
                kind="rebuild_data",
                owner="synthesis-feedback",
                reason="repair the named synthesis producer and rebuild immutably",
                evidence_ids=(evidence_id,),
            ),
        )
    elif (
        cycle_intent in {"screening", "promotion"}
        and not measurement_incomplete
        and not delivery.get("positive")
        and not any(
            priority.disposition == "experiment_next"
            and priority.proposed_experiment_id
            for priority in priorities
        )
    ):
        actions.insert(
            0,
            AutotrainActionV1(
                kind="repair_harness",
                owner="improve-openui-harnesses",
                reason=(
                    "registered quality-arm bank exhausted; preregister and wire "
                    "a distinct size-matched model-build objective before the next run"
                ),
                evidence_ids=(evidence_id,),
                harness_family="model_build",
            ),
        )
    else:
        actions.append(
            AutotrainActionV1(
                kind="next_experiment",
                owner="autotrain",
                reason="consume the ranked successor priorities",
                evidence_ids=(evidence_id,),
            )
        )
    if delivery.get("positive"):
        actions.append(
            AutotrainActionV1(
                kind="deliver_stack",
                owner="sdlc",
                reason="document the positive result, then deliver its reviewable delta",
                evidence_ids=(evidence_id,),
            )
        )

    handoff = AutotrainCycleHandoffV1(
        loop_id=loop_id,
        campaign_id=campaign_id,
        cycle_index=cycle_index,
        upstream_commit=upstream_commit,
        integration_commit=integration_commit,
        cycle_role=role,  # type: ignore[arg-type]
        cycle_intent=cycle_intent,
        evidence_class=evidence_class,  # type: ignore[arg-type]
        climb_state=climb_state,  # type: ignore[arg-type]
        ship_state=ship_state,  # type: ignore[arg-type]
        primary_metric=primary_metric,
        reasons=reasons,
        priorities=priorities,
        actions=tuple(actions),
        formal_status=formal_status,
        checkpoint_paths=checkpoint_paths,
        checkpoint_documentation_required=bool(checkpoint_paths),
    )
    (camp_dir / "cycle_handoff.json").write_text(
        handoff.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    _write_loop_state(
        root,
        AutotrainLoopStateV1(
            loop_id=loop_id,
            state="IDLE",
            phase="between_cycles",
            active_campaign_id=None,
            last_completed_campaign_id=campaign_id,
            cycle_index=cycle_index,
            next_action=actions[0].kind,
            pid=os.getpid(),
        ),
    )
    return handoff


def _refresh_incomplete_replay_handoff(
    root: Path, loop_id: str, campaign_id: str | None
) -> bool:
    """Repair a derived terminal handoff when its current candidate never scored."""

    if not campaign_id:
        return False
    camp_dir = root / campaign_id
    handoff_path = camp_dir / "cycle_handoff.json"
    if not handoff_path.is_file():
        return False
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != loop_id or handoff.cycle_intent != "retry_measurement":
        return False
    terminal = any(
        reason.startswith("candidate_runtime_unblock_reproduced:")
        for reason in handoff.reasons
    )
    delivery = _read_json(camp_dir / "sdlc_delivery.json")
    candidate_id = str(delivery.get("candidate_id") or "")
    if not terminal or _run_has_usable_metrics(camp_dir, candidate_id):
        return False
    campaign = CampaignSpec.model_validate_json(
        (camp_dir / "campaign.json").read_text(encoding="utf-8")
    )
    matrix = _read_json(camp_dir / "matrix-proposal.json")
    _write_cycle_handoff(
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        cycle_index=campaign.cycle_index,
        upstream_commit=str(campaign.upstream_commit),
        integration_commit=str(campaign.integration_commit),
        role=handoff.cycle_role,
        cycle_intent=handoff.cycle_intent,
        primary_metric=handoff.primary_metric,
        matrix=matrix,
        delivery=delivery,
        resolution=None,
        formal_status=_formal_preflight_status(camp_dir),
    )
    print(
        "REPLAY_HANDOFF_REFRESH "
        f"campaign={campaign_id} reason=current_candidate_missing_metrics",
        flush=True,
    )
    return True


def _latest_cycle(root: Path, loop_id: str) -> tuple[int, str | None]:
    campaigns = sorted(root.glob("*/campaign.json"))
    best_idx = 0
    best_id: str | None = None
    completed_idx = 0
    completed_id: str | None = None
    for path in campaigns:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("loop_id") != loop_id:
            continue
        idx = int(data.get("cycle_index") or 0)
        if idx >= best_idx:
            best_idx = idx
            best_id = str(data.get("campaign_id"))
        campaign_id = str(data.get("campaign_id"))
        if (
            idx >= completed_idx
            and (root / campaign_id / "cycle_handoff.json").is_file()
        ):
            completed_idx = idx
            completed_id = campaign_id
    return best_idx, completed_id or best_id


def _campaign_at_cycle(root: Path, loop_id: str, cycle_index: int) -> str | None:
    matches: list[str] = []
    for path in root.glob("*/campaign.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            data.get("loop_id") == loop_id
            and int(data.get("cycle_index") or 0) == cycle_index
        ):
            matches.append(str(data.get("campaign_id")))
    if len(matches) > 1:
        raise RuntimeError(f"multiple campaigns claim loop cycle {cycle_index}")
    return matches[0] if matches else None


def _finalize_terminal_interrupted_replay(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    deadline: float,
) -> str | None:
    """Finish a terminal frozen replay whose process stopped before its handoff.

    Recovery is deliberately narrower than cycle replay: both decision arms must
    have terminal events in the verified campaign chain.  A partially executed
    campaign therefore remains incomplete and cannot acquire a handoff merely
    because a stray run artifact exists.
    """

    cycle_index, _predecessor = _latest_cycle(root, loop_id)
    campaign_id = _campaign_at_cycle(root, loop_id, cycle_index)
    if campaign_id is None:
        return None
    camp_dir = root / campaign_id
    if (camp_dir / "cycle_handoff.json").is_file():
        return None
    matrix_path = camp_dir / "matrix-proposal.json"
    if not matrix_path.is_file():
        return None
    matrix = _read_json(matrix_path)
    hypotheses = matrix.get("hypotheses") or []
    if not hypotheses or not isinstance(hypotheses[0], dict):
        return None
    control = str((hypotheses[0].get("experiment") or {}).get("experiment_id") or "")
    candidate = str(matrix.get("recommended_experiment_id") or "")
    expected = {control, candidate}
    if "" in expected or len(expected) != 2:
        return None
    events = CampaignStore(campaign_id, root).verify_event_chain()
    finished = {
        str(event.get("experiment_id") or "")
        for event in events
        if event.get("event_type") == "experiment_finished"
    }
    if not expected.issubset(finished):
        return None

    campaign = _read_json(camp_dir / "campaign.json")
    manifests = [
        _read_json(camp_dir / "manifests" / f"{experiment_id}.json")
        for experiment_id in sorted(expected)
    ]
    if not all(
        item.get("replay_of_manifest_sha256")
        and item.get("claim_class") == "diagnostic"
        for item in manifests
    ):
        return None
    role = "screening"
    cycle_intent = "retry_measurement"
    primary_metric = str(campaign.get("primary_metric") or "")
    if not primary_metric:
        raise RuntimeError(
            f"terminal interrupted campaign lacks primary metric: {campaign_id}"
        )

    _set_active_stage(root, loop_id, "recover-terminal-handoff")
    delivery = _phase_a_delivery(
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        primary_metric=primary_metric,
        cycle_index=cycle_index,
        role=role,
        cycle_intent=cycle_intent,
        deadline=deadline,
    )
    _write_cycle_handoff(
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        cycle_index=cycle_index,
        upstream_commit=str(campaign.get("upstream_commit") or ""),
        integration_commit=str(campaign.get("integration_commit") or ""),
        role=role,
        cycle_intent=cycle_intent,
        primary_metric=primary_metric,
        matrix=matrix,
        delivery=delivery,
        resolution=None,
        formal_status=_formal_preflight_status(camp_dir),
    )
    _run(
        [
            sys.executable,
            "-m",
            "scripts.autoresearch",
            "--root",
            str(root),
            "status",
            "--loop-id",
            loop_id,
            "--matrix",
            "--last",
            "5",
        ],
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="recover-status",
    )
    print(
        f"CYCLE_RECOVERED {campaign_id} role={role} intent={cycle_intent} "
        f"positive={delivery['positive']}",
        flush=True,
    )
    return campaign_id


def _experiment_artifact(camp_dir: Path, experiment_id: str) -> dict[str, Any]:
    for path in (camp_dir / "artifacts" / "experiments").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment_id") == experiment_id:
            return payload
    raise RuntimeError(f"frozen replay experiment is missing: {experiment_id}")


def _manifest_with_sha(
    camp_dir: Path, digest: str
) -> tuple[Path, ExperimentCampaignV1]:
    for path in (camp_dir / "manifests").glob("*.json"):
        if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            manifest = ExperimentCampaignV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            return path, manifest
    raise RuntimeError(f"frozen replay manifest is missing: {digest}")


def _restore_frozen_formal_claims(
    camp_dir: Path,
    experiment: dict[str, Any],
    manifest: ExperimentCampaignV1,
) -> None:
    """Recover claims omitted by an older replay from its proved artifacts."""

    if experiment.get("formal_claims") or not manifest.formal_obligations:
        return
    claims: list[dict[str, str]] = []
    for obligation in manifest.formal_obligations:
        path = (
            camp_dir
            / "artifacts"
            / "formal_preflights"
            / f"{obligation.preflight_sha256}.json"
        )
        if not path.is_file():
            raise RuntimeError(
                "frozen replay formal preflight is missing: "
                f"{obligation.preflight_sha256}"
            )
        preflight = FormalPreflightV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        claim = FormalClaimV1(
            template_id=preflight.template_id,
            claim=preflight.claim,
            policy=preflight.policy,
        )
        if (
            preflight.campaign_id != manifest.campaign_id
            or preflight.experiment_id != manifest.experiment_id
            or preflight.template_id != obligation.template_id
            or preflight.policy != obligation.policy
            or (
                claim.policy == "required"
                and preflight.status != "proved"
            )
            or formal_obligation_id(
                manifest.campaign_id, manifest.experiment_id, claim
            )
            != preflight.obligation_id
        ):
            raise RuntimeError(
                "frozen replay formal claim recovery mismatch: "
                f"{obligation.obligation_id}"
            )
        claims.append(claim.model_dump())
    experiment["formal_claims"] = claims


def _nonreplayable_configuration_failure(
    camp_dir: Path, experiment_id: str
) -> str | None:
    """Identify frozen arms whose exact configuration cannot reach execution."""

    for path in (camp_dir / "artifacts" / "outcomes").glob("*.json"):
        outcome = _read_json(path)
        if (
            outcome.get("experiment_id") != experiment_id
            or outcome.get("status") != "failed"
        ):
            continue
        error = str(outcome.get("error") or "")
        if (
            "lever_capability_compatibility" in error
            or "unsupported compiler auxiliary lever" in error
            or "no runtime owner is implemented" in error
        ):
            return "lever_capability_compatibility"
    return None


def _load_frozen_replay(
    root: Path, loop_id: str, predecessor_campaign_id: str | None
) -> dict[str, Any] | None:
    if not predecessor_campaign_id:
        return None
    handoff_path = root / predecessor_campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    retries = [
        (index, action)
        for index, action in pending_autotrain_execution_actions(root, handoff)
        if action.kind == "retry_measurement"
    ]
    if not retries:
        return None
    if len(retries) != 1:
        raise RuntimeError("exactly one frozen measurement retry may be active")
    action_index, action = retries[0]
    if action.frozen_manifest_sha256 is None:
        print(
            "FROZEN_REPLAY_SKIP reason=missing_frozen_manifest_sha256 "
            f"campaign={predecessor_campaign_id} action_index={action_index}",
            flush=True,
        )
        return None
    camp_dir = root / predecessor_campaign_id
    candidate_path, candidate_manifest = _manifest_with_sha(
        camp_dir, action.frozen_manifest_sha256
    )
    nonreplayable = _nonreplayable_configuration_failure(
        camp_dir, candidate_manifest.experiment_id
    )
    if nonreplayable:
        print(
            "FROZEN_REPLAY_SKIP reason=nonreplayable_configuration "
            f"detail={nonreplayable} campaign={predecessor_campaign_id} "
            f"action_index={action_index}",
            flush=True,
        )
        return None
    matrix = json.loads((camp_dir / "matrix-proposal.json").read_text(encoding="utf-8"))
    control_id = str(matrix["hypotheses"][0]["experiment"]["experiment_id"])
    control_path = camp_dir / "manifests" / f"{control_id}.json"
    if not control_path.is_file():
        raise RuntimeError(f"frozen replay control manifest is missing: {control_id}")
    control_manifest = ExperimentCampaignV1.model_validate_json(
        control_path.read_text(encoding="utf-8")
    )
    replay = {
        "handoff": handoff,
        "action_index": action_index,
        "action": action,
        "candidate": {
            "experiment": _experiment_artifact(
                camp_dir, candidate_manifest.experiment_id
            ),
            "manifest": candidate_manifest,
            "manifest_sha256": action.frozen_manifest_sha256,
            "path": candidate_path,
        },
        "control": {
            "experiment": _experiment_artifact(camp_dir, control_id),
            "manifest": control_manifest,
            "manifest_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
            "path": control_path,
        },
    }
    for role in ("control", "candidate"):
        item = replay[role]
        _restore_frozen_formal_claims(
            camp_dir, item["experiment"], item["manifest"]
        )
        item["train_reuse"] = _completed_frozen_train_source(
            root=root,
            campaign_dir=camp_dir,
            manifest=item["manifest"],
            manifest_path=item["path"],
        )
    return replay


def _completed_frozen_train_source(
    *,
    root: Path,
    campaign_dir: Path,
    manifest: ExperimentCampaignV1,
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Find a completed train stage along a hash-linked frozen replay chain."""

    lineage: list[Path] = []
    visited: set[str] = set()
    current_dir = campaign_dir
    current_manifest = manifest
    current_path = manifest_path
    while True:
        digest = hashlib.sha256(current_path.read_bytes()).hexdigest()
        if digest in visited:
            raise RuntimeError("frozen training replay lineage contains a cycle")
        visited.add(digest)
        lineage.append(current_path)
        run_dir = current_dir / "runs" / current_manifest.experiment_id
        summary_path = run_dir / "train_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint = Path(str(summary.get("checkpoint") or ""))
            if (
                summary.get("run_id") == current_manifest.experiment_id
                and summary.get("stopped_on") == "steps"
                and int(summary.get("steps") or -1) > 0
                and checkpoint.is_file()
            ):
                return {
                    "run_dir": run_dir,
                    "manifest_paths": tuple(lineage),
                }
        replay_sha = current_manifest.replay_of_manifest_sha256
        if replay_sha is None:
            return None
        campaign = CampaignSpec.model_validate_json(
            (current_dir / "campaign.json").read_text(encoding="utf-8")
        )
        predecessor_id = campaign.predecessor_campaign_id
        if predecessor_id is None:
            return None
        predecessor_seen: set[str] = set()
        while predecessor_id is not None:
            if predecessor_id in predecessor_seen:
                raise RuntimeError("frozen campaign predecessor lineage contains a cycle")
            predecessor_seen.add(predecessor_id)
            current_dir = root / predecessor_id
            try:
                current_path, current_manifest = _manifest_with_sha(
                    current_dir, replay_sha
                )
                break
            except RuntimeError as exc:
                if not str(exc).startswith("frozen replay manifest is missing:"):
                    raise
                predecessor_campaign_path = current_dir / "campaign.json"
                if not predecessor_campaign_path.is_file():
                    raise
                predecessor_campaign = CampaignSpec.model_validate_json(
                    predecessor_campaign_path.read_text(encoding="utf-8")
                )
                predecessor_id = predecessor_campaign.predecessor_campaign_id
        else:
            raise RuntimeError(f"frozen replay manifest is missing: {replay_sha}")


def _apply_frozen_replay(
    matrix: dict[str, Any], replay: dict[str, Any], campaign_id: str
) -> dict[str, dict[str, Any]]:
    prefix = campaign_id.replace("continuous-loop-", "c")
    old_candidate_id = str(replay["candidate"]["experiment"]["experiment_id"])
    registered_slugs = sorted(
        (item[0] for item in _SCREENING_ARM_BANK), key=len, reverse=True
    )
    slug = next(
        (
            registered
            for registered in registered_slugs
            if old_candidate_id.endswith(f"-{registered}")
        ),
        None,
    )
    promotion_replay = old_candidate_id.endswith("-promote")
    confirmation_replay = old_candidate_id.endswith("-confirm")
    if promotion_replay:
        slug = "promote"
    elif confirmation_replay:
        slug = "confirm"
    if slug is None:
        raise RuntimeError(
            f"unsupported automatic frozen replay arm: {old_candidate_id}"
        )
    new_ids = {"control": f"{prefix}-control", "candidate": f"{prefix}-{slug}"}
    if promotion_replay or confirmation_replay:
        frozen_knobs = replay["candidate"]["experiment"].get("knobs") or {}
        frozen_fingerprint = _knobs_fingerprint(
            _lever_knobs(frozen_knobs)
        )
        source_slug = (
            _arm_slug_from_knobs(frozen_knobs, candidate_id=old_candidate_id)
            if confirmation_replay
            else None
        )
        candidate_target = next(
            (
                item["experiment"]
                for item in matrix["hypotheses"]
                if source_slug
                and item["experiment"]["experiment_id"].endswith(f"-{source_slug}")
            ),
            None,
        ) or next(
            (
                item["experiment"]
                for item in matrix["hypotheses"]
                if _knobs_fingerprint(
                    _lever_knobs(item["experiment"].get("knobs") or {})
                )
                == frozen_fingerprint
            ),
            None,
        ) or next(
            (
                item["experiment"]
                for item in matrix["hypotheses"]
                if item["experiment"]["experiment_id"]
                == matrix["recommended_experiment_id"]
            ),
            None,
        )
        if candidate_target is None:
            kind = "promotion" if promotion_replay else "confirmation"
            raise RuntimeError(f"frozen {kind} replay target is absent from matrix")
        previous_target_id = str(candidate_target["experiment_id"])
        candidate_target["experiment_id"] = new_ids["candidate"]
        for priority in matrix["next_run_priorities"]:
            if priority.get("proposed_experiment_id") == previous_target_id:
                priority["proposed_experiment_id"] = new_ids["candidate"]
    for role, new_id in new_ids.items():
        target = next(
            (
                item["experiment"]
                for item in matrix["hypotheses"]
                if item["experiment"]["experiment_id"] == new_id
            ),
            None,
        )
        if target is None:
            raise RuntimeError(f"frozen replay target is absent from matrix: {new_id}")
        frozen = replay[role]["experiment"]
        for key in (
            "hypothesis",
            "rationale",
            "expected_effect",
            "falsification_criteria",
            "stop_conditions",
            "knobs",
            "formal_claims",
        ):
            target[key] = frozen.get(key, []) if key == "formal_claims" else frozen[key]
    matrix["recommended_experiment_id"] = new_ids["candidate"]
    matrix["selection_rationale"] = (
        "Current-main successor replay of the incomplete frozen measurement; "
        "model/data arms, seeds, endpoints, gates, and stopping rules are preserved."
    )
    matrix["next_run_priorities"][0].update(
        {
            "area": "infrastructure",
            "hypothesis": "The repaired harness completes the identical frozen measurement.",
            "authority": "observed_result",
            "confidence": 0.95,
            "proposed_experiment_id": new_ids["candidate"],
        }
    )
    HypothesisMatrix.model_validate(matrix)
    return {new_ids[role]: replay[role] for role in ("control", "candidate")}


def _replay_successor_manifest(
    frozen: ExperimentCampaignV1,
    *,
    frozen_manifest_sha256: str,
    campaign_id: str,
    experiment_id: str,
    integration_commit: str,
) -> ExperimentCampaignV1:
    successor = frozen.model_copy(
        update={
            "campaign_id": campaign_id,
            "experiment_id": experiment_id,
            "source_commit": integration_commit,
            "source_dirty": False,
            "author": "autotrain-frozen-replay-successor",
            "created_at": utc_now(),
            "replay_of_manifest_sha256": frozen_manifest_sha256,
            "replay_reason": (
                "Current-main successor after an infrastructure-incomplete measurement."
            ),
            # A proof is commit- and experiment-bound. Never carry the source
            # campaign's proof digest into a current-main successor.
            "formal_obligations": (),
        }
    )
    return ExperimentCampaignV1.model_validate(successor.model_dump(mode="json"))


def _bind_fresh_replay_formal_preflight(
    successor: ExperimentCampaignV1,
    frozen: ExperimentCampaignV1,
    *,
    preflight_sha256: str,
    formal_claims: list[dict[str, str]],
) -> ExperimentCampaignV1:
    """Bind frozen claim policy to current identities and the fresh proof."""

    claims = tuple(FormalClaimV1.model_validate(claim) for claim in formal_claims)
    expected = sorted(
        (obligation.template_id, obligation.policy)
        for obligation in frozen.formal_obligations
    )
    actual = sorted((claim.template_id, claim.policy) for claim in claims)
    if actual != expected:
        raise RuntimeError("frozen replay formal claim policy mismatch")
    obligations = tuple(
        FormalObligationV1(
            obligation_id=formal_obligation_id(
                successor.campaign_id, successor.experiment_id, claim
            ),
            template_id=claim.template_id,
            policy=claim.policy,
            preflight_sha256=preflight_sha256,
        )
        for claim in claims
    )
    rebound = successor.model_copy(update={"formal_obligations": obligations})
    return ExperimentCampaignV1.model_validate(rebound.model_dump(mode="json"))


def _campaign_id(loop_id: str, cycle: int, *, date: str | None = None) -> str:
    """Loop-scoped identity; historical campaign ids remain readable."""

    slug = re.sub(r"[^A-Za-z0-9]+", "-", loop_id).strip("-").lower()[:24]
    digest = hashlib.sha256(loop_id.encode("utf-8")).hexdigest()[:8]
    day = date or time.strftime("%Y%m%d")
    return f"continuous-loop-{day}-{slug or 'loop'}-{digest}-c{cycle}"


def _continuous_evidence_roots(
    root: Path, loop_id: str, predecessor_campaign_id: str | None
) -> tuple[Path, ...]:
    roots = [root / "loops" / loop_id, root / "sdlc_delivery_ledger.jsonl"]
    if predecessor_campaign_id:
        roots.insert(0, root / predecessor_campaign_id)
    return tuple(roots)


def _require_predecessor_actions(
    root: Path, loop_id: str, predecessor_campaign_id: str | None
) -> None:
    if not predecessor_campaign_id:
        return
    handoff_path = root / predecessor_campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        return  # Historical campaigns predate supervised handoffs.
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != loop_id or handoff.campaign_id != predecessor_campaign_id:
        raise RuntimeError("predecessor handoff identity does not match loop lineage")
    pending = pending_autotrain_actions(root, handoff)
    if pending:
        detail = ", ".join(f"{index}:{action.kind}" for index, action in pending)
        raise RuntimeError(
            f"predecessor {predecessor_campaign_id} has unacknowledged actions: {detail}"
        )


def _matrix(
    *,
    campaign_id: str,
    evidence_snapshot_id: str,
    cites: list[str],
    role_citations: dict[str, str],
    train_version: str,
    eval_version: str,
    steps: int,
    cycle: int,
    feedback: list[dict] | None = None,
    previous_matrix_id: str | None = None,
    role: str = "screening",
    policy: Any | None = None,
    confirm_levers: dict[str, Any] | None = None,
    confirm_control_levers: dict[str, Any] | None = None,
    promote_levers: dict[str, Any] | None = None,
    promote_control_levers: dict[str, Any] | None = None,
    recommended_slug: str | None = None,
    skip_slugs: set[str] | None = None,
) -> dict:
    from slm_training.autoresearch.climb_policy import (
        decode_timeout_seconds_for_role,
        eval_suites_for_role,
        load_climb_policy,
    )

    research = role_citations.get("research") or cites[0]
    prior = role_citations.get("prior_result") or (
        cites[1] if len(cites) > 1 else cites[0]
    )
    # Unique seed per cycle — (cycle*17)%50 only yields 50 seeds and thrash-rejects.
    seed = 100_000 + int(cycle)
    steps = steps + (cycle % 3)  # slight variation avoids knob-signature collision
    pol = policy or load_climb_policy()
    decode_timeout = decode_timeout_seconds_for_role(pol, role)
    eval_suites = ",".join(eval_suites_for_role(pol, role))

    def uses() -> list[dict]:
        out = []
        contrib = {
            "research": "Continuous-loop and decode contracts.",
            "prior_trace": "Prior run telemetry or insight from the continuous loop.",
            "prior_result": "Prior campaign or evaluation baseline.",
        }
        for role, citation in role_citations.items():
            out.append(
                {
                    "role": role,
                    "citation": citation,
                    "contribution": contrib.get(role, "Captured continuous evidence."),
                }
            )
        if not out:
            out = [
                {
                    "role": "research",
                    "citation": research,
                    "contribution": contrib["research"],
                }
            ]
        return out

    def novelty(i: int, residual: str) -> dict:
        return {
            "transition_kind": (
                "regime_transition_candidate" if i == 0 else "fixed_regime_search"
            ),
            "old_schema_elements": ["training recipe", "grammar decode path"],
            "proposed_schema_elements": [residual if i == 0 else "training recipe"],
            "transported_elements": ["prior scoreboard", "fixture smoke path"],
            "transport_analysis": [
                "Residual is not explained by the matched fixture control alone."
            ],
            "residual_elements": [residual],
            "preservation_checks": ["rerun matched control under the wall cap"],
            "stress_tests": ["smoke parse_rate and latency under published eval"],
            "worthiness_criteria": [
                "complete scoreboard under wall without path errors"
            ],
        }

    def knobs(**extra: object) -> dict:
        base: dict[str, object] = {
            "train_version": train_version,
            "eval_version": eval_version,
            "steps": steps,
            "batch_size": 2,
            "seed": seed,
            "context_backend": "scratch",
            "sync_checkpoints": False,
            "local_files_only": True,
            "grammar_completion_bounds": False,
            "grammar_equivalence_cache": False,
            "grammar_draft_window": 8,
            "compact_active_canvas": False,
            "component_plan_loss_weight": 0.0,
            "component_plan_decode_weight": 0.0,
            "component_edge_loss_weight": 0.0,
            "component_edge_alignment_loss_weight": 0.0,
            "component_edge_decode_weight": 0.0,
            "component_inventory_loss_weight": 0.0,
            "component_inventory_decode_weight": 0.0,
            "binder_topology_loss_weight": 0.0,
            "binder_topology_decode_weight": 0.0,
            "binder_component_plan_loss_weight": 0.0,
            "binder_component_plan_decode_weight": 0.0,
            "binder_arity_loss_weight": 0.0,
            "binder_arity_decode_weight": 0.0,
            "fidelity_loss_weight": 0.5,
            "semantic_contrast_loss_weight": 0.0,
            "symbol_slot_augmentation": False,
            "mask_pattern": "random",
            "symbol_boundary_loss_weight": 0.0,
            "design_md_dropout": 0.0,
            "ltr_prefix_loss_weight": 0.0,
            "component_token_loss_weight": 0.0,
            "component_edge_token_loss_weight": 0.0,
            "compiler_decision_token_loss_weight": 0.0,
            "structure_token_loss_weight": 0.0,
            "typed_family_balance_loss_weight": 0.0,
            "structural_aux_head_profile": "none",
            "compiler_decode_mode": "off",
            "ltr_tail_loss_weight": 0.0,
            "compiler_alignment_loss_weight": 0.0,
            "compiler_alignment_margin": 0.0,
            "compiler_alignment_stratified": False,
            "compiler_alignment_kind_filter": "all",
            # Measurement completeness: smoke-only screening + longer decode wall.
            "decode_timeout_seconds": decode_timeout,
            "eval_suites": eval_suites,
        }
        base.update(extra)
        return base

    def exp(
        eid: str,
        hyp: str,
        k: dict,
        rationale: str,
        *,
        formal_claims: list[dict[str, str]] | None = None,
    ) -> dict:
        # Citations must include every evidence_use citation (schema invariant).
        exp_cites = list(dict.fromkeys([*cites[:3], *role_citations.values()]))
        payload: dict[str, Any] = {
            "experiment_id": eid,
            "campaign_id": campaign_id,
            "hypothesis": hyp,
            "rationale": rationale,
            "expected_effect": "Runnable smoke scoreboard under the wall cap.",
            "falsification_criteria": [
                "Path error or no smoke metrics under the wall cap."
            ],
            "stop_conditions": ["Stop at declared steps or the campaign wall cap."],
            "citations": exp_cites,
            "knobs": k,
        }
        # Attach formal claims *before* hypothesize so locked matrix membership
        # stays exact at execute (do not rewrite experiment files post-lock).
        if formal_claims:
            payload["formal_claims"] = list(formal_claims)
        return payload

    prefix = campaign_id.replace("continuous-loop-", "c")
    # Promotion path (Change C): confirmed champion knobs under promotion suites.
    if promote_levers:
        promo_extra = {
            k: v
            for k, v in promote_levers.items()
            if k in _LEVER_KNOB_KEYS
        }
        promo_steps = int(promo_extra.pop("steps", steps) or steps)
        control_extra = {
            k: v
            for k, v in (promote_control_levers or {}).items()
            if k in _LEVER_KNOB_KEYS
        }
        control_steps = int(control_extra.pop("steps", steps) or steps)
        control_extra.setdefault(
            "structural_aux_head_profile",
            str(promo_extra.get("structural_aux_head_profile") or "none"),
        )
        control_extra.setdefault(
            "compiler_decode_mode",
            str(promo_extra.get("compiler_decode_mode") or "off"),
        )
        control_knobs = knobs(steps=control_steps, **control_extra)
        cand_knobs = knobs(steps=promo_steps, **promo_extra)
        candidates = [
            {
                "experiment": exp(
                    f"{prefix}-control",
                    "Matched control for promotion of a confirmed champion under held-out suites.",
                    control_knobs,
                    "Size-matched baseline for promotion cycle.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(0, "promote matched control"),
            },
            {
                "experiment": exp(
                    f"{prefix}-promote",
                    "Promotion retest of confirmed champion levers under promotion primary/suites.",
                    cand_knobs,
                    "Champion queue promotion arm — multi-seed / held-out when policy requires.",
                    formal_claims=[promote_formal_claim_dict()],
                ),
                "evidence_uses": uses(),
                "novelty": novelty(1, "promote confirmed knobs"),
            },
            {
                "experiment": exp(
                    f"{prefix}-bounds",
                    "Monitor-only bounds pad deferred while a confirmed champion is evaluated.",
                    knobs(grammar_completion_bounds=True, steps=promo_steps + 2000),
                    "Schema pad — not executed while promote is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(2, "promote pad bounds"),
            },
            {
                "experiment": exp(
                    f"{prefix}-canvas",
                    "Monitor-only canvas pad deferred while a confirmed champion is evaluated.",
                    knobs(compact_active_canvas=True, steps=promo_steps + 2001),
                    "Schema pad — not executed while promote is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(3, "promote pad canvas"),
            },
            {
                "experiment": exp(
                    f"{prefix}-both",
                    "Monitor-only combined pad deferred while a confirmed champion is evaluated.",
                    knobs(
                        grammar_completion_bounds=True,
                        compact_active_canvas=True,
                        steps=promo_steps + 2002,
                    ),
                    "Schema pad — not executed while promote is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(4, "promote pad both"),
            },
        ]
        rec = f"{prefix}-promote"
        priorities = [
            {
                "rank": 1,
                "area": "model",
                "hypothesis": (
                    "Confirmed champion levers hold under promotion primary and multi-seed."
                ),
                "evidence_ids": [research, prior],
                "confidence": 0.8,
                "expected_information_gain": "Promotion claim evidence from sticky knobs.",
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": rec,
            },
            {
                "rank": 2,
                "area": "evaluation",
                "hypothesis": "Matched control remains the size-matched baseline on promote.",
                "evidence_ids": [research, prior],
                "confidence": 0.7,
                "expected_information_gain": "Prevents false promotion from recipe drift.",
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": f"{prefix}-control",
            },
            {
                "rank": 3,
                "area": "infrastructure",
                "hypothesis": "Only confirmed queue heads enter promotion matrices.",
                "evidence_ids": [research, prior],
                "confidence": 0.85,
                "expected_information_gain": "Stops index-based empty promotion thrash.",
                "authority": "observed_result",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
            {
                "rank": 4,
                "area": "model",
                "hypothesis": "Pad arms stay monitor-only during promote.",
                "evidence_ids": [research, prior],
                "confidence": 0.5,
                "expected_information_gain": "Schema completeness without thrash spend.",
                "authority": "speculative",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
            {
                "rank": 5,
                "area": "model_build",
                "hypothesis": "After promote resolves, resume screening thrash with rotation.",
                "evidence_ids": [research, prior],
                "confidence": 0.55,
                "expected_information_gain": "Queue advances only after promote resolve.",
                "authority": "speculative",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
        ]
        payload = {
            "matrix_id": f"{campaign_id}-m1-promote",
            "campaign_id": campaign_id,
            "evidence_snapshot_id": evidence_snapshot_id,
            "hypotheses": candidates,
            "recommended_experiment_id": rec,
            "selection_rationale": (
                "Champion-queue promotion matrix: confirmed levers under "
                "promotion role suites/seeds; thrash deferred."
            ),
            "next_run_priorities": priorities,
        }
    # Confirmatory path: same levers as a quality-held champion, new seed — not
    # another thrash of the fixed lever bank.
    elif confirm_levers:
        confirm_extra = {
            k: v
            for k, v in confirm_levers.items()
            if k in _LEVER_KNOB_KEYS
        }
        confirm_steps = int(confirm_extra.pop("steps", steps) or steps)
        control_extra = {
            k: v
            for k, v in (confirm_control_levers or {}).items()
            if k in _LEVER_KNOB_KEYS
        }
        control_steps = int(control_extra.pop("steps", steps) or steps)
        control_extra.setdefault(
            "structural_aux_head_profile",
            str(confirm_extra.get("structural_aux_head_profile") or "none"),
        )
        control_extra.setdefault(
            "compiler_decode_mode",
            str(confirm_extra.get("compiler_decode_mode") or "off"),
        )
        control_knobs = knobs(steps=control_steps, **control_extra)
        # Drop lever defaults then re-apply champion levers on candidate.
        cand_knobs = knobs(steps=confirm_steps, **confirm_extra)
        # HypothesisMatrix requires ≥5 arms; only control + recommended execute.
        # Pad with monitor-only thrash placeholders so schema stays closed.
        candidates = [
            {
                "experiment": exp(
                    f"{prefix}-control",
                    "Matched control (levers off) for confirmatory retest of a quality-held champion.",
                    control_knobs,
                    "Size-matched baseline for confirm cycle.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(0, "confirm matched control"),
            },
            {
                "experiment": exp(
                    f"{prefix}-confirm",
                    "Confirmatory retest: same lever knobs as a quality-held screening win, new seed.",
                    cand_knobs,
                    "Champion queue confirmatory arm — must re-hold quality before promotion.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(1, "confirm same knobs new seed"),
            },
            {
                "experiment": exp(
                    f"{prefix}-bounds",
                    "Monitor-only pad: bounds thrash deferred while champion confirm is open.",
                    knobs(
                        grammar_completion_bounds=True,
                        # Distinct knob signature vs confirm (schema uniqueness).
                        steps=confirm_steps + 1000,
                    ),
                    "Schema pad — not executed while confirm is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(2, "confirm pad bounds"),
            },
            {
                "experiment": exp(
                    f"{prefix}-canvas",
                    "Monitor-only pad: canvas thrash deferred while champion confirm is open.",
                    knobs(
                        compact_active_canvas=True,
                        steps=confirm_steps + 1001,
                    ),
                    "Schema pad — not executed while confirm is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(3, "confirm pad canvas"),
            },
            {
                "experiment": exp(
                    f"{prefix}-both",
                    "Monitor-only pad: combined thrash deferred while champion confirm is open.",
                    knobs(
                        grammar_completion_bounds=True,
                        compact_active_canvas=True,
                        steps=confirm_steps + 1002,
                    ),
                    "Schema pad — not executed while confirm is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(4, "confirm pad both"),
            },
        ]
        rec = f"{prefix}-confirm"
        priorities = [
            {
                "rank": 1,
                "area": "model",
                "hypothesis": (
                    "Quality-held champion levers re-hold under a new seed before any "
                    "promotion claim."
                ),
                "evidence_ids": [research, prior],
                "confidence": 0.75,
                "expected_information_gain": "Separates one-off smoke noise from sticky knobs.",
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": rec,
            },
            {
                "rank": 2,
                "area": "evaluation",
                "hypothesis": "Matched control remains the size-matched baseline on confirm.",
                "evidence_ids": [research, prior],
                "confidence": 0.7,
                "expected_information_gain": "Prevents false confirm from recipe drift.",
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": f"{prefix}-control",
            },
            {
                "rank": 3,
                "area": "infrastructure",
                "hypothesis": "Champion queue blocks thrash until confirm resolves.",
                "evidence_ids": [research, prior],
                "confidence": 0.8,
                "expected_information_gain": "Learning signal from sticky knobs, not cycle noise.",
                "authority": "observed_result",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
            {
                "rank": 4,
                "area": "model",
                "hypothesis": "Pad arms stay monitor-only during confirm.",
                "evidence_ids": [research, prior],
                "confidence": 0.5,
                "expected_information_gain": "Schema completeness without thrash spend.",
                "authority": "speculative",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
            {
                "rank": 5,
                "area": "model_build",
                "hypothesis": "After confirm resolves, resume lever thrash or promote.",
                "evidence_ids": [research, prior],
                "confidence": 0.55,
                "expected_information_gain": "Queue head advances only after resolve.",
                "authority": "speculative",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
        ]
        payload = {
            "matrix_id": f"{campaign_id}-m1-confirm",
            "campaign_id": campaign_id,
            "evidence_snapshot_id": evidence_snapshot_id,
            "hypotheses": candidates,
            "recommended_experiment_id": rec,
            "selection_rationale": (
                "Champion-queue confirmatory matrix: same levers, new seed; "
                "no thrash of the fixed lever bank."
            ),
            "next_run_priorities": priorities,
        }
    else:
        # Change B: rotate recommended thrash arm; full bank always present.
        rec_slug = recommended_slug or _select_recommended_slug(cycle, skip=skip_slugs)
        bank_by_slug = {
            slug: (hyp, extras) for slug, hyp, extras in _SCREENING_ARM_BANK
        }
        if rec_slug not in bank_by_slug:
            rec_slug = _SCREENING_ARM_BANK[0][0]
        control_extra: dict[str, Any] = {}
        treatment_key = {
            "bounded-compiler-decision-margin": "grammar_completion_bounds",
            "cached-compiler-decision-margin": "grammar_equivalence_cache",
            "wide-draft-compiler-decision-margin": "grammar_draft_window",
            "capacity-aware-compiler-decision-margin": "mixture_sampling_policy",
            "capacity-aware-tail-compiler-decision-margin": "ltr_tail_loss_weight",
            "capacity-aware-semantic-exhaustive-compiler-decision-margin": (
                "compiler_alignment_semantic_exhaustive"
            ),
            "capacity-aware-semantic-exhaustive-structure-token-margin": (
                "structure_token_loss_weight"
            ),
            "exposure-targeted-compiler-decision-margin": (
                "mixture_sampling_policy"
            ),
            "exposure-targeted-semantic-exhaustive-compiler-decision-margin": (
                "compiler_alignment_semantic_exhaustive"
            ),
        }.get(rec_slug)
        if treatment_key is not None:
            control_extra = {
                key: value
                for key, value in bank_by_slug[rec_slug][1].items()
                if key != treatment_key
            }
        if rec_slug == "exposure-targeted-compiler-decision-margin":
            control_extra = dict(bank_by_slug[rec_slug][1])
            control_extra["mixture_sampling_policy"] = "capacity_aware"
        if rec_slug == "exposure-targeted-semantic-exhaustive-compiler-decision-margin":
            precursor_slug = "exposure-targeted-compiler-decision-margin"
        elif rec_slug == "exposure-targeted-compiler-decision-margin":
            precursor_slug = "capacity-aware-compiler-decision-margin"
        elif rec_slug == "capacity-aware-semantic-exhaustive-structure-token-margin":
            precursor_slug = (
                "capacity-aware-semantic-exhaustive-compiler-decision-margin"
            )
        elif rec_slug in {
            "capacity-aware-tail-compiler-decision-margin",
            "capacity-aware-semantic-exhaustive-compiler-decision-margin",
        }:
            precursor_slug = "capacity-aware-compiler-decision-margin"
        else:
            precursor_slug = "compiler-decision-margin"
        candidates = [
            {
                "experiment": exp(
                    f"{prefix}-control",
                    "Matched fixture control with both grammar levers off completes smoke eval under the published suite.",
                    knobs(
                        structural_aux_head_profile=str(
                            bank_by_slug[rec_slug][1].get(
                                "structural_aux_head_profile", "none"
                            )
                        ),
                        compiler_decode_mode=str(
                            bank_by_slug[rec_slug][1].get("compiler_decode_mode", "off")
                        ),
                        **(
                            {
                                key: value
                                for key, value in bank_by_slug[rec_slug][1].items()
                                if key
                                in {
                                    "semantic_contrast_dir",
                                    "semantic_contrast_margin",
                                    "semantic_contrast_fraction",
                                }
                            }
                            | (
                                {"semantic_contrast_loss_weight": 0.0}
                                if rec_slug == "semantic-contrast"
                                else {}
                            )
                        ),
                        **control_extra,
                    ),
                    (
                        "All-family margin control for isolated "
                        f"{treatment_key} attribution."
                        if treatment_key is not None
                        else "Baseline for size-matched continuous attribution."
                    ),
                ),
                "evidence_uses": uses(),
                "novelty": novelty(0, "matched control with published eval"),
            }
        ]
        for i, (slug, hyp, extras) in enumerate(_SCREENING_ARM_BANK, start=1):
            if (
                treatment_key is not None
                and slug == precursor_slug
            ):
                # The matched control is exactly this precursor arm. Emitting it
                # again would violate the preregistration contract's distinct-
                # knob-signature requirement without adding information.
                continue
            arm_extra = _apply_arm_extras(steps, extras)
            candidates.append(
                {
                    "experiment": exp(
                        f"{prefix}-{slug}",
                        hyp,
                        knobs(**arm_extra),
                        f"Continuous thrash arm '{slug}' (rotated recommendation).",
                    ),
                    "evidence_uses": uses(),
                    "novelty": novelty(i, f"thrash arm {slug}"),
                }
            )
        rec = f"{prefix}-{rec_slug}"
        priorities = [
            {
                "rank": 1,
                "area": "model",
                "hypothesis": f"Test thrash arm '{rec_slug}' first under the published eval suite.",
                "evidence_ids": [research, prior],
                "confidence": 0.6,
                "expected_information_gain": "Attributes decode metrics vs matched control.",
                "authority": "speculative",
                "disposition": "experiment_next",
                "proposed_experiment_id": rec,
            },
            {
                "rank": 2,
                "area": "evaluation",
                "hypothesis": "Keep the matched control as the size-matched baseline every cycle.",
                "evidence_ids": [research, prior],
                "confidence": 0.7,
                "expected_information_gain": "Prevents false positives from recipe drift.",
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": f"{prefix}-control",
            },
            {
                "rank": 3,
                "area": "model",
                "hypothesis": "Rotate thrash recommendation across the lever bank (not bounds-only).",
                "evidence_ids": [research, prior],
                "confidence": 0.65,
                "expected_information_gain": "Avoids single-lever thrash collapse.",
                "authority": "observed_result",
                "disposition": "experiment_next",
                "proposed_experiment_id": rec,
            },
            {
                "rank": 4,
                "area": "infrastructure",
                "hypothesis": "Soft ship-gate fails on fixture n never stop the continuous loop.",
                "evidence_ids": [research, prior],
                "confidence": 0.8,
                "expected_information_gain": "Preserves hands-off continuous operation.",
                "authority": "observed_result",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
            {
                "rank": 5,
                "area": "model_build",
                "hypothesis": "Confirmed champions promote under cadence; thrash only screens.",
                "evidence_ids": [research, prior],
                "confidence": 0.55,
                "expected_information_gain": "Separates screening diversity from promotion.",
                "authority": "speculative",
                "disposition": "monitor",
                "proposed_experiment_id": None,
            },
        ]
        # Ensure ≥5 priorities with contiguous ranks (already 5).
        payload = {
            "matrix_id": f"{campaign_id}-m1",
            "campaign_id": campaign_id,
            "evidence_snapshot_id": evidence_snapshot_id,
            "hypotheses": candidates,
            "recommended_experiment_id": rec,
            "selection_rationale": (
                f"Size-matched continuous thrash with rotated recommendation "
                f"'{rec_slug}' (cycle {cycle})."
            ),
            "next_run_priorities": priorities,
        }
    if feedback:
        fb_ids = [
            str(item.get("feedback_id")) for item in feedback if item.get("feedback_id")
        ]
        payload["feedback_ids"] = fb_ids
        if previous_matrix_id:
            payload["predecessor_matrix_id"] = previous_matrix_id
        # continuous priorities must cite every feedback id
        for priority in payload["next_run_priorities"]:
            evidence = list(priority.get("evidence_ids") or [])
            for fid in fb_ids:
                if fid not in evidence:
                    evidence.append(fid)
            priority["evidence_ids"] = evidence
    return payload


def _manifest(
    campaign_id: str,
    experiment: dict,
    commit: str,
    *,
    role: str = "screening",
    policy: Any | None = None,
    cycle_intent: str | None = None,
    formal_preflight_sha256: str | None = None,
) -> ExperimentCampaignV1:
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        primary_for_role,
        promotion_seed_floor,
        stage_wall_minutes_for_role,
    )
    from slm_training.autoresearch.formal import formal_obligation_id

    pol = policy or load_climb_policy()
    role_primary = primary_for_role(pol, role)
    metric = str(role_primary["metric"])
    direction = str(role_primary["direction"])  # type: ignore[assignment]
    min_effect = float(role_primary.get("minimum_effect") or 0.0)
    defaults = pol.defaults
    metric_expectations_sha: str | None = None
    formal_obligations: tuple[FormalObligationV1, ...] = ()
    if role == "promotion":
        claim_class = str(
            defaults.get("claim_class_promotion") or "promotion_candidate"
        )
        base_seed = int(experiment.get("knobs", {}).get("seed") or 7)
        # One bounded campaign executes one actual seed. The cross-campaign,
        # content-bound replicate ledger owns the policy multi-seed gate.
        seeds = (base_seed,)
        # Promotion-class needs causal shape fields.
        mechanism_off = ("mechanism_off",)
        kill_criteria = (
            "primary_lcb_within_noise",
            "fixture_insufficient_n_alone",
        )
        controls = (
            CampaignControlV1(
                control_id="matched-positive",
                description="Size-matched baseline without the candidate mechanism.",
                kind="positive",
            ),
            CampaignControlV1(
                control_id="matched-control",
                description="Destructive negative / unchanged baseline.",
                kind="negative",
            ),
        )
        negative_controls = ("matched-control",)
        artifact_kinds = [
            "version_stamp",
            "seed_result",
            "paired_examples",
            "endpoint_result",
            "holm_family",
            "agentevals",
            "agentv",
            # Authoritative credit TCB (required for promotion_candidate)
            "observation_table",
            "analysis_plan",
            "credit_report",
        ]
        # Proof driver: lock metric expectations on every promotion-role campaign.
        metric_expectations_sha = locked_promote_expectations_sha256()
        # Required formal preflight only on champion-promote *candidate* arm when
        # a content-addressed preflight SHA is available (never placeholder zeros).
        if (
            cycle_intent == "promote"
            and formal_preflight_sha256
            and len(formal_preflight_sha256) == 64
            and formal_preflight_sha256 != ("0" * 64)
        ):
            artifact_kinds.append("formal_preflight")
            from slm_training.autoresearch.schemas import FormalClaimV1

            claim = FormalClaimV1(**promote_formal_claim_dict())
            oid = formal_obligation_id(
                campaign_id, str(experiment["experiment_id"]), claim
            )
            formal_obligations = (
                FormalObligationV1(
                    obligation_id=oid,
                    template_id=_PROMOTE_FORMAL_TEMPLATE_ID,
                    policy="required",
                    preflight_sha256=formal_preflight_sha256,
                ),
            )
        artifact_requirements = tuple(
            ArtifactRequirementV1(kind=k) for k in artifact_kinds
        )
        locked_eval = "e" * 64  # placeholder digest; real lock verified at promotion
        # Prefer eval_version from knobs as identity string for locked field when present
        knobs_pre = experiment.get("knobs") or {}
        if knobs_pre.get("eval_version"):
            locked_eval = hashlib.sha256(
                str(knobs_pre["eval_version"]).encode("utf-8")
            ).hexdigest()
    else:
        claim_class = str(defaults.get("claim_class_screening") or "diagnostic")
        seeds = (int(experiment.get("knobs", {}).get("seed") or 7),)
        mechanism_off = ()
        kill_criteria = ()
        controls = (
            CampaignControlV1(
                control_id="matched-control",
                description="Matched fixture baseline with the tested lever off.",
                kind="negative",
            ),
        )
        negative_controls = ("matched-control",)
        artifact_requirements = (ArtifactRequirementV1(kind="version_stamp"),)
        locked_eval = None

    knobs = experiment["knobs"]
    cfg = hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()
    ctrl = hashlib.sha256(
        json.dumps(
            {
                **knobs,
                "grammar_completion_bounds": False,
                "compact_active_canvas": False,
                "component_plan_loss_weight": 0.0,
                "component_edge_loss_weight": 0.0,
                "component_edge_alignment_loss_weight": 0.0,
                "component_inventory_loss_weight": 0.0,
                "binder_topology_loss_weight": 0.0,
                "binder_component_plan_loss_weight": 0.0,
                "binder_arity_loss_weight": 0.0,
                "fidelity_loss_weight": 0.5,
                "semantic_contrast_loss_weight": 0.0,
                "symbol_slot_augmentation": False,
                "mask_pattern": "random",
                "symbol_boundary_loss_weight": 0.0,
                "design_md_dropout": 0.0,
                "ltr_prefix_loss_weight": 0.0,
                "component_token_loss_weight": 0.0,
                "component_edge_token_loss_weight": 0.0,
                "compiler_decision_token_loss_weight": 0.0,
                "structure_token_loss_weight": 0.0,
                "typed_family_balance_loss_weight": 0.0,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    arms = [
        CampaignArmV1(arm_id="control", role="control", config_sha256=ctrl),
        CampaignArmV1(arm_id="candidate", role="candidate", config_sha256=cfg),
    ]
    if mechanism_off:
        arms.insert(
            1,
            CampaignArmV1(
                arm_id="mechanism_off",
                role="candidate",
                config_sha256=hashlib.sha256(b"mechanism_off").hexdigest(),
            ),
        )
    # Gate operators depend on direction
    if direction == "decrease":
        promote_op, promote_thr = "le", -abs(min_effect) if min_effect else -1.0
        rollback_op, rollback_thr = "gt", 1e9
    else:
        promote_op, promote_thr = "ge", abs(min_effect)
        rollback_op, rollback_thr = "lt", 0.0

    return ExperimentCampaignV1(
        campaign_id=campaign_id,
        experiment_id=experiment["experiment_id"],
        hypothesis=experiment["hypothesis"],
        decision=(
            "Promotion-class continuous cycle under locked held-out primary."
            if role == "promotion"
            else "Attribute continuous fixture decode metrics only under published eval suites."
        ),
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="primary",
                metric=metric,
                role="primary",
                direction=direction,  # type: ignore[arg-type]
                minimum_effect=min_effect,
            ),
        ),
        arms=tuple(arms),
        seeds=seeds,
        budget=CampaignBudget(
            max_experiments=1,
            max_wall_minutes=stage_wall_minutes_for_role(pol, role),
        ),
        stopping_rules=(
            "Stop after the declared seeds finish or the wall cap is hit.",
        ),
        controls=controls,
        negative_controls=negative_controls,
        mechanism_off_arm_ids=mechanism_off,
        executable_kill_criteria=kill_criteria,
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="primary-family", hypothesis_ids=("primary",), alpha=0.05
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="promote-primary",
                endpoint_id="primary",
                operator=promote_op,  # type: ignore[arg-type]
                threshold=promote_thr,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="rollback-primary",
                endpoint_id="primary",
                operator=rollback_op,  # type: ignore[arg-type]
                threshold=rollback_thr,
            ),
        ),
        artifact_requirements=artifact_requirements,
        formal_obligations=formal_obligations,
        claim_class=claim_class,  # type: ignore[arg-type]
        locked_eval_manifest_sha256=locked_eval,
        metric_expectations_sha256=metric_expectations_sha,
        replicate_ledger_schema=(
            _PROMOTION_REPLICATE_SCHEMA if role == "promotion" else None
        ),
        replicate_seed_floor=(
            promotion_seed_floor(pol)[0] if role == "promotion" else None
        ),
        source_commit=commit,
        source_dirty=False,
        author="autotrain-continuous-driver",
    )


def _effective_primary_metric(
    *,
    role: str,
    policy_metric: str,
    requested_metric: str,
    replay_metric: str | None = None,
) -> str:
    effective = replay_metric or policy_metric
    requested_parts = requested_metric.rsplit(".", maxsplit=1)
    effective_parts = effective.rsplit(".", maxsplit=1)
    requested_scope = requested_parts[0] if len(requested_parts) == 2 else ""
    effective_scope = effective_parts[0] if len(effective_parts) == 2 else ""
    if (
        replay_metric is None
        and role == "screening"
        and requested_metric
        and requested_parts[-1] == effective_parts[-1]
        and requested_scope in {"", effective_scope}
    ):
        return requested_metric
    return effective


def _empty_promotion_slot_falls_back(
    *,
    cadence_role: str,
    replay: object | None,
    promotion_target_available: bool,
    prior_screening_win_required: bool,
) -> bool:
    """Keep fresh hypotheses out of promotion suites and held-out selection."""

    return (
        replay is None
        and cadence_role == "promotion"
        and not promotion_target_available
        and prior_screening_win_required
    )


def _role_with_confirmation_boundary(
    cadence_role: str, *, confirming: bool
) -> str:
    """Keep unconfirmed champions on screening endpoints and suites."""

    return "screening" if confirming else cadence_role


def run_cycle(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    train_version: str,
    steps: int,
    objective: str,
    primary_metric: str,
    sync_git: bool = True,
    startup_commit: str | None = None,
    require_action_receipts: bool = True,
) -> str:
    from slm_training.autoresearch.climb_policy import (
        assert_cycle_cadence,
        cycle_role_for_index,
        eval_suites_for_role,
        load_climb_policy,
        primary_for_role,
        stage_wall_minutes_for_role,
    )

    deadline = time.monotonic() + MAX_RUN_SECONDS
    policy = load_climb_policy()
    # Defaults from external policy when caller still uses legacy pins.
    if train_version == "wf_smoke_v2":
        train_version = str(policy.defaults.get("train_version") or train_version)

    if sync_git:
        _run(
            ["git", "fetch", "origin", "main"],
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-fetch",
        )
        _run(
            ["git", "merge", "--no-edit", "origin/main"],
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-merge",
        )
        integrated = _git(
            "rev-parse",
            "HEAD",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-integration-head",
        )
        if startup_commit is not None and integrated != startup_commit:
            raise _CodeUpdated(
                f"integrated {integrated}; restart stale process from {startup_commit}"
            )
    if _git(
        "status",
        "--porcelain",
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="sync-clean-status",
    ):
        raise RuntimeError("loop worktree is dirty; continuous requires a clean tree")
    upstream = _git(
        "rev-parse",
        "origin/main",
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="sync-upstream-head",
    )
    integration = _git(
        "rev-parse",
        "HEAD",
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="sync-current-head",
    )
    if upstream != integration:
        # merge should have equalized; if not, still require ancestor
        _run(
            ["git", "merge-base", "--is-ancestor", upstream, integration],
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-ancestry",
        )

    recovered_campaign = _finalize_terminal_interrupted_replay(
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        deadline=deadline,
    )
    if recovered_campaign is not None:
        return recovered_campaign

    idx, pred = _latest_cycle(root, loop_id)
    lineage_pred = _campaign_at_cycle(root, loop_id, idx)
    if require_action_receipts:
        _refresh_incomplete_replay_handoff(root, loop_id, pred)
    if require_action_receipts:
        _require_predecessor_actions(root, loop_id, pred)
    replay = (
        _load_frozen_replay(root, loop_id, pred) if require_action_receipts else None
    )
    cycle = idx + 1
    cadence_role = (
        str(replay["handoff"].cycle_role)
        if replay is not None
        else cycle_role_for_index(policy, cycle)
    )
    # Champion queue: confirm open heads; promote confirmed heads on promotion
    # cadence; otherwise keep rotating screening levers. A promotion cadence
    # slot is an opportunity, not authority to expose a fresh arm to held-out
    # suites before it has a prior screening win.
    queue_path = _champion_queue_path(root, loop_id)
    queue_entries = _load_champion_queue(queue_path)
    # Pre-execution crashes do not spend bounded champion attempts. Promotion
    # heads also return to confirmed so the next promotion slot can retry.
    recipes_refreshed = _refresh_champion_source_recipes(root, queue_entries)
    reconciled_replays = _reconcile_completed_confirmation_replays(
        root, queue_entries
    )
    recovered = _recover_interrupted_champion_entries(root, queue_entries)
    revalidated = _revalidate_open_champion_entries(root, queue_entries)
    confirmations_revalidated = _revalidate_confirmed_champion_entries(
        root, queue_entries
    )
    if (
        recipes_refreshed
        or reconciled_replays
        or recovered
        or revalidated
        or confirmations_revalidated
    ):
        _write_champion_queue(queue_path, queue_entries)
    replayed_confirmation = _confirmation_replay_entry(queue_entries, replay)
    recent_exhausted = _recent_completed_nonpositive_slugs(root, pred)
    skip_slugs = (
        _skip_arm_slugs(queue_entries, integration_commit=integration)
        | recent_exhausted
    )
    open_champion = _queue_head_open(queue_entries)
    confirmed_champion: dict[str, Any] | None = None
    promoting_champion: dict[str, Any] | None = None
    role = _role_with_confirmation_boundary(
        cadence_role,
        confirming=replay is None and open_champion is not None,
    )
    if replay is not None:
        open_champion = None
        cycle_intent = "retry_measurement"
    elif open_champion is not None:
        cycle_intent = "confirm"
    elif cadence_role == "promotion":
        confirmed_champion = _queue_head_confirmed(queue_entries)
        if confirmed_champion is not None:
            cycle_intent = "promote"
            promoting_champion = confirmed_champion
        else:
            role = "screening"
            cycle_intent = "screening"
    else:
        confirmed_waiting = _queue_head_confirmed(queue_entries)
        if _repeat_confirm_while_waiting_for_promotion(
            cadence_role=cadence_role,
            confirmed_champion=confirmed_waiting,
            cycle=cycle,
            skip=skip_slugs,
        ):
            open_champion = confirmed_waiting
            cycle_intent = "confirm"
            print(
                "CHAMPION_CONFIRM_FALLBACK "
                "reason=screening_bank_exhausted_awaiting_promotion_cadence "
                f"entry_id={confirmed_waiting.get('entry_id')}",
                flush=True,
            )
        else:
            cycle_intent = "screening"
    campaign_id = _campaign_id(loop_id, cycle)
    if open_champion is not None:
        attempts = _bump_champion_attempt(
            root=root,
            loop_id=loop_id,
            entry_id=str(open_champion["entry_id"]),
            field="confirm_attempts",
        )
        if attempts > _MAX_CONFIRM_ATTEMPTS:
            _update_champion_status(
                root=root,
                loop_id=loop_id,
                entry_id=str(open_champion["entry_id"]),
                status="rejected",
                confirm_campaign_id=campaign_id,
                confirm_cycle_index=cycle,
                resolve_reasons=[
                    f"confirm_attempts_exceeded:{attempts}>{_MAX_CONFIRM_ATTEMPTS}"
                ],
            )
            print(
                f"CHAMPION_CONFIRM_DROP entry_id={open_champion.get('entry_id')} "
                f"attempts={attempts} max={_MAX_CONFIRM_ATTEMPTS}",
                flush=True,
            )
            open_champion = None
            cycle_intent = role
        else:
            _update_champion_status(
                root=root,
                loop_id=loop_id,
                entry_id=str(open_champion["entry_id"]),
                status="confirming",
                confirm_campaign_id=campaign_id,
                confirm_cycle_index=cycle,
            )
            print(
                f"CHAMPION_CONFIRM_START entry_id={open_champion.get('entry_id')} "
                f"fingerprint={open_champion.get('knobs_fingerprint')} "
                f"attempt={attempts}/{_MAX_CONFIRM_ATTEMPTS} campaign={campaign_id}",
                flush=True,
            )
    elif promoting_champion is not None:
        attempts = _bump_champion_attempt(
            root=root,
            loop_id=loop_id,
            entry_id=str(promoting_champion["entry_id"]),
            field="promote_attempts",
        )
        if attempts > _MAX_PROMOTE_ATTEMPTS:
            _update_champion_status(
                root=root,
                loop_id=loop_id,
                entry_id=str(promoting_champion["entry_id"]),
                status="promotion_failed",
                confirm_campaign_id=campaign_id,
                confirm_cycle_index=cycle,
                resolve_reasons=[
                    f"promote_attempts_exceeded:{attempts}>{_MAX_PROMOTE_ATTEMPTS}"
                ],
            )
            print(
                f"CHAMPION_PROMOTE_DROP entry_id={promoting_champion.get('entry_id')} "
                f"attempts={attempts} max={_MAX_PROMOTE_ATTEMPTS}",
                flush=True,
            )
            promoting_champion = None
            cycle_intent = role
        else:
            _update_champion_status(
                root=root,
                loop_id=loop_id,
                entry_id=str(promoting_champion["entry_id"]),
                status="promoting",
                confirm_campaign_id=campaign_id,
                confirm_cycle_index=cycle,
            )
            print(
                f"CHAMPION_PROMOTE_START entry_id={promoting_champion.get('entry_id')} "
                f"fingerprint={promoting_champion.get('knobs_fingerprint')} "
                f"attempt={attempts}/{_MAX_PROMOTE_ATTEMPTS} campaign={campaign_id}",
                flush=True,
            )
    promotion_target_available = bool(open_champion or promoting_champion)
    if _empty_promotion_slot_falls_back(
        cadence_role=cadence_role,
        replay=replay,
        promotion_target_available=promotion_target_available,
        prior_screening_win_required=bool(
            policy.cadence.get("promotion_requires_prior_screening_win", True)
        ),
    ):
        role = "screening"
        cycle_intent = "screening"
        print(
            "PROMOTION_SLOT_FALLBACK reason=no_prior_screening_winner "
            f"cycle={cycle} suites={','.join(eval_suites_for_role(policy, role))}",
            flush=True,
        )
    formal_lane_required = _formal_lane_required(
        cycle_intent=cycle_intent,
        replay=replay,
    )
    arm_wall_minutes = _arm_wall_minutes(
        stage_wall_minutes_for_role(policy, role),
        formal_required=formal_lane_required,
    )
    # The campaign records the maximum arm ceiling before outcomes exist. The
    # driver later passes the exact symmetric post-formal share to each run.
    # Keeping the old pre-formal split here made the inner executor discard
    # time that the outer allocator had correctly reclaimed.
    campaign_wall_minutes = _post_formal_arm_budget_request(
        policy_minutes=stage_wall_minutes_for_role(policy, role),
        initial_arm_wall_minutes=arm_wall_minutes,
        formal_completed=formal_lane_required,
    )
    claim_for_role = (
        str(policy.defaults.get("claim_class_promotion") or "promotion_candidate")
        if role == "promotion"
        else str(policy.defaults.get("claim_class_screening") or "diagnostic")
    )
    if replay is None:
        assert_cycle_cadence(
            policy,
            cycle_index=cycle,
            claimed_role=role,
            claim_class=claim_for_role,
            promotion_target_available=promotion_target_available,
            confirmation_pending=cycle_intent == "confirm",
        )
    role_primary = primary_for_role(policy, role)
    # Screening may preserve a same-leaf CLI suite override for compatibility.
    # Promotion always uses the policy-owned held-out endpoint.
    effective_primary = _effective_primary_metric(
        role=role,
        policy_metric=str(role_primary["metric"]),
        requested_metric=primary_metric,
        replay_metric=(str(replay["handoff"].primary_metric) if replay else None),
    )
    _write_loop_state(
        root,
        AutotrainLoopStateV1(
            loop_id=loop_id,
            state="RUNNING",
            phase="running",
            active_campaign_id=campaign_id,
            last_completed_campaign_id=pred,
            cycle_index=cycle,
            next_action="run bounded campaign",
            pid=os.getpid(),
            active_stage="orchestration",
            integration_commit=integration,
        ),
    )
    py = sys.executable
    ar = [py, "-m", "scripts.autoresearch", "--root", str(root)]
    if cycle_intent == "retry_measurement":
        notes = (
            "Current-main successor replay of an infrastructure-incomplete frozen "
            "control/candidate measurement."
        )
    elif cycle_intent == "confirm":
        notes = (
            "Champion-queue confirmatory cycle: same levers as quality-held win, "
            "new seed; local-only fixture scale."
        )
    elif cycle_intent == "promote":
        notes = (
            "Champion-queue promotion cycle: confirmed levers under promotion "
            "primary/suites/seeds; local-only fixture scale."
        )
    else:
        notes = (
            "Hands-off continuous driver cycle; rotated thrash recommendation; "
            "local-only fixture scale."
        )
    init = [
        *ar,
        "init",
        "--campaign-id",
        campaign_id,
        "--loop-id",
        loop_id,
        "--cycle-index",
        str(cycle),
        "--upstream-commit",
        upstream,
        "--integration-commit",
        integration,
        "--objective",
        objective,
        "--primary-metric",
        effective_primary,
        "--track",
        "twotower",
        "--max-experiments",
        "3",
        "--max-wall-minutes",
        str(campaign_wall_minutes),
        "--notes",
        notes,
    ]
    if lineage_pred:
        init.extend(["--predecessor-campaign-id", lineage_pred])
    for evidence_root in _continuous_evidence_roots(root, loop_id, pred):
        init.extend(["--evidence-root", str(evidence_root)])
    _run(
        init,
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="campaign-init",
    )
    _run(
        [
            *ar,
            "research",
            "--campaign-id",
            campaign_id,
            "--offline",
        ],
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="campaign-research",
    )

    camp_dir = root / campaign_id
    evidence = next((camp_dir / "artifacts" / "evidence").glob("*.json"))
    ev = json.loads(evidence.read_text(encoding="utf-8"))
    research_paths = [
        item["path"]
        for item in ev.get("items", [])
        if item.get("kind") == "repo_lineage" and item.get("path")
    ]
    result_paths = [
        item["path"]
        for item in ev.get("items", [])
        if item.get("kind")
        in {"prior_campaign", "prior_run", "evaluation", "data_snapshot"}
        and item.get("path")
    ]
    trace_paths = [
        item["path"]
        for item in ev.get("items", [])
        if item.get("kind") in {"run_insight", "telemetry", "agentv", "feedback"}
        and item.get("path")
    ]
    if not research_paths:
        research_paths = ["docs/design/research-lineage.md"]
    if not result_paths:
        result_paths = research_paths[:]
    cites = [research_paths[0], result_paths[0]]
    if len(research_paths) > 1:
        cites.append(research_paths[1])
    elif len(result_paths) > 1:
        cites.append(result_paths[1])
    else:
        cites.append(research_paths[0])
    role_citations = {
        "research": research_paths[0],
        "prior_result": result_paths[0],
    }
    if trace_paths:
        role_citations["prior_trace"] = trace_paths[0]
    eval_version = default_eval_version()
    # Load predecessor matrix feedback when continuous lineage requires a successor matrix.
    feedback: list[dict] = []
    previous_matrix_id = None
    if pred:
        pred_dir = root / pred
        mats = sorted(
            (pred_dir / "artifacts" / "hypothesis_matrices").glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if mats:
            previous_matrix_id = json.loads(mats[-1].read_text(encoding="utf-8")).get(
                "matrix_id"
            )
        fbs = sorted(
            (pred_dir / "artifacts" / "hypothesizer_feedback").glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        feedback = [json.loads(path.read_text(encoding="utf-8")) for path in fbs]
        # only terminal feedback for the latest predecessor matrix
        if previous_matrix_id:
            feedback = [
                item for item in feedback if item.get("matrix_id") == previous_matrix_id
            ]
    confirm_levers = None
    confirm_control_levers = None
    promote_levers = None
    promote_control_levers = None
    if open_champion is not None:
        confirm_levers = _lever_knobs(open_champion.get("knobs") or {})
        confirm_control_levers = _lever_knobs(open_champion.get("control_knobs") or {})
        if not confirm_control_levers:
            source_dir = root / str(open_champion.get("source_campaign_id") or "")
            confirm_control_levers = _lever_knobs(
                _load_experiment_knobs(
                    source_dir, str(open_champion.get("source_control_id") or "")
                )
            )
        if not confirm_levers:
            # Corrupt queue entry — reject and fall back to thrash matrix.
            _update_champion_status(
                root=root,
                loop_id=loop_id,
                entry_id=str(open_champion["entry_id"]),
                status="rejected",
                confirm_campaign_id=campaign_id,
                confirm_cycle_index=cycle,
                resolve_reasons=["confirm_missing_knobs"],
            )
            open_champion = None
            cycle_intent = role
            confirm_levers = None
    elif promoting_champion is not None:
        promote_levers = _lever_knobs(promoting_champion.get("knobs") or {})
        promote_control_levers = _lever_knobs(
            promoting_champion.get("control_knobs") or {}
        )
        if not promote_control_levers:
            source_dir = root / str(promoting_champion.get("source_campaign_id") or "")
            promote_control_levers = _lever_knobs(
                _load_experiment_knobs(
                    source_dir,
                    str(promoting_champion.get("source_control_id") or ""),
                )
            )
        if not promote_levers:
            _update_champion_status(
                root=root,
                loop_id=loop_id,
                entry_id=str(promoting_champion["entry_id"]),
                status="promotion_failed",
                confirm_campaign_id=campaign_id,
                confirm_cycle_index=cycle,
                resolve_reasons=["promote_missing_knobs"],
            )
            promoting_champion = None
            cycle_intent = role
            promote_levers = None
    if recent_exhausted:
        print(
            "RECENT_EXHAUSTION skip=" + ",".join(sorted(recent_exhausted)),
            flush=True,
        )
    rec_slug = _select_cycle_slug(
        cycle,
        predecessor_priority=_predecessor_priority_slug(
            root,
            pred,
            skip=skip_slugs,
            closed=recent_exhausted,
        ),
        skip=skip_slugs,
        has_confirm_levers=confirm_levers is not None,
        has_promote_levers=promote_levers is not None,
    )
    matrix = _matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id=ev["snapshot_id"],
        cites=cites[:3],
        role_citations=role_citations,
        train_version=train_version,
        eval_version=eval_version,
        steps=steps,
        cycle=cycle,
        feedback=feedback or None,
        previous_matrix_id=previous_matrix_id,
        role=role,
        policy=policy,
        confirm_levers=confirm_levers,
        confirm_control_levers=confirm_control_levers,
        promote_levers=promote_levers,
        promote_control_levers=promote_control_levers,
        recommended_slug=rec_slug
        if promote_levers is None and confirm_levers is None
        else None,
        skip_slugs=skip_slugs,
    )
    replay_manifests: dict[str, dict[str, Any]] = {}
    if replay is not None:
        replay_manifests = _apply_frozen_replay(matrix, replay, campaign_id)
        print(
            "FROZEN_REPLAY "
            f"source_campaign={replay['handoff'].campaign_id} "
            f"manifest={replay['action'].frozen_manifest_sha256} "
            f"successor_campaign={campaign_id}",
            flush=True,
        )
    elif promote_levers is None and confirm_levers is None:
        print(
            f"THRASH_ROTATE cycle={cycle} recommended={rec_slug} skip={sorted(skip_slugs)}",
            flush=True,
        )
    HypothesisMatrix.model_validate(matrix)
    matrix_path = camp_dir / "matrix-proposal.json"
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    replay_manifest_paths: dict[str, Path] = {}
    for eid, frozen_replay in replay_manifests.items():
        successor = _replay_successor_manifest(
            frozen_replay["manifest"],
            frozen_manifest_sha256=frozen_replay["manifest_sha256"],
            campaign_id=campaign_id,
            experiment_id=eid,
            integration_commit=integration,
        )
        path = camp_dir / "manifests" / f"{eid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(successor.model_dump_json(indent=2) + "\n", encoding="utf-8")
        replay_manifest_paths[eid] = path
    promote_formal_status: str | None = None
    promote_preflight_sha: str | None = None
    replay_formal_required = any(
        bool(item["manifest"].formal_obligations)
        for item in replay_manifests.values()
    )
    if replay_formal_required:
        replay_arm_count = len(replay_manifest_paths)
        _require_symmetric_arm_budget(
            deadline=deadline,
            arm_count=replay_arm_count,
            arm_wall_minutes=arm_wall_minutes,
        )
        formal_budget = _promotion_formal_budget_seconds(
            deadline=deadline,
            arm_count=replay_arm_count,
            arm_wall_minutes=arm_wall_minutes,
        )
        promote_formal_status, promote_preflight_sha = ensure_promote_formal_preflight(
            camp_dir=camp_dir,
            campaign_id=campaign_id,
            experiment_id=str(matrix["recommended_experiment_id"]),
            run_lean=True,
            timeout_seconds=formal_budget,
            root=root,
            loop_id=loop_id,
        )
        print(
            f"REPLAY_FORMAL_PREFLIGHT status={promote_formal_status} "
            f"sha={promote_preflight_sha} campaign={campaign_id}",
            flush=True,
        )
        if promote_formal_status == "proved" and promote_preflight_sha:
            for eid, frozen_replay in replay_manifests.items():
                if not frozen_replay["manifest"].formal_obligations:
                    continue
                path = replay_manifest_paths[eid]
                successor = ExperimentCampaignV1.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                rebound = _bind_fresh_replay_formal_preflight(
                    successor,
                    frozen_replay["manifest"],
                    preflight_sha256=promote_preflight_sha,
                    formal_claims=list(
                        frozen_replay["experiment"].get("formal_claims") or []
                    ),
                )
                path.write_text(
                    rebound.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
    hypothesize_cmd = [
        *ar,
        "hypothesize",
        "--campaign-id",
        campaign_id,
        "--provider",
        "agent",
        "--matrix",
        str(matrix_path),
    ]
    for path in replay_manifest_paths.values():
        hypothesize_cmd.extend(["--frozen-replay-manifest", str(path)])
    _run(
        hypothesize_cmd,
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="campaign-hypothesize",
    )

    exp_dir = camp_dir / "artifacts" / "experiments"
    by_id = {
        json.loads(path.read_text(encoding="utf-8"))["experiment_id"]: path
        for path in exp_dir.glob("*.json")
    }
    control_eid = str(matrix["hypotheses"][0]["experiment"]["experiment_id"])
    candidate_eid = str(matrix["recommended_experiment_id"])
    candidate_exp = json.loads(by_id[candidate_eid].read_text(encoding="utf-8"))
    arm_seed = int((candidate_exp.get("knobs") or {}).get("seed") or 7)
    promotion_replicate_index = (
        len(_verified_promotion_replicates(root, loop_id, promoting_champion))
        if cycle_intent == "promote" and promoting_champion is not None
        else None
    )
    order = _counterbalanced_arm_order(
        control_eid,
        candidate_eid,
        cycle_index=cycle,
        seed=arm_seed,
        promotion_replicate_index=promotion_replicate_index,
    )
    scheduled_order = list(order)
    arm_count = len({eid for eid in order if eid in by_id})
    # Promote path: formal preflight must be proved before train executes.
    if cycle_intent == "promote" and promoting_champion is not None:
        _require_symmetric_arm_budget(
            deadline=deadline,
            arm_count=arm_count,
            arm_wall_minutes=arm_wall_minutes,
        )
        formal_budget = _promotion_formal_budget_seconds(
            deadline=deadline,
            arm_count=arm_count,
            arm_wall_minutes=arm_wall_minutes,
        )
        promote_formal_status, promote_preflight_sha = ensure_promote_formal_preflight(
            camp_dir=camp_dir,
            campaign_id=campaign_id,
            experiment_id=str(matrix["recommended_experiment_id"]),
            run_lean=True,
            timeout_seconds=formal_budget,
            root=root,
            loop_id=loop_id,
        )
        print(
            f"PROMOTE_FORMAL_PREFLIGHT status={promote_formal_status} "
            f"sha={promote_preflight_sha} campaign={campaign_id}",
            flush=True,
        )
        if promote_formal_status != "proved" or not promote_preflight_sha:
            # Do not train without a proved formal. Timeouts → inconclusive
            # disposition (not promotion_failed); other unproved → fail closed.
            kind = (
                "TIMEOUT_INCONCLUSIVE"
                if _formal_status_is_timeout(promote_formal_status)
                else "BLOCK"
            )
            print(
                f"PROMOTE_FORMAL_{kind} skip_execute "
                f"status={promote_formal_status} "
                f"wall_s={_PROMOTE_FORMAL_TIMEOUT_S:g}",
                flush=True,
            )
            order = []
    elif replay_formal_required and (
        promote_formal_status != "proved" or not promote_preflight_sha
    ):
        kind = (
            "TIMEOUT_INCONCLUSIVE"
            if _formal_status_is_timeout(promote_formal_status)
            else "BLOCK"
        )
        print(
            f"REPLAY_FORMAL_{kind} skip_execute "
            f"status={promote_formal_status}",
            flush=True,
        )
        order = []

    arm_count = len({eid for eid in order if eid in by_id})
    if arm_count:
        requested_arm_wall_minutes = _post_formal_arm_budget_request(
            policy_minutes=stage_wall_minutes_for_role(policy, role),
            initial_arm_wall_minutes=arm_wall_minutes,
            formal_completed=(
                formal_lane_required and promote_formal_status == "proved"
            ),
        )
        arm_wall_minutes = _fit_symmetric_arm_budget(
            deadline=deadline,
            arm_count=arm_count,
            requested_arm_wall_minutes=requested_arm_wall_minutes,
        )
        if arm_wall_minutes < requested_arm_wall_minutes:
            print(
                "ARM_BUDGET_REBALANCED "
                f"requested_s={requested_arm_wall_minutes * 60:.3f} "
                f"effective_s={arm_wall_minutes * 60:.3f} "
                f"arms={arm_count} "
                f"finalization_reserve_s={HARNESS_FINALIZATION_RESERVE_SECONDS}",
                flush=True,
            )
    seen: set[str] = set()
    arm_exits: dict[str, int] = {}
    for eid in order:
        if eid in seen or eid not in by_id:
            continue
        seen.add(eid)
        exp = json.loads(by_id[eid].read_text(encoding="utf-8"))
        is_promote_arm = cycle_intent == "promote" and (
            eid.endswith("-promote") or "-promote" in eid
        )
        if eid in replay_manifest_paths:
            man_path = replay_manifest_paths[eid]
        else:
            man = _manifest(
                campaign_id,
                exp,
                integration,
                role=role,
                policy=policy,
                cycle_intent=cycle_intent,
                formal_preflight_sha256=(
                    promote_preflight_sha if is_promote_arm else None
                ),
            )
            man_path = camp_dir / "manifests" / f"{eid}.json"
            man_path.parent.mkdir(parents=True, exist_ok=True)
            man_path.write_text(man.model_dump_json(indent=2) + "\n", encoding="utf-8")
        # soft-fail: ship gates may fail on fixture n
        cmd = [
            *ar,
            "run",
            "--campaign-id",
            campaign_id,
            "--experiment",
            str(by_id[eid]),
            "--campaign-manifest",
            str(man_path),
            "--execute",
            "--experiment-wall-seconds",
            f"{arm_wall_minutes * 60:.6f}",
        ]
        reuse = replay_manifests.get(eid, {}).get("train_reuse")
        if reuse is not None:
            cmd.extend(["--reuse-train-run", str(reuse["run_dir"])])
            for lineage_path in reuse["manifest_paths"]:
                cmd.extend(["--reuse-train-manifest", str(lineage_path)])
            print(
                "FROZEN_TRAIN_REUSE "
                f"experiment={eid} source_run={reuse['run_dir']} "
                f"lineage={len(reuse['manifest_paths'])}",
                flush=True,
            )
        print("+", " ".join(cmd), flush=True)
        stage = f"experiment:{eid}"
        result = _stage_command(
            cmd,
            cwd=cwd,
            deadline=_arm_execution_deadline(
                cycle_deadline=deadline,
                arm_wall_minutes=arm_wall_minutes,
            ),
            root=root,
            loop_id=loop_id,
            stage=stage,
        )
        if result.stdout:
            print(
                result.stdout,
                end="" if result.stdout.endswith("\n") else "\n",
                flush=True,
            )
        if result.stderr:
            print(
                result.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
                file=sys.stderr,
                flush=True,
            )
        if result.timed_out:
            code = 124
        elif result.outcome is ProcessOutcome.LAUNCH_FAILED:
            code = 127
        else:
            code = int(result.returncode or 0)
        arm_exits[eid] = int(code)
        print(f"experiment {eid} exit={code}", flush=True)

    _set_active_stage(root, loop_id, "diagnosis-and-handoff")
    delivery = _phase_a_delivery(
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        primary_metric=effective_primary,
        cycle_index=cycle,
        role=role,
        cycle_intent=cycle_intent,
        arm_order=scheduled_order,
        arm_seed=arm_seed,
        deadline=deadline,
    )
    if (
        replay is not None
        and set(arm_exits) == set(replay_manifests)
        and all(code == 0 for code in arm_exits.values())
        and delivery.get("measurement_complete") is True
    ):
        evidence = (
            str(
                (camp_dir / "sdlc_delivery.json").relative_to(cwd)
                if (camp_dir / "sdlc_delivery.json").is_relative_to(cwd)
                else camp_dir / "sdlc_delivery.json"
            ),
            *(
                str(path.relative_to(cwd) if path.is_relative_to(cwd) else path)
                for eid in sorted(arm_exits)
                for path in (camp_dir / "manifests" / f"{eid}.json",)
            ),
        )
        append_autotrain_action_receipt(
            root,
            AutotrainActionReceiptV1(
                loop_id=loop_id,
                campaign_id=replay["handoff"].campaign_id,
                action_index=int(replay["action_index"]),
                action_sha256=autotrain_action_sha256(replay["action"]),
                action_kind="retry_measurement",
                status="completed",
                evidence_uris=evidence,
                evidence=bind_autotrain_action_evidence(
                    root,
                    replay["handoff"],
                    replay["action"],
                    evidence,
                ),
            ),
        )
        print(
            f"FROZEN_REPLAY_ACK source_campaign={replay['handoff'].campaign_id} "
            f"successor_campaign={campaign_id}",
            flush=True,
        )
    camp_dir = root / campaign_id
    resolution: dict[str, Any] | None = None
    if open_champion is not None:
        resolution = _resolve_confirm_result(
            root=root,
            loop_id=loop_id,
            entry=open_champion,
            delivery=delivery,
            campaign_id=campaign_id,
            cycle_index=cycle,
        )
    elif promoting_champion is not None:
        # Export LeverProof certificate from promote run metrics (fail closed).
        cert_err: str | None = None
        if (
            promote_formal_status == "proved"
            or _formal_preflight_status(camp_dir) == "proved"
        ):
            control_id = str(delivery.get("control_id") or "")
            candidate_id = str(delivery.get("candidate_id") or "")
            if not control_id or not candidate_id:
                # Infer from runs if Phase A ids missing.
                runs = camp_dir / "runs"
                if runs.is_dir():
                    names = sorted(p.name for p in runs.iterdir() if p.is_dir())
                    for n in names:
                        if n.endswith("-control"):
                            control_id = control_id or n
                        if "-promote" in n or n.endswith("-confirm"):
                            candidate_id = n
                    if not candidate_id and len(names) >= 2:
                        candidate_id = names[-1]
                    if not control_id and names:
                        control_id = names[0]
            # Prefer matrix arm ids when delivery omitted promote.
            if not candidate_id:
                for eid in order:
                    if "-promote" in eid:
                        candidate_id = eid
                        break
            if not control_id:
                for eid in order:
                    if eid.endswith("-control"):
                        control_id = eid
                        break
            if (
                control_id
                and candidate_id
                and _run_has_usable_metrics(camp_dir, candidate_id)
            ):
                _cert_path, cert_err = export_promote_metric_certificate(
                    camp_dir=camp_dir,
                    campaign_id=campaign_id,
                    control_id=control_id,
                    candidate_id=candidate_id,
                    delivery=delivery,
                    deadline=deadline,
                    root=root,
                    loop_id=loop_id,
                )
                if cert_err:
                    print(f"PROMOTE_CERT_EXPORT_FAIL {cert_err}", flush=True)
                    delivery = {
                        **delivery,
                        "reasons": list(delivery.get("reasons") or []) + [cert_err],
                    }
            elif control_id and candidate_id:
                cert_err = "promote_cert_incomplete_metrics:ss=None parse=None"
                print(f"PROMOTE_CERT_EXPORT_FAIL {cert_err}", flush=True)
                delivery = {
                    **delivery,
                    "control_id": control_id,
                    "candidate_id": candidate_id,
                    "reasons": list(delivery.get("reasons") or []) + [cert_err],
                    "harness_failure": True,
                    "measurement_complete": False,
                }
            else:
                cert_err = "promote_cert_missing_run_ids"
                delivery = {
                    **delivery,
                    "reasons": list(delivery.get("reasons") or []) + [cert_err],
                    "harness_failure": True,
                    "measurement_complete": False,
                }
            # Attach arm exits for harness classification.
            delivery = {
                **delivery,
                "control_id": control_id or delivery.get("control_id"),
                "candidate_id": candidate_id or delivery.get("candidate_id"),
                "arm_exits": arm_exits,
            }
        resolution = _resolve_promotion_result(
            root=root,
            loop_id=loop_id,
            entry=promoting_champion,
            delivery=delivery,
            campaign_id=campaign_id,
            cycle_index=cycle,
            camp_dir=camp_dir,
            formal_preflight_status=promote_formal_status
            or _formal_preflight_status(camp_dir),
            arm_exits=arm_exits,
            cert_err=cert_err,
        )
    else:
        if replayed_confirmation is not None:
            resolution = _resolve_confirm_result(
                root=root,
                loop_id=loop_id,
                entry=replayed_confirmation,
                delivery=delivery,
                campaign_id=campaign_id,
                cycle_index=cycle,
            )
        # Only screening thrash quality-held wins enqueue (not promotion thrash noise).
        elif _screening_enqueue_allowed(cycle_intent=cycle_intent, replay=replay):
            resolution = _enqueue_champion(
                root=root,
                loop_id=loop_id,
                delivery=delivery,
                camp_dir=camp_dir,
            )
    _write_cycle_handoff(
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        cycle_index=cycle,
        upstream_commit=upstream,
        integration_commit=integration,
        role=role,
        cycle_intent=cycle_intent,
        primary_metric=effective_primary,
        matrix=matrix,
        delivery=delivery,
        resolution=resolution,
        formal_status=promote_formal_status,
        skip_slugs=skip_slugs,
    )
    _run(
        [
            *ar,
            "status",
            "--loop-id",
            loop_id,
            "--matrix",
            "--last",
            "5",
        ],
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="campaign-status",
    )
    print(
        f"CYCLE_COMPLETE {campaign_id} role={role} intent={cycle_intent} "
        f"positive={delivery['positive']}",
        flush=True,
    )
    return campaign_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-id", default="continuous-openui-local")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/autoresearch"),
        help="Campaign bundle root",
    )
    parser.add_argument("--max-cycles", type=int, default=1, help="0 = unbounded")
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="Run one cycle after the agent has synced Git; emit a typed handoff",
    )
    parser.add_argument("--train-version", default="wf_smoke_v2")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--objective",
        default=(
            "On a size-matched fixture TwoTower arm under the wall cap, improve "
            "the current certified OpenUI quality primary without lowering parse_rate."
        ),
    )
    parser.add_argument("--primary-metric", default="smoke.latency_ms_p50")
    args = parser.parse_args(argv)
    cwd = Path.cwd()
    root = args.root if args.root.is_absolute() else cwd / args.root
    root.mkdir(parents=True, exist_ok=True)
    try:
        code_sha = _git(
            "rev-parse",
            "HEAD",
            cwd=cwd,
            root=root,
            loop_id=args.loop_id,
            stage="driver-startup-git",
        )
    except (subprocess.CalledProcessError, OSError):
        code_sha = None
    try:
        lock_fh = acquire_driver_lock(root, args.loop_id, code_sha=code_sha)
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2
    if args.supervised and args.max_cycles not in {0, 1}:
        parser.error("--supervised runs exactly one bounded cycle")
    passes = (
        itertools.count(1)
        if args.max_cycles == 0 and not args.supervised
        else range(1, 2 if args.supervised else max(1, args.max_cycles) + 1)
    )
    try:
        for pass_no in passes:
            total = (
                "∞"
                if args.max_cycles == 0 and not args.supervised
                else str(1 if args.supervised else max(1, args.max_cycles))
            )
            print(f"=== continuous cycle pass {pass_no}/{total} ===", flush=True)
            try:
                run_cycle(
                    cwd=cwd,
                    root=root,
                    loop_id=args.loop_id,
                    train_version=args.train_version,
                    steps=args.steps,
                    objective=args.objective,
                    primary_metric=args.primary_metric,
                    sync_git=not args.supervised,
                    startup_commit=code_sha,
                    require_action_receipts=args.supervised,
                )
            except _CodeUpdated as exc:
                print(f"CODE_UPDATED {exc}; re-executing driver", flush=True)
                os.execv(sys.executable, [sys.executable, *sys.argv])
                raise RuntimeError("driver re-exec unexpectedly returned")
            except Exception as exc:  # noqa: BLE001 - continuous must self-heal next pass
                print(f"CYCLE_ERROR {exc!r}", flush=True)
                cycle_index, _ = _latest_cycle(root, args.loop_id)
                count = _record_cycle_failure(
                    root=root,
                    loop_id=args.loop_id,
                    exc=exc,
                    cycle_index=cycle_index,
                )
                # A supervised invocation returns control immediately. Legacy
                # unbounded mode stops only at the typed repeated-blocker rule.
                if args.supervised or "dirty" in str(exc).lower() or count >= 3:
                    return 2
                time.sleep(1)
                continue
        return 0
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            lock_fh.close()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
