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
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from slm_training.autoresearch.engine import default_eval_version
from slm_training.autoresearch.hillclimb import (
    assert_warm_start_launch,
    climb_champion_checkpoint_path,
    load_climb_champion,
    maybe_advance_climb_champion,
)

# Re-exported for the suite, which reads them as attributes of this module
# (`run_autotrain_continuous.DEFAULT_MAX_CUMULATIVE_EPOCHS`). Their only
# in-module callers moved to scripts/autotrain_champion_*.py.
from slm_training.autoresearch.hillclimb import (  # noqa: F401
    DEFAULT_MAX_CUMULATIVE_EPOCHS,
    seed_climb_champion,
)
from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
    select_best_by_primary_then_smallest,
)
from slm_training.autoresearch.formal import formal_obligation_id
from slm_training.autoresearch.thrash_regime import (
    DECODE_RESIDUAL_SLUGS,
    LATENCY_PRIMARY_LEAF,
    REGIME_CLIMB,
    REGIME_ISOLATE,
    ThrashRegimeDecision,
    compose_climb_control_levers,
    compose_isolate_control_levers,
    compose_treatment_levers,
    decide_screening_regime,
    select_recommended_slug_for_regime,
)
from slm_training.autoresearch.thrash_residuals import (
    SlugStats,
    pick_soft_ranked_slug,
    rank_absolute_regimes,
    residual_boosts_from_observations,
    screening_tie_saturation,
)
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
    HarnessSignalV1,
    HypothesisFeedback,
    HypothesisMatrix,
    NextRunPriorityV1,
    utc_now,
)
from slm_training.autoresearch.storage import (
    CampaignStore,
    append_autotrain_action_receipt,
    autotrain_action_sha256,
    autotrain_loop_state_lock,
    bind_autotrain_action_evidence,
    pending_autotrain_actions,
    pending_autotrain_execution_actions,
)
from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.harness_core.versioning import build_version_stamp
from scripts.autotrain_paths import (  # noqa: F401
    DRIVER_LOCK_BASENAME as _DRIVER_LOCK_BASENAME,
)
from scripts.autotrain_paths import (  # noqa: F401
    LOOP_OWNED_GENERATED_PATHS as _LOOP_OWNED_GENERATED_PATHS,
)
from scripts.autotrain_paths import (  # noqa: F401
    LOOP_OWNED_GENERATED_SUFFIXES as _LOOP_OWNED_GENERATED_SUFFIXES,
)
from scripts.autotrain_paths import (  # noqa: F401
    PROMOTE_EXPECTATIONS_REL as _PROMOTE_EXPECTATIONS_REL,
)
from scripts.autotrain_paths import (  # noqa: F401
    SCREENING_EXPECTATIONS_REL as _SCREENING_EXPECTATIONS_REL,
)
from scripts.autotrain_paths import (  # noqa: F401
    champion_queue_path as _champion_queue_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    continuous_docs_paths as _continuous_docs_paths,
)
from scripts.autotrain_paths import (  # noqa: F401
    continuous_evidence_roots as _continuous_evidence_roots,
)
from scripts.autotrain_paths import (  # noqa: F401
    driver_lock_path as _driver_lock_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    heal_retired_versions_path as _heal_retired_versions_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    hillclimb_review_path as _hillclimb_review_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    is_continuous_closeout_path as _is_continuous_closeout_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    is_foreign_dirty_path as _is_foreign_dirty_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    is_loop_owned_generated_path as _is_loop_owned_generated_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    loop_campaign_dirs as _loop_campaign_dirs,
)
from scripts.autotrain_paths import (  # noqa: F401
    loop_champion_dir as _loop_champion_dir,
)
from scripts.autotrain_paths import (  # noqa: F401
    loop_state_path as _loop_state_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    merge_head_path as _merge_head_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    normalize_repo_relpath as _normalize_repo_relpath,
)
from scripts.autotrain_paths import (  # noqa: F401
    porcelain_paths as _porcelain_paths,
)
from scripts.autotrain_paths import (  # noqa: F401
    promotion_replicate_ledger_path as _promotion_replicate_ledger_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    terminal_verdict_path as _terminal_verdict_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    promote_expectations_path as promote_expectations_path,
)
from scripts.autotrain_paths import (  # noqa: F401
    screening_expectations_path as screening_expectations_path,
)
from scripts.autotrain_io import (  # noqa: F401
    read_json as _read_json,
)
from scripts.autotrain_metrics import (  # noqa: F401
    EVAL_NLL_RECORDS_NAME as _EVAL_NLL_RECORDS_NAME,
)
from scripts.autotrain_metrics import (  # noqa: F401
    EVAL_NLL_RECORDS_SCHEMA as _EVAL_NLL_RECORDS_SCHEMA,
)
from scripts.autotrain_metrics import (  # noqa: F401
    METRIC_LEAVES as _METRIC_LEAVES,
)
from scripts.autotrain_metrics import (  # noqa: F401
    effective_primary_metric as _effective_primary_metric,
)
from scripts.autotrain_metrics import (  # noqa: F401
    find_nested_key as _find_nested_key,
)
from scripts.autotrain_metrics import (  # noqa: F401
    finite_metric as _finite_metric,
)
from scripts.autotrain_metrics import (  # noqa: F401
    metric_from_eval as _metric_from_eval,
)
from scripts.autotrain_metrics import (  # noqa: F401
    metric_leaf as _metric_leaf,
)
from scripts.autotrain_metrics import (  # noqa: F401
    primary_harness_family as _primary_harness_family,
)
from scripts.autotrain_metrics import (  # noqa: F401
    rate_to_pm as _rate_to_pm,
)
from scripts.autotrain_metrics import (  # noqa: F401
    raw_metric_observations as _raw_metric_observations,
)
from scripts.autotrain_metrics import (  # noqa: F401
    read_eval_nll_records as _read_eval_nll_records,
)
from scripts.autotrain_metrics import (  # noqa: F401
    run_has_usable_metrics as _run_has_usable_metrics,
)
from scripts.autotrain_metrics import (  # noqa: F401
    run_metrics as _run_metrics,
)
from scripts.autotrain_metrics import (  # noqa: F401
    run_suite_metrics as _run_suite_metrics,
)
from scripts.autotrain_measurement import (  # noqa: F401
    EPS as _EPS,
)
from scripts.autotrain_measurement import (  # noqa: F401
    candidate_mpr_positive as _candidate_mpr_positive,
)
from scripts.autotrain_measurement import (  # noqa: F401
    candidate_ship_state as _candidate_ship_state,
)
from scripts.autotrain_measurement import (  # noqa: F401
    has_primary_metric_win as _has_primary_metric_win,
)
from scripts.autotrain_measurement import (  # noqa: F401
    measurement_is_complete as _measurement_is_complete,
)
from scripts.autotrain_measurement import (  # noqa: F401
    multi_arm_measurement as _multi_arm_measurement,
)
from scripts.autotrain_measurement import (  # noqa: F401
    quality_metrics_identical as _quality_metrics_identical,
)
from scripts.autotrain_measurement import (  # noqa: F401
    yaml_mapping_equal as _yaml_mapping_equal,
)
from scripts.autotrain_tradeoff import (  # noqa: F401
    LATENCY_REGRESSION_ABS_MS as _LATENCY_REGRESSION_ABS_MS,
)
from scripts.autotrain_tradeoff import (  # noqa: F401
    LATENCY_REGRESSION_BUDGET as _LATENCY_REGRESSION_BUDGET,
)
from scripts.autotrain_tradeoff import (  # noqa: F401
    MIN_MPR_FOR_LATENCY_WIN as _MIN_MPR_FOR_LATENCY_WIN,
)
from scripts.autotrain_tradeoff import (  # noqa: F401
    TIMEOUT_BAND_HI_MS as _TIMEOUT_BAND_HI_MS,
)
from scripts.autotrain_tradeoff import (  # noqa: F401
    TIMEOUT_BAND_LO_MS as _TIMEOUT_BAND_LO_MS,
)
from scripts.autotrain_tradeoff import (  # noqa: F401
    classify_metric_tradeoff as _classify_metric_tradeoff,
)
from scripts.autotrain_tradeoff import (  # noqa: F401
    in_timeout_band as _in_timeout_band,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    BANK_EXHAUST_MARKERS as _BANK_EXHAUST_MARKERS,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    BANK_EXHAUST_MSG as _BANK_EXHAUST_MSG,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    CodeUpdated as _CodeUpdated,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    HARNESS_INCOMPLETE_REASON_PREFIXES as _HARNESS_INCOMPLETE_REASON_PREFIXES,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    OPEN_NUMERIC_LITERAL_RE as _OPEN_NUMERIC_LITERAL_RE,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    QUALITY_ENQUEUE_PREFIXES as _QUALITY_ENQUEUE_PREFIXES,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    diagnosis_target as _diagnosis_target,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    exception_is_soft_continuous as _exception_is_soft_continuous,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    has_numeric_literal_close_starvation as _has_numeric_literal_close_starvation,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    is_decisive_causal_terminal as _is_decisive_causal_terminal,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    quality_held_reasons as _quality_held_reasons,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    reason_is_harness_incomplete as _reason_is_harness_incomplete,
)
from scripts.autotrain_diagnosis import (  # noqa: F401
    reasons_are_harness_incomplete_only as _reasons_are_harness_incomplete_only,
)
from scripts.autotrain_levers import (  # noqa: F401
    EXPERIMENT_ONLY_KNOB_CATEGORIES as _EXPERIMENT_ONLY_KNOB_CATEGORIES,
)
from scripts.autotrain_levers import (  # noqa: F401
    FINGERPRINT_EXCLUDE_KEYS as _FINGERPRINT_EXCLUDE_KEYS,
)
from scripts.autotrain_levers import (  # noqa: F401
    LEVER_KNOB_KEYS as _LEVER_KNOB_KEYS,
)
from scripts.autotrain_levers import (  # noqa: F401
    bank_lever_categories as _bank_lever_categories,
)
from scripts.autotrain_levers import (  # noqa: F401
    knobs_fingerprint as _knobs_fingerprint,
)
from scripts.autotrain_levers import (  # noqa: F401
    lever_knobs as _lever_knobs,
)
from scripts.autotrain_levers import (  # noqa: F401
    load_experiment_knobs as _load_experiment_knobs,
)
from scripts.autotrain_levers import (  # noqa: F401
    matrix_experiment_knobs as _matrix_experiment_knobs,
)
from scripts.autotrain_levers import (  # noqa: F401
    matrix_treatment_signature as _matrix_treatment_signature,
)
from scripts.autotrain_levers import (  # noqa: F401
    short_lever_token as _short_lever_token,
)
from scripts.autotrain_levers import (  # noqa: F401
    thrash_lever_signature as _thrash_lever_signature,
)
from scripts.autotrain_arms import (  # noqa: F401
    PROCESS_ARM_FLAG_KEYS as _PROCESS_ARM_FLAG_KEYS,
)
from scripts.autotrain_arms import (  # noqa: F401
    PROCESS_ROLES as _PROCESS_ROLES,
)
from scripts.autotrain_arms import (  # noqa: F401
    apply_arm_extras as _apply_arm_extras,
)
from scripts.autotrain_arms import (  # noqa: F401
    arm_completed_n as _arm_completed_n,
)
from scripts.autotrain_arms import (  # noqa: F401
    arm_eval_version as _arm_eval_version,
)
from scripts.autotrain_arms import (  # noqa: F401
    arm_swaps_train_corpus as _arm_swaps_train_corpus,
)
from scripts.autotrain_arms import (  # noqa: F401
    arm_trainable_params as _arm_trainable_params,
)
from scripts.autotrain_arms import (  # noqa: F401
    capacity_view as _capacity_view,
)
from scripts.autotrain_arms import (  # noqa: F401
    compose_atom_extras as _compose_atom_extras,
)
from scripts.autotrain_arms import (  # noqa: F401
    counterbalanced_arm_order as _counterbalanced_arm_order,
)
from scripts.autotrain_arms import (  # noqa: F401
    current_rung_label as _current_rung_label,
)
from scripts.autotrain_arms import (  # noqa: F401
    finalize_compose_extras as _finalize_compose_extras,
)
from scripts.autotrain_arms import (  # noqa: F401
    is_process_arm as _is_process_arm,
)
from scripts.autotrain_arms import (  # noqa: F401
    latency_only_arm as _latency_only_arm,
)
from scripts.autotrain_arms import (  # noqa: F401
    size_match_skip_reason as _size_match_skip_reason,
)
from scripts.autotrain_campaign import (  # noqa: F401
    PROMOTE_AUTHORITY_HARNESS_COMPONENT as _PROMOTE_AUTHORITY_HARNESS_COMPONENT,
)
from scripts.autotrain_campaign import (  # noqa: F401
    campaign_at_cycle as _campaign_at_cycle,
)
from scripts.autotrain_campaign import (  # noqa: F401
    campaign_id as _campaign_id,
)
from scripts.autotrain_campaign import (  # noqa: F401
    campaign_power_feasibility as _campaign_power_feasibility,
)
from scripts.autotrain_campaign import (  # noqa: F401
    campaign_started_experiment as _campaign_started_experiment,
)
from scripts.autotrain_campaign import (  # noqa: F401
    experiment_artifact as _experiment_artifact,
)
from scripts.autotrain_campaign import (  # noqa: F401
    experiment_campaign_component_version as _experiment_campaign_component_version,
)
from scripts.autotrain_campaign import (  # noqa: F401
    lineage_campaign_ids as _lineage_campaign_ids,
)
from scripts.autotrain_campaign import (  # noqa: F401
    warm_start_policy as _warm_start_policy,
)
from scripts.autotrain_screening import (  # noqa: F401
    LOCAL_I10_ROOT_CAP as _LOCAL_I10_ROOT_CAP,
)
from scripts.autotrain_screening import (  # noqa: F401
    exhaust_screening_losers as _exhaust_screening_losers,
)
from scripts.autotrain_screening import (  # noqa: F401
    fit_screening_candidate_count as _fit_screening_candidate_count,
)
from scripts.autotrain_screening import (  # noqa: F401
    latest_train_telemetry_payload as _latest_train_telemetry_payload,
)
from scripts.autotrain_screening import (  # noqa: F401
    local_i10_train_version as _local_i10_train_version,
)
from scripts.autotrain_screening import (  # noqa: F401
    local_rebuild_data_argv as _local_rebuild_data_argv,
)
from scripts.autotrain_screening import (  # noqa: F401
    local_rebuild_screening_eval_argv as _local_rebuild_screening_eval_argv,
)
from scripts.autotrain_screening import (  # noqa: F401
    policy_default_decode_floor_seconds as _policy_default_decode_floor_seconds,
)
from scripts.autotrain_screening import (  # noqa: F401
    screening_enqueue_allowed as _screening_enqueue_allowed,
)
from scripts.autotrain_screening import (  # noqa: F401
    screening_max_gpu_hours as _screening_max_gpu_hours,
)
from scripts.autotrain_screening import (  # noqa: F401
    screening_multi_arm_ids as _screening_multi_arm_ids,
)
from scripts.autotrain_screening import (  # noqa: F401
    screening_regime_decision as _screening_regime_decision,
)
from scripts.autotrain_screening import (  # noqa: F401
    screening_thrash_steps as _screening_thrash_steps,
)
from scripts.autotrain_screening import (  # noqa: F401
    screening_train_device as _screening_train_device,
)
from scripts.autotrain_decode_timing import (  # noqa: F401
    predecessor_compiler_ms_timeout as _predecessor_compiler_ms_timeout,
)
from scripts.autotrain_decode_timing import (  # noqa: F401
    predecessor_decode_p95_seconds as _predecessor_decode_p95_seconds,
)
from scripts.autotrain_decode_timing import (  # noqa: F401
    predecessor_timeout_cause as _predecessor_timeout_cause,
)
from scripts.autotrain_timeout_state import (  # noqa: F401
    FORMAL_TIMEOUT_STATUSES as _FORMAL_TIMEOUT_STATUSES,
)
from scripts.autotrain_timeout_state import (  # noqa: F401
    arm_decode_timeout_count as _arm_decode_timeout_count,
)
from scripts.autotrain_timeout_state import (  # noqa: F401
    delivery_is_thrash_timeout_residual as _delivery_is_thrash_timeout_residual,
)
from scripts.autotrain_timeout_state import (  # noqa: F401
    formal_status_is_timeout as _formal_status_is_timeout,
)
from scripts.autotrain_timeout_state import (  # noqa: F401
    has_finalized_decode_timeout as _has_finalized_decode_timeout,
)
from scripts.autotrain_timeout_state import (  # noqa: F401
    is_reproduced_timeout_retirement as _is_reproduced_timeout_retirement,
)
from scripts.autotrain_timeout_state import (  # noqa: F401
    require_predecessor_actions as _require_predecessor_actions,
)
from scripts.autotrain_records import (  # noqa: F401
    OBSERVED_PAIRED_SD_SCHEMA as _OBSERVED_PAIRED_SD_SCHEMA,
)
from scripts.autotrain_records import (  # noqa: F401
    OBSERVED_PAIRED_SD_SOURCE as _OBSERVED_PAIRED_SD_SOURCE,
)
from scripts.autotrain_records import (  # noqa: F401
    last_cycle_failure_message as _last_cycle_failure_message,
)
from scripts.autotrain_records import (  # noqa: F401
    last_heal_receipt_outcome as _last_heal_receipt_outcome,
)
from scripts.autotrain_records import (  # noqa: F401
    record_cycle_failure as _record_cycle_failure,
)
from scripts.autotrain_records import (  # noqa: F401
    record_cycle_recovery as _record_cycle_recovery,
)
from scripts.autotrain_records import (  # noqa: F401
    record_observed_paired_sd as _record_observed_paired_sd,
)
from scripts.autotrain_records import (  # noqa: F401
    write_loop_state as _write_loop_state,
)
from scripts.autotrain_records import (  # noqa: F401
    write_loop_state_unlocked as _write_loop_state_unlocked,
)
from scripts.autotrain_candidate_state import (  # noqa: F401
    confirm_candidate_blocked as _confirm_candidate_blocked,
)
from scripts.autotrain_candidate_state import (  # noqa: F401
    confirmation_quality_reheld as _confirmation_quality_reheld,
)
from scripts.autotrain_candidate_state import (  # noqa: F401
    delivery_parse_mpr_held as _delivery_parse_mpr_held,
)
from scripts.autotrain_candidate_state import (  # noqa: F401
    is_confirm_candidate_win as _is_confirm_candidate_win,
)
from scripts.autotrain_candidate_state import (  # noqa: F401
    queued_candidate_priorities as _queued_candidate_priorities,
)
from scripts.autotrain_candidate_state import (  # noqa: F401
    role_with_confirmation_boundary as _role_with_confirmation_boundary,
)
from scripts.autotrain_confirmation import (  # noqa: F401
    completed_confirmation_priorities as _completed_confirmation_priorities,
)
from scripts.autotrain_confirmation import (  # noqa: F401
    confirmation_replay_entry as _confirmation_replay_entry,
)
from scripts.autotrain_confirmation import (  # noqa: F401
    consecutive_frozen_replays as _consecutive_frozen_replays,
)
from scripts.autotrain_confirmation import (  # noqa: F401
    reconcile_completed_confirmation_replays as _reconcile_completed_confirmation_replays,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    REGIME_PARKED_STATUS as _REGIME_PARKED_STATUS,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    RETRYABLE_PROMOTE_STATUSES as _RETRYABLE_PROMOTE_STATUSES,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    clear_loop_blocker as _clear_loop_blocker,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    park_champion_epochs_if_needed as _park_champion_epochs_if_needed,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    queue_head_confirmed as _queue_head_confirmed,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    queue_head_open as _queue_head_open,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    should_enqueue_champion as _should_enqueue_champion,
)
from scripts.autotrain_champion_queue import (  # noqa: F401
    write_champion_queue as _write_champion_queue,
)
from scripts.autotrain_champion_lifecycle import (  # noqa: F401
    CONTROL_RUN_SUFFIXES as _CONTROL_RUN_SUFFIXES,
)
from scripts.autotrain_champion_lifecycle import (  # noqa: F401
    ensure_climb_champion as _ensure_climb_champion,
)
from scripts.autotrain_champion_lifecycle import (  # noqa: F401
    first_complete_control_run as _first_complete_control_run,
)
from scripts.autotrain_champion_lifecycle import (  # noqa: F401
    refresh_champion_source_recipes as _refresh_champion_source_recipes,
)
from scripts.autotrain_champion_lifecycle import (  # noqa: F401
    seed_baseline_champion as _seed_baseline_champion,
)
from scripts.autotrain_champion_repair import (  # noqa: F401
    recover_interrupted_champion_entries as _recover_interrupted_champion_entries,
)
from scripts.autotrain_champion_repair import (  # noqa: F401
    reopen_harness_blocked_champions as _reopen_harness_blocked_champions,
)
from scripts.autotrain_champion_repair import (  # noqa: F401
    revalidate_confirmed_champion_entries as _revalidate_confirmed_champion_entries,
)
from scripts.autotrain_park import (  # noqa: F401
    STALL_FINGERPRINT as _STALL_FINGERPRINT,
)
from scripts.autotrain_park import (  # noqa: F401
    park_loop_stalled as _park_loop_stalled,
)
from scripts.autotrain_park import (  # noqa: F401
    park_screening_n_deficit as _park_screening_n_deficit,
)
from scripts.autotrain_park import (  # noqa: F401
    terminal_park_on_exhaust as _terminal_park_on_exhaust,
)
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    INTERRUPT_AFTER_SECONDS,
    KILL_GRACE_SECONDS,
    MAX_HARNESS_WALL_SECONDS,
    MAX_RUN_SECONDS,
)
# Budget constants re-exported under their original private names: the
# extracted functions consume them, and the test suite reads them off this
# module, so they are part of its surface even though nothing here calls them.
from scripts.autotrain_ledgers import (
    dynamic_thrash_arms_path as _dynamic_thrash_arms_path,
)
from scripts.autotrain_ledgers import (
    slug_stats_path as _slug_stats_path,
)
from scripts.autotrain_ledgers import (
    iter_loop_deliveries as _iter_loop_deliveries,
)
from scripts.autotrain_ledgers import (
    load_residual_observations as _load_residual_observations,
)
from scripts.autotrain_ledgers import (
    append_hillclimb_iteration as _append_hillclimb_iteration,
)
from scripts.autotrain_ledgers import (
    append_historical_reclassification as _append_historical_reclassification,
)
from scripts.autotrain_ledgers import (
    append_interesting_residual as _append_interesting_residual,
)
from scripts.autotrain_provenance import (
    checkpoint_path_for_candidate as _checkpoint_path_for_candidate,
)
# Re-exported under its original private name: the cycle's provenance write
# moved with it, so the runner no longer calls it, but the suite exercises it
# as `run_autotrain_continuous._auto_no_bump_version_registry`.
from scripts.autotrain_provenance import (  # noqa: F401
    auto_no_bump_version_registry as _auto_no_bump_version_registry,
)
from scripts.autotrain_provenance import (
    append_checkpoint_doc_notes as _append_checkpoint_doc_notes,
)
from scripts.autotrain_budget import (  # noqa: F401
    ARM_BUDGET_SCHEDULE_MARGIN_SECONDS as _ARM_BUDGET_SCHEDULE_MARGIN_SECONDS,
)
from scripts.autotrain_budget import (  # noqa: F401
    COLD_START_STEPS_PER_SEC as _COLD_START_STEPS_PER_SEC,
)
from scripts.autotrain_budget import (  # noqa: F401
    COLD_START_STEPS_PER_SEC_EVIDENCE as _COLD_START_STEPS_PER_SEC_EVIDENCE,
)
from scripts.autotrain_budget import (  # noqa: F401
    SCREENING_THRASH_STEPS_MAX_DEFAULT as _SCREENING_THRASH_STEPS_MAX_DEFAULT,
)
from scripts.autotrain_budget import (
    STEPS_PER_SEC_SAFETY as _STEPS_PER_SEC_SAFETY,
)
from scripts.autotrain_budget import (
    arm_execution_deadline as _arm_execution_deadline,
)
from scripts.autotrain_budget import (
    arm_wall_minutes as _arm_wall_minutes,
)
from scripts.autotrain_budget import (
    arm_wall_seconds as _arm_wall_seconds,
)
from scripts.autotrain_budget import (
    fit_screening_steps as _fit_screening_steps,
)
from scripts.autotrain_budget import (
    fit_symmetric_arm_budget as _fit_symmetric_arm_budget,
)
from scripts.autotrain_budget import (
    remaining_timeout as _remaining_timeout,
)
from scripts.autotrain_budget import (
    require_symmetric_arm_budget as _require_symmetric_arm_budget,
)
from scripts.autotrain_budget import (
    screening_thrash_steps_max as _screening_thrash_steps_max,
)
from scripts.autotrain_budget import (
    steps_per_sec_from_train_payload as _steps_per_sec_from_train_payload,
)
from scripts.autotrain_budget import (
    thrash_timing_block as _thrash_timing_block,
)

# Locked continuous promote metric programs (SHA bound on campaign lock).
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






def _screening_suite_records() -> int | None:
    """Record count of the resolved smoke screening suite (volume ceiling)."""

    try:
        from slm_training.autoresearch.engine import default_eval_version
        from slm_training.data.store import DataStore

        records = (
            DataStore().resolve_path("eval", default_eval_version())
            / "suites"
            / "smoke"
            / "records.jsonl"
        )
        if records.is_file():
            return sum(
                1
                for line in records.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    except Exception:  # noqa: BLE001 — telemetry input only, never fatal
        return None
    return None


def _screening_n_report(policy: Any | None = None) -> tuple[int, dict[str, Any] | None]:
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        screening_smoke_n_for_policy,
        stage_wall_minutes_for_role,
    )

    pol = policy if policy is not None else load_climb_policy()
    arm = _arm_wall_seconds(
        policy_minutes=float(stage_wall_minutes_for_role(pol, "screening")),
        formal_required=False,
    )
    return screening_smoke_n_for_policy(
        pol, arm_wall_seconds=arm, suite_records=_screening_suite_records()
    )


class _SeedAppendResult(NamedTuple):
    """Honest accounting for one seed-file growth attempt."""

    paths: list[Path]
    seed_path: Path
    lines_before: int
    smoke_n_before: int
    smoke_n_after: int
    need: int
    appended: int

    @property
    def deficit_unfilled(self) -> bool:
        """A deficit existed and the sampler produced nothing: a failed heal."""
        return self.need > 0 and self.appended == 0


def _append_deficit_smoke_seeds(cwd: Path, *, n_min: int) -> _SeedAppendResult:
    """Append unused extra smoke fixtures to the tracked seed file.

    Wraps the sampler (``extra_smoke_fixtures_for_deficit``) with before/after
    counts so the caller can verify growth instead of trusting the return
    value; an empty sampler result against a real deficit is reported as
    ``deficit_unfilled`` — never silently as success.
    """
    from slm_training.autoresearch.screening_sample_size import (
        extra_smoke_fixtures_for_deficit,
    )

    seed_path = cwd / "src/slm_training/resources/test_seeds.jsonl"
    lines = seed_path.read_text(encoding="utf-8").splitlines()
    existing: set[str] = set()
    smoke_n = 0
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        existing.add(str(rec.get("id") or ""))
        suite = str((rec.get("meta") or {}).get("suite") or rec.get("split") or "")
        if suite == "smoke":
            smoke_n += 1
    lines_before = sum(1 for line in lines if line.strip())
    need = max(0, int(n_min) - smoke_n)
    extras = extra_smoke_fixtures_for_deficit(existing_ids=existing, need=need)
    if not extras:
        return _SeedAppendResult(
            [], seed_path, lines_before, smoke_n, smoke_n, need, 0
        )
    with seed_path.open("a", encoding="utf-8") as fh:
        if lines and lines[-1].strip():
            fh.write("\n")
        for rec in extras:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return _SeedAppendResult(
        [seed_path],
        seed_path,
        lines_before,
        smoke_n,
        smoke_n + len(extras),
        need,
        len(extras),
    )




_SCREENING_EVAL_HEAL_ID = "rebuild_screening_eval"


def _self_heal_rebuild_screening_eval(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str | None,
) -> str | None:
    """Grow smoke to the Lean floor, publish under resources/, commit.

    Fail-closed (``docs/design/autotrain-fail-closed-self-healing.md`` §3):
    the heal counts only when the postcondition probe
    (:func:`verify_driver_heal`) observes the resolved smoke suite grow
    (``smoke_n_after > smoke_n_before``) and the policy resolver no longer
    demands generation (``must_generate == False``). An empty sampler against
    a real deficit, a publish conflict, or an unchanged suite each leave a
    ``heal_postcondition_failed`` receipt and return ``None`` — no sidecar,
    no commit of the fresh suite id, no action ack.
    """
    from slm_training.autoresearch.heal.fail_closed import (
        allocate_screening_suite_id,
        count_records,
        record_count_probe,
        verify_driver_heal,
    )
    from slm_training.data.store import DataStore
    from slm_training.levers import DEFAULT_TRAIN_DATA_DIR

    n, report = _screening_n_report()
    if not isinstance(report, dict) or not report.get("must_generate"):
        return None
    n_min = int(report.get("n_min") or 6)
    smoke_n_before = int(_screening_suite_records() or 0)
    seeds = _append_deficit_smoke_seeds(cwd, n_min=n_min)
    counts_before = {
        "smoke_n": smoke_n_before,
        "seed_smoke_n": seeds.smoke_n_before,
    }

    def _failed(
        stage: str,
        *,
        probe_path: Path,
        must_exceed: int,
        counts_after: dict[str, int],
        conditions: dict[str, bool] | None = None,
        note: str = "",
    ) -> None:
        receipt = verify_driver_heal(
            root=root,
            loop_id=loop_id,
            campaign_id=campaign_id,
            heal_id=_SCREENING_EVAL_HEAL_ID,
            verify=record_count_probe(probe_path, must_exceed=must_exceed),
            cwd=cwd,
            counts_before=counts_before,
            counts_after=counts_after,
            extra_conditions=conditions,
            note=f"stage={stage} {note}".strip(),
        )
        print(
            f"SELF_HEAL_REBUILD_SCREENING_EVAL_FAIL stage={stage} "
            f"outcome={receipt.outcome} n_min={n_min} "
            f"counts_before={json.dumps(counts_before, sort_keys=True)} "
            f"counts_after={json.dumps(counts_after, sort_keys=True)}",
            flush=True,
        )
        return None

    if seeds.deficit_unfilled:
        # The sampler had nothing to add: the seed file did not grow, so no
        # bigger suite can be built from it. Probe the seed file itself.
        return _failed(
            "seed_sampler_empty",
            probe_path=seeds.seed_path,
            must_exceed=seeds.lines_before,
            counts_after={
                "smoke_n": smoke_n_before,
                "seed_smoke_n": seeds.smoke_n_after,
            },
            conditions={"seed_deficit_filled": False},
            note=f"need={seeds.need} appended=0",
        )
    eval_root = cwd / "src/slm_training/resources/data/eval"
    eval_version = allocate_screening_suite_id(eval_root, n_min)
    train_manifest = cwd / DEFAULT_TRAIN_DATA_DIR / "manifest.json"
    out_dir = cwd / "outputs" / "data" / "eval" / eval_version
    published = eval_root / eval_version
    published_records = published / "suites" / "smoke" / "records.jsonl"
    if not (out_dir / "manifest.json").is_file() and not published_records.is_file():
        argv = _local_rebuild_screening_eval_argv(
            eval_version=eval_version, train_manifest=train_manifest
        )
        print(
            f"SELF_HEAL_REBUILD_SCREENING_EVAL start version={eval_version} "
            f"n_min={n_min} argv={argv}",
            flush=True,
        )
        result = run_bounded_process(
            argv,
            interrupt_after_seconds=float(INTERRUPT_AFTER_SECONDS),
            kill_grace_seconds=float(KILL_GRACE_SECONDS),
            cwd=str(cwd),
        )
        if result.outcome != ProcessOutcome.COMPLETED or result.returncode != 0:
            print(
                f"SELF_HEAL_REBUILD_SCREENING_EVAL_FAIL outcome={result.outcome} "
                f"code={result.returncode}",
                flush=True,
            )
            return None
    store = DataStore(root=cwd)
    if not published_records.is_file():
        try:
            store.publish("eval", eval_version)
        except FileExistsError as exc:
            # A conflict on a freshly allocated id means the allocator and
            # the store disagree: that is a failed heal, never a silent pass.
            return _failed(
                "publish_conflict",
                probe_path=published_records,
                must_exceed=count_records(published_records),
                counts_after={
                    "smoke_n": smoke_n_before,
                    "seed_smoke_n": seeds.smoke_n_after,
                },
                conditions={"published_fresh_suite": False},
                note=f"version={eval_version} {exc!r}"[:300],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"SELF_HEAL_REBUILD_SCREENING_EVAL_FAIL publish={exc!r}", flush=True)
            return None
    n_after, report_after = _screening_n_report()
    smoke_n_after = int(_screening_suite_records() or 0)
    must_generate_after = bool(
        isinstance(report_after, dict) and report_after.get("must_generate")
    )
    counts_after = {
        "smoke_n": smoke_n_after,
        "seed_smoke_n": seeds.smoke_n_after,
        "published_records": count_records(published_records),
    }
    receipt = verify_driver_heal(
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        heal_id=_SCREENING_EVAL_HEAL_ID,
        verify=record_count_probe(published_records, must_exceed=smoke_n_before),
        cwd=cwd,
        counts_before=counts_before,
        counts_after=counts_after,
        extra_conditions={
            "must_generate_false": not must_generate_after,
            "resolver_reports_growth": smoke_n_after > smoke_n_before,
        },
        note=f"stage=published version={eval_version}",
    )
    if receipt.outcome != "healed":
        print(
            f"SELF_HEAL_REBUILD_SCREENING_EVAL_FAIL stage=postcondition "
            f"outcome={receipt.outcome} version={eval_version} "
            f"counts_before={json.dumps(counts_before, sort_keys=True)} "
            f"counts_after={json.dumps(counts_after, sort_keys=True)} "
            f"must_generate_after={must_generate_after}",
            flush=True,
        )
        return None
    sidecar = published / "screening_sample_size.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "screening_sample_size/v1",
                "eval_version": eval_version,
                "smoke_n": n_after,
                "smoke_n_before": smoke_n_before,
                "smoke_n_after": smoke_n_after,
                "report": report_after,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    commit_paths = [
        cwd / "src/slm_training/resources/test_seeds.jsonl",
        sidecar,
        *sorted(p for p in published.rglob("*") if p.is_file()),
    ]
    _git_commit_paths(
        cwd,
        commit_paths,
        message=(
            f"data(eval): persist screening smoke n>={n_min} ({eval_version})"
        ),
        root=root,
        loop_id=loop_id,
        stage="self-heal-screening-eval",
    )
    if campaign_id:
        handoff_path = root / campaign_id / "cycle_handoff.json"
        if handoff_path.is_file():
            handoff = AutotrainCycleHandoffV1.model_validate_json(
                handoff_path.read_text(encoding="utf-8")
            )
            pending = [
                (index, action)
                for index, action in pending_autotrain_actions(root, handoff)
                if action.kind == "rebuild_data"
                and "screening suite" in action.reason
            ]
            for index, _action in pending:
                _ack_rebuild_data_action(
                    root,
                    handoff,
                    action_index=index,
                    evidence_uris=[
                        f"src/slm_training/resources/data/eval/{eval_version}/"
                        "screening_sample_size.json"
                    ],
                    counts=(smoke_n_before, smoke_n_after),
                )
    print(
        f"SELF_HEAL_REBUILD_SCREENING_EVAL version={eval_version} "
        f"smoke_n={n_after} smoke_n_before={smoke_n_before} "
        f"smoke_n_after={smoke_n_after}",
        flush=True,
    )
    return _SCREENING_EVAL_HEAL_ID


# Cold-start steps/s prior until a train_summary exists. Measured 2026-09-02
# on the 4-CPU (no GPU) autotrain box with the canonical trainer
# (``python -m scripts.train_model``) on ``wf_smoke_v2`` (101 records) at the
# exact screening launch shape from ``autoresearch/engine.py`` (scratch
# context, batch 2 / 3, lr 3e-4, seed 0, ``--no-full-state-checkpoint``) and
# at the E53 recipe architecture (d_model 192, 6 heads, 3+6 layers, scratch
# context: the frozen SmolLM2 ``hf`` tower is not installable here), while
# sibling agents shared the CPUs. 12 complete 200-step runs spanned
# 2.906-19.633 steps/s (median 16.091; the spread is CPU contention, not
# recipe variance). The prior is the slowest run rounded down to 0.1 so a
# cold fit never overshoots the train floor under contention. Telemetry from
# ``*/runs/*/train_summary.json`` replaces it after the first arm. Provenance:
# ``docs/design/p8-screening-cold-start-steps-prior-20260902.md`` (+ ``.json``).


























def _fit_screening_decode_timeout_seconds(
    policy: Any,
    *,
    arm_wall_seconds: float | None = None,
    formal_required: bool = False,
    telemetry_root: Path | None = None,
    requested_steps: int = 20,
    predecessor_campaign_id: str | None = None,
    decode_floor_evidence: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Fit the screening decode budget to measured per-record cost.

    Budget feedback loop (policy ``measurement.thrash_timing``): the
    predecessor's per-record decode p95 is the decode floor handed to
    ``screening_smoke_n_for_policy`` (``decode_floor_source=measured_p95``;
    ``policy_default`` only when no predecessor scoreboard exists). The number
    of decoded quality-probe records is what fits the eval share,
    ``n_probe = max(1, floor(eval_share / floor))``, and the per-record timeout
    is ``min(eval_share / n_probe, p95 x (1 + p95_margin))`` -- timeouts ~
    p95(work) + margin, never wall++. The Pareto rule then recalibrates from
    the predecessor's incomplete rate: above ``incomplete_rate_high`` the
    floor is inflated by the margin (n_probe shrinks; with n_probe already 1
    and a train-bound wall the decode share is kept whole instead of growing
    the train floor -- ``shrink_steps``); below ``incomplete_rate_low`` the
    floor is deflated (n_probe grows). Timeouts whose cause is budget
    (``p95 > applied timeout``) are classified ``budget_timeout``. Unused
    eval allocation is reassigned to the train floor (never past the wall).
    """

    import math

    from slm_training.autoresearch.climb_policy import (
        decode_timeout_seconds_for_role,
        screening_smoke_n_for_policy,
        stage_wall_minutes_for_role,
    )

    thrash = _thrash_timing_block(policy)
    configured = float(decode_timeout_seconds_for_role(policy, "screening"))
    if arm_wall_seconds is None:
        arm_wall_seconds = _arm_wall_seconds(
            policy_minutes=float(stage_wall_minutes_for_role(policy, "screening")),
            formal_required=formal_required,
        )
    min_train = float(thrash.get("min_train_floor_seconds") or 20.0)
    overhead = float(thrash.get("eval_overhead_seconds") or 8.0)
    usable = max(0.0, float(arm_wall_seconds) - min_train - overhead)
    evidence = (
        dict(decode_floor_evidence)
        if decode_floor_evidence is not None
        else _predecessor_decode_p95_seconds(telemetry_root, predecessor_campaign_id)
    )
    raw_p95 = evidence.get("p95_seconds")
    p95 = (
        float(raw_p95) if isinstance(raw_p95, (int, float)) and raw_p95 > 0 else None
    )
    measured = p95 is not None
    policy_floor = _policy_default_decode_floor_seconds(policy)
    decode_floor = p95 if p95 is not None else policy_floor
    decode_floor_source = "measured_p95" if measured else "policy_default"
    smoke_n, sample_size_report = screening_smoke_n_for_policy(
        policy,
        arm_wall_seconds=arm_wall_seconds,
        per_record_decode_floor_seconds=decode_floor if decode_floor > 0 else None,
        suite_records=_screening_suite_records(),
    )
    try:
        margin = max(0.0, float(thrash.get("p95_margin", 0.15)))
    except (TypeError, ValueError):
        margin = 0.15
    try:
        rate_high = float(thrash.get("incomplete_rate_high", 0.15))
        rate_low = float(thrash.get("incomplete_rate_low", 0.05))
    except (TypeError, ValueError):
        rate_high, rate_low = 0.15, 0.05
    raw_rate = evidence.get("incomplete_rate")
    rate = float(raw_rate) if isinstance(raw_rate, (int, float)) else None
    pareto: dict[str, Any] = {
        "incomplete_rate": rate,
        "incomplete_rate_high": rate_high,
        "incomplete_rate_low": rate_low,
        "p95_margin": margin,
        "decoded_records": int(evidence.get("decoded_records") or 0),
        "timeout_records": int(evidence.get("timeout_records") or 0),
        "decision": "cold_start",
        "reason": "no_measured_predecessor_decode_cost",
        "effective_floor_seconds": None,
    }
    margin_cap: float | None = None
    if p95 is not None:
        n_probe_base = max(1, int(math.floor(usable / p95))) if usable > 0 else 1
        margin_cap = p95 * (1.0 + margin)
        effective_floor = p95
        if rate is None:
            decision, reason = "hold", "incomplete_rate_unmeasured"
        elif rate > rate_high:
            effective_floor = p95 * (1.0 + margin)
            decision, reason = "shrink", (
                f"incomplete_rate={rate:.3f}>high={rate_high:.3f};"
                "floor_inflated_by_margin"
            )
        elif rate < rate_low:
            effective_floor = p95 / (1.0 + margin)
            decision, reason = "grow", (
                f"incomplete_rate={rate:.3f}<low={rate_low:.3f};"
                "floor_deflated_by_margin"
            )
        else:
            decision, reason = "hold", (
                f"incomplete_rate={rate:.3f}_within_band"
                f"[{rate_low:.3f},{rate_high:.3f}]"
            )
        n_probe = (
            max(1, int(math.floor(usable / effective_floor))) if usable > 0 else 1
        )
        cap = margin_cap
        if decision == "shrink" and n_probe == 1 and cap > usable:
            # Train-bound: one probe record cannot fit p95+margin under the
            # train floor. Keep the whole eval share for decode (no slack
            # reassigned to the train floor, so fitted steps shrink) rather
            # than raising any wall.
            decision = "shrink_steps"
            reason += ";train_bound_keep_eval_share_for_decode"
            cap = max(1.0, usable)
        pareto.update(
            decision=decision,
            reason=reason,
            effective_floor_seconds=float(effective_floor),
        )
    else:
        # Cold start: no measured cost. Sizing the probe at the whole suite is
        # a deadlock — decoding 96 records cannot fit the eval share, the arm
        # is killed at the wall (exit 124), no p95 is ever recorded, and the
        # next cycle cold-starts again. Measured 2026-09-02: the control
        # decoded 80 of 96 records and was interrupted, so it produced an NLL
        # but no document counts, and every cycle scored
        # ``measurement_incomplete``. Spend a bounded probe instead: enough to
        # be decidable, cheap enough to finish and yield the p95 the next
        # cycle fits from.
        n_probe_base = max(0, min(int(smoke_n), _COLD_START_PROBE_RECORDS))
        n_probe = n_probe_base
        cap = configured
    if n_probe > 0 and usable > 0:
        per_probe = usable / float(n_probe)
    else:
        per_probe = usable if usable > 0 else 1.0
    fitted = max(1.0, min(per_probe, cap))
    projected_eval = float(fitted) * float(n_probe)
    allocated_eval = usable
    residual = max(0.0, allocated_eval - projected_eval)
    grown_floor = min_train + residual
    wall_headroom = max(0.0, float(arm_wall_seconds) - overhead - projected_eval)
    grown_floor = min(grown_floor, wall_headroom)
    if grown_floor + projected_eval + overhead > float(arm_wall_seconds) + 1e-9:
        grown_floor = max(0.0, float(arm_wall_seconds) - overhead - projected_eval)
    train_device = _screening_train_device()
    payload = _latest_train_telemetry_payload(telemetry_root)
    sps = _steps_per_sec_from_train_payload(payload) if payload else None
    fitted_steps, steps_fit = _fit_screening_steps(
        floor_seconds=grown_floor,
        measured_steps_per_sec=sps,
        steps_max=_screening_thrash_steps_max(thrash),
    )
    steps_fit["telemetry_path"] = (
        None if payload is None else payload.get("_telemetry_path")
    )
    steps_fit["requested_steps"] = int(requested_steps)
    timeout_cause = _predecessor_timeout_cause(
        telemetry_root, predecessor_campaign_id, evidence=evidence
    )
    meta = {
        "arm_wall_seconds": float(arm_wall_seconds),
        "configured_decode_timeout_seconds": configured,
        "fitted_decode_timeout_seconds": float(fitted),
        "smoke_n": float(smoke_n),
        "n_probe": int(n_probe),
        "n_probe_base": int(n_probe_base),
        "decode_floor_seconds": float(decode_floor),
        "decode_floor_source": decode_floor_source,
        "decode_floor_evidence": evidence,
        "p95_margin": float(margin),
        "p95_margin_timeout_seconds": margin_cap,
        "eval_share_seconds": float(usable),
        "pareto": pareto,
        "timeout_cause": timeout_cause,
        "min_train_floor_seconds": min_train,
        "grown_train_floor_seconds": float(grown_floor),
        "eval_overhead_seconds": overhead,
        "eval_budget_seconds": projected_eval,
        "allocated_eval_seconds": float(allocated_eval),
        "eval_slack_reassigned_seconds": float(residual),
        "clamp_bound": 1.0 if fitted + 1e-9 < configured else 0.0,
        "exceeds_configured": bool(fitted > configured + 1e-9),
        "train_device": train_device,
        "fitted_steps": int(fitted_steps),
        "steps_fit": steps_fit,
        "screening_sample_size": (
            dict(sample_size_report) if sample_size_report else None
        ),
    }
    return float(fitted), meta


def _write_thrash_timing(
    camp_dir: Path,
    *,
    loop_id: str,
    campaign_id: str,
    cycle_index: int | None,
    role: str,
    measurement_complete: bool,
    arm_wall_seconds: float | None,
    decode_fit: dict[str, Any] | None,
    reasons: list[str],
    control_metrics: dict[str, Any] | None,
    candidate_metrics: dict[str, Any] | None,
    thrash_regime: Mapping[str, Any] | None = None,
) -> Path:
    """Durable thrash timing / completeness row for Pareto recalibration."""

    incomplete_reasons = [
        str(r)
        for r in reasons
        if str(r).startswith(
            (
                "measurement_incomplete",
                "empty_metrics",
                "primary_metric_unavailable",
                "harness_failure",
            )
        )
    ]
    has_ss = isinstance(
        (control_metrics or {}).get("structural_similarity"), (int, float)
    ) and isinstance(
        (candidate_metrics or {}).get("structural_similarity"), (int, float)
    )
    payload = {
        "schema": "thrash_timing/v1",
        "loop_id": loop_id,
        "campaign_id": campaign_id,
        "cycle_index": cycle_index,
        "role": role,
        "complete": bool(measurement_complete and has_ss),
        "measurement_complete": bool(measurement_complete),
        "has_dual_arm_ss": has_ss,
        "arm_wall_seconds": arm_wall_seconds,
        "decode_fit": decode_fit,
        "fitted_steps": (decode_fit or {}).get("fitted_steps")
        if isinstance(decode_fit, dict)
        else None,
        "steps_fit": (decode_fit or {}).get("steps_fit")
        if isinstance(decode_fit, dict)
        else None,
        "train_device": (decode_fit or {}).get("train_device")
        if isinstance(decode_fit, dict)
        else None,
        # Budget feedback loop (measured decode floor + Pareto recalibration).
        "decode_floor_source": (decode_fit or {}).get("decode_floor_source")
        if isinstance(decode_fit, dict)
        else None,
        "decode_floor_seconds": (decode_fit or {}).get("decode_floor_seconds")
        if isinstance(decode_fit, dict)
        else None,
        "n_probe": (decode_fit or {}).get("n_probe")
        if isinstance(decode_fit, dict)
        else None,
        "fitted_decode_timeout_seconds": (decode_fit or {}).get(
            "fitted_decode_timeout_seconds"
        )
        if isinstance(decode_fit, dict)
        else None,
        "pareto": (decode_fit or {}).get("pareto")
        if isinstance(decode_fit, dict)
        else None,
        "timeout_cause": (decode_fit or {}).get("timeout_cause")
        if isinstance(decode_fit, dict)
        else None,
        "incomplete_reasons": incomplete_reasons,
        "thrash_regime": dict(thrash_regime) if thrash_regime else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = camp_dir / "thrash_timing.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # camp_dir is root/campaign_id; parent is autoresearch root
    root = camp_dir.parent
    ledger = root / "loops" / loop_id / "thrash_timing.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _formal_lane_required(*, cycle_intent: str, replay: dict[str, Any] | None) -> bool:
    """Reserve formal time only for work whose locked plan requires it."""

    if cycle_intent == "promote":
        return True
    if replay is None:
        return False
    return any(
        bool(replay[arm]["manifest"].formal_obligations)
        for arm in ("control", "candidate")
    )




# Leave schedule margin after fit so the fit→execute deadline check cannot
# fail from monotonic clock drift or float round-trip (observed: remaining
# 159.788399 < required 159.788414 → both promote arms skipped as
# deadline_reserve with zero runs).




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




def _promotion_formal_budget_seconds(
    *, deadline: float, arm_count: int, arm_wall_minutes: float
) -> float:
    """Return only wall time left after reserving complete arms + finalization."""

    reserved = arm_count * arm_wall_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    available = _remaining_timeout(deadline) - reserved
    if available <= 0:
        raise subprocess.TimeoutExpired("promotion formal preflight budget", reserved)
    return min(float(_PROMOTE_FORMAL_TIMEOUT_S), available)


# Chunked promotion measurement (docs/design/chunked-promotion-eval-20260902.md):
# a promotion suite wider than one bounded run is decoded as a locked number of
# resumable ``scripts.evaluate_model`` runs, each its own <= MAX_HARNESS_WALL
# subprocess on the same checkpoint and suite.  Exhausting the locked run
# budget is measurement incomplete, never a model verdict.
_PROMOTION_CHUNK_LEDGER_SCHEMA = "autotrain_promotion_chunks/v1"
# Fixed per-run cost outside record decode (model load, suite setup, scoreboard
# finalization) charged against every chunk run when the policy has no
# ``thrash_timing.eval_overhead_seconds``.
_PROMOTION_CHUNK_OVERHEAD_SECONDS_DEFAULT = 8.0
# evaluate_model exits a chunk may end with and still be resumed: 0 complete,
# 2 (autoresearch stopped), 8 typed gate rejection, 10 resume pending.
_PROMOTION_CHUNK_RESUMABLE_EXITS = frozenset({0, 2, 8, 10})
_PROMOTION_CHUNK_LEDGER_NAME = "promotion_chunks.json"


def _promotion_suite_records(suite: str) -> int | None:
    """Record count of ``suite`` in the resolved default eval version."""

    try:
        from slm_training.autoresearch.engine import default_eval_version
        from slm_training.data.store import DataStore

        records = (
            DataStore().resolve_path("eval", default_eval_version())
            / "suites"
            / suite
            / "records.jsonl"
        )
        if records.is_file():
            return sum(
                1
                for line in records.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    except Exception:  # noqa: BLE001 — plan input only, never fatal
        return None
    return None


def _measured_promotion_decode_p95_seconds(
    root: Path | None,
) -> tuple[float | None, str | None]:
    """Newest measured eval ``latency_ms_p95`` (seconds) under the campaigns root.

    Promotion suites are preferred (``eval_held_out.json``) and the smoke suite
    is the fallback; a run whose p95 is null (all timeouts) is skipped.
    """

    if root is None or not Path(root).is_dir():
        return None, None
    candidates: list[tuple[float, Path]] = []
    for name in ("eval_held_out.json", "eval_smoke.json"):
        for path in Path(root).glob(f"*/runs/*/{name}"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
    for _mtime, path in sorted(candidates, key=lambda row: row[0], reverse=True):
        payload = _read_json(path)
        value = payload.get("latency_ms_p95") if isinstance(payload, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value) / 1000.0, str(path)
    return None, None


def _promotion_chunk_plan(
    policy: Any,
    *,
    root: Path | None = None,
    harness_wall_seconds: float = float(MAX_HARNESS_WALL_SECONDS),
) -> dict[str, Any]:
    """Lock the promotion chunk plan before any arm executes.

    ``records_per_run`` is fitted from the measured decode p95 (grown by the
    policy p95 margin, never below the locked per-record timeout) against the
    harness wall minus the fixed per-run overhead.  ``run_n`` is the number of
    resumable runs that cover every planned record of every promotion suite;
    it is the locked chunk budget.
    """

    from slm_training.autoresearch.climb_policy import (
        decode_timeout_seconds_for_role,
        eval_suites_for_role,
    )
    from slm_training.autoresearch.experiment_campaign import (
        PROMOTION_CHUNK_PLAN_SCHEMA,
    )

    measurement = getattr(policy, "measurement", None)
    measurement = measurement if isinstance(measurement, Mapping) else {}
    suites = [str(suite) for suite in eval_suites_for_role(policy, "promotion")]
    suite_n = max(1, int(measurement.get("promotion_suite_n") or 24))
    timeout = float(decode_timeout_seconds_for_role(policy, "promotion"))
    p95_seconds, p95_source = _measured_promotion_decode_p95_seconds(root)
    thrash = _thrash_timing_block(policy)
    overhead = float(
        thrash.get("eval_overhead_seconds") or _PROMOTION_CHUNK_OVERHEAD_SECONDS_DEFAULT
    )
    margin = max(0.0, float(thrash.get("p95_margin") or 0.0))
    per_record = max(timeout, float(p95_seconds or 0.0) * (1.0 + margin))
    usable = max(0.0, float(harness_wall_seconds) - overhead)
    records_per_run = max(1, int(usable // per_record))
    suite_record_plan: dict[str, dict[str, int | None]] = {}
    total = 0
    for suite in suites:
        available = _promotion_suite_records(suite)
        planned = suite_n if available is None else min(suite_n, available)
        suite_record_plan[suite] = {"available": available, "planned": planned}
        total += planned
    total = max(1, total)
    run_n = max(1, -(-total // records_per_run))
    return {
        "schema": PROMOTION_CHUNK_PLAN_SCHEMA,
        "suites": suites,
        "suite_n": suite_n,
        "suite_record_plan": suite_record_plan,
        "total_record_n": total,
        "decode_timeout_seconds": timeout,
        "measured_decode_p95_seconds": p95_seconds,
        "measured_decode_p95_source": p95_source,
        "p95_margin": margin,
        "per_record_seconds": per_record,
        "chunk_wall_seconds": float(harness_wall_seconds),
        "chunk_overhead_seconds": overhead,
        "records_per_run": records_per_run,
        "run_n": run_n,
        "authority": "locked_before_execution",
    }


def _set_command_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    """Return ``cmd`` with ``flag value`` set exactly once."""

    out = list(cmd)
    if flag in out:
        index = out.index(flag)
        if index + 1 < len(out):
            out[index + 1] = value
            return out
        out.append(value)
        return out
    out.extend([flag, value])
    return out


def _command_flag_value(cmd: Sequence[str], flag: str) -> str | None:
    if flag in cmd:
        index = list(cmd).index(flag)
        if index + 1 < len(cmd):
            return str(cmd[index + 1])
    return None


def _promotion_chunk_eval_command(
    *,
    root: Path,
    campaign_id: str,
    experiment_path: Path,
    run_dir: Path,
    plan: Mapping[str, Any],
) -> list[str]:
    """The locked arm's evaluate command, re-armed as one resumable chunk.

    The command is compiled from the same typed knobs the arm ran with (same
    checkpoint, suites, timeout, eval limit); only the resume flags and the
    chunk wall are (re)set, so no chunk can drift from the locked measurement.
    """

    from slm_training.autoresearch.engine import (
        compile_commands,
        is_latency_probe_command,
    )
    from slm_training.autoresearch.schemas import ExperimentSpec

    store = CampaignStore(campaign_id, root)
    campaign = store.load_campaign()
    experiment = ExperimentSpec.model_validate_json(
        Path(experiment_path).read_text(encoding="utf-8")
    )
    evaluates = [
        list(command)
        for command in compile_commands(campaign, experiment, output_root=root)
        if "scripts.evaluate_model" in command
        and not is_latency_probe_command(command)
    ]
    if not evaluates:
        raise ValueError(f"no evaluate_model command compiled for {experiment_path}")
    cmd = evaluates[-1]
    if "--partial-scoreboard" not in cmd:
        cmd.append("--partial-scoreboard")
    cmd = _set_command_flag(cmd, "--resume-run", str(run_dir))
    cmd = _set_command_flag(
        cmd, "--max-records-this-run", str(int(plan["records_per_run"]))
    )
    return _set_command_flag(
        cmd, "--evaluation-wall-seconds", f"{float(plan['chunk_wall_seconds']):.6f}"
    )


def _promotion_scoreboard_state(run_dir: Path) -> dict[str, Any]:
    """Completion state of an arm's (possibly partial) ``scoreboard.json``."""

    path = Path(run_dir) / "scoreboard.json"
    if not path.is_file():
        return {"exists": False, "complete": False, "pending": None, "decoded": None}
    board = _read_json(path)
    resume = board.get("resume") if isinstance(board, dict) else None
    resume = resume if isinstance(resume, dict) else {}

    def _total(key: str) -> int | None:
        rows = resume.get(key)
        if not isinstance(rows, dict):
            return None
        return sum(int(value) for value in rows.values() if isinstance(value, int))

    return {
        "exists": True,
        # A scoreboard without the key predates resumable evals: complete.
        "complete": board.get("measurement_complete") is not False,
        "pending": _total("pending_record_n"),
        "decoded": _total("decoded_this_run_n"),
    }


def _run_promotion_eval_chunks(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str,
    camp_dir: Path,
    plan: Mapping[str, Any],
    experiment_paths: Mapping[str, Path],
    arm_order: Sequence[str],
) -> dict[str, Any]:
    """Finish every executed arm's promotion measurement under the locked plan.

    Runs are launched in sequence, each its own bounded subprocess (fresh
    ``MAX_RUN_SECONDS`` deadline; eval wall = ``plan.chunk_wall_seconds``) on
    the arm's locked checkpoint and suites, replaying stored records and
    decoding the next ``records_per_run``.  The loop stops at a complete merged
    scoreboard or when ``run_n`` runs are spent (``chunk_budget_exhausted``).
    The ledger is written to ``camp_dir/promotion_chunks.json``.
    """

    run_budget = max(1, int(plan["run_n"]))
    ledger: dict[str, Any] = {
        "schema": _PROMOTION_CHUNK_LEDGER_SCHEMA,
        "campaign_id": campaign_id,
        "plan": dict(plan),
        "arms": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for eid in arm_order:
        run_dir = camp_dir / "runs" / eid
        arm: dict[str, Any] = {
            "run_dir": str(run_dir),
            "run_budget": run_budget,
            "runs_used": 0,
            "runs": [],
            "status": "pending",
        }
        ledger["arms"][eid] = arm
        try:
            cmd = _promotion_chunk_eval_command(
                root=root,
                campaign_id=campaign_id,
                experiment_path=Path(experiment_paths[eid]),
                run_dir=run_dir,
                plan=plan,
            )
        except Exception as exc:  # noqa: BLE001 — typed harness failure, never a verdict
            arm["status"] = "harness_failure"
            arm["error"] = f"{type(exc).__name__}: {exc}"[:400]
            continue
        checkpoint = _command_flag_value(cmd, "--checkpoint")
        checkpoint_path = (
            Path(checkpoint) if checkpoint else run_dir / "checkpoints" / "last.pt"
        )
        if not checkpoint_path.is_file():
            arm["status"] = "no_checkpoint"
            arm["checkpoint"] = str(checkpoint_path)
            continue
        arm["command"] = list(cmd)
        while True:
            state = _promotion_scoreboard_state(run_dir)
            if state["exists"] and state["complete"]:
                arm["status"] = "complete"
                break
            if arm["runs_used"] >= run_budget:
                arm["status"] = "chunk_budget_exhausted"
                break
            index = int(arm["runs_used"]) + 1
            print(
                f"PROMOTION_CHUNK arm={eid} run={index}/{run_budget} "
                f"pending={state['pending']} "
                f"records_per_run={int(plan['records_per_run'])} "
                f"wall_s={float(plan['chunk_wall_seconds']):.0f}",
                flush=True,
            )
            result = _stage_command(
                cmd,
                cwd=cwd,
                # Each chunk is its own bounded run under the repository cap.
                deadline=time.monotonic() + float(MAX_RUN_SECONDS),
                root=root,
                loop_id=loop_id,
                stage=f"promotion-chunk:{eid}:{index}",
            )
            if result.stderr:
                print(result.stderr[-4000:], file=sys.stderr, flush=True)
            if result.timed_out:
                code = 124
            elif result.outcome is ProcessOutcome.LAUNCH_FAILED:
                code = 127
            else:
                code = int(result.returncode or 0)
            arm["runs_used"] = index
            after = _promotion_scoreboard_state(run_dir)
            arm["runs"].append(
                {
                    "index": index,
                    "exit_code": code,
                    "timed_out": bool(result.timed_out),
                    "duration_seconds": float(
                        getattr(result, "duration_seconds", 0.0) or 0.0
                    ),
                    "pending_before": state["pending"],
                    "pending_after": after["pending"],
                    "decoded_this_run_n": after["decoded"],
                    "measurement_complete": bool(after["exists"] and after["complete"]),
                }
            )
            print(
                f"PROMOTION_CHUNK_DONE arm={eid} run={index}/{run_budget} "
                f"exit={code} decoded={after['decoded']} pending={after['pending']} "
                f"complete={bool(after['exists'] and after['complete'])}",
                flush=True,
            )
            if code not in _PROMOTION_CHUNK_RESUMABLE_EXITS or not after["exists"]:
                arm["status"] = "harness_failure"
                arm["error"] = f"chunk {index} exit={code} scoreboard={after['exists']}"
                break
    ledger["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    camp_dir.mkdir(parents=True, exist_ok=True)
    (camp_dir / _PROMOTION_CHUNK_LEDGER_NAME).write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ledger


def _attach_promotion_chunks(
    delivery: dict[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Fold chunk-stage outcomes into the delivery as typed reasons.

    An exhausted chunk budget or a chunk harness failure marks the measurement
    incomplete (retryable, refunded); it is never a model reject.
    """

    reasons = list(delivery.get("reasons") or [])
    incomplete = False
    for eid, arm in (ledger.get("arms") or {}).items():
        status = str(arm.get("status") or "")
        if status == "chunk_budget_exhausted":
            reasons.append(
                f"measurement_incomplete:{eid}:chunk_budget_exhausted:"
                f"runs={arm.get('runs_used')}/{arm.get('run_budget')}"
            )
            incomplete = True
        elif status == "harness_failure":
            reasons.append(
                f"harness_failure:{eid}:promotion_chunk:{arm.get('error') or 'exit'}"
            )
            incomplete = True
        elif status == "no_checkpoint":
            reasons.append(f"measurement_incomplete:{eid}:no_checkpoint_for_chunks")
            incomplete = True
    out = {**delivery, "promotion_chunks": dict(ledger), "reasons": reasons}
    if incomplete:
        out["measurement_complete"] = False
    return out


def _promotion_measurement_incomplete_reasons(
    camp_dir: Path,
    *,
    control_id: str,
    candidate_id: str,
    delivery: Mapping[str, Any],
) -> list[str]:
    """Typed reasons a promotion measurement is still partial evidence.

    A partial scoreboard (``measurement_complete: false``) or a spent chunk
    budget can never be disposed as a model verdict: the disposition is
    ``promotion_inconclusive`` (retryable) until a later run merges the suite
    to completion or the locked chunk budget is exhausted for good.
    """

    reasons: list[str] = []
    ledger = delivery.get("promotion_chunks")
    if isinstance(ledger, Mapping):
        for eid, arm in (ledger.get("arms") or {}).items():
            status = str(arm.get("status") or "")
            if status == "chunk_budget_exhausted":
                reasons.append(
                    f"measurement_incomplete:{eid}:chunk_budget_exhausted:"
                    f"runs={arm.get('runs_used')}/{arm.get('run_budget')}"
                )
    for run_id in (control_id, candidate_id):
        if not run_id:
            continue
        state = _promotion_scoreboard_state(camp_dir / "runs" / run_id)
        if state["exists"] and not state["complete"]:
            reasons.append(
                f"measurement_incomplete:{run_id}:partial_scoreboard:"
                f"pending={state['pending']}"
            )
    return list(dict.fromkeys(reasons))


def _merged_promotion_power_feasibility(
    camp_dir: Path,
    *,
    control_id: str,
    candidate_id: str,
    locked: dict[str, Any] | None,
    primary_metric: str,
) -> dict[str, Any] | None:
    """Power feasibility at the final merged n, not the planned n.

    The locked report admits the planned geometry; the disposition must judge
    the records actually completed in *both* arms of the primary suite (the
    paired sign test cannot use more pairs than the smaller arm completed).
    When either scoreboard is missing or still partial the locked report is
    returned unchanged (the incomplete path decides, never this gate).
    """

    if not isinstance(locked, dict):
        return None
    suite = primary_metric.rsplit(".", 1)[0] if "." in primary_metric else "held_out"
    merged_n: int | None = None
    for run_id in (control_id, candidate_id):
        if not run_id:
            return dict(locked)
        path = camp_dir / "runs" / run_id / "scoreboard.json"
        if not path.is_file():
            return dict(locked)
        board = _read_json(path)
        if not isinstance(board, dict) or board.get("measurement_complete") is False:
            return dict(locked)
        suites = board.get("suites")
        row = suites.get(suite) if isinstance(suites, dict) else None
        completed = row.get("completed_document_n") if isinstance(row, dict) else None
        if type(completed) is not int or completed < 0:
            return dict(locked)
        merged_n = completed if merged_n is None else min(merged_n, completed)
    if merged_n is None:
        return dict(locked)
    from slm_training.autoresearch import evidence_ledger as _ev

    report = _ev.power_feasibility_report(
        max(1, int(merged_n)), _ev.parse_alpha(locked.get("alpha"))
    )
    if merged_n < 1:
        report["decisive"] = False
    return {
        **report,
        "locked_n": locked.get("n"),
        "locked_decisive": locked.get("decisive"),
        "merged_n": int(merged_n),
        "merged_suite": suite,
        "source": "merged_scoreboard",
    }












def _bind_expected_arms(
    *,
    root: Path,
    campaign_id: str,
    matrix_path: Path,
    control_id: str,
    candidate_id: str,
    arm_order: Sequence[str],
    candidate_ids: Sequence[str] | None = None,
    selection_rule: str | None = None,
) -> dict[str, Any]:
    """Bind the exact paired decision arms before execution starts.

    The matrix and handoff are human-facing projections; this event is the
    append-only authority used to distinguish an unstarted arm from a model
    result when a bounded cycle runs out of wall time.
    """

    extras = [str(item) for item in (candidate_ids or ()) if str(item)]
    seen: list[str] = []
    for arm_id in (str(control_id), str(candidate_id), *extras):
        if arm_id and arm_id not in seen:
            seen.append(arm_id)
    expected = tuple(seen)
    if len(expected) < 2 or str(control_id) not in expected:
        raise ValueError("decision arms must contain distinct control/candidate ids")
    order = tuple(str(item) for item in arm_order)
    if set(order) != set(expected) or len(order) != len(expected):
        raise ValueError("arm order must contain exactly the bound decision arms")
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_id = str(matrix_payload.get("matrix_id") or "")
    if not matrix_id:
        raise ValueError("hypothesis matrix is missing matrix_id")
    detail: dict[str, Any] = {
        "matrix_id": matrix_id,
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "expected_arm_ids": list(expected),
        "arm_order": list(order),
    }
    if len(expected) > 2:
        detail["shared_control"] = True
    if selection_rule:
        detail["selection_rule"] = str(selection_rule)
    store = CampaignStore(campaign_id, root)
    for event in reversed(store.verify_event_chain()):
        if event.get("event_type") != "decision_arms_bound":
            continue
        if event.get("detail") != detail:
            raise RuntimeError(
                "decision arm binding already exists with different content"
            )
        return event
    return store.append_event(
        "decision_arms_bound",
        status="bound",
        detail=detail,
    )




def _lock_screening_multi_arm_campaign(
    *,
    root: Path,
    campaign_id: str,
    experiments: Sequence[Mapping[str, Any]],
    control_id: str,
    candidate_ids: Sequence[str],
    seeds: Sequence[int],
    selection_rule: str,
    commit: str,
    role: str,
    policy: Any,
) -> ExperimentCampaignV1:
    by_eid = {str(exp["experiment_id"]): dict(exp) for exp in experiments}
    rec = str(candidate_ids[0])
    base = _manifest(
        campaign_id, by_eid[rec], commit, role=role, policy=policy
    )
    arms = [
        CampaignArmV1(
            arm_id=control_id,
            role="control",
            config_sha256=hashlib.sha256(
                json.dumps(
                    by_eid[control_id].get("knobs") or {},
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest(),
        )
    ]
    for eid in candidate_ids:
        arms.append(
            CampaignArmV1(
                arm_id=eid,
                role="candidate",
                config_sha256=hashlib.sha256(
                    json.dumps(
                        by_eid[eid].get("knobs") or {},
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest(),
            )
        )
    locked = base.model_copy(
        update={
            "experiment_id": rec,
            "arms": tuple(arms),
            "seeds": tuple(int(s) for s in seeds),
            "selection_rule": selection_rule,
            "stopping_rules": (
                *base.stopping_rules,
                f"Select winner by {selection_rule} after locked arms finish.",
            ),
        }
    )
    store = CampaignStore(campaign_id, root)
    store.lock_experiment_campaign(locked)
    man_path = store.root / "manifests" / f"{rec}.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(locked.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return locked




















def _run_arm_eval_nll(
    run_dir: Path,
    *,
    test_dir: Path | None = None,
    checkpoint: Path | None = None,
    model: Any = None,
    nll_config: Any = None,
    eval_nll: float | None = None,
    definition_hash: str | None = None,
    records: Mapping[str, float] | None = None,
    eval_version: str | None = None,
) -> dict[str, Any]:
    """Write canonical ``suites.smoke.eval_nll`` (arm loss weights ignored).

    Diagnostic only: does not touch ship gates. Uses
    ``evaluate_loss_suites`` with a cycle-shared baseline ``nll_config`` over
    the whole smoke suite under ``test_dir``. Per-record broad NLL rows are
    persisted to ``eval_nll_records.json`` (``{record_id: nll}`` plus the
    definition hash) so screening can pair arms record-by-record; the suite
    mean lands in the scoreboard exactly as before.
    """
    from slm_training.autoresearch.climb_policy import screening_nll_definition_hash

    digest = definition_hash or screening_nll_definition_hash()
    value = eval_nll
    report: dict[str, Any] | None = None
    per_record: dict[str, float] | None = (
        {str(k): float(v) for k, v in records.items()} if records is not None else None
    )
    if value is None:
        from slm_training.evals.denoising_nll import DenoisingNLLConfig
        from slm_training.evals.loss_suites import (
            LOSS_SUITE_VERSION,
            evaluate_loss_suites,
            load_suite_spec,
        )

        if model is None:
            if checkpoint is None:
                raise ValueError("eval_nll requires model, checkpoint, or eval_nll")
            from slm_training.models.twotower import TwoTowerModel

            model = TwoTowerModel.from_checkpoint(checkpoint, device="cpu")
        if test_dir is None:
            raise ValueError("eval_nll compute path requires test_dir")
        spec = load_suite_spec(LOSS_SUITE_VERSION)
        cfg = nll_config or DenoisingNLLConfig(
            suite_version=LOSS_SUITE_VERSION,
            mask_rates=tuple(
                float(r)
                for r in (spec.get("mask_rates") or [0.15, 0.30, 0.50, 0.70, 0.85])
            ),
            mask_seed=int(spec.get("mask_seed", 0) or 0),
            compute_legal_support=False,
        )
        report = evaluate_loss_suites(
            model,
            Path(test_dir),
            nll_config=cfg,
            base_suite="smoke",
            ood_suite="smoke",
        )
        broad = (report.get("categories") or {}).get("broad") or {}
        mean = (broad.get("aggregate") or {}).get("mean_nll")
        if mean is None:
            mean = (report.get("aggregate") or {}).get("weighted_nll")
        if not isinstance(mean, (int, float)):
            raise ValueError("evaluate_loss_suites did not yield a finite smoke NLL")
        value = float(mean)
        from slm_training.evals.loss_suites import per_record_nll_map

        per_record = per_record_nll_map(report)
    scoreboard_path = Path(run_dir) / "scoreboard.json"
    scoreboard = _read_json(scoreboard_path)
    suites = scoreboard.get("suites")
    if not isinstance(suites, dict):
        suites = {}
        scoreboard["suites"] = suites
    smoke = suites.get("smoke")
    if not isinstance(smoke, dict):
        smoke = {}
        suites["smoke"] = smoke
    smoke["eval_nll"] = float(value)
    smoke["eval_nll_definition_hash"] = digest
    smoke["eval_nll_claim_class"] = "diagnostic"
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    records_path: Path | None = None
    if per_record is not None:
        smoke["eval_nll_n_records"] = len(per_record)
        records_path = Path(run_dir) / _EVAL_NLL_RECORDS_NAME
        records_path.write_text(
            json.dumps(
                {
                    "schema": _EVAL_NLL_RECORDS_SCHEMA,
                    "definition_hash": digest,
                    "claim_class": "diagnostic",
                    "suite": "smoke",
                    "eval_version": eval_version,
                    "n_records": len(per_record),
                    "mean_nll": float(value),
                    "records": {k: per_record[k] for k in sorted(per_record)},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    scoreboard_path.write_text(
        json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "eval_nll": float(value),
        "definition_hash": digest,
        "scoreboard": str(scoreboard_path),
        "report": report,
        "records": per_record,
        "records_path": str(records_path) if records_path else None,
    }






def _attach_screening_eval_nll(
    run_dir: Path,
    *,
    exit_code: int | None = None,
    eval_version: str | None = None,
) -> dict[str, Any] | None:
    """Compute canonical smoke.eval_nll for any arm that has a checkpoint.

    Independent of the decode-heavy quality eval: runs whether that eval
    succeeded, crashed, or timed out (``exit_code`` is logged only), needs no
    prior scoreboard, scores the whole smoke suite the arm was assigned, and
    persists per-record rows. Idempotent once both the aggregate and the
    per-record file exist.
    """

    run_dir = Path(run_dir)
    scoreboard_path = run_dir / "scoreboard.json"
    suites = (
        _read_json(scoreboard_path).get("suites") if scoreboard_path.is_file() else None
    )
    smoke = suites.get("smoke") if isinstance(suites, dict) else None
    has_aggregate = isinstance(smoke, dict) and isinstance(
        smoke.get("eval_nll"), (int, float)
    )
    if has_aggregate and (run_dir / _EVAL_NLL_RECORDS_NAME).is_file():
        return None
    summary = _read_json(run_dir / "train_summary.json")
    checkpoint = Path(str(summary.get("checkpoint") or ""))
    if not checkpoint.is_file():
        print(
            f"EVAL_NLL_SKIP run={run_dir.name} reason=missing_checkpoint"
            f" exit={exit_code}",
            flush=True,
        )
        return None
    from slm_training.data.store import DataStore

    eval_id = eval_version or _arm_eval_version(run_dir) or default_eval_version()
    test_dir = DataStore().resolve_path("eval", eval_id)
    try:
        out = _run_arm_eval_nll(
            run_dir,
            test_dir=test_dir,
            checkpoint=checkpoint,
            eval_version=eval_id,
        )
    except Exception as exc:  # noqa: BLE001 — NLL is diagnostic, never abort quality
        print(
            f"EVAL_NLL_SKIP run={run_dir.name} exit={exit_code} err={exc!r}",
            flush=True,
        )
        return None
    n_records = len(out.get("records") or {})
    print(
        f"EVAL_NLL run={run_dir.name} nll={out.get('eval_nll')}"
        f" n_records={n_records} eval_version={eval_id} exit={exit_code}",
        flush=True,
    )
    return out


# Phase A positive classification: latency is never a free win over quality.
# Smoke fixture n≈3 → one meaningful program is ~1/3. Below that a latency
# blip is not a real win (parse-only / empty-meaning arms).
# Quality improvements may pay up to this latency regression (relative or abs).
# ~12s wall-band noise must not mint positives.
_WIN_REASON_PREFIXES = (
    "primary_metric_win:",
    "quality_metric_win:",
    "efficiency_win:",
    "executable_unblock:",
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
# Recipe levers that define "same knobs" for confirmatory retest. Measurement
# knobs (seed, decode_timeout, eval_suites) are re-sampled from role policy.
# Dedup identity ignores cycle-local steps jitter (continuous does steps+(cycle%3)).
# Soft bound: confirm/promote attempts before the queue head is rejected.
_MAX_CONFIRM_ATTEMPTS = 2
_MAX_PROMOTE_ATTEMPTS = 2
_CAUSAL_FAMILY_ATTEMPT_CAP = 2
# Screening thrash bank — rotate recommended arm each cycle (Change B).
# Each entry: (slug, hypothesis fragment, knob extras relative to control).
# Special key "_steps_factor" multiplies base steps (depth confound).
# Data-volume arms (RC7, docs/design/autotrain-recovery-2-p9-20260902.md).
# The matched control trains on the climb policy ``defaults.train_version``
# (policy.v3: the certified, eval-decontaminated train bucket
# ``openui_verified_train_v2``); each data arm swaps only the corpus id at
# identical model size. A data arm whose corpus equals the control corpus is
# a self-control (delta identically 0) and ``_all_screening_arm_bank`` drops
# it, so ``data-certified`` is live only when a loop trains its control on a
# smaller corpus (legacy ``wf_smoke_v2`` pins). ``openui_verified_v1`` is
# never a train arm: it carries the validation/test families the certified
# smoke suites are sampled from.
_DATA_ARM_CERTIFIED_TRAIN_VERSION = "openui_verified_train_v2"  # 1,054 records
# ``hillclimb_strict_v2`` is NOT a legal data arm against the certified smoke
# suites. Measured 2026-09-02 against e938_role_safe_all_targets_smoke96_v1:
# 6 identical programs / 3 identical prompts / 16 shared root families with
# the smoke suite, and 7 / 2 / 5 with held_out. Its own decontamination
# indexed e938_..._v2 and the smoke6/smoke24 snapshots, never smoke96_v1 or
# heldout24_v1, so the overlap was never gated. An arm trained on it would
# score the eval it memorized: a leakage win, not a capability win. Its
# synthesis feedback is also uncleared for SFT (blocking-class
# ``eval_leakage_source`` on three families plus dup_share 0.55-0.93), so the
# arm could not even train. Re-admit it only after a rebuild that
# decontaminates against the live suites and clears the SFT gate.
_LEAKED_TRAIN_VERSIONS: frozenset[str] = frozenset(
    {"hillclimb_strict_v2", "wf_smoke_v2"}
)
#: The one leaked corpus that used to be a screening *arm* (``data-strict``).
#: Only this one is reverse-classified from historical knobs; ``wf_smoke_v2``
#: was a control corpus, so naming it there would relabel legacy control rows.
_EX_DATA_ARM_LEAKED_TRAIN_VERSION = "hillclimb_strict_v2"
# Control recipe values the size-matched recipe arms are defined against
# (``_matrix`` control: batch_size=2; ModelBuildConfig.lr default 3e-4, the
# control never sets ``lr``).
_CONTROL_RECIPE_LR = 3e-4
_CONTROL_RECIPE_BATCH_SIZE = 2
# ``steps-fill``: the fitter sizes control steps at ``floor * sps * 0.9``;
# the fill arm spends the remaining 10 % of the fitted train floor (charged as
# wall time, never parameters). ``_apply_arm_extras`` applies the ``base + 10``
# depth-confound minimum only to depth arms (factor >= 2); a fill factor gets
# ``base + 1`` so it never overshoots the floor.
_STEPS_FILL_FACTOR = round(1.0 / _STEPS_PER_SEC_SAFETY, 4)
#: Decoded-probe records to spend on a cold-start cycle, before any per-record
#: decode cost has been measured. The exact sign test at alpha = 1/20 needs 6
#: pairs, so 6 is the smallest probe that can be decisive on its own; sizing
#: the cold probe at the whole suite instead deadlocks the loop (see the
#: cold-start branch of the decode fit).
_COLD_START_PROBE_RECORDS = 6
# ``noise_rate`` (policy ``recipe_tweak_knobs``) is deliberately absent: it is
# a ``StubModel``-only lever (``harnesses/model_build/plugin.py``) and not an
# ``ExperimentKnobs`` field, so a noise-rate arm could never move weights.
# ExperimentKnobs-only keys with no ``lever_catalog()`` row (category used
# when classifying bank arms; ``train_version`` selects the data corpus).
_SCREENING_ARM_BANK: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "data-certified",
        "Training on the certified, eval-decontaminated openui_verified_train_v1 bucket (1,083 records) instead of the loop control corpus lowers smoke eval_nll at fixed model size without lowering parse_rate.",
        {"train_version": _DATA_ARM_CERTIFIED_TRAIN_VERSION},
    ),
    (
        "lr-x2",
        "Doubling the learning rate (3e-4 -> 6e-4) at fixed steps and model size lowers smoke eval_nll without lowering parse_rate.",
        {"lr": _CONTROL_RECIPE_LR * 2},
    ),
    (
        "lr-x0.5",
        "Halving the learning rate (3e-4 -> 1.5e-4) at fixed steps and model size lowers smoke eval_nll without lowering parse_rate.",
        {"lr": _CONTROL_RECIPE_LR / 2},
    ),
    (
        "batch-x2",
        "Doubling batch_size (2 -> 4) at fixed steps and model size lowers smoke eval_nll without lowering parse_rate.",
        {"batch_size": _CONTROL_RECIPE_BATCH_SIZE * 2},
    ),
    (
        "steps-fill",
        "Filling the fitted train floor (steps x 1/0.9, wall-time charged, no extra parameters) lowers smoke eval_nll without lowering parse_rate.",
        {"_steps_factor": _STEPS_FILL_FACTOR},
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
        "slot-component-fidelity-coupling",
        "Coupling implemented slot-component ownership with stronger placeholder-fidelity supervision improves component_type_recall and placeholder_fidelity without lowering parse_rate or structural_similarity.",
        {
            "slot_component_loss_weight": 1.0,
            "slot_component_decode_weight": 1.0,
            "fidelity_loss_weight": 1.5,
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "slot-component-inventory-coupling",
        "Coupling implemented slot-component ownership with component-inventory supervision improves component_type_recall and structural_similarity without lowering parse_rate or binder_reference_f1.",
        {
            "slot_component_loss_weight": 1.0,
            "slot_component_decode_weight": 1.0,
            "component_inventory_loss_weight": 1.0,
            "component_inventory_decode_weight": 1.0,
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "slot-component-exposure-cap",
        "Targeted rare-component exposure with a per-template cap plus implemented slot-component supervision improves component_type_recall and structural_similarity without buying capacity.",
        {
            "slot_component_loss_weight": 1.0,
            "slot_component_decode_weight": 1.0,
            "mixture_sampling_policy": "exposure_targeted",
            "mixture_per_template_cap": 2,
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "slot-contract-context",
        "Providing the canonical slot contract in context improves component_type_recall, placeholder_fidelity, and structural_similarity without weakening constrained decoding or increasing model size.",
        {
            "slot_contract_in_context": True,
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "constraint-graph",
        "Grammar constraint-graph conditioning improves structural_similarity and meaningful-program rate at fixed model size while preserving fail-closed constrained decoding.",
        {
            "constraint_graph_mode": "grammar",
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
            # train_loop requires batch_size >= 3 for contrast base-corpus row
            "batch_size": 3,
        },
    ),
    (
        "semantic-contrast-compiler-margin",
        "Combining hard-valid semantic contrast with grammar-oracle compiler alignment improves structural_similarity and exact structural agreement at fixed model size.",
        {
            "semantic_contrast_dir": "src/slm_training/resources/data/eval/openui_hard_valid_v1",
            "semantic_contrast_loss_weight": 0.25,
            "semantic_contrast_margin": 1.0,
            "semantic_contrast_fraction": 0.5,
            "batch_size": 3,
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_kind_filter": "all",
            "compiler_decode_mode": "tree",
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
        "scaffold-prefix-structure",
        "Prefix-weighted LTR plus STRUCT-token reconstruction couples early scaffold formation to grammar structure tokens without lowering parse_rate or binder_reference_f1.",
        {
            "ltr_prefix_loss_weight": 1.0,
            "structure_token_loss_weight": 1.0,
        },
    ),
    (
        "scaffold-prefix-tail",
        "Prefix- and tail-weighted LTR jointly improve early scaffold formation and legal termination without lowering parse_rate or binder_reference_f1.",
        {
            "ltr_prefix_loss_weight": 1.0,
            "ltr_tail_loss_weight": 1.0,
        },
    ),
    (
        "component-token",
        "Direct component-token reconstruction weighting improves component_type_recall and structural_similarity without lowering parse_rate or binder_reference_f1.",
        {"component_token_loss_weight": 1.0},
    ),
    (
        "component-token-prefix",
        "Joint component-token and prefix-LTR weighting improves component recall while stabilizing early scaffold formation without lowering parse_rate.",
        {
            "component_token_loss_weight": 1.0,
            "ltr_prefix_loss_weight": 1.0,
        },
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
        "literal-close-structure",
        "Tail-weighted LTR plus STRUCT-token reconstruction couples legal termination to grammar structure without lowering parse_rate or binder_reference_f1.",
        {
            "ltr_tail_loss_weight": 2.0,
            "structure_token_loss_weight": 1.0,
        },
    ),
    (
        "literal-close-component-token",
        "Tail-weighted LTR plus component-token reconstruction improves termination and component recall without lowering parse_rate.",
        {
            "ltr_tail_loss_weight": 2.0,
            "component_token_loss_weight": 1.0,
        },
    ),
    (
        "literal-close-typed-balance",
        "Tail-weighted LTR with count-normalized typed-family balance stabilizes component mix while fixing legal termination.",
        {
            "ltr_tail_loss_weight": 2.0,
            "typed_family_balance_loss_weight": 0.25,
        },
    ),
    (
        "symbol-boundary-structure",
        "Opaque-symbol boundary supervision plus STRUCT-token reconstruction improves binder fidelity and scaffold structure without lowering parse_rate.",
        {
            "symbol_boundary_loss_weight": 1.0,
            "structure_token_loss_weight": 1.0,
        },
    ),
    (
        "semantic-contrast-structure",
        "Hard-valid semantic contrast with STRUCT-token reconstruction improves structural_similarity at fixed size without lowering parse_rate.",
        {
            "semantic_contrast_dir": "src/slm_training/resources/data/eval/openui_hard_valid_v1",
            "semantic_contrast_loss_weight": 0.25,
            "semantic_contrast_margin": 1.0,
            "semantic_contrast_fraction": 0.5,
            "batch_size": 3,
            "structure_token_loss_weight": 1.0,
        },
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
    (
        "solver-energy-rerank",
        "A trained candidate-energy reranker over legal compiler decisions improves smoke structural_similarity without lowering parse_rate or binder_reference_f1.",
        {
            "solver_energy_loss_weight": 1.0,
            "solver_energy_decode_weight": 1.0,
            "structural_aux_head_profile": "solver-energy",
            "compiler_decode_mode": "tree",
        },
    ),
    (
        "legal-edit-hazard",
        "A flow-matching hazard head over legal compiler decisions improves smoke structural_similarity without lowering parse_rate or binder_reference_f1.",
        {
            "legal_edit_hazard_loss_weight": 1.0,
            "legal_edit_hazard_decode_weight": 1.0,
            "structural_aux_head_profile": "legal-edit-hazard",
            "compiler_decode_mode": "tree",
        },
    ),
)
# Latency-only arms (RC7): ``bounds`` / ``canvas`` / ``both`` set only decode
# cost levers and cannot move trained weights, so under a quality/NLL
# screening primary they are a guaranteed null (evidence_ledger.v1: ``bounds``
# n_complete=50, mean_delta=0.0, m2_delta=0.0; ``canvas`` n_complete=5, all
# null). ``steps`` (x2 depth confound) and ``batch1`` are training-recipe
# levers whose preregistered hypotheses are latency / cost claims, so they
# ride with this bank. ``_all_screening_arm_bank`` prepends the bank only when
# the screening role primary leaf is ``latency_ms_p50`` (historical rotation
# order preserved under that primary). Slugs are stable so ledger history
# stays attached. Arms with ``compiler_*_loss_weight`` knobs stay in the
# screening bank: ``lever_catalog()`` labels them ``decode`` by name prefix,
# but the weights enter the training objective (``twotower.py``
# compiler-alignment and decision-token losses).
_LATENCY_ARM_BANK: tuple[tuple[str, str, dict[str, Any]], ...] = (
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
)
# Static data-volume arm slugs (corpus swap is their only lever).
_STATIC_DATA_ARM_SLUGS: frozenset[str] = frozenset(
    slug
    for slug, _, extras in _SCREENING_ARM_BANK
    if set(k for k in extras if not str(k).startswith("_")) == {"train_version"}
)
# Permanent thrash-arm closure requires this many *distinct seeds* of complete
# non-positive measurement. A single fixture-noise null must not close the
# approach forever (aligns with climb recipe_null_cap / promotion min_seeds).
_DEFAULT_ARM_CLOSE_MIN_NULL_SEEDS = 2
# Handoff reasons use "quality-arm bank"; runtime uses "screening arm bank".
# Loop-local thrash successors synthesized when the static bank is multi-seed
# exhausted. Persistent under loops/<id>/dynamic_thrash_arms.jsonl so the
# continuous driver self-heals without a human re-prompt.
_DYNAMIC_THRASH_ARMS: list[tuple[str, str, dict[str, Any]]] = []
_DYNAMIC_THRASH_LOADED_FOR: str | None = None
_THRASH_COMPOSE_ATOMS: tuple[tuple[str, Any], ...] = (
    ("ltr_tail_loss_weight", 2.0),
    ("ltr_prefix_loss_weight", 1.0),
    ("structure_token_loss_weight", 1.0),
    ("component_token_loss_weight", 1.0),
    ("symbol_boundary_loss_weight", 1.0),
    ("typed_family_balance_loss_weight", 0.25),
    ("design_md_dropout", 0.25),
    ("compiler_decision_token_loss_weight", 1.0),
    ("grammar_completion_bounds", True),
    ("compact_active_canvas", True),
)
_SELF_HEAL_BANK_BATCH = 5












# Test / driver hook: when set, wins over the climb policy's screening primary.
_SCREENING_PRIMARY_LEAF_OVERRIDE: str | None = None


def _screening_primary_leaf() -> str:
    """Leaf of the screening role primary metric (``smoke.eval_nll`` -> ``eval_nll``)."""
    if _SCREENING_PRIMARY_LEAF_OVERRIDE:
        return str(_SCREENING_PRIMARY_LEAF_OVERRIDE).rsplit(".", 1)[-1]
    try:
        from slm_training.autoresearch.climb_policy import (
            load_climb_policy,
            primary_for_role,
        )

        metric = str(primary_for_role(load_climb_policy(), "screening").get("metric") or "")
    except Exception:  # noqa: BLE001 — bank must stay computable without policy
        return ""
    return metric.rsplit(".", 1)[-1]


def _latency_arms_active() -> bool:
    """Latency-only arms are legal screening candidates only under a latency primary."""
    return _screening_primary_leaf() == LATENCY_PRIMARY_LEAF


def _arm_is_self_control(extras: Mapping[str, Any] | None) -> bool:
    """True for a data arm whose only lever is the control's own train corpus.

    Such an arm trains the control recipe twice (delta identically 0), so it
    is never a legal screening candidate — the policy default corpus decides.
    """
    public = {k: v for k, v in (extras or {}).items() if not str(k).startswith("_")}
    if set(public) != {"train_version"}:
        return False
    return str(public["train_version"] or "") == _default_screening_train_version()








def _all_screening_arm_bank() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Legal screening bank for the active role primary.

    Static preregistered arms minus self-control data arms; the latency-only
    bank is prepended only under a ``latency_ms_p50`` primary; loop-local
    self-heal successors follow.
    """
    bank = tuple(row for row in _SCREENING_ARM_BANK if not _arm_is_self_control(row[2]))
    if _latency_arms_active():
        bank = _LATENCY_ARM_BANK + bank
    if not _DYNAMIC_THRASH_ARMS:
        return bank
    return bank + tuple(_DYNAMIC_THRASH_ARMS)


def _recent_exhaustion_cycle_window() -> int:
    return max(len(_SCREENING_ARM_BANK), len(_all_screening_arm_bank()), 1)


# Back-compat alias for tests that walked the static bank length.
_RECENT_EXHAUSTION_CYCLE_WINDOW = len(_SCREENING_ARM_BANK)

# Cycle status returned when a parked terminal verdict short-circuits run_cycle.






def _screening_bank_fingerprint(policy_sha256: str | None = None) -> str:
    """Deterministic identity of the legal screening domain for park/resume.

    sha256 over canonical JSON of the sorted legal screening bank (slug +
    public knobs from ``_all_screening_arm_bank``, process arms excluded),
    the climb-policy sha256, and ``MAX_RUN_MINUTES``. Heal/resume arms are
    execution vehicles and have their own park/resume gate
    (``_selectable_process_arm``); including them made leftover=[] park
    then immediately unpark when the spent heal snapshot was tombstoned.
    """
    from slm_training.levers import MAX_RUN_MINUTES
    from slm_training.lineage.records import canonical_json

    if policy_sha256 is None:
        try:
            from slm_training.autoresearch.climb_policy import load_climb_policy

            policy_sha256 = load_climb_policy().sha256
        except Exception:  # noqa: BLE001 — fingerprint must stay computable
            policy_sha256 = ""
    arms = sorted(
        (
            (
                slug,
                {k: v for k, v in extras.items() if not str(k).startswith("_")},
            )
            for slug, _, extras in _all_screening_arm_bank()
            if not _is_process_arm(extras)
        ),
        key=lambda item: item[0],
    )
    body = {
        "schema": "autotrain_bank_fingerprint/v1",
        "arms": [[slug, knobs] for slug, knobs in arms],
        "climb_policy_sha256": str(policy_sha256 or ""),
        "max_run_minutes": int(MAX_RUN_MINUTES),
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


_PROCESS_ARM_KNOB_KEYS = frozenset({"heal_resume", "process_arm", "process_role"})




def _train_version_has_complete_nonpositive(
    root: Path, predecessor_campaign_id: str | None, train_version: str
) -> bool:
    """True when this exact heal snapshot already has a complete non-positive."""
    if not predecessor_campaign_id or not train_version:
        return False
    for camp_id in _lineage_campaign_ids(root, predecessor_campaign_id):
        camp_dir = root / camp_id
        delivery = _read_json(camp_dir / "sdlc_delivery.json")
        if delivery.get("positive") is not False:
            continue
        if delivery.get("measurement_complete") is not True:
            continue
        candidate_id = str(delivery.get("candidate_id") or "")
        knobs = _load_experiment_knobs(camp_dir, candidate_id) if candidate_id else {}
        if not knobs:
            knobs = _matrix_experiment_knobs(
                _read_json(camp_dir / "matrix-proposal.json"), candidate_id
            )
        if str((knobs or {}).get("train_version") or "") == train_version:
            return True
    return False


def _selectable_process_arm(
    root: Path, loop_id: str, *, predecessor_campaign_id: str | None
) -> bool:
    """True when a process arm is registered and still a screening candidate.

    Registration is not selectability: a spent heal snapshot stays in the
    dynamic bank until retired, and unparking on mere presence is the
    park/resume spin (leftover=[] → REGIME_RESUMED heal_resume_arm_open).
    """
    _load_dynamic_thrash_arms(root, loop_id)
    retired = _retired_heal_versions(root, loop_id)
    closed = (
        _recent_completed_nonpositive_slugs(root, predecessor_campaign_id)
        if predecessor_campaign_id
        else set()
    )
    for slug, _, extras in _all_screening_arm_bank():
        if not _is_process_arm(extras):
            continue
        version = str(extras.get("train_version") or "")
        if version and version in retired:
            continue
        if slug in closed:
            if not version or _train_version_has_complete_nonpositive(
                root, predecessor_campaign_id, version
            ):
                continue
        return True
    return False


def _check_regime_parked(
    *, root: Path, loop_id: str, cwd: Path | None = None
) -> str | None:
    """Deterministic park/resume predicate over a persisted terminal verdict.

    Returns ``_REGIME_PARKED_STATUS`` while the bank identity matches the
    verdict's fingerprint; archives the verdict, restores the loop state, and
    returns ``None`` once the fingerprint moves (bank/policy/budget changed).
    A completed heal snapshot whose resume-arm registration was lost is
    re-registered here — an acked rebuild_data must never park forever.
    A spent (null-closed or tombstoned) heal arm is not a resume reason.
    """
    path = _terminal_verdict_path(root, loop_id)
    if not path.is_file():
        return None
    verdict = _read_json(path)
    predecessor = str(verdict.get("campaign_id") or "") or None
    _load_dynamic_thrash_arms(root, loop_id)
    if not _selectable_process_arm(
        root, loop_id, predecessor_campaign_id=predecessor
    ):
        _recover_heal_resume_arm(
            root, loop_id, cwd=cwd, predecessor_campaign_id=predecessor
        )
    if _selectable_process_arm(
        root, loop_id, predecessor_campaign_id=predecessor
    ):
        resolved = path.with_name(
            f"terminal_verdict.resolved.c{int(verdict.get('cycle_index') or 0)}.json"
        )
        path.replace(resolved)
        print("REGIME_RESUMED reason=heal_resume_arm_open", flush=True)
        _clear_loop_blocker(root, loop_id, reason="regime_resumed_heal_resume_arm")
        return None
    stored = verdict.get("bank_fingerprint")
    if stored and stored == _screening_bank_fingerprint():
        print(
            f"REGIME_PARKED loop={loop_id} "
            f"constraint={verdict.get('binding_constraint')}",
            flush=True,
        )
        return _REGIME_PARKED_STATUS
    resolved = path.with_name(
        f"terminal_verdict.resolved.c{int(verdict.get('cycle_index') or 0)}.json"
    )
    path.replace(resolved)
    print("REGIME_RESUMED reason=bank_identity_changed", flush=True)
    _clear_loop_blocker(root, loop_id, reason="regime_resumed_bank_identity_changed")
    return None


def _park_screening_saturation(
    *,
    root: Path,
    loop_id: str,
    campaign_id: str,
    cycle_index: int,
    policy: Any,
    ranked_regimes: Sequence[str],
    cwd: Path | None = None,
) -> str:
    """Persist the typed terminal verdict once bounded residual recovery closes."""

    from slm_training.autoresearch import evidence_ledger as _ev

    handoff_path = root / campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        raise RuntimeError(
            "screening objective saturated without a typed predecessor handoff"
        )
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    actions = _capability_objective_refresh_actions(
        root=root,
        campaign_id=campaign_id,
        preserved_actions=tuple(
            action for action in handoff.actions if action.kind == "document"
        ),
    )
    handoff_path.write_text(
        handoff.model_copy(update={"actions": actions}).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    verdict = _ev.build_regime_exhausted_verdict(
        campaign_id=campaign_id,
        loop_id=loop_id,
        cycle_index=cycle_index,
        binding_constraint="screening_objective_saturated",
        closed_slugs=sorted(set(ranked_regimes)),
        policy_sha256=policy.sha256,
        resume_predicate=(
            f"a feedback-grounded current-rung ({_current_rung_label()}) data "
            "and capability objective is preregistered under unchanged I10 "
            "rung gates"
        ),
        bank_fingerprint=_screening_bank_fingerprint(policy_sha256=policy.sha256),
    )
    path = _terminal_verdict_path(root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_loop_state(
        root,
        AutotrainLoopStateV1(
            loop_id=loop_id,
            state="BLOCKED",
            phase="blocked",
            active_campaign_id=None,
            last_completed_campaign_id=campaign_id,
            cycle_index=cycle_index,
            next_action=actions[0].kind,
            blocker_fingerprint="screening_objective_saturated",
            blocker_count=1,
            pid=os.getpid(),
        ),
    )
    print(
        f"REGIME_PARK loop={loop_id} campaign={campaign_id} "
        "constraint=screening_objective_saturated",
        flush=True,
    )
    # Park is not a report-only stop: execute the just-queued local rebuild
    # in this process so the next cycle has a process arm instead of sleeping.
    if cwd is not None:
        try:
            kind = _self_heal_rebuild_data(
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                campaign_id=campaign_id,
            )
            if kind:
                print(
                    f"SELF_HEAL_REBUILD_DATA_ON_PARK campaign={campaign_id} kind={kind}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 — park already persisted
            print(f"SELF_HEAL_REBUILD_DATA_ON_PARK_WARN err={exc!r}", flush=True)
    return _REGIME_PARKED_STATUS




def _latest_hypothesis_feedback(
    root: Path, campaign_id: str
) -> HypothesisFeedback:
    """Load the terminal typed feedback that grounds an objective change.

    Incomplete retries may have no hypothesizer event; walk predecessor
    campaigns until a recorded feedback exists.
    """

    seen: set[str] = set()
    current: str | None = campaign_id
    while current and current not in seen:
        seen.add(current)
        store = CampaignStore(current, root)
        try:
            events = store.verify_event_chain()
        except Exception:  # noqa: BLE001 — missing/broken chain, try predecessor
            events = []
        for event in reversed(events):
            if event.get("event_type") != "hypothesizer_feedback_recorded":
                continue
            digest = str(event.get("artifact_sha256") or "")
            path = store.root / "artifacts" / "hypothesizer_feedback" / f"{digest}.json"
            if not path.is_file():
                continue
            feedback = HypothesisFeedback.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if feedback.campaign_id == current:
                return feedback
        spec = _read_json(root / current / "campaign.json")
        nxt = str(spec.get("predecessor_campaign_id") or "") if spec else ""
        current = nxt or None
    raise RuntimeError(
        "screening objective change requires terminal HypothesisFeedback"
    )




def _capability_objective_refresh_actions(
    *,
    root: Path,
    campaign_id: str,
    preserved_actions: Sequence[AutotrainActionV1] = (),
) -> tuple[AutotrainActionV1, ...]:
    """Route exhausted smoke search into the existing rung/data/research loop.

    Every action targets the policy's *current* rung. Naming a later rung
    (e.g. simplified-NL while grammar_2_ast is uncertified) turns the pending
    action into an I10 skip that no legal heal can execute.
    """

    feedback = _latest_hypothesis_feedback(root, campaign_id)
    evidence_ids = (feedback.feedback_id, f"campaign:{campaign_id}")
    rung = _current_rung_label()
    return (
        AutotrainActionV1(
            kind="rebuild_data",
            owner="synthesis-feedback",
            reason=(
                f"rebuild the current-rung ({rung}) training corpus from the "
                "climb-policy plan; preserve I10 rung gates and inspect the "
                "quality report before any new training"
            ),
            evidence_ids=evidence_ids,
        ),
        *preserved_actions,
        AutotrainActionV1(
            kind="next_experiment",
            owner="autotrain",
            reason=(
                "after the data receipt and objective change, invoke the configured "
                "Researcher once with the terminal HypothesisFeedback and "
                f"preregister a size-matched capability objective for the current "
                f"rung ({rung}); do not rotate the exhausted decoder-lever bank"
            ),
            evidence_ids=evidence_ids,
        ),
    )






def _load_dynamic_thrash_arms(root: Path, loop_id: str) -> None:
    """Load loop-local thrash successors into the process bank cache."""
    global _DYNAMIC_THRASH_LOADED_FOR, _DYNAMIC_THRASH_ARMS
    key = f"{root.resolve()}::{loop_id}"
    if _DYNAMIC_THRASH_LOADED_FOR == key:
        return
    path = _dynamic_thrash_arms_path(root, loop_id)
    loaded: list[tuple[str, str, dict[str, Any]]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            slug = str(row.get("slug") or "")
            hyp = str(row.get("hypothesis") or "")
            extras = row.get("extras") if isinstance(row.get("extras"), dict) else {}
            if slug and hyp:
                extras = dict(extras)
                extras["_thrash_slug"] = slug
                loaded.append((slug, hyp, extras))
    _DYNAMIC_THRASH_ARMS = loaded
    _DYNAMIC_THRASH_LOADED_FOR = key


def _append_dynamic_thrash_arms(
    root: Path,
    loop_id: str,
    arms: list[tuple[str, str, dict[str, Any]]],
) -> None:
    if not arms:
        return
    path = _dynamic_thrash_arms_path(root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for slug, hyp, extras in arms:
            payload = {
                "schema": "autotrain_dynamic_thrash_arm/v1",
                "slug": slug,
                "hypothesis": hyp,
                "extras": {
                    k: v for k, v in extras.items() if not str(k).startswith("_")
                },
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
            live = dict(extras)
            live["_thrash_slug"] = slug
            _DYNAMIC_THRASH_ARMS.append((slug, hyp, live))




def _known_thrash_lever_signatures() -> set[str]:
    return {
        _thrash_lever_signature(extras) for _, _, extras in _all_screening_arm_bank()
    }




def _synthesize_thrash_arms(
    *,
    known_slugs: set[str],
    closed: set[str],
    batch: int = _SELF_HEAL_BANK_BATCH,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Compose size-matched thrash successors from lever atoms (self-heal).

    Does not recycle multi-seed-closed slugs **or** knob signatures already in
    the effective bank (static or prior dynamic). Pair then triple compositions.
    """
    out: list[tuple[str, str, dict[str, Any]]] = []
    known_sigs = _known_thrash_lever_signatures()
    atoms = list(_THRASH_COMPOSE_ATOMS)

    def _try_add(slug: str, hyp: str, extras: dict[str, Any]) -> bool:
        if slug in known_slugs or slug in closed:
            return False
        finalized = _finalize_compose_extras(extras, slug=slug)
        sig = _thrash_lever_signature(finalized)
        if sig in known_sigs:
            return False
        known_sigs.add(sig)
        out.append((slug, hyp, finalized))
        return True

    # Pairs first.
    for i, (k1, v1) in enumerate(atoms):
        for k2, v2 in atoms[i + 1 :]:
            if k1 == k2:
                continue
            slug = f"compose-{_short_lever_token(k1)}-{_short_lever_token(k2)}"
            extras = {**_compose_atom_extras(k1, v1), **_compose_atom_extras(k2, v2)}
            hyp = (
                f"Joint {k1}={v1!r} with {k2}={v2!r} improves smoke "
                f"structural_similarity without lowering parse_rate "
                f"(loop self-heal thrash successor {slug})."
            )
            if _try_add(slug, hyp, extras) and len(out) >= max(1, int(batch)):
                return out
    # Triples when pairs are exhausted / collide with static bank recipes.
    for i, (k1, v1) in enumerate(atoms):
        for j, (k2, v2) in enumerate(atoms[i + 1 :], start=i + 1):
            for k3, v3 in atoms[j + 1 :]:
                slug = (
                    f"compose-{_short_lever_token(k1)}-"
                    f"{_short_lever_token(k2)}-{_short_lever_token(k3)}"
                )
                extras = {
                    **_compose_atom_extras(k1, v1),
                    **_compose_atom_extras(k2, v2),
                    **_compose_atom_extras(k3, v3),
                }
                hyp = (
                    f"Joint {k1}/{k2}/{k3} supervision improves smoke "
                    f"structural_similarity without lowering parse_rate "
                    f"(loop self-heal thrash successor {slug})."
                )
                if _try_add(slug, hyp, extras) and len(out) >= max(1, int(batch)):
                    return out
    return out


class _ThrashBankHeal(NamedTuple):
    """Bank-heal outcome. Truthy iff at least one open arm remains.

    ``composed`` is True only when new dynamic successors were written.
    """

    available: bool
    composed: bool = False

    def __bool__(self) -> bool:
        return self.available


def _self_heal_thrash_bank_exhaust(
    root: Path,
    loop_id: str,
    *,
    closed: set[str],
    skip: set[str],
    predecessor_campaign_id: str | None = None,
) -> _ThrashBankHeal:
    """Ensure thrash can continue after static bank multi-seed exhaust.

    Truthy when at least one open arm is available. ``composed`` is set only
    when this call wrote new successors.
    """
    _load_dynamic_thrash_arms(root, loop_id)
    bank = _all_screening_arm_bank()
    known = {slug for slug, _, _ in bank}
    open_now = {slug for slug, _, _ in bank if slug not in closed and slug not in skip}
    pred = predecessor_campaign_id or _latest_cycle(root, loop_id)[1]
    if _terminal_park_on_exhaust() and _open_slugs_are_snapshot_leftovers(open_now):
        # Isolate OFAT bank is done. Remaining snapshot slugs are I10
        # leftovers, not compose fodder — unless a process arm is still
        # selectable (closed slug spelling hides an unused train_version).
        if _selectable_process_arm(
            root, loop_id, predecessor_campaign_id=pred
        ):
            print(
                "SELF_HEAL_BANK_EXHAUST heal_open "
                "reason=selectable_process_arm",
                flush=True,
            )
            return _ThrashBankHeal(True, False)
        print(
            "SELF_HEAL_BANK_EXHAUST parked reason=snapshot_leftovers",
            flush=True,
        )
        return _ThrashBankHeal(False)
    if open_now:
        return _ThrashBankHeal(True, False)
    if _terminal_park_on_exhaust():
        # Terminal policy retires compose filler: an exhausted bank concludes
        # with the typed regime_exhausted verdict instead of synthetic arms.
        print(
            "SELF_HEAL_BANK_EXHAUST parked reason=terminal_park_on_exhaust",
            flush=True,
        )
        return _ThrashBankHeal(False)
    synthesized = _synthesize_thrash_arms(
        known_slugs=known | closed | skip, closed=closed
    )
    if not synthesized:
        print(
            "SELF_HEAL_BANK_EXHAUST exhausted "
            "reason=no_untried_size_matched_compose_pairs",
            flush=True,
        )
        return _ThrashBankHeal(False)
    _append_dynamic_thrash_arms(root, loop_id, synthesized)
    print(
        "SELF_HEAL_BANK_EXHAUST "
        f"added={[slug for slug, _, _ in synthesized]} "
        "reason=static_bank_multi_seed_exhausted_compose_successors",
        flush=True,
    )
    open_after = {
        slug
        for slug, _, _ in _all_screening_arm_bank()
        if slug not in closed and slug not in skip
    }
    return _ThrashBankHeal(bool(open_after), True)








def _abort_in_progress_merge(*, cwd: Path, root: Path, loop_id: str) -> bool:
    """Fail closed: drop MERGE_HEAD so thrash is not foreign_dirty forever."""
    if _merge_head_path(cwd) is None:
        return False
    try:
        _run(
            ["git", "merge", "--abort"],
            cwd=cwd,
            root=root,
            loop_id=loop_id,
            stage="self-heal-merge-abort",
        )
        print("SELF_HEAL_MERGE_ABORT reason=fail_closed_clean_tree", flush=True)
        return True
    except Exception as abort_exc:  # noqa: BLE001
        print(f"SELF_HEAL_MERGE_ABORT_FAIL err={abort_exc!r}", flush=True)
        return False


def _git_is_ancestor(
    ancestor: str,
    descendant: str,
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    deadline: float | None = None,
) -> bool:
    try:
        _run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="self-heal-ancestry-is-ancestor",
        )
        return True
    except Exception:  # noqa: BLE001 — nonzero means not an ancestor
        return False


def _upstream_commit_for_init(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    upstream: str,
    integration: str,
    deadline: float | None = None,
) -> str:
    """origin/main when it is in HEAD; otherwise HEAD so init can proceed.

    After SELF_HEAL_GIT_ANCESTRY_SKIP the worktree has diverged from squash-
    merged main; passing origin/main as --upstream-commit fails
    ``integration_commit does not contain upstream_commit``. Do not rewrite
    upstream when HEAD already contains main.
    """
    if _git_is_ancestor(
        upstream,
        integration,
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        deadline=deadline,
    ):
        return upstream
    print(
        "CYCLE_COMMITS reason=diverged_unmergeable upstream=HEAD",
        flush=True,
    )
    return integration


def _integrate_origin_main(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    deadline: float | None = None,
) -> str:
    """Fetch origin/main and integrate when possible; never leave MERGE_HEAD.

    Already-integrated (origin/main ancestor of HEAD): skip. Fast-forward when
    HEAD is behind main. Diverged: try merge; on conflict or stamp-hook
    failure abort and continue on local HEAD. Unmergeable main is not
    CYCLE_ERROR — I6 fail-closed is dirt, not a squash-diverged worktree.
    """
    _abort_in_progress_merge(cwd=cwd, root=root, loop_id=loop_id)
    try:
        _run(
            ["git", "fetch", "origin", "main"],
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="self-heal-ancestry-fetch",
        )
        head = _git(
            "rev-parse",
            "HEAD",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="self-heal-ancestry-head",
        )
        upstream = _git(
            "rev-parse",
            "origin/main",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="self-heal-ancestry-upstream",
        )
    except Exception as fetch_exc:  # noqa: BLE001
        _abort_in_progress_merge(cwd=cwd, root=root, loop_id=loop_id)
        print(
            f"SELF_HEAL_GIT_ANCESTRY_SKIP reason=origin_main_unavailable "
            f"err={fetch_exc!r}",
            flush=True,
        )
        return "git_ancestry_skip"
    if head == upstream or _git_is_ancestor(
        upstream,
        head,
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        deadline=deadline,
    ):
        print(
            "SELF_HEAL_GIT_ANCESTRY_SKIP reason=already_integrated",
            flush=True,
        )
        return "git_ancestry_already_integrated"
    if _git_is_ancestor(
        head,
        upstream,
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        deadline=deadline,
    ):
        _run(
            ["git", "merge", "--ff-only", "origin/main"],
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="self-heal-ancestry-ff",
        )
        print("SELF_HEAL_GIT_ANCESTRY merged origin/main reason=fast_forward", flush=True)
        return "git_ancestry_fast_forward"
    try:
        _run(
            ["git", "merge", "--no-edit", "origin/main"],
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="self-heal-ancestry-merge",
        )
        print("SELF_HEAL_GIT_ANCESTRY merged origin/main", flush=True)
        return "git_ancestry_merge"
    except Exception:  # noqa: BLE001 — conflict or hook; abort and continue
        finished = _self_heal_incomplete_merge(cwd=cwd, root=root, loop_id=loop_id)
        if finished:
            print(
                "SELF_HEAL_GIT_ANCESTRY merged origin/main via conflict resolve",
                flush=True,
            )
            return finished
        _abort_in_progress_merge(cwd=cwd, root=root, loop_id=loop_id)
        print(
            "SELF_HEAL_GIT_ANCESTRY_SKIP reason=diverged_unmergeable",
            flush=True,
        )
        return "git_ancestry_skip"


def _unmerged_paths(
    cwd: Path,
    *,
    root: Path | None = None,
    loop_id: str | None = None,
) -> list[str]:
    git_kw: dict[str, Any] = {"cwd": cwd}
    if root is not None and loop_id is not None:
        git_kw.update(root=root, loop_id=loop_id)
    try:
        out = _git(
            "diff",
            "--name-only",
            "--diff-filter=U",
            stage="self-heal-unmerged-list" if root is not None else None,
            **git_kw,
        )
    except Exception:  # noqa: BLE001
        return []
    return [p.strip() for p in out.splitlines() if p.strip()]


def _self_heal_incomplete_merge(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
) -> str | None:
    """Finish interrupted origin/main merges that freeze thrash as foreign_dirty.

    Agents often ``git merge origin/main`` into the live continuous worktree to
    land self-heal driver fixes; version-registry conflicts leave ``UU`` paths
    and the supervisor hard-backs-off forever. That is soft thrash dirt, not a
    human control plane.

    Policy (merge in progress only — never invent a merge):
    - continuous closeout docs keep *ours* (loop-local evidence)
    - every other unmerged path takes *theirs* (incoming origin/main harness)
    - then ``git commit`` to complete the merge
    - if the commit (or resolve) fails, ``git merge --abort`` so MERGE_HEAD
      cannot park the loop as foreign_dirty_tree
    """
    if _merge_head_path(cwd) is None:
        return None
    unmerged = _unmerged_paths(cwd, root=root, loop_id=loop_id)
    try:
        for rel in unmerged:
            side = "--ours" if _is_continuous_closeout_path(rel) else "--theirs"
            _run(
                ["git", "checkout", side, "--", rel],
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                stage="self-heal-merge-checkout",
            )
            _run(
                ["git", "add", "--", rel],
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                stage="self-heal-merge-add",
            )
        # Re-check; refuse to commit if anything still unmerged.
        still = _unmerged_paths(cwd, root=root, loop_id=loop_id)
        if still:
            print(
                f"SELF_HEAL_INCOMPLETE_MERGE_HARD still_unmerged={still[:8]}",
                flush=True,
            )
            _abort_in_progress_merge(cwd=cwd, root=root, loop_id=loop_id)
            return None
        _run(
            [
                "git",
                "commit",
                "--no-edit",
                "-m",
                "self-heal: complete origin/main merge for continuous thrash",
            ],
            cwd=cwd,
            root=root,
            loop_id=loop_id,
            stage="self-heal-merge-commit",
        )
        print(
            f"SELF_HEAL_INCOMPLETE_MERGE resolved={unmerged or ['clean']} "
            "reason=prefer_origin_main_for_non_closeout",
            flush=True,
        )
        return "git_merge_complete"
    except Exception as heal_exc:  # noqa: BLE001
        print(f"SELF_HEAL_INCOMPLETE_MERGE_FAIL err={heal_exc!r}", flush=True)
        _abort_in_progress_merge(cwd=cwd, root=root, loop_id=loop_id)
        return None


def _self_heal_git_ancestry(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    exc: BaseException,
) -> str | None:
    """Integrate origin/main after a merge-base ancestry CYCLE_ERROR.

    Unmergeable divergence (squash-merged main vs exclusive worktree commits)
    is skipped, never raised: merge --abort and continue on local HEAD.
    """
    message = str(exc)
    cmd_text = ""
    if isinstance(exc, subprocess.CalledProcessError):
        cmd_text = " ".join(str(x) for x in (exc.cmd or ()))
    blob = f"{message} {cmd_text}"
    if "merge-base" not in blob and "is-ancestor" not in blob:
        return None
    return _integrate_origin_main(cwd=cwd, root=root, loop_id=loop_id)




def _is_bank_exhaust_repair_action(action: AutotrainActionV1) -> bool:
    """True when repair_harness is thrash bank exhaust (soft if compose can reopen)."""
    if action.kind != "repair_harness":
        return False
    reason = str(action.reason or "").lower()
    return any(m in reason for m in _BANK_EXHAUST_MARKERS)


def _self_heal_bank_exhaust_repair(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str | None,
    integration_commit: str | None = None,
) -> str | None:
    """Compose thrash successors and retire bank-exhaust repair_harness.

    Compose alone is not enough: the predecessor handoff still blocks on
    repair_harness until rewritten to next_experiment. Returns None when no
    open arms remain after compose (true hard stop).
    """
    if not campaign_id:
        return None
    handoff_path = root / campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != loop_id or handoff.campaign_id != campaign_id:
        return None
    pending = list(pending_autotrain_actions(root, handoff))
    bank_pending = [(i, a) for i, a in pending if _is_bank_exhaust_repair_action(a)]
    if not bank_pending:
        return None
    other_hard = [
        (i, a)
        for i, a in pending
        if a.kind in _HARD_PREREQUISITE_ACTION_KINDS
        and not _is_bank_exhaust_repair_action(a)
        and a.kind != "document"
    ]
    if other_hard:
        return None

    closed = _recent_completed_nonpositive_slugs(root, campaign_id)
    entries = _load_champion_queue(_champion_queue_path(root, loop_id))
    if integration_commit:
        _reopen_harness_blocked_champions(
            root, entries, integration_commit=integration_commit
        )
        _write_champion_queue(_champion_queue_path(root, loop_id), entries)
    skip = _skip_arm_slugs(
        entries, integration_commit=integration_commit, include_causal_cap=False
    )
    opened = _self_heal_thrash_bank_exhaust(
        root, loop_id, closed=closed, skip=skip | closed
    )
    if not opened:
        print(
            f"SELF_HEAL_BANK_EXHAUST_HARD campaign={campaign_id} "
            "reason=no_untried_size_matched_compose_pairs",
            flush=True,
        )
        return None

    evidence_id = f"campaign:{campaign_id}"
    kept: list[AutotrainActionV1] = []
    for action in handoff.actions:
        if action.kind == "repair_harness" and _is_bank_exhaust_repair_action(action):
            continue
        if action.kind == "retry_measurement":
            continue
        kept.append(action)
    if not any(a.kind == "document" for a in kept):
        kept.insert(
            0,
            AutotrainActionV1(
                kind="document",
                owner="documenting-experiment-results",
                reason="persist thrash bank-exhaust closeout under docs/design",
                evidence_ids=(evidence_id,),
            ),
        )
    if not any(a.kind == "next_experiment" for a in kept):
        kept.append(
            AutotrainActionV1(
                kind="next_experiment",
                owner="autotrain",
                reason=(
                    "consume composed thrash successors after static quality-arm "
                    "bank exhaust (continuous self-heal; not a model attribution)"
                ),
                evidence_ids=(evidence_id,),
            )
        )
    rebuilt = handoff.model_copy(
        update={
            "actions": tuple(kept),
            "reasons": tuple(
                list(handoff.reasons)
                + ["self_heal:bank_exhaust_compose→next_experiment"]
            ),
        }
    )
    handoff_path.write_text(rebuilt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"SELF_HEAL_BANK_EXHAUST_REPAIR campaign={campaign_id} "
        f"actions={[a.kind for a in rebuilt.actions]}",
        flush=True,
    )
    doc_kind = _self_heal_document_actions(
        cwd=cwd, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    pending_after = pending_autotrain_actions(root, rebuilt)
    hard_after = [
        (i, a) for i, a in pending_after if a.kind in _HARD_PREREQUISITE_ACTION_KINDS
    ]
    if hard_after:
        return None
    return doc_kind or "bank_exhaust_compose"


def _self_heal_thrash_timeout_repair(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str | None,
) -> str | None:
    """Unblock thrash when stuck on decode/wall-timeout repair_harness.

    Continuous thrash often finalizes AgentV with decode timeouts under the arm
    wall. That is a thrash residual / budget signal, not a hard model_build
    stop that should freeze the loop until a human is prompted. Rewrite the
    predecessor handoff to next_experiment (+ document closeout) so thrash
    continues. Real harness crashes (missing AgentV, import errors) stay hard.
    """
    if not campaign_id:
        return None
    handoff_path = root / campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != loop_id or handoff.campaign_id != campaign_id:
        return None
    pending = list(pending_autotrain_actions(root, handoff))
    if not pending:
        return None
    repair_pending = [(i, a) for i, a in pending if a.kind == "repair_harness"]
    if not repair_pending:
        return None
    # Any non-repair hard prereq still blocks (formal/stop/deliver/rebuild).
    other_hard = [
        (i, a)
        for i, a in pending
        if a.kind in _HARD_PREREQUISITE_ACTION_KINDS and a.kind != "repair_harness"
    ]
    if other_hard:
        return None
    delivery_path = root / campaign_id / "sdlc_delivery.json"
    delivery: dict[str, Any] = {}
    if delivery_path.is_file():
        try:
            loaded = _read_json(delivery_path)
            if isinstance(loaded, dict):
                delivery = loaded
        except Exception:  # noqa: BLE001
            delivery = {}
    if not _delivery_is_thrash_timeout_residual(delivery, handoff):
        return None
    if handoff.cycle_role not in {"screening", "promotion"} and str(
        handoff.cycle_intent or ""
    ) not in {"screening", "retry_measurement", "confirm"}:
        # Still allow thrash-like intents above; otherwise leave hard.
        pass
    # Rebuild actions: drop repair/retry, keep document, ensure next_experiment.
    evidence_id = f"campaign:{campaign_id}"
    kept: list[AutotrainActionV1] = []
    for action in handoff.actions:
        if action.kind in {"repair_harness", "retry_measurement"}:
            continue
        kept.append(action)
    if not any(a.kind == "document" for a in kept):
        kept.insert(
            0,
            AutotrainActionV1(
                kind="document",
                owner="documenting-experiment-results",
                reason=("persist thrash timeout-residual closeout under docs/design"),
                evidence_ids=(evidence_id,),
            ),
        )
    if not any(a.kind == "next_experiment" for a in kept):
        kept.append(
            AutotrainActionV1(
                kind="next_experiment",
                owner="autotrain",
                reason=(
                    "retire thrash decode/wall-timeout residual and consume the "
                    "next distinct ranked hypothesis (continuous self-heal; not "
                    "a model attribution)"
                ),
                evidence_ids=(evidence_id,),
            )
        )
    rebuilt = handoff.model_copy(
        update={
            "actions": tuple(kept),
            "reasons": tuple(
                list(handoff.reasons)
                + [
                    "self_heal:thrash_timeout_residual_bypass:"
                    "repair_harness→next_experiment"
                ]
            ),
        }
    )
    handoff_path.write_text(rebuilt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"SELF_HEAL_THRASH_TIMEOUT_REPAIR campaign={campaign_id} "
        f"actions={[a.kind for a in rebuilt.actions]}",
        flush=True,
    )
    # Document closeout may still be pending on the rewritten handoff.
    doc_kind = _self_heal_document_actions(
        cwd=cwd, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    # Verify no hard prereqs remain.
    pending_after = pending_autotrain_actions(root, rebuilt)
    hard_after = [
        (i, a) for i, a in pending_after if a.kind in _HARD_PREREQUISITE_ACTION_KINDS
    ]
    if hard_after:
        return None
    return doc_kind or "thrash_timeout_repair_bypass"


def _self_heal_env_repair_rewrite(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str | None,
) -> str | None:
    """Retire an environment-incomplete ``repair_harness`` after a verified heal.

    A missing JS bridge / AgentV dependency install is an *environment*
    condition misclassified at emission as a harness crash. When a heal
    playbook has actually performed the documented install **and its verify
    probe passed** (``heal_receipts.jsonl`` outcome ``healed`` for this
    blocker's cross-campaign fingerprint), rewrite the pending action to
    ``next_experiment`` with the receipt as evidence — the
    ``SELF_HEAL_THRASH_TIMEOUT_REPAIR`` precedent, never a receipt ack.

    Code-class crashes (repo-internal import errors) are untouched: they stay
    hard until the owner skill lands a real fix. The rewrite reason carries
    the heal-receipt sha so the successor's provenance names its evidence.
    """
    if not campaign_id:
        return None
    handoff_path = root / campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != loop_id or handoff.campaign_id != campaign_id:
        return None
    pending = list(pending_autotrain_actions(root, handoff))
    repair_pending = [(i, a) for i, a in pending if a.kind == "repair_harness"]
    if not repair_pending:
        return None
    other_hard = [
        (i, a)
        for i, a in pending
        if a.kind in _HARD_PREREQUISITE_ACTION_KINDS and a.kind != "repair_harness"
    ]
    if other_hard:
        return None

    from slm_training.autoresearch.heal import load_heal_receipts
    from slm_training.autoresearch.heal.classify import classify_blocker
    from slm_training.autoresearch.heal.escalation import blocker_fingerprint

    # Recency binding: fingerprints are cross-campaign, so only receipts
    # recorded at/after this handoff's creation may authorize its rewrite —
    # a historical heal never clears a later regression of the same blocker.
    handoff_created = str(handoff.created_at or "")
    healed_by_fingerprint = {
        receipt.blocker_fingerprint: receipt
        for receipt in load_heal_receipts(root, loop_id)
        if receipt.outcome == "healed"
        and receipt.verify_result is not None
        and receipt.recorded_at
        and (not handoff_created or receipt.recorded_at >= handoff_created)
    }
    matched: list[tuple[int, AutotrainActionV1, str]] = []
    for index, action in repair_pending:
        reason = str(action.reason or "")
        if classify_blocker("repair_harness", reason) != "environment":
            return None  # any code-class repair keeps the whole handoff hard
        receipt = healed_by_fingerprint.get(
            blocker_fingerprint("repair_harness", reason)
        )
        if receipt is None:
            return None  # no verified heal — stays hard
        matched.append((index, action, receipt.sha256()))
    if not matched:
        return None

    evidence_id = f"campaign:{campaign_id}"
    receipt_shas = sorted({sha for _, _, sha in matched})
    kept: list[AutotrainActionV1] = []
    for action in handoff.actions:
        if action.kind in {"repair_harness", "retry_measurement"}:
            continue
        kept.append(action)
    if not any(a.kind == "document" for a in kept):
        kept.insert(
            0,
            AutotrainActionV1(
                kind="document",
                owner="documenting-experiment-results",
                reason="persist environment-repair closeout under docs/design",
                evidence_ids=(evidence_id,),
            ),
        )
    if not any(a.kind == "next_experiment" for a in kept):
        kept.append(
            AutotrainActionV1(
                kind="next_experiment",
                owner="autotrain",
                reason=(
                    "environment-incomplete repair verified by heal receipt "
                    f"{receipt_shas[0][:12]}; replay the frozen arm on the "
                    "restored environment (continuous self-heal; not a model "
                    "attribution)"
                ),
                evidence_ids=(evidence_id,),
            )
        )
    rebuilt = handoff.model_copy(
        update={
            "actions": tuple(kept),
            "reasons": tuple(
                list(handoff.reasons)
                + [
                    "self_heal:env_repair_verified:"
                    + ",".join(sha[:12] for sha in receipt_shas)
                ]
            ),
        }
    )
    handoff_path.write_text(rebuilt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"SELF_HEAL_ENV_REPAIR campaign={campaign_id} "
        f"receipts={receipt_shas} actions={[a.kind for a in rebuilt.actions]}",
        flush=True,
    )
    doc_kind = _self_heal_document_actions(
        cwd=cwd, root=root, loop_id=loop_id, campaign_id=campaign_id
    )
    pending_after = pending_autotrain_actions(root, rebuilt)
    hard_after = [
        (i, a) for i, a in pending_after if a.kind in _HARD_PREREQUISITE_ACTION_KINDS
    ]
    if hard_after:
        return None
    return doc_kind or "env_repair_rewrite"


def _self_heal_cycle_error(
    *,
    root: Path,
    loop_id: str,
    exc: BaseException,
    integration_commit: str | None = None,
    cwd: Path | None = None,
) -> str | None:
    """Attempt in-pipeline recovery for known continuous blockers.

    Returns a heal kind string on success (caller should continue the loop),
    or None when the error remains hard.
    """
    message = str(exc)
    work_cwd = cwd or Path.cwd()
    # Ordinary thrash closeout: unacked document / continuous-only dirty tree /
    # thrash timeout residual repair_harness that must not freeze the loop.
    if "unacknowledged actions" in message or "repair_harness" in message:
        pred = _latest_cycle(root, loop_id)[1]
        # Prefer thrash-timeout repair bypass before ordinary document heal.
        timeout_kind = _self_heal_thrash_timeout_repair(
            cwd=work_cwd, root=root, loop_id=loop_id, campaign_id=pred
        )
        if timeout_kind:
            return timeout_kind
        bank_kind = _self_heal_bank_exhaust_repair(
            cwd=work_cwd,
            root=root,
            loop_id=loop_id,
            campaign_id=pred,
            integration_commit=integration_commit,
        )
        if bank_kind:
            return bank_kind
        kind = _self_heal_closeout_blockers(
            cwd=work_cwd, root=root, loop_id=loop_id, campaign_id=pred
        )
        if kind:
            # Only claim heal when no hard prerequisites remain on predecessor.
            if pred:
                handoff_path = root / pred / "cycle_handoff.json"
                if handoff_path.is_file():
                    handoff = AutotrainCycleHandoffV1.model_validate_json(
                        handoff_path.read_text(encoding="utf-8")
                    )
                    hard = [
                        (i, a)
                        for i, a in pending_autotrain_actions(root, handoff)
                        if a.kind in _HARD_PREREQUISITE_ACTION_KINDS
                    ]
                    if hard:
                        # One more attempt: thrash timeout residual may still apply.
                        timeout_kind = _self_heal_thrash_timeout_repair(
                            cwd=work_cwd,
                            root=root,
                            loop_id=loop_id,
                            campaign_id=pred,
                        )
                        if timeout_kind:
                            return timeout_kind
                        return None
            return kind
    if "loop worktree is dirty" in message:
        try:
            owned_kind = _self_heal_loop_owned_generated_dirt(
                cwd=work_cwd, root=root, loop_id=loop_id
            )
        except Exception:  # noqa: BLE001 — fall through to closeout heal
            owned_kind = None
        if owned_kind:
            return owned_kind
        try:
            porcelain = _git(
                "status",
                "--porcelain",
                cwd=work_cwd,
                root=root,
                loop_id=loop_id,
                stage="self-heal-serena-dirty",
            )
            serena_kind = _maybe_restore_serena_project_yml(
                cwd=work_cwd,
                paths=_porcelain_paths(porcelain),
                git_kw={
                    "cwd": work_cwd,
                    "root": root,
                    "loop_id": loop_id,
                    "stage": "self-heal-serena-yml",
                },
            )
            if serena_kind:
                return serena_kind
        except Exception:  # noqa: BLE001 — fall through to closeout heal
            pass
        dirty_kind = _self_heal_continuous_dirty_tree(
            cwd=work_cwd, root=root, loop_id=loop_id
        )
        if dirty_kind:
            return dirty_kind
        # Document closeout may produce the dirt; heal both.
        close_kind = _self_heal_closeout_blockers(
            cwd=work_cwd,
            root=root,
            loop_id=loop_id,
            campaign_id=_latest_cycle(root, loop_id)[1],
        )
        if close_kind:
            return close_kind
    # Bank exhaust: synthesize thrash successors and/or rearm harness heads.
    if _BANK_EXHAUST_MSG in message or "screening arm bank exhausted" in message:
        pred = _latest_cycle(root, loop_id)[1]
        closed = _recent_completed_nonpositive_slugs(root, pred)
        entries = _load_champion_queue(_champion_queue_path(root, loop_id))
        if integration_commit:
            _reopen_harness_blocked_champions(
                root, entries, integration_commit=integration_commit
            )
            _write_champion_queue(_champion_queue_path(root, loop_id), entries)
        skip = _skip_arm_slugs(
            entries, integration_commit=integration_commit, include_causal_cap=False
        )
        if _self_heal_thrash_bank_exhaust(
            root,
            loop_id,
            closed=closed,
            skip=skip | closed,
            predecessor_campaign_id=pred,
        ):
            return "thrash_bank_compose"
        if _queue_head_confirmed(entries) is not None:
            return "promote_head_available"
        return None
    if "conflicts with supplied feedback" in message:
        # Tip already strips thrash feedback; treat as soft and continue.
        return "feedback_conflict_soft"
    if "campaign already exists with different spec" in message:
        return "campaign_identity_soft"
    # Empty-bank steering used bare next() on the static bank after compose
    # successors were selected; treat residual StopIteration as bank heal.
    if isinstance(exc, StopIteration):
        pred = _latest_cycle(root, loop_id)[1]
        closed = _recent_completed_nonpositive_slugs(root, pred)
        if _self_heal_thrash_bank_exhaust(root, loop_id, closed=closed, skip=closed):
            return "stopiteration_bank_heal"
    # Compose arms that collide with static recipes fail matrix validation.
    if "knob signatures must be distinct" in message:
        if _self_heal_dedupe_dynamic_thrash_arms(root, loop_id):
            return "knob_signature_dedupe"
    return None


def _self_heal_dedupe_dynamic_thrash_arms(root: Path, loop_id: str) -> bool:
    """Drop dynamic thrash arms whose lever signature matches the static bank.

    Also compose fresh unique successors so thrash can continue.
    """
    global _DYNAMIC_THRASH_ARMS, _DYNAMIC_THRASH_LOADED_FOR
    _load_dynamic_thrash_arms(root, loop_id)
    static_sigs = {
        _thrash_lever_signature(extras) for _, _, extras in _SCREENING_ARM_BANK
    }
    kept: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set(static_sigs)
    dropped: list[str] = []
    for slug, hyp, extras in _DYNAMIC_THRASH_ARMS:
        sig = _thrash_lever_signature(extras)
        if sig in seen:
            dropped.append(slug)
            continue
        seen.add(sig)
        kept.append((slug, hyp, extras))
    path = _dynamic_thrash_arms_path(root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rewrite file with only unique dynamic arms.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for slug, hyp, extras in kept:
            payload = {
                "schema": "autotrain_dynamic_thrash_arm/v1",
                "slug": slug,
                "hypothesis": hyp,
                "extras": {
                    k: v for k, v in extras.items() if not str(k).startswith("_")
                },
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
    tmp.replace(path)
    _DYNAMIC_THRASH_ARMS = kept
    _DYNAMIC_THRASH_LOADED_FOR = f"{root.resolve()}::{loop_id}"
    print(
        f"SELF_HEAL_KNOB_DEDUPE dropped={dropped} kept={[s for s, _, _ in kept]}",
        flush=True,
    )
    # Ensure at least one open unique recipe remains.
    closed = _recent_completed_nonpositive_slugs(root, _latest_cycle(root, loop_id)[1])
    if _self_heal_thrash_bank_exhaust(root, loop_id, closed=closed, skip=closed):
        return True
    return bool(kept)




# ---------------------------------------------------------------------------
# Document + dirty-tree self-heal (ordinary thrash closeout — never agent-prompt)
# ---------------------------------------------------------------------------

_HARD_PREREQUISITE_ACTION_KINDS = frozenset(
    {
        "stop_campaign",
        "repair_harness",
        "repair_formal",
        "rebuild_data",
        "deliver_stack",
    }
)






# Tracked mirrors the driver itself mutates (evidence-store sync, similar).
# Restore to HEAD — never treat as human WIP, never commit every cycle.






_SERENA_PROJECT_YML = ".serena/project.yml"




def _maybe_restore_serena_project_yml(
    *,
    cwd: Path,
    paths: Sequence[str],
    git_kw: Mapping[str, Any],
) -> str | None:
    """Restore comment-stripped Serena project.yml; semantic edits stay parked."""
    foreign = [_normalize_repo_relpath(p) for p in paths if _is_foreign_dirty_path(p)]
    if foreign != [_SERENA_PROJECT_YML]:
        return None
    work = cwd / _SERENA_PROJECT_YML
    if not work.is_file():
        return None
    try:
        head = _git("show", f"HEAD:{_SERENA_PROJECT_YML}", **git_kw)
    except Exception:  # noqa: BLE001 — missing HEAD path is not this heal
        return None
    work_text = work.read_text(encoding="utf-8")
    if not _yaml_mapping_equal(head, work_text):
        return None
    _git(
        "restore",
        "--source=HEAD",
        "--worktree",
        "--staged",
        "--",
        _SERENA_PROJECT_YML,
        stage="self-heal-serena-yml" if git_kw.get("root") is not None else None,
        **{k: v for k, v in git_kw.items() if k != "stage"},
    )
    print("SELF_HEAL_SERENA_PROJECT_YML reason=comment_whitespace_strip", flush=True)
    return "serena_project_yml_comment_strip"






def _render_continuous_cycle_docs(
    *,
    campaign_id: str,
    loop_id: str,
    handoff: AutotrainCycleHandoffV1,
    delivery: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Honest fixture-screening closeout payload (not a ship claim)."""
    reasons = list(delivery.get("reasons") or handoff.reasons or [])
    # Stamp the eval-comparability components so this record lands in a real
    # cross-version partition of the evidence ledger instead of "unstamped".
    # Without this, `eval_key_from_stamp` returns None and the record's delta is
    # pooled with every other unstamped cycle regardless of the scorer/gate
    # version it actually ran under. Never let stamping fail the closeout.
    try:
        from slm_training.autoresearch.evidence_ledger import EVAL_KEY_COMPONENTS

        version_stamp: dict[str, Any] | None = build_version_stamp(*EVAL_KEY_COMPONENTS)
    except Exception:
        version_stamp = None
    payload: dict[str, Any] = {
        "schema": "continuous_cycle_results/v1",
        "campaign_id": campaign_id,
        "loop_id": loop_id,
        "cycle_index": handoff.cycle_index,
        "cycle_role": handoff.cycle_role,
        "cycle_intent": handoff.cycle_intent,
        "positive": bool(delivery.get("positive")),
        "stack_layer": bool(delivery.get("stack_layer")),
        "measurement_complete": delivery.get("measurement_complete"),
        "primary_metric": handoff.primary_metric,
        "control_metrics": delivery.get("control_metrics"),
        "candidate_metrics": delivery.get("candidate_metrics"),
        "reasons": reasons,
        "evidence_class": handoff.evidence_class,
        "honesty": "fixture_screening_only_not_ship",
        "auto": True,
    }
    if version_stamp is not None:
        payload["version_stamp"] = version_stamp
    # Embed the rich delivery record (candidate_id/arm_seed/policy_sha256) so
    # future ledger mining never falls back to reasons-string recovery.
    if delivery.get("schema") == "autotrain_sdlc_delivery/v1":
        payload["delivery"] = dict(delivery)
    md = (
        f"# Continuous cycle `{campaign_id}`\n\n"
        f"- loop_id: `{loop_id}`\n"
        f"- cycle_index: `{handoff.cycle_index}`\n"
        f"- role/intent: `{handoff.cycle_role}` / `{handoff.cycle_intent}`\n"
        f"- primary_metric: `{handoff.primary_metric}`\n"
        f"- positive: **{payload['positive']}**\n"
        f"- stack_layer: **{payload['stack_layer']}**\n"
        f"- measurement_complete: `{payload['measurement_complete']}`\n"
        f"- evidence_class: `{handoff.evidence_class}`\n"
        f"- reasons: {', '.join(str(r) for r in reasons) or '—'}\n"
        f"- control_metrics: `{payload['control_metrics']}`\n"
        f"- candidate_metrics: `{payload['candidate_metrics']}`\n\n"
    )
    from slm_training.autoresearch.hillclimb import hillclimb_iteration_report

    hill = hillclimb_iteration_report(
        campaign_id=campaign_id,
        cycle_index=handoff.cycle_index,
        positive=bool(payload["positive"]),
        measurement_complete=payload.get("measurement_complete"),
        reasons=reasons,
        control_metrics=payload.get("control_metrics")
        if isinstance(payload.get("control_metrics"), dict)
        else None,
        candidate_metrics=payload.get("candidate_metrics")
        if isinstance(payload.get("candidate_metrics"), dict)
        else None,
        primary_metric=str(handoff.primary_metric or ""),
    )
    payload["hillclimb"] = hill
    md += (
        "## Hill-climb this cycle\n\n"
        f"- went well: {', '.join(hill['went_well']) or '—'}\n"
        f"- went wrong: {', '.join(hill['went_wrong']) or '—'}\n"
        f"- speculate: {', '.join(hill['speculate']) or '—'}\n"
        f"- deltas: `{hill.get('deltas')}`\n\n"
        "Auto-documented by the continuous driver self-heal closeout. "
        "Fixture screening only — not a ship claim.\n"
    )
    return md, payload








def _stagnation_skip_slugs(root: Path, loop_id: str) -> set[str]:
    path = _hillclimb_review_path(root, loop_id)
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, dict):
        return set()
    slugs = payload.get("skip_slugs") or []
    return {str(s) for s in slugs if s}


def _apply_stagnation_review(
    *,
    root: Path,
    loop_id: str,
    iterations: Sequence[Mapping[str, Any]],
    campaign_id: str,
) -> dict[str, Any] | None:
    from slm_training.autoresearch.hillclimb import (
        HILLCLIMB_STAGNATION_CADENCE,
        stagnation_review,
    )

    review = stagnation_review(iterations, cadence=HILLCLIMB_STAGNATION_CADENCE)
    if review is None:
        return None
    prior_path = _hillclimb_review_path(root, loop_id)
    prior: dict[str, Any] = {}
    if prior_path.is_file():
        try:
            loaded = json.loads(prior_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prior = loaded
        except json.JSONDecodeError:
            prior = {}
    if prior.get("campaign_ids") == review.get("campaign_ids"):
        return prior
    skip: set[str] = set(prior.get("skip_slugs") or [])
    for cid in review.get("campaign_ids") or []:
        camp = root / str(cid)
        delivery = _read_json(camp / "sdlc_delivery.json") if camp.is_dir() else {}
        cand = str(delivery.get("candidate_id") or "")
        slug = _slug_from_candidate_id(cand) if cand else None
        if slug:
            skip.add(slug)
    review["skip_slugs"] = sorted(skip)
    review["applied_at_campaign"] = campaign_id
    review["applied"] = list(review.get("viable_applies") or [])
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "HILLCLIMB_STAGNATION_REVIEW "
        f"loop={loop_id} cadence={review.get('cadence')} "
        f"applied={review.get('applied')} skip_n={len(skip)}",
        flush=True,
    )
    return review


def _persist_hillclimb_cycle_outputs(
    *,
    root: Path,
    loop_id: str,
    campaign_id: str,
    delivery: Mapping[str, Any],
    cycle_index: int | None,
    primary_metric: str,
) -> dict[str, Any]:
    from slm_training.autoresearch.hillclimb import hillclimb_iteration_report

    report = hillclimb_iteration_report(
        campaign_id=campaign_id,
        cycle_index=cycle_index,
        positive=bool(delivery.get("positive")),
        measurement_complete=delivery.get("measurement_complete"),
        reasons=list(delivery.get("reasons") or []),
        control_metrics=delivery.get("control_metrics")
        if isinstance(delivery.get("control_metrics"), dict)
        else None,
        candidate_metrics=delivery.get("candidate_metrics")
        if isinstance(delivery.get("candidate_metrics"), dict)
        else None,
        primary_metric=primary_metric,
    )
    rows = _append_hillclimb_iteration(root, loop_id, report)
    print(
        "HILLCLIMB_ITERATION "
        f"campaign={campaign_id} progress={report['progress']} "
        f"well={';'.join(report['went_well']) or '—'} "
        f"wrong={';'.join(report['went_wrong']) or '—'} "
        f"speculate={';'.join(report['speculate']) or '—'}",
        flush=True,
    )
    _apply_stagnation_review(
        root=root,
        loop_id=loop_id,
        iterations=rows,
        campaign_id=campaign_id,
    )
    return report


def _git_commit_paths(
    cwd: Path,
    paths: Sequence[Path],
    *,
    message: str,
    root: Path | None = None,
    loop_id: str | None = None,
    stage: str = "self-heal-git-commit",
) -> bool:
    """Stage only the given paths and commit if there is a staged diff."""
    rels: list[str] = []
    for path in paths:
        resolved = path if path.is_absolute() else cwd / path
        if not resolved.is_file():
            continue
        try:
            rels.append(str(resolved.resolve().relative_to(cwd.resolve())))
        except ValueError:
            rels.append(str(path))
    if not rels:
        return False
    stage_kw: dict[str, Any] = {"cwd": cwd}
    if root is not None and loop_id is not None:
        stage_kw.update(root=root, loop_id=loop_id)
    add_stage = f"{stage}-add" if root is not None else None
    staged_stage = f"{stage}-staged" if root is not None else None
    commit_stage = f"{stage}-commit" if root is not None else None
    _run(
        ["git", "add", "--", *rels],
        stage=add_stage,
        **stage_kw,
    )
    staged = _git(
        "diff",
        "--cached",
        "--name-only",
        stage=staged_stage,
        **stage_kw,
    )
    if not staged.strip():
        return False
    _run(
        [
            "git",
            "commit",
            "-m",
            message,
            "--",
            *rels,
        ],
        stage=commit_stage,
        **stage_kw,
    )
    return True


def _ack_document_action(
    root: Path,
    handoff: AutotrainCycleHandoffV1,
    *,
    action_index: int,
    evidence_uris: Sequence[str],
) -> None:
    action = handoff.actions[action_index]
    if action.kind != "document":
        raise ValueError(f"refusing to auto-ack non-document action: {action.kind}")
    uris = tuple(evidence_uris)
    evidence = bind_autotrain_action_evidence(root, handoff, action, uris)
    append_autotrain_action_receipt(
        root,
        AutotrainActionReceiptV1(
            loop_id=handoff.loop_id,
            campaign_id=handoff.campaign_id,
            action_index=action_index,
            action_sha256=autotrain_action_sha256(action),
            action_kind="document",
            status="completed",
            evidence_uris=uris,
            evidence=evidence,
        ),
    )


# Wall-capped local CPU I10 heal. Policy min_unique_roots=32 is promotion-scale.
# Rung-honest: the heal rebuild compiles the climb-policy plan for the
# *current* rung (I10 — never a skipped rung's corpus, see _current_rung_label).
_HEAL_RESUME_SLUG = "current-rung-data-heal"




def _sample_adequacy_report(cwd: Path) -> dict | None:
    """Compute the cycle's sample-adequacy report from live build evidence.

    Observes the most recent local heal build's stats (falling back to the
    policy's committed fixture corpus) plus the latest data-adequacy ladder
    classification when one has been measured. Returns the report as a JSON
    dict, or None when no stats evidence exists.
    """
    from slm_training.autoresearch.climb_policy import (
        data_intervention_action,
        load_climb_policy,
    )
    from slm_training.autoresearch.sample_adequacy import (
        compute_sample_adequacy,
        observation_from_train_stats,
    )
    from slm_training.harnesses.experiments.data_adequacy_ladder import (
        load_ladder_classification,
    )

    spec = data_intervention_action(load_climb_policy())
    heal_stats = sorted(
        (cwd / "outputs" / "data" / "train").glob("continuous_i10_*/stats.json"),
        key=lambda path: path.stat().st_mtime,
    )
    fixture_stats = (
        cwd
        / "src/slm_training/resources/data/train"
        / str(spec.get("train_version") or "wf_smoke_v2")
        / "stats.json"
    )
    stats_path = heal_stats[-1] if heal_stats else fixture_stats
    if not stats_path.is_file():
        return None
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    flat: bool | None = None
    source: str | None = None
    ladders = sorted(
        (cwd / "outputs" / "ladders").glob("**/data_adequacy_ladder.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if ladders:
        try:
            flat, source = load_ladder_classification(ladders[-1])
        except ValueError:
            flat, source = None, None
    observation = observation_from_train_stats(
        stats, marginal_gain_flat=flat, marginal_gain_source=source
    )
    return compute_sample_adequacy(observation).model_dump(mode="json")




def _rebuild_data_artifact_sources(train_dir: Path) -> dict[str, Path] | None:
    """Map receipt names to files written by build_train_data."""
    quality = train_dir / "quality_report.json"
    feedback = train_dir / "synthesis_feedback.json"
    manifest = train_dir / "data_manifest.json"
    if not manifest.is_file():
        manifest = train_dir / "manifest.json"
    if not (quality.is_file() and feedback.is_file() and manifest.is_file()):
        return None
    return {
        "quality_report.json": quality,
        "synthesis_feedback.json": feedback,
        "data_manifest.json": manifest,
    }


def _ack_rebuild_data_action(
    root: Path,
    handoff: AutotrainCycleHandoffV1,
    *,
    action_index: int,
    evidence_uris: Sequence[str],
    counts: tuple[int, int] | None = None,
) -> None:
    """Acknowledge one ``rebuild_data`` action with bound evidence.

    ``counts`` is the heal's ``(records_before, records_after)`` postcondition:
    when given, an ack is refused unless the count actually grew — a sidecar
    path alone is never evidence that data changed.
    """
    action = handoff.actions[action_index]
    if action.kind != "rebuild_data":
        raise ValueError(f"refusing to ack non-rebuild_data action: {action.kind}")
    if counts is not None:
        before, after = int(counts[0]), int(counts[1])
        if after <= before:
            raise ValueError(
                "refusing to ack rebuild_data without a count postcondition: "
                f"records_before={before} records_after={after}"
            )
    uris = tuple(evidence_uris)
    evidence = bind_autotrain_action_evidence(root, handoff, action, uris)
    append_autotrain_action_receipt(
        root,
        AutotrainActionReceiptV1(
            loop_id=handoff.loop_id,
            campaign_id=handoff.campaign_id,
            action_index=action_index,
            action_sha256=autotrain_action_sha256(action),
            action_kind="rebuild_data",
            status="completed",
            evidence_uris=uris,
            evidence=evidence,
        ),
    )


def _register_i10_heal_arm(
    root: Path, loop_id: str, *, train_version: str
) -> None:
    """Add or refresh a selectable I10 successor so the park fingerprint moves."""
    global _DYNAMIC_THRASH_ARMS, _DYNAMIC_THRASH_LOADED_FOR
    _load_dynamic_thrash_arms(root, loop_id)
    hyp = (
        f"Size-matched TwoTower on the local current-rung "
        f"({_current_rung_label()}) data rebuild {train_version} improves "
        "smoke.structural_similarity versus the matched control without "
        "rotating the exhausted OFAT bank."
    )
    extras = {
        "train_version": train_version,
        "heal_resume": True,
        "process_arm": True,
        "process_role": "heal_resume",
    }
    path = _dynamic_thrash_arms_path(root, loop_id)
    kept: list[tuple[str, str, dict[str, Any]]] = [
        (slug, h, ex)
        for slug, h, ex in _DYNAMIC_THRASH_ARMS
        if slug != _HEAL_RESUME_SLUG
    ]
    kept.append((_HEAL_RESUME_SLUG, hyp, extras))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for slug, h, ex in kept:
            fh.write(
                json.dumps(
                    {
                        "schema": "autotrain_dynamic_thrash_arm/v1",
                        "slug": slug,
                        "hypothesis": h,
                        "extras": {
                            k: v
                            for k, v in ex.items()
                            if not str(k).startswith("_")
                        },
                        "created_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    _DYNAMIC_THRASH_ARMS = [
        (slug, h, {**dict(ex), "_thrash_slug": slug}) for slug, h, ex in kept
    ]
    _DYNAMIC_THRASH_LOADED_FOR = f"{root.resolve()}::{loop_id}"




def _retired_heal_versions(root: Path, loop_id: str) -> set[str]:
    """Train versions whose heal arm was measured and retired (tombstones)."""
    path = _heal_retired_versions_path(root, loop_id)
    if not path.is_file():
        return set()
    versions: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        version = str(row.get("train_version") or "")
        if version:
            versions.add(version)
    return versions


def _recover_heal_resume_arm(
    root: Path,
    loop_id: str,
    *,
    cwd: Path | None = None,
    predecessor_campaign_id: str | None = None,
) -> bool:
    """Re-register a lost heal-resume arm from a completed on-disk snapshot.

    A crash (or historical bug) between the rebuild_data ack and arm
    registration leaves a healed snapshot with no selectable successor — a
    permanent park. Recovery is idempotent: versions already measured and
    retired (tombstoned) or null-closed are never re-registered.
    """
    base = (cwd or Path.cwd()) / "outputs" / "data" / "train"
    retired = _retired_heal_versions(root, loop_id)
    candidates = sorted(
        (
            child
            for child in base.glob("continuous_i10_*")
            if child.is_dir() and not child.name.endswith("_harness")
        ),
        key=lambda child: child.stat().st_mtime,
        reverse=True,
    )
    for train_dir in candidates:
        if _rebuild_data_artifact_sources(train_dir) is None:
            continue
        prepared = train_dir.with_name(train_dir.name + "_harness")
        version = prepared.name if prepared.is_dir() else train_dir.name
        if version in retired or train_dir.name in retired:
            continue
        _register_i10_heal_arm(root, loop_id, train_version=version)
        if predecessor_campaign_id and _HEAL_RESUME_SLUG in (
            _recent_completed_nonpositive_slugs(root, predecessor_campaign_id)
        ):
            _retire_i10_heal_arm(root, loop_id, reason="recovered_spent_snapshot")
            continue
        print(
            f"SELF_HEAL_PARK_RECOVER_HEAL_ARM version={version}", flush=True
        )
        return True
    return False


def _retire_i10_heal_arm(root: Path, loop_id: str, *, reason: str) -> bool:
    """Drop the heal process arm after a complete dual-arm measurement.

    Process arms outrank OFAT selection; leaving a fixture-n-rejected heal
    selectable rematches the same incomplete-evidence win forever. Retired
    versions are tombstoned so park recovery never resurrects them.
    """
    global _DYNAMIC_THRASH_ARMS, _DYNAMIC_THRASH_LOADED_FOR
    _load_dynamic_thrash_arms(root, loop_id)
    if not any(slug == _HEAL_RESUME_SLUG for slug, _, _ in _DYNAMIC_THRASH_ARMS):
        return False
    path = _dynamic_thrash_arms_path(root, loop_id)
    retired_versions = [
        str(extras.get("train_version") or "")
        for slug, _, extras in _DYNAMIC_THRASH_ARMS
        if slug == _HEAL_RESUME_SLUG and extras.get("train_version")
    ]
    if retired_versions:
        tombstones = _heal_retired_versions_path(root, loop_id)
        tombstones.parent.mkdir(parents=True, exist_ok=True)
        with tombstones.open("a", encoding="utf-8") as fh:
            for version in retired_versions:
                fh.write(
                    json.dumps(
                        {
                            "schema": "autotrain_heal_retired_version/v1",
                            "train_version": version,
                            "reason": reason,
                            "retired_at": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    kept = [
        (slug, hyp, extras)
        for slug, hyp, extras in _DYNAMIC_THRASH_ARMS
        if slug != _HEAL_RESUME_SLUG
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for slug, hyp, extras in kept:
            fh.write(
                json.dumps(
                    {
                        "schema": "autotrain_dynamic_thrash_arm/v1",
                        "slug": slug,
                        "hypothesis": hyp,
                        "extras": {
                            k: v
                            for k, v in extras.items()
                            if not str(k).startswith("_")
                        },
                        "created_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    _DYNAMIC_THRASH_ARMS = [
        (slug, hyp, {**dict(extras), "_thrash_slug": slug})
        for slug, hyp, extras in kept
    ]
    _DYNAMIC_THRASH_LOADED_FOR = f"{root.resolve()}::{loop_id}"
    print(
        f"HEAL_RESUME_RETIRE slug={_HEAL_RESUME_SLUG} reason={reason}",
        flush=True,
    )
    return True


def _prepare_i10_train_dir_for_sft(train_dir: Path) -> Path:
    """Filter NL/repair prompts and clear open synthesis feedback for SFT.

    ``train_model`` loads prompts via ``parse_harness_task``. Cap0 I10 rebuilds
    still emit Repair/NL rows; keep only harness-parseable prompts under a
    ``*_harness`` sibling when needed so heal arms are train-loadable.
    """
    from datetime import datetime, timezone

    from slm_training.autoresearch.hillclimb import (
        assert_synthesis_feedback_cleared_for_sft,
        open_synthesis_recommendations,
    )
    from slm_training.dsl.harness_dsl import parse_harness_task
    from slm_training.harnesses.model_build.data import load_train_records

    records_path = train_dir / "records.jsonl"
    if not records_path.is_file():
        return train_dir
    kept: list[dict[str, Any]] = []
    dropped = 0
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            parse_harness_task(str(row.get("prompt") or ""))
        except Exception:  # noqa: BLE001 — drop non-harness prompts for SFT
            dropped += 1
            continue
        kept.append(row)
    out_dir = train_dir
    if dropped:
        out_dir = train_dir.parent / f"{train_dir.name}_harness"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "records.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in kept),
            encoding="utf-8",
        )
        for name in (
            "synthesis_feedback.json",
            "quality_report.json",
            "rejected.jsonl",
            "data_manifest.json",
            "manifest.json",
            "stats.json",
        ):
            src = train_dir / name
            if src.is_file():
                (out_dir / name).write_bytes(src.read_bytes())
        man_path = out_dir / "manifest.json"
        if man_path.is_file():
            try:
                man = json.loads(man_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                man = {}
            man["version"] = out_dir.name
            man["record_count"] = len(kept)
            man["derived_from"] = train_dir.name
            man["derive_note"] = (
                "filtered to HARNESS_V1 prompts for symbol-only SFT loader; "
                f"dropped={dropped}"
            )
            man["derived_at"] = datetime.now(timezone.utc).isoformat()
            man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        print(
            f"I10_SFT_FILTER version={out_dir.name} kept={len(kept)} "
            f"dropped={dropped}",
            flush=True,
        )
    feedback_path = out_dir / "synthesis_feedback.json"
    actions_path = out_dir / "synthesis_feedback_actions.json"
    if feedback_path.is_file() and not actions_path.is_file():
        feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
        seen: dict[str, dict[str, Any]] = {}
        for rec in open_synthesis_recommendations(feedback):
            code = str(rec.get("code") or "")
            if code and code not in seen:
                seen[code] = dict(rec)
        actions = [
            {
                "code": code,
                "target_kind": rec.get("target_kind"),
                "target": rec.get("target"),
                "note": (
                    f"I10 continuous snapshot {out_dir.name}: recorded {code} "
                    f"for {rec.get('target_kind')}:{rec.get('target')}. "
                    "Fixture climb uses admitted harness rows only; next rebuild "
                    "must cut expansion / fix quarantine producers. Gates unchanged."
                ),
                "evidence": rec.get("evidence") or {},
            }
            for code, rec in seen.items()
        ]
        if actions:
            actions_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "train_version": out_dir.name,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "honesty_mode": "fixture_or_scratch",
                        "actions": actions,
                        "experiment_candidates_filed": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    try:
        load_train_records(out_dir)
        assert_synthesis_feedback_cleared_for_sft(out_dir)
    except Exception as exc:  # noqa: BLE001 — keep original on prepare failure
        print(f"I10_SFT_PREPARE_WARN dir={out_dir} err={exc!r}", flush=True)
        return train_dir
    return out_dir


def _self_heal_rebuild_data(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str | None,
) -> str | None:
    """Run a wall-capped local CPU data rebuild and ack rebuild_data.

    Does not fake a receipt: missing quality artifacts leave the action pending.
    A successful heal registers an I10 resume arm so the parked fingerprint
    moves and the next cycle can train instead of rematching smoke.
    """
    if not campaign_id:
        return None
    handoff_path = root / campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != loop_id or handoff.campaign_id != campaign_id:
        return None
    pending = [
        (index, action)
        for index, action in pending_autotrain_actions(root, handoff)
        if action.kind == "rebuild_data"
    ]
    if not pending:
        print(
            f"SELF_HEAL_REBUILD_DATA_SKIP campaign={campaign_id} "
            "reason=no_pending_rebuild_data",
            flush=True,
        )
        return None
    if any("screening suite" in action.reason for _i, action in pending):
        return _self_heal_rebuild_screening_eval(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=campaign_id
        )
    cycle_index = int(handoff.cycle_index or 0)
    train_version = _local_i10_train_version(loop_id, cycle_index)
    train_dir = cwd / "outputs" / "data" / "train" / train_version
    adequacy = _sample_adequacy_report(cwd)
    if adequacy is not None and adequacy.get("verdict") == (
        "saturated_change_trajectory"
    ):
        # Measured flat marginal gain: rebuilding is the wrong lever. Leave
        # the action pending for a trajectory decision instead of faking
        # progress with another same-distribution corpus.
        print(
            f"SAMPLE_ADEQUACY_SATURATED campaign={campaign_id} "
            f"source={adequacy.get('marginal_gain_source')}",
            flush=True,
        )
        return None
    sources = _rebuild_data_artifact_sources(train_dir)
    if sources is None:
        argv = _local_rebuild_data_argv(
            train_version=train_version, adequacy=adequacy
        )
        print(
            f"SELF_HEAL_REBUILD_DATA start campaign={campaign_id} "
            f"version={train_version} argv={argv}",
            flush=True,
        )
        result = run_bounded_process(
            argv,
            interrupt_after_seconds=float(INTERRUPT_AFTER_SECONDS),
            kill_grace_seconds=float(KILL_GRACE_SECONDS),
            cwd=str(cwd),
        )
        if result.outcome != ProcessOutcome.COMPLETED or result.returncode != 0:
            print(
                f"SELF_HEAL_REBUILD_DATA_FAIL campaign={campaign_id} "
                f"outcome={result.outcome} code={result.returncode}",
                flush=True,
            )
            return None
        sources = _rebuild_data_artifact_sources(train_dir)
    if sources is None:
        print(
            f"SELF_HEAL_REBUILD_DATA_FAIL campaign={campaign_id} "
            "missing=quality/feedback/manifest",
            flush=True,
        )
        return None
    camp_dir = root / campaign_id
    camp_dir.mkdir(parents=True, exist_ok=True)
    evidence_uris: list[str] = []
    for name, src in sources.items():
        dest = camp_dir / name
        dest.write_bytes(src.read_bytes())
        evidence_uris.append(name)
    if adequacy is not None:
        (camp_dir / "sample_adequacy.json").write_text(
            json.dumps(adequacy, indent=2) + "\n", encoding="utf-8"
        )
        # sample_adequacy.json is not a rebuild_data receipt name
    for index, _action in pending:
        _ack_rebuild_data_action(
            root, handoff, action_index=index, evidence_uris=evidence_uris
        )
    prepared = _prepare_i10_train_dir_for_sft(train_dir)
    register_version = prepared.name if prepared != train_dir else train_version
    closed = _recent_completed_nonpositive_slugs(root, campaign_id)
    open_slugs = _thrash_bank_open_slugs(closed)
    if not open_slugs or _open_slugs_are_snapshot_leftovers(open_slugs):
        _register_i10_heal_arm(root, loop_id, train_version=register_version)
    print(
        f"SELF_HEAL_REBUILD_DATA campaign={campaign_id} "
        f"version={register_version} files={evidence_uris}",
        flush=True,
    )
    return "rebuild_data"








def _self_heal_document_actions(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str | None,
) -> str | None:
    """Write/commit docs/design closeout and ack pending document actions.

    Returns a heal kind when at least one document action was completed.
    Never acks repair_harness / formal / deliver_stack / rebuild_data / stop.
    """
    if not campaign_id:
        return None
    handoff_path = root / campaign_id / "cycle_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    if handoff.loop_id != loop_id or handoff.campaign_id != campaign_id:
        return None
    pending_docs = [
        (index, action)
        for index, action in pending_autotrain_actions(root, handoff)
        if action.kind == "document"
    ]
    if not pending_docs:
        return None

    delivery_path = root / campaign_id / "sdlc_delivery.json"
    delivery: dict[str, Any] = {}
    if delivery_path.is_file():
        try:
            loaded = _read_json(delivery_path)
            if isinstance(loaded, dict):
                delivery = loaded
        except Exception:  # noqa: BLE001 — still document what we have
            delivery = {}

    md_path, json_path = _continuous_docs_paths(cwd, campaign_id)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_text, payload = _render_continuous_cycle_docs(
        campaign_id=campaign_id,
        loop_id=loop_id,
        handoff=handoff,
        delivery=delivery,
    )
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    touched: list[Path] = [md_path, json_path]
    if handoff.checkpoint_documentation_required:
        touched.extend(
            _append_checkpoint_doc_notes(
                cwd,
                campaign_id=campaign_id,
                checkpoint_paths=tuple(handoff.checkpoint_paths or ()),
                loop_id=loop_id,
            )
        )
        if handoff.checkpoint_paths and not any(
            p.name in {"MODEL_CARD.md", "README.md"} for p in touched
        ):
            print(
                "SELF_HEAL_DOCUMENT_WARN "
                f"campaign={campaign_id} reason=checkpoint_docs_missing_templates",
                flush=True,
            )
            # Still ack design docs; iron-law checkpoint note attempted.
    committed = _git_commit_paths(
        cwd,
        touched,
        message=f"docs(autotrain): continuous {loop_id} {campaign_id} closeout",
        root=root,
        loop_id=loop_id,
        stage="self-heal-document",
    )
    evidence_uris: list[str] = []
    for path in touched:
        try:
            rel = str(path.resolve().relative_to(cwd.resolve()))
        except ValueError:
            rel = str(path)
        # Evidence must be git-tracked (ls-files --error-unmatch fails if not).
        try:
            _git(
                "ls-files",
                "--error-unmatch",
                rel,
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                stage="self-heal-document-tracked",
            )
        except Exception:  # noqa: BLE001 — untracked path is not valid evidence
            continue
        evidence_uris.append(rel)
    if not evidence_uris:
        print(
            f"SELF_HEAL_DOCUMENT_FAIL campaign={campaign_id} reason=no_tracked_evidence "
            f"committed={committed}",
            flush=True,
        )
        return None
    for index, _action in pending_docs:
        _ack_document_action(
            root,
            handoff,
            action_index=index,
            evidence_uris=evidence_uris,
        )
    print(
        f"SELF_HEAL_DOCUMENT campaign={campaign_id} "
        f"files={evidence_uris} acked={len(pending_docs)}",
        flush=True,
    )
    return "document_closeout"


def _self_heal_loop_owned_generated_dirt(
    *,
    cwd: Path,
    root: Path | None = None,
    loop_id: str | None = None,
) -> str | None:
    """Restore driver-written tracked mirrors so they never hard-block thrash."""
    git_kw: dict[str, Any] = {"cwd": cwd}
    if root is not None and loop_id is not None:
        git_kw.update(root=root, loop_id=loop_id)
    porcelain = _git(
        "status",
        "--porcelain",
        stage="self-heal-owned-dirt-status" if root is not None else None,
        **git_kw,
    )
    if not porcelain.strip():
        return None
    owned = [
        path
        for path in _porcelain_paths(porcelain)
        if _is_loop_owned_generated_path(path)
    ]
    if not owned:
        return None
    _git(
        "restore",
        "--source=HEAD",
        "--worktree",
        "--staged",
        "--",
        *owned,
        stage="self-heal-owned-dirt" if root is not None else None,
        **git_kw,
    )
    print(f"SELF_HEAL_LOOP_OWNED_DIRT files={owned}", flush=True)
    return "loop_owned_generated_dirt"


def _self_heal_continuous_dirty_tree(
    *,
    cwd: Path,
    root: Path | None = None,
    loop_id: str | None = None,
) -> str | None:
    """Commit continuous closeout dirt only; leave foreign WIP hard-failing."""
    # _git requires root/loop_id/stage as a triple — omit all when root missing.
    git_kw: dict[str, Any] = {"cwd": cwd}
    if root is not None and loop_id is not None:
        git_kw.update(root=root, loop_id=loop_id)
    porcelain = _git(
        "status",
        "--porcelain",
        stage="self-heal-dirty-status" if root is not None else None,
        **git_kw,
    )
    if not porcelain.strip():
        return None
    paths = _porcelain_paths(porcelain)
    if not paths:
        return None
    serena_kind = _maybe_restore_serena_project_yml(
        cwd=cwd, paths=paths, git_kw=git_kw
    )
    if serena_kind:
        porcelain = _git(
            "status",
            "--porcelain",
            stage="self-heal-dirty-status" if root is not None else None,
            **git_kw,
        )
        if not porcelain.strip():
            return serena_kind
        paths = _porcelain_paths(porcelain)
        if not paths:
            return serena_kind
    closeout = [p for p in paths if _is_continuous_closeout_path(p)]
    foreign = [p for p in paths if _is_foreign_dirty_path(p)]
    if foreign and not closeout:
        print(
            f"SELF_HEAL_DIRTY_TREE_SKIP foreign={foreign[:8]}",
            flush=True,
        )
        return None
    if not closeout:
        return None
    abs_paths = [cwd / p for p in closeout]
    committed = _git_commit_paths(
        cwd,
        abs_paths,
        message=f"docs(autotrain): continuous {loop_id or 'loop'} self-heal closeout",
        root=root,
        loop_id=loop_id,
        stage="self-heal-dirty",
    )
    if not committed:
        return None
    print(f"SELF_HEAL_DIRTY_TREE files={closeout}", flush=True)
    return "dirty_tree_closeout"


def _self_heal_closeout_blockers(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str | None = None,
) -> str | None:
    """Run document + dirty-tree heals; return last successful heal kind."""
    kinds: list[str] = []
    target = campaign_id or _latest_cycle(root, loop_id)[1]
    try:
        doc_kind = _self_heal_document_actions(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=target
        )
        if doc_kind:
            kinds.append(doc_kind)
    except Exception as exc:  # noqa: BLE001 — surface, keep trying dirty heal
        print(f"SELF_HEAL_DOCUMENT_WARN err={exc!r}", flush=True)
    try:
        owned_kind = _self_heal_loop_owned_generated_dirt(
            cwd=cwd, root=root, loop_id=loop_id
        )
        if owned_kind:
            kinds.append(owned_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_LOOP_OWNED_DIRT_WARN err={exc!r}", flush=True)
    try:
        dirty_kind = _self_heal_continuous_dirty_tree(
            cwd=cwd, root=root, loop_id=loop_id
        )
        if dirty_kind:
            kinds.append(dirty_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_DIRTY_TREE_WARN err={exc!r}", flush=True)
    return kinds[-1] if kinds else None


def self_heal_unblock_loop(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    integration_commit: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Single owner for continuous soft-unblock (never chat-prompt thrash).

    Returns::

        {
          "soft_healed": [kind, ...],
          "hard_pending": [{"campaign_id", "index", "kind", "reason"}, ...],
          "blocker_cleared": bool,
          "predecessor_campaign_id": str | None,
        }

    Soft: incomplete origin/main merge, document, continuous-only dirt,
    loop-owned generated mirrors, thrash timeout residual repair_harness,
    bank exhaust, local-CPU rebuild_data.
    Hard: true harness crash, formal, deliver_stack, foreign dirt (non-merge WIP).
    """
    soft_healed: list[str] = []
    hard_pending: list[dict[str, Any]] = []
    pred = campaign_id or _latest_cycle(root, loop_id)[1]
    parked = _check_regime_parked(root=root, loop_id=loop_id, cwd=cwd) is not None

    # 0) Incomplete merge (UU) from landing origin/main into the thrash worktree.
    try:
        merge_kind = _self_heal_incomplete_merge(cwd=cwd, root=root, loop_id=loop_id)
        if merge_kind:
            soft_healed.append(merge_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK merge_warn={exc!r}", flush=True)

    # 1) Hygiene — restore driver-written mirrors, then commit closeout docs.
    try:
        owned_kind = _self_heal_loop_owned_generated_dirt(
            cwd=cwd, root=root, loop_id=loop_id
        )
        if owned_kind:
            soft_healed.append(owned_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK owned_dirt_warn={exc!r}", flush=True)
    try:
        dirty_kind = _self_heal_continuous_dirty_tree(
            cwd=cwd, root=root, loop_id=loop_id
        )
        if dirty_kind:
            soft_healed.append(dirty_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK dirty_warn={exc!r}", flush=True)

    # Foreign dirt is hard (tree still dirty after continuous-only attempt).
    try:
        porcelain = _git(
            "status",
            "--porcelain",
            cwd=cwd,
            root=root,
            loop_id=loop_id,
            stage="self-heal-unblock-dirty-check",
        )
    except Exception:  # noqa: BLE001
        porcelain = ""
    if porcelain.strip():
        paths = _porcelain_paths(porcelain)
        serena_kind = _maybe_restore_serena_project_yml(
            cwd=cwd,
            paths=paths,
            git_kw={
                "cwd": cwd,
                "root": root,
                "loop_id": loop_id,
                "stage": "self-heal-serena-yml",
            },
        )
        if serena_kind:
            soft_healed.append(serena_kind)
            porcelain = _git(
                "status",
                "--porcelain",
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                stage="self-heal-unblock-dirty-check",
            )
            paths = _porcelain_paths(porcelain) if porcelain.strip() else []
        foreign = [p for p in paths if _is_foreign_dirty_path(p)]
        try:
            from slm_training.autoresearch.heal.fail_closed import lease_covers
            lease = root / "loops" / loop_id / "wip_lease.json"
            leased = [p for p in foreign if lease_covers(lease, p)]
            if leased:
                soft_healed.append("wip_lease_deferred")
                foreign = [p for p in foreign if p not in leased]
        except Exception as exc:  # noqa: BLE001
            print(f"SELF_HEAL_UNBLOCK lease_warn={exc!r}", flush=True)
        if foreign:
            hard_pending.append(
                {
                    "campaign_id": pred,
                    "index": -1,
                    "kind": "foreign_dirty_tree",
                    "reason": f"non-closeout dirty paths: {foreign[:8]}",
                }
            )

    # 2) Thrash timeout residual repair_harness → next_experiment.
    try:
        timeout_kind = _self_heal_thrash_timeout_repair(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=pred
        )
        if timeout_kind:
            soft_healed.append(timeout_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK timeout_warn={exc!r}", flush=True)

    # 2a) Environment-incomplete repair_harness with a verified heal receipt →
    # next_experiment (SELF_HEAL_ENV_REPAIR). Code-class crashes stay hard.
    try:
        env_kind = _self_heal_env_repair_rewrite(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=pred
        )
        if env_kind:
            soft_healed.append(env_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK env_repair_warn={exc!r}", flush=True)

    # 2b) Quality-arm bank exhaust repair_harness → compose + next_experiment.
    # Parked I10 off-ramp must not be unparked by compose filler.
    if not parked:
        try:
            bank_kind = _self_heal_bank_exhaust_repair(
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                campaign_id=pred,
                integration_commit=integration_commit,
            )
            if bank_kind:
                soft_healed.append(bank_kind)
        except Exception as exc:  # noqa: BLE001
            print(f"SELF_HEAL_UNBLOCK bank_repair_warn={exc!r}", flush=True)

    # 2c0) Screening-n suite deficit: generate/publish smoke, never screen at 3.
    try:
        screen_kind = _self_heal_rebuild_screening_eval(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=pred
        )
        if screen_kind:
            soft_healed.append(screen_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK screening_eval_warn={exc!r}", flush=True)

    # 2c) Local CPU rebuild_data (I10 / bank-exhaust park). Never fake.
    try:
        rebuild_kind = _self_heal_rebuild_data(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=pred
        )
        if rebuild_kind:
            soft_healed.append(rebuild_kind)
            parked = _check_regime_parked(root=root, loop_id=loop_id, cwd=cwd) is not None
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK rebuild_warn={exc!r}", flush=True)

    # 3) Document closeout.
    try:
        doc_kind = _self_heal_document_actions(
            cwd=cwd, root=root, loop_id=loop_id, campaign_id=pred
        )
        if doc_kind:
            soft_healed.append(doc_kind)
    except Exception as exc:  # noqa: BLE001
        print(f"SELF_HEAL_UNBLOCK document_warn={exc!r}", flush=True)

    # 4) Bank exhaust when no open screening arms.
    if not parked:
        try:
            closed = _recent_completed_nonpositive_slugs(root, pred)
            entries = _load_champion_queue(_champion_queue_path(root, loop_id))
            if integration_commit:
                _reopen_harness_blocked_champions(
                    root, entries, integration_commit=integration_commit
                )
                _write_champion_queue(_champion_queue_path(root, loop_id), entries)
            skip = _skip_arm_slugs(
                entries, integration_commit=integration_commit, include_causal_cap=False
            )
            bank_heal = _self_heal_thrash_bank_exhaust(
                root,
                loop_id,
                closed=closed,
                skip=skip | closed,
                predecessor_campaign_id=pred,
            )
            if bank_heal.composed:
                soft_healed.append("thrash_bank_compose")
        except Exception as exc:  # noqa: BLE001
            print(f"SELF_HEAL_UNBLOCK bank_warn={exc!r}", flush=True)

    # 5) Enumerate remaining hard prereqs on predecessor.
    if pred:
        handoff_path = root / pred / "cycle_handoff.json"
        if handoff_path.is_file():
            try:
                handoff = AutotrainCycleHandoffV1.model_validate_json(
                    handoff_path.read_text(encoding="utf-8")
                )
                delivery: dict[str, Any] = {}
                delivery_path = root / pred / "sdlc_delivery.json"
                if delivery_path.is_file():
                    try:
                        loaded = _read_json(delivery_path)
                        if isinstance(loaded, dict):
                            delivery = loaded
                    except Exception:  # noqa: BLE001
                        delivery = {}
                pending = list(pending_autotrain_actions(root, handoff))
                if delivery.get("stack_layer") is False:
                    pending = [(i, a) for i, a in pending if a.kind != "deliver_stack"]
                for index, action in pending:
                    if action.kind == "document":
                        hard_pending.append(
                            {
                                "campaign_id": pred,
                                "index": index,
                                "kind": "document",
                                "reason": str(action.reason or "document unacked"),
                            }
                        )
                        continue
                    if action.kind == "repair_harness":
                        if _delivery_is_thrash_timeout_residual(delivery, handoff):
                            # Should have been healed; treat residual as soft miss.
                            continue
                        if _is_bank_exhaust_repair_action(action):
                            # Compose failed or rewrite missed — typed hard stop.
                            hard_pending.append(
                                {
                                    "campaign_id": pred,
                                    "index": index,
                                    "kind": "repair_harness",
                                    "reason": (
                                        "bank_exhaust_no_successors: "
                                        + str(action.reason or "arm bank exhausted")
                                    ),
                                }
                            )
                            continue
                        hard_pending.append(
                            {
                                "campaign_id": pred,
                                "index": index,
                                "kind": "repair_harness",
                                "reason": str(action.reason or "harness repair"),
                            }
                        )
                        continue
                    if action.kind in _HARD_PREREQUISITE_ACTION_KINDS:
                        hard_pending.append(
                            {
                                "campaign_id": pred,
                                "index": index,
                                "kind": action.kind,
                                "reason": str(action.reason or action.kind),
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                print(f"SELF_HEAL_UNBLOCK enumerate_warn={exc!r}", flush=True)

    # Drop document hard_pending if we just acked (re-check).
    if pred and any(h.get("kind") == "document" for h in hard_pending):
        try:
            handoff = AutotrainCycleHandoffV1.model_validate_json(
                (root / pred / "cycle_handoff.json").read_text(encoding="utf-8")
            )
            still_doc = any(
                a.kind == "document"
                for _, a in pending_autotrain_actions(root, handoff)
            )
            if not still_doc:
                hard_pending = [h for h in hard_pending if h.get("kind") != "document"]
        except Exception:  # noqa: BLE001
            pass

    # Annotate every hard blocker with its heal class so the supervisor's
    # playbook dispatch and the escalation ledger never re-derive it.
    if hard_pending:
        try:
            from slm_training.autoresearch.heal.classify import classify_blocker

            for entry in hard_pending:
                entry["blocker_class"] = classify_blocker(
                    str(entry.get("kind") or ""), str(entry.get("reason") or "")
                )
        except Exception as exc:  # noqa: BLE001
            print(f"SELF_HEAL_UNBLOCK classify_warn={exc!r}", flush=True)

    blocker_cleared = not hard_pending
    if blocker_cleared:
        _clear_loop_blocker(
            root,
            loop_id,
            reason=(
                "unblock:" + ",".join(soft_healed) if soft_healed else "unblock:clean"
            ),
        )
        _record_cycle_recovery(
            root=root,
            loop_id=loop_id,
            soft_healed=soft_healed,
            predecessor_campaign_id=pred,
        )
    else:
        # Typed hard block — one state write, no restart thrash on soft classes.
        cycle_index, _ = _latest_cycle(root, loop_id)
        detail = "; ".join(
            f"{h.get('kind')}@{h.get('campaign_id')}:{h.get('reason', '')[:80]}"
            for h in hard_pending
        )
        _write_loop_state(
            root,
            AutotrainLoopStateV1(
                loop_id=loop_id,
                state="BLOCKED",
                phase="blocked",
                cycle_index=max(0, cycle_index),
                next_action=f"hard_pending:{detail[:200]}",
                blocker_fingerprint=hashlib.sha256(detail.encode("utf-8")).hexdigest(),
                blocker_count=len(hard_pending),
                pid=os.getpid(),
                heartbeat_at=utc_now(),
            ),
        )
        print(f"SELF_HEAL_UNBLOCK hard_pending={hard_pending}", flush=True)

    report = {
        "soft_healed": soft_healed,
        "hard_pending": hard_pending,
        "blocker_cleared": blocker_cleared,
        "predecessor_campaign_id": pred,
    }
    print(
        f"SELF_HEAL_UNBLOCK cleared={blocker_cleared} "
        f"soft={soft_healed} hard_n={len(hard_pending)} pred={pred}",
        flush=True,
    )
    return report










def _set_active_stage(root: Path, loop_id: str, stage: str) -> None:
    path = _loop_state_path(root, loop_id)
    with autotrain_loop_state_lock(root, loop_id):
        if not path.is_file():
            return
        try:
            state = AutotrainLoopStateV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return
        _write_loop_state_unlocked(
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
    with autotrain_loop_state_lock(root, loop_id):
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
        _write_loop_state_unlocked(
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


def _clear_active_stage(root: Path, loop_id: str) -> None:
    path = _loop_state_path(root, loop_id)
    with autotrain_loop_state_lock(root, loop_id):
        if not path.is_file():
            return
        try:
            state = AutotrainLoopStateV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return
        _write_loop_state_unlocked(
            root,
            state.model_copy(
                update={
                    "active_stage": None,
                    "child_pid": None,
                    "stage_started_at": None,
                    "heartbeat_at": utc_now(),
                }
            ),
        )




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
        if (
            not campaign_id
            or not control_id
            or not candidate_id
            or not camp_dir.is_dir()
        ):
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
                    row.get("source_role") or handoff.get("cycle_role") or "screening"
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




_PROMOTE_AUTHORITY_STATUSES = frozenset({"promoted", "climb_accepted"})




def current_promote_authority() -> dict[str, str]:
    """Identity of current climb-promote rules + harness (for recert stamps)."""
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        promote_authority_sha256,
    )

    climb = load_climb_policy()
    try:
        locked = locked_promote_expectations_sha256()
    except OSError:
        locked = ""
    harness_version = _experiment_campaign_component_version()
    sha = promote_authority_sha256(
        climb_policy_sha256=str(climb.sha256),
        locked_expectations_sha256=locked,
        harness_component_version=harness_version,
        harness_component=_PROMOTE_AUTHORITY_HARNESS_COMPONENT,
    )
    return {
        "schema": "autotrain_promote_authority/v1",
        "sha256": sha,
        "climb_policy_sha256": str(climb.sha256),
        "locked_expectations_sha256": locked,
        "harness_component": _PROMOTE_AUTHORITY_HARNESS_COMPONENT,
        "harness_component_version": harness_version,
    }


def _stamp_promote_authority(row: dict[str, Any], authority: dict[str, str]) -> None:
    row["promote_authority_sha256"] = authority["sha256"]
    row["promote_authority"] = {
        "schema": authority.get("schema"),
        "climb_policy_sha256": authority.get("climb_policy_sha256"),
        "locked_expectations_sha256": authority.get("locked_expectations_sha256"),
        "harness_component": authority.get("harness_component"),
        "harness_component_version": authority.get("harness_component_version"),
    }
    row["promote_authority_stamped_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )




def _paper_recert_promote_entry(
    root: Path,
    row: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Re-run current dispose rules on stored promote evidence.

    Returns ``(action, reasons, disposition)`` where action is one of:
    ``keep`` (still climb-valid), ``fail`` (promotion_failed), or
    ``requeue`` (needs a live promote re-run under current rules).
    """
    campaign_id = str(
        row.get("promotion_campaign_id") or row.get("confirm_campaign_id") or ""
    )
    if not campaign_id:
        return (
            "requeue",
            ["recert_required:missing_promotion_campaign"],
            None,
        )
    camp = root / campaign_id
    if not camp.is_dir():
        return (
            "requeue",
            [f"recert_required:missing_campaign_dir:{campaign_id}"],
            None,
        )

    delivery = _read_json(camp / "sdlc_delivery.json")
    formal = str(
        row.get("formal_preflight_status") or _formal_preflight_status(camp) or ""
    )
    certificate = _load_promote_certificate(camp)
    if formal != "proved" or certificate is None:
        return (
            "requeue",
            [
                f"recert_required:incomplete_evidence:formal={formal!r}:"
                f"certificate={'yes' if certificate is not None else 'no'}"
            ],
            None,
        )

    control_id = str(delivery.get("control_id") or "")
    candidate_id = str(delivery.get("candidate_id") or "")
    ctrl_metrics = delivery.get("control_metrics")
    cand_metrics = delivery.get("candidate_metrics")
    if not isinstance(ctrl_metrics, dict) or not isinstance(cand_metrics, dict):
        ctrl_metrics = (
            _run_metrics(camp, control_id, prefer_held_out=True) if control_id else {}
        )
        cand_metrics = (
            _run_metrics(camp, candidate_id, prefer_held_out=True)
            if candidate_id
            else {}
        )
    if not isinstance(ctrl_metrics, dict) or not isinstance(cand_metrics, dict):
        return (
            "requeue",
            ["recert_required:missing_dual_arm_metrics"],
            None,
        )

    try:
        locked = locked_promote_expectations_sha256()
    except OSError as exc:
        return (
            "fail",
            [f"promote_locked_expectations_unreadable:{exc}"],
            None,
        )

    reasons_in = list(delivery.get("reasons") or row.get("resolve_reasons") or [])
    phase_a_positive = bool(delivery.get("positive"))
    phase_a_quality = _quality_held_reasons(reasons_in) or any(
        str(r).startswith("primary_metric_win:")
        or str(r).startswith("quality_metric_win:")
        for r in reasons_in
    )
    from slm_training.autoresearch.climb_policy import load_climb_policy

    climb = load_climb_policy()
    disposition = dispose_champion_promote(
        formal_preflight_status=formal,
        certificate=certificate,
        locked_expectations_sha256=locked,
        phase_a_positive=phase_a_positive,
        phase_a_quality_held=phase_a_quality,
        control_metrics=ctrl_metrics,  # type: ignore[arg-type]
        candidate_metrics=cand_metrics,  # type: ignore[arg-type]
        promotion_primary=dict(climb.promotion_primary),
        promotion_dispose=dict(climb.promotion_dispose),
    )
    status = str(disposition.get("status") or "")
    reasons = [str(r) for r in disposition.get("reasons") or []]
    if status == "climb_accepted":
        return "keep", reasons, disposition
    if disposition.get("inconclusive") or status == "promotion_inconclusive":
        return (
            "requeue",
            ["recert_required:promotion_inconclusive", *reasons],
            disposition,
        )
    if status in {"promotion_failed", "rejected"}:
        return "fail", reasons, disposition
    return (
        "requeue",
        [f"recert_required:unhandled_dispose_status:{status}", *reasons],
        disposition,
    )


def _recertify_promoted_champion_entries(
    root: Path,
    loop_id: str,
    entries: list[dict[str, Any]],
    *,
    authority: dict[str, str] | None = None,
) -> bool:
    """Re-certify climb promotions when promote authority (policy/harness) changes.

    Built into every cycle startup so harness or climb-policy updates cannot leave
    historical ``promoted`` / ``climb_accepted`` rows authoritative under stale
    dispose rules. Actions:

    - stamp matches current authority → no-op
    - paper dispose still ``climb_accepted`` → restamp keep
    - paper dispose fails → ``promotion_failed`` + audit
    - incomplete evidence under current rules → requeue as ``confirmed`` for live
      promote re-run (``recert_required``)
    """
    from slm_training.autoresearch.climb_policy import load_climb_policy

    climb = load_climb_policy()
    dispose_cfg = dict(climb.promotion_dispose)
    if not bool(dispose_cfg.get("recertify_on_authority_change", True)):
        return False

    authority = authority or current_promote_authority()
    auth_sha = str(authority.get("sha256") or "")
    if not auth_sha:
        return False

    changed = False
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for row in entries:
        if row.get("status") not in _PROMOTE_AUTHORITY_STATUSES:
            continue
        stamped = str(row.get("promote_authority_sha256") or "")
        if stamped and stamped == auth_sha:
            continue

        before_status = str(row.get("status") or "")
        before_stamp = stamped or None
        action, reasons, disposition = _paper_recert_promote_entry(root, row)
        base_reasons = [
            "promote_authority_changed"
            if stamped
            else "promote_authority_unstamped_legacy",
            f"promote_authority_was={before_stamp or 'none'}",
            f"promote_authority_now={auth_sha[:16]}",
            f"harness_component_version={authority.get('harness_component_version')}",
            *reasons,
        ]

        if action == "keep":
            _stamp_promote_authority(row, authority)
            if disposition is not None:
                row["cert_policy"] = disposition.get("cert_policy")
                row["primary_improvement"] = disposition.get("primary_improvement")
                row["promotion_primary_met"] = disposition.get("promotion_primary_met")
            row["resolve_reasons"] = [
                "recertified_under_current_promote_authority",
                *base_reasons,
            ]
            row["historical_recertified_at"] = now
            row["historical_reclassification"] = (
                f"{before_status}→{before_status}:recert_keep"
            )
            changed = True
            print(
                "CHAMPION_PROMOTE_RECERT_KEEP "
                f"entry_id={row.get('entry_id')} "
                f"authority={auth_sha[:12]}",
                flush=True,
            )
            _append_historical_reclassification(
                root,
                loop_id,
                {
                    "schema": "autotrain_historical_reclassification/v1",
                    "kind": "champion_queue",
                    "repair": "promote_authority_recert",
                    "loop_id": loop_id,
                    "entry_id": row.get("entry_id"),
                    "campaign_id": row.get("promotion_campaign_id")
                    or row.get("confirm_campaign_id"),
                    "before_status": before_status,
                    "after_status": before_status,
                    "action": "keep",
                    "authority_sha256": auth_sha,
                    "reclassified_at": now,
                    "reasons": base_reasons[:12],
                },
            )
            continue

        if action == "fail":
            row["status"] = "promotion_failed"
            row["resolved_at"] = now
            row["promotion_primary_met"] = False
            row["resolve_reasons"] = [
                "historical_reclassification:"
                f"{before_status}→promotion_failed:recert_under_current_policy",
                *base_reasons,
            ]
            row["historical_reclassified_at"] = now
            row["historical_reclassification"] = (
                f"{before_status}→promotion_failed:recert_under_current_policy"
            )
            # Clear authority stamp — failed under current rules.
            row.pop("promote_authority_sha256", None)
            changed = True
            print(
                "CHAMPION_PROMOTE_RECERT_FAIL "
                f"entry_id={row.get('entry_id')} "
                f"was={before_status}",
                flush=True,
            )
            _append_historical_reclassification(
                root,
                loop_id,
                {
                    "schema": "autotrain_historical_reclassification/v1",
                    "kind": "champion_queue",
                    "repair": "promote_authority_recert",
                    "loop_id": loop_id,
                    "entry_id": row.get("entry_id"),
                    "campaign_id": row.get("promotion_campaign_id")
                    or row.get("confirm_campaign_id"),
                    "before_status": before_status,
                    "after_status": "promotion_failed",
                    "action": "fail",
                    "authority_sha256": auth_sha,
                    "reclassified_at": now,
                    "reasons": base_reasons[:12],
                },
            )
            # Append ledger event for observability.
            ledger = root / "loops" / loop_id / "learning_certificate_ledger.jsonl"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "schema": "autotrain_learning_event/v1",
                            "loop_id": loop_id,
                            "campaign_id": row.get("promotion_campaign_id")
                            or row.get("confirm_campaign_id"),
                            "entry_id": row.get("entry_id"),
                            "knobs_fingerprint": row.get("knobs_fingerprint"),
                            "outcome": "promotion_failed",
                            "reasons": base_reasons[:20],
                            "recert": True,
                            "promote_authority_sha256": auth_sha,
                            "finished_at": now,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            continue

        # requeue for live promote under current authority
        row["status"] = "confirmed"
        row.pop("resolved_at", None)
        row["recert_required"] = True
        row["recert_required_at"] = now
        row["recert_from_status"] = before_status
        row["resolve_reasons"] = [
            "recert_required_under_current_promote_authority",
            *base_reasons,
        ]
        row["historical_reclassified_at"] = now
        row["historical_reclassification"] = (
            f"{before_status}→confirmed:recert_required"
        )
        row.pop("promote_authority_sha256", None)
        changed = True
        print(
            "CHAMPION_PROMOTE_RECERT_REQUEUE "
            f"entry_id={row.get('entry_id')} was={before_status}",
            flush=True,
        )
        _append_historical_reclassification(
            root,
            loop_id,
            {
                "schema": "autotrain_historical_reclassification/v1",
                "kind": "champion_queue",
                "repair": "promote_authority_recert",
                "loop_id": loop_id,
                "entry_id": row.get("entry_id"),
                "campaign_id": row.get("promotion_campaign_id")
                or row.get("confirm_campaign_id"),
                "before_status": before_status,
                "after_status": "confirmed",
                "action": "requeue",
                "authority_sha256": auth_sha,
                "reclassified_at": now,
                "reasons": base_reasons[:12],
            },
        )
    return changed










def _arm_slug_from_knobs(
    knobs: dict[str, Any], *, candidate_id: str = ""
) -> str | None:
    """Map knobs / candidate id to thrash arm slug."""
    if knobs.get("_thrash_slug"):
        return str(knobs["_thrash_slug"])
    if knobs.get("constraint_graph_mode") == "grammar":
        return "constraint-graph"
    if knobs.get("slot_contract_in_context"):
        return "slot-contract-context"
    if (
        knobs.get("semantic_contrast_loss_weight")
        and knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
    ):
        return "semantic-contrast-compiler-margin"
    if (
        knobs.get("compiler_alignment_loss_weight")
        and knobs.get("compiler_alignment_kind_filter") == "all"
        and knobs.get("mixture_sampling_policy") == "exposure_targeted"
        and knobs.get("compiler_alignment_semantic_exhaustive")
    ):
        return "exposure-targeted-semantic-exhaustive-compiler-decision-margin"
    if (
        knobs.get("slot_component_loss_weight")
        and knobs.get("mixture_sampling_policy") == "exposure_targeted"
        and knobs.get("mixture_per_template_cap")
    ):
        return "slot-component-exposure-cap"
    if knobs.get("slot_component_loss_weight") and knobs.get(
        "component_inventory_loss_weight"
    ):
        return "slot-component-inventory-coupling"
    if (
        knobs.get("slot_component_loss_weight")
        and float(knobs.get("fidelity_loss_weight") or 0.5) != 0.5
    ):
        return "slot-component-fidelity-coupling"
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
    if knobs.get("ltr_prefix_loss_weight") and knobs.get("ltr_tail_loss_weight"):
        return "scaffold-prefix-tail"
    if knobs.get("ltr_prefix_loss_weight") and knobs.get("structure_token_loss_weight"):
        return "scaffold-prefix-structure"
    if knobs.get("component_token_loss_weight") and knobs.get("ltr_prefix_loss_weight"):
        return "component-token-prefix"
    if knobs.get("ltr_tail_loss_weight") and knobs.get("structure_token_loss_weight"):
        return "literal-close-structure"
    if knobs.get("ltr_tail_loss_weight") and knobs.get("component_token_loss_weight"):
        return "literal-close-component-token"
    if knobs.get("ltr_tail_loss_weight") and knobs.get(
        "typed_family_balance_loss_weight"
    ):
        return "literal-close-typed-balance"
    if knobs.get("symbol_boundary_loss_weight") and knobs.get(
        "structure_token_loss_weight"
    ):
        return "symbol-boundary-structure"
    if knobs.get("semantic_contrast_loss_weight") and knobs.get(
        "structure_token_loss_weight"
    ):
        return "semantic-contrast-structure"
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
    if knobs.get("solver_energy_loss_weight") or knobs.get(
        "solver_energy_decode_weight"
    ):
        return "solver-energy-rerank"
    if knobs.get("legal_edit_hazard_loss_weight") or knobs.get(
        "legal_edit_hazard_decode_weight"
    ):
        return "legal-edit-hazard"
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
    if knobs.get("batch_size") == _CONTROL_RECIPE_BATCH_SIZE * 2:
        return "batch-x2"
    lr_raw = knobs.get("lr")
    if lr_raw is not None:
        try:
            lr_value = float(lr_raw)
        except (TypeError, ValueError):
            lr_value = None
        if lr_value is not None:
            if math.isclose(lr_value, _CONTROL_RECIPE_LR * 2, rel_tol=1e-6):
                return "lr-x2"
            if math.isclose(lr_value, _CONTROL_RECIPE_LR / 2, rel_tol=1e-6):
                return "lr-x0.5"
    # Data-volume arms: a corpus swap with no other lever. The control corpus
    # itself never names an arm (``steps-fill`` and controls carry it too).
    train_version = str(knobs.get("train_version") or "")
    if (
        train_version
        and not _is_process_arm(knobs)
        and train_version != _default_screening_train_version()
    ):
        if train_version == _EX_DATA_ARM_LEAKED_TRAIN_VERSION:
            # Historical ledger rows still name the withdrawn arm's corpus;
            # classify them so the ledger stays readable, never so the arm
            # becomes selectable. ``wf_smoke_v2`` is deliberately absent: it
            # was a legacy *control* corpus, never a data arm, and naming it
            # here would relabel every legacy control row as an arm.
            return f"data-leaked:{train_version}"
        if train_version == _DATA_ARM_CERTIFIED_TRAIN_VERSION:
            return "data-certified"
    cid = candidate_id or ""
    recovered = _slug_from_candidate_id(cid)
    if recovered:
        return recovered
    if cid.endswith("-steps"):
        return "steps"
    return None


def _slug_from_candidate_id(candidate_id: str) -> str | None:
    """Recover a bank slug after hypothesis knobs drop ``_thrash_slug``.

    Train-version-only snapshot arms have no lever match. The old
    ``"-steps" in candidate_id`` fallback also collapsed
    ``simplified-nl-c52-steps40`` onto ``steps``, so those identities
    never closed.
    """
    cid = str(candidate_id or "")
    if not cid:
        return None
    matches = [
        slug
        for slug, _, _ in _all_screening_arm_bank()
        if cid == slug or cid.endswith(f"-{slug}")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _default_screening_train_version() -> str:
    try:
        from slm_training.autoresearch.climb_policy import load_climb_policy

        return str(load_climb_policy().defaults.get("train_version") or "wf_smoke_v2")
    except Exception:  # noqa: BLE001 — identity close must stay computable
        return "wf_smoke_v2"


def _slug_is_snapshot_arm(slug: str, extras: Mapping[str, Any] | None = None) -> bool:
    """True when the arm is a data-snapshot leftover, not an isolate OFAT lever."""
    if extras is None:
        extras = next(
            (row[2] for row in _all_screening_arm_bank() if row[0] == slug),
            {},
        )
    extras = extras or {}
    if _is_process_arm(extras):
        return False
    if slug in _STATIC_DATA_ARM_SLUGS:
        # Preregistered data-volume arms are isolate OFAT levers, not I10
        # snapshot leftovers.
        return False
    train_version = str(extras.get("train_version") or "")
    if not train_version:
        return False
    if train_version in {"wf_smoke_v2", "hillclimb_strict_v1"}:
        return False
    return train_version != _default_screening_train_version()


def _open_slugs_are_snapshot_leftovers(open_slugs: set[str]) -> bool:
    """Park signal: isolate OFAT bank is closed; only snapshot slugs remain."""
    if not open_slugs:
        return True
    extras_by_slug = {slug: extras for slug, _, extras in _all_screening_arm_bank()}
    return all(
        _slug_is_snapshot_arm(slug, extras_by_slug.get(slug)) for slug in open_slugs
    )




def _skip_arm_slugs(
    entries: list[dict[str, Any]],
    *,
    integration_commit: str | None = None,
    include_causal_cap: bool = True,
) -> set[str]:
    """Deprioritize arms only while a champion is still open (not forever).

    Permanent skip of rejected/promotion_failed starved bounds/canvas and left
    only steps/batch1 thrash — which could not re-enter the champion path.

    ``include_causal_cap=False`` drops same-integration family saturation so a
    multi-seed-open thrash bank cannot hard-die solely from confirm CAP.
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
            and _is_decisive_causal_terminal(row)
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
    if include_causal_cap:
        skip.update(
            slug
            for slug, count in terminal_counts.items()
            if count >= _CAUSAL_FAMILY_ATTEMPT_CAP
        )
    return skip


def _thrash_bank_open_slugs(closed: set[str]) -> set[str]:
    """Open isolate slugs. Snapshot train_version arms are I10 leftovers.

    When ``park_on_exhaust`` is on they are not isolate screening candidates —
    otherwise UCB rematches c96/c120 after c48/c52 while any OFAT slug is still
    technically open. A process/heal arm from a local rebuild_data heal is
    the I10 successor and stays selectable.
    """
    open_slugs = {slug for slug, _, _ in _all_screening_arm_bank() if slug not in closed}
    if not _terminal_park_on_exhaust():
        return open_slugs
    extras_by_slug = {slug: extras for slug, _, extras in _all_screening_arm_bank()}
    return {
        slug
        for slug in open_slugs
        if _is_process_arm(extras_by_slug.get(slug))
        or not _slug_is_snapshot_arm(slug, extras_by_slug.get(slug))
    }


def _arm_close_min_null_seeds(policy: Any | None = None) -> int:
    """Distinct complete-null seeds required before a thrash arm is closed."""
    base = _DEFAULT_ARM_CLOSE_MIN_NULL_SEEDS
    payload: dict[str, Any] = {}
    try:
        from slm_training.autoresearch.climb_policy import load_climb_policy

        pol = policy if policy is not None else load_climb_policy()
        payload = getattr(pol, "payload", None) or {}
        block = (
            payload.get("screening_arm_closure") if isinstance(payload, dict) else None
        )
        if isinstance(block, dict) and block.get("min_complete_null_seeds") is not None:
            base = max(1, int(block["min_complete_null_seeds"]))
        else:
            # Align with recipe_null_cap when screening_arm_closure is absent.
            cap = payload.get("recipe_null_cap") if isinstance(payload, dict) else None
            if isinstance(cap, dict) and cap.get("max_nulls_per_family") is not None:
                base = max(1, int(cap["max_nulls_per_family"]))
    except Exception:  # noqa: BLE001 — fail closed to default
        return base
    try:
        gate = payload.get("power_gate") if isinstance(payload, dict) else None
        if isinstance(gate, dict) and gate.get("enabled"):
            from slm_training.autoresearch import evidence_ledger as _ev

            measurement = (
                payload.get("measurement") if isinstance(payload, dict) else None
            )
            # Auto mode resolves the certified screening-n range; unset stays 0
            # so power_scaled_null_seeds keeps its no-op fallback.
            if (measurement or {}).get("screening_smoke_n"):
                from slm_training.autoresearch.climb_policy import (
                    screening_smoke_n_for_policy,
                )

                per_cycle_n = screening_smoke_n_for_policy(pol)[0]
            else:
                per_cycle_n = 0
            base = _ev.power_scaled_null_seeds(
                base, per_cycle_n, _ev.parse_alpha(gate.get("alpha"))
            )
    except Exception:  # noqa: BLE001 — power floor never blocks closure math
        pass
    return base


def _evidence_ranked_slug(
    candidates: list[str],
    *,
    stats: dict[str, Any],
    boosts: dict[str, float],
) -> str | None:
    """Posterior-UCB pick over the committed evidence ledger (policy-gated).

    Returns ``None`` when the policy does not enable posterior selection or on
    any failure, so the caller falls through to the legacy soft rank.
    """
    try:
        from slm_training.autoresearch import evidence_ledger as _ev
        from slm_training.autoresearch.climb_policy import load_climb_policy

        payload = getattr(load_climb_policy(), "payload", None) or {}
        selection = payload.get("selection")
        if not isinstance(selection, dict) or selection.get("mode") != "posterior_ucb":
            return None
        live_stats = {
            slug: (float(entry.n_complete), float(entry.mean_delta_ss))
            for slug, entry in stats.items()
            if getattr(entry, "n_complete", 0) > 0
        }
        return _ev.pick_evidence_ranked_slug(
            candidates,
            _ev.load_ledger(),
            exploration_c=float(selection.get("exploration_c", 1.0)),
            prior_scale=float(selection.get("prior_scale", 0.05)),
            staleness_decay=float(
                selection.get("staleness_decay", _ev.STALENESS_DECAY)
            ),
            staleness_floor=float(
                selection.get("staleness_floor", _ev.STALENESS_FLOOR)
            ),
            residual_boosts=boosts,
            live_stats=live_stats,
            rotation_order=candidates,
            eval_key=_ev.current_eval_key(),
        )
    except Exception as exc:  # noqa: BLE001 — selection upgrade must fail open
        print(f"EVIDENCE_RANK_WARN {exc}", flush=True)
        return None




def _recent_completed_nonpositive_slugs(
    root: Path,
    predecessor_campaign_id: str | None,
    *,
    max_cycles: int | None = None,
    min_null_seeds: int | None = None,
) -> set[str]:
    """Close thrash approaches only after multi-seed complete non-positives.

    ``max_cycles`` remains an explicit diagnostic/test bound. Production walks
    the content-linked lineage to its root so aging alone never reopens an
    approach. A **single** complete non-positive is not enough to close the
    arm: fixture n is noisy, and climb policy already requires multi-seed
    evidence before treating a family as recipe-null exhausted
    (``recipe_null_cap.max_nulls_per_family``, default 2).

    A complete **positive** after nulls clears the null-seed tally for that
    slug (the approach is not permanently dead if it later wins). Snapshot
    clones that share ``train_version`` close together once that identity
    has enough distinct-seed nulls.
    """

    required = (
        max(1, int(min_null_seeds))
        if min_null_seeds is not None
        else _arm_close_min_null_seeds()
    )
    # slug -> set of seeds with complete non-positive (since last positive)
    null_seeds: dict[str, set[int | None]] = {}
    # train_version -> seeds; snapshot clones share identity, not slug spelling
    tv_null_seeds: dict[str, set[int | None]] = {}
    for camp_id in _lineage_campaign_ids(
        root, predecessor_campaign_id, max_cycles=max_cycles
    ):
        camp_dir = root / camp_id
        handoff = _read_json(camp_dir / "cycle_handoff.json")
        delivery = _read_json(camp_dir / "sdlc_delivery.json")
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
        complete_enough = (
            delivery.get("measurement_complete") is True or runtime_terminal
        )
        intent_ok = (
            intent in {"screening", "promotion", "confirm", "retry_measurement"}
            or runtime_terminal
        )
        if not (candidate_id and complete_enough and intent_ok):
            continue
        if stored_positive is not True and stored_positive is not False:
            continue
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
        if not slug:
            continue
        seed_raw = knobs.get("seed")
        try:
            seed: int | None = int(seed_raw) if seed_raw is not None else None
        except (TypeError, ValueError):
            seed = None
        train_version = str(knobs.get("train_version") or "")
        snapshot_tv = _slug_is_snapshot_arm(
            slug, {"train_version": train_version}
        )
        # Incomplete / harness outcomes never close a thrash approach — even if
        # a buggy delivery marked measurement_complete or positive=False.
        if not runtime_terminal and (
            any(_reason_is_harness_incomplete(item) for item in handoff_reasons)
            or any(
                _reason_is_harness_incomplete(item)
                for item in (delivery.get("reasons") or [])
            )
        ):
            continue
        if delivery.get("harness_failure") is True and not runtime_terminal:
            continue
        # Fixture-n screening wins stay positive=False (no stack) but are still
        # confirm candidates — do not burn the thrash approach as a null seed.
        confirm_win = False
        if (
            not runtime_terminal
            and delivery.get("measurement_complete") is True
            and intent in {"screening", "retry_measurement"}
        ):
            confirm_win = _is_confirm_candidate_win(
                {
                    **dict(delivery),
                    "positive": stored_positive,
                    "reasons": list(delivery.get("reasons") or []),
                    "measurement_complete": True,
                }
            )
        if stored_positive is True or confirm_win:
            # Win re-opens the approach; clear prior null tally for this slug.
            null_seeds[slug] = set()
            if snapshot_tv:
                tv_null_seeds[train_version] = set()
            continue
        if snapshot_tv:
            # Snapshot close is by train_version identity, not slug spelling:
            # every heal snapshot shares the resume slug, so slug-level nulls
            # would permanently close all future (distinct) snapshots.
            tv_null_seeds.setdefault(train_version, set()).add(seed)
        else:
            null_seeds.setdefault(slug, set()).add(seed)

    closed = {slug for slug, seeds in null_seeds.items() if len(seeds) >= required}
    # Snapshot clones share a train_version but differ by slug spelling. Close
    # every bank slug on an identity that already has enough complete nulls.
    exhausted_versions = {
        train_version
        for train_version, seeds in tv_null_seeds.items()
        if len(seeds) >= required
    }
    if exhausted_versions:
        for slug, _, extras in _all_screening_arm_bank():
            if extras.get("train_version") in exhausted_versions:
                closed.add(slug)
    return closed








def _load_slug_stats(root: Path, loop_id: str) -> dict[str, SlugStats]:
    path = _slug_stats_path(root, loop_id)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("slugs") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, SlugStats] = {}
    for slug, row in raw.items():
        if not isinstance(row, dict):
            continue
        try:
            out[str(slug)] = SlugStats(
                slug=str(slug),
                n_complete=int(row.get("n_complete") or 0),
                n_positive=int(row.get("n_positive") or 0),
                mean_delta_ss=float(row.get("mean_delta_ss") or 0.0),
                binder_fail_rate=float(row.get("binder_fail_rate") or 0.0),
                incomplete_n=int(row.get("incomplete_n") or 0),
                residual_hits=int(row.get("residual_hits") or 0),
            )
        except (TypeError, ValueError):
            continue
    return out










def _sync_reproduced_timeout_retirements(
    root: Path,
    loop_id: str,
    predecessor_campaign_id: str | None,
    *,
    policy: Any,
    train_version: str,
    eval_version: str,
    primary_metric: str,
    direction: str,
    claim_class: str,
    data_generation: Mapping[str, Any] | None = None,
) -> tuple[set[str], tuple[str, ...]]:
    """Backfill and enforce exact reproduced-timeout retirements."""

    from slm_training.autoresearch.climb_policy import (
        load_loop_exhausted_ledger,
        loop_data_eval_identity,
        save_loop_exhausted_ledger,
    )

    def _identity_extra(raw: object) -> dict[str, Any]:
        return {"data_generation": raw or None}

    ledger = load_loop_exhausted_ledger(root, loop_id, policy)
    before = len(ledger.entries)
    retired: dict[tuple[str, str], tuple[str, int, str]] = {}
    reintroduced: set[str] = set()
    chain = _lineage_campaign_ids(root, predecessor_campaign_id)
    for campaign_id in chain:
        camp_dir = root / campaign_id
        delivery = _read_json(camp_dir / "sdlc_delivery.json")
        handoff = _read_json(camp_dir / "cycle_handoff.json")
        matrix = _read_json(camp_dir / "matrix-proposal.json")
        candidate_id = str(delivery.get("candidate_id") or "")
        control_id = str(delivery.get("control_id") or "")
        knobs = _load_experiment_knobs(
            camp_dir, candidate_id
        ) or _matrix_experiment_knobs(matrix, candidate_id)
        slug = _arm_slug_from_knobs(knobs, candidate_id=candidate_id)
        signature = _matrix_treatment_signature(matrix, candidate_id, control_id)
        if not slug or not signature:
            continue
        identity = loop_data_eval_identity(
            policy,
            claim_class=claim_class,
            train_version=str(knobs.get("train_version") or train_version),
            eval_version=str(knobs.get("eval_version") or eval_version),
            primary_metric=str(handoff.get("primary_metric") or primary_metric),
            direction=direction,
            extra=_identity_extra(knobs.get("data_generation")),
        )
        try:
            cycle_index = int(
                delivery.get("cycle_index") or handoff.get("cycle_index") or 0
            )
        except (TypeError, ValueError):
            cycle_index = 0
        key = (signature, identity)
        prior = retired.get(key)
        if prior is not None and cycle_index > prior[1]:
            reintroduced.add(campaign_id)
        if not _is_reproduced_timeout_retirement(handoff, delivery):
            continue
        manifest = camp_dir / "manifests" / f"{candidate_id}.json"
        manifest_sha = (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.is_file()
            else "missing"
        )
        ledger.record_null(
            knob_signature_sha256=signature,
            data_eval_identity=identity,
            claim_class=claim_class,
            reason="reproduced_decode_timeout_retirement",
            note=(f"slug={slug};campaign={campaign_id};manifest_sha256={manifest_sha}"),
        )
        retired[key] = (slug, cycle_index, campaign_id)
    if len(ledger.entries) != before:
        save_loop_exhausted_ledger(ledger, root, loop_id, policy)
    current_identity = loop_data_eval_identity(
        policy,
        claim_class=claim_class,
        train_version=train_version,
        eval_version=eval_version,
        primary_metric=primary_metric,
        direction=direction,
        extra=_identity_extra(data_generation),
    )
    retired_slugs = {
        slug
        for (signature, identity), (slug, _cycle, _campaign) in retired.items()
        if identity == current_identity
        and ledger.is_exhausted(
            knob_signature_sha256=signature,
            data_eval_identity=identity,
            claim_class=claim_class,
        )
    }
    if reintroduced:
        for campaign_id in chain:
            for path in (root / campaign_id / "artifacts" / "harness_signals").glob(
                "*.json"
            ):
                if (
                    _read_json(path).get("code")
                    == "screening_selector_reintroduced_retired_arm"
                ):
                    reintroduced.clear()
                    break
            if not reintroduced:
                break
    return retired_slugs, tuple(sorted(reintroduced))


def _persist_selector_harness_signal(
    root: Path,
    campaign_id: str,
    loop_id: str,
    source_campaigns: Sequence[str],
) -> None:
    """Persist one content-addressed signal for a proven selector regression."""

    if not source_campaigns:
        return
    signal = HarnessSignalV1(
        family="autoresearch",
        code="screening_selector_reintroduced_retired_arm",
        evidence_uri=f"loops/{loop_id}/exhausted_knob_ledger.json",
        reproduced_on_frozen_input=True,
        primary=True,
    )
    store = CampaignStore(campaign_id, root=root)
    path = store.write_artifact("harness_signals", signal)
    store.append_event(
        "harness_signal_recorded",
        status="reproduced",
        artifact_sha256=path.stem,
        detail={"source_campaigns": list(source_campaigns)},
    )


def _screening_saturation_state(
    root: Path,
    loop_id: str,
    *,
    policy: Any,
    excluded_slugs: set[str],
) -> dict[str, Any] | None:
    block = (getattr(policy, "payload", None) or {}).get("screening_saturation")
    if not isinstance(block, dict) or block.get("mode") != "residual_multiseed":
        return None
    threshold = max(1, int(block.get("complete_tie_streak") or 15))
    deliveries = _iter_loop_deliveries(root, loop_id, limit=240)
    streak, trigger_cycle = screening_tie_saturation(deliveries, threshold=threshold)
    if trigger_cycle is None:
        return None
    positive_policy = (getattr(policy, "payload", None) or {}).get(
        "positive_classification"
    ) or {}
    ranked = rank_absolute_regimes(
        deliveries,
        through_cycle=trigger_cycle,
        max_regimes=max(1, int(block.get("max_regimes") or 2)),
        decode_cost_slugs=DECODE_RESIDUAL_SLUGS,
        excluded_slugs=excluded_slugs,
        minimum_latency_gain_fraction=float(
            positive_policy.get("minimum_efficiency_gain_fraction") or 0.05
        ),
    )
    completed_after = {
        slug
        for row in deliveries
        if row.get("measurement_complete") is True
        and int(row.get("cycle_index") or 0) > trigger_cycle
        and (
            slug := _arm_slug_from_knobs(
                _load_experiment_knobs(
                    root / str(row.get("campaign_id") or ""),
                    str(row.get("candidate_id") or ""),
                ),
                candidate_id=str(row.get("candidate_id") or ""),
            )
        )
    }
    pending = [slug for slug in ranked if slug not in completed_after]
    return {
        "schema": "screening_saturation_recovery/v1",
        "mode": "residual_multiseed",
        "tie_streak": streak,
        "threshold": threshold,
        "trigger_cycle": trigger_cycle,
        "ranked_regimes": ranked,
        "completed_after_trigger": sorted(completed_after & set(ranked)),
        "pending_regimes": pending,
    }










def _select_recommended_slug(
    cycle: int,
    skip: set[str] | None = None,
    *,
    root: Path | None = None,
    loop_id: str | None = None,
) -> str:
    """Rotate screening arms without reopening a multi-seed-closed approach.

    When residual / slug-stat ledgers exist under the loop root, soft-rank open
    candidates (boost interesting residuals; still never reopen skipped arms).
    """
    skip = skip or set()
    pred = (
        _latest_cycle(root, loop_id)[1] if root is not None and loop_id else None
    )
    for slug, _, extras in _all_screening_arm_bank():
        if not _is_process_arm(extras):
            continue
        if slug not in skip:
            print(f"HEAL_RESUME_SELECT cycle={cycle} slug={slug}", flush=True)
            return slug
        if (
            root is not None
            and loop_id
            and _selectable_process_arm(
                root, loop_id, predecessor_campaign_id=pred
            )
        ):
            print(
                f"HEAL_RESUME_SELECT cycle={cycle} slug={slug} "
                "reason=unused_train_version",
                flush=True,
            )
            return slug
    # Evidence-triggered and successor quality arms do not perturb the stable
    # cycle-number rotation of the original screening bank. They become the
    # fail-forward path only after older approaches are closed.
    successor_slugs = (
        "binder-arity",
        "binder-component-plan",
        "slot-component-coverage",
        "slot-component-fidelity-coupling",
        "slot-component-inventory-coupling",
        "slot-component-exposure-cap",
        "slot-contract-context",
        "constraint-graph",
        "literal-margin",
        "literal-close",
        "fidelity",
        "edge-alignment",
        "semantic-contrast",
        "semantic-contrast-compiler-margin",
        "slot-augmentation",
        "mixed-mask",
        "symbol-boundary",
        "design-dropout",
        "scaffold-prefix",
        "scaffold-prefix-structure",
        "scaffold-prefix-tail",
        "component-token",
        "component-token-prefix",
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
        "literal-close-structure",
        "literal-close-component-token",
        "literal-close-typed-balance",
        "symbol-boundary-structure",
        "semantic-contrast-structure",
        "solver-energy-rerank",
        "legal-edit-hazard",
    )
    legacy_quality_slugs = {
        "component-plan",
        "component-edge",
        "component-inventory",
        "binder-topology",
        "component-structure",
    }
    full_bank = _all_screening_arm_bank()
    bank = tuple(row for row in full_bank if row[0] not in successor_slugs)
    n = len(bank)
    if n == 0:
        # Dynamic-only bank: rotate full effective bank.
        bank = full_bank
        n = len(bank)
    start = (max(1, int(cycle)) - 1) % max(n, 1)
    ordered = [bank[(start + i) % n][0] for i in range(n)] if n else []
    skip = skip or set()
    open_candidates: list[str] = []
    if legacy_quality_slugs.issubset(skip):
        for slug in successor_slugs:
            if slug not in skip:
                open_candidates.append(slug)
    for slug in ordered:
        if slug not in skip:
            open_candidates.append(slug)
    for slug in successor_slugs:
        if slug not in skip and slug not in open_candidates:
            open_candidates.append(slug)
    # Self-heal thrash successors (compose-*) live in the full bank after static.
    for slug, _, _ in full_bank:
        if (
            slug not in skip
            and slug.startswith("compose-")
            and slug not in open_candidates
        ):
            open_candidates.append(slug)
    if not open_candidates:
        raise RuntimeError(_BANK_EXHAUST_MSG)

    # Soft residual re-rank when loop ledgers are available (Layer 1).
    if root is not None and loop_id:
        try:
            stats = _load_slug_stats(root, loop_id)
            observations = _load_residual_observations(root, loop_id)
            boosts = residual_boosts_from_observations(
                observations,
                max_age=40,
                latest_cycle=int(cycle),
            )
            evidence_pick = _evidence_ranked_slug(
                open_candidates, stats=stats, boosts=boosts
            )
            if evidence_pick:
                print(
                    f"EVIDENCE_RANK cycle={cycle} slug={evidence_pick} "
                    f"mode=posterior_ucb",
                    flush=True,
                )
                return evidence_pick
            if stats or boosts:
                picked = pick_soft_ranked_slug(
                    open_candidates,
                    stats=stats,
                    residual_boosts=boosts,
                    rotation_order=open_candidates,
                )
                if picked:
                    if boosts.get(picked) or (
                        stats.get(picked) and stats[picked].prior_score() != 0.0
                    ):
                        print(
                            f"THRASH_SOFT_RANK cycle={cycle} slug={picked} "
                            f"boost={boosts.get(picked, 0.0):.3f}",
                            flush=True,
                        )
                    return picked
        except Exception as exc:  # noqa: BLE001 — never block thrash on soft rank
            print(f"THRASH_SOFT_RANK_WARN err={exc!r}", flush=True)

    return open_candidates[0]


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
    if _terminal_park_on_exhaust():
        # Terminal policy retires the confirm filler: an exhausted bank parks
        # under the typed verdict instead of burning confirmation seeds.
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
    thrash_regime: ThrashRegimeDecision | None = None,
    root: Path | None = None,
    loop_id: str | None = None,
) -> str | None:
    """Select only for screening; confirm/promote carry frozen recipes."""

    if has_confirm_levers or has_promote_levers:
        return None
    bank_slugs = [slug for slug, _, _ in _all_screening_arm_bank()]
    decision = thrash_regime or decide_screening_regime(
        climb_baseline_knobs=None,
        compiler_ms_timeout=False,
    )

    def _isolate(c: int, s: set[str]) -> str:
        return _select_recommended_slug(c, skip=s, root=root, loop_id=loop_id)

    if decision.timeout_residual:
        # Timeout residual outranks generic predecessor quality priorities.
        return select_recommended_slug_for_regime(
            decision=decision,
            cycle=cycle,
            skip=skip,
            bank_slugs=bank_slugs,
            isolate_selector=_isolate,
        )
    extras_by_slug = {slug: extras for slug, _, extras in _all_screening_arm_bank()}
    process_open = [
        slug
        for slug, extras in extras_by_slug.items()
        if slug not in skip and _is_process_arm(extras)
    ]
    if predecessor_priority and predecessor_priority not in skip:
        # A leftover rematch (c78/c96 after a derailed heal) must not outrank
        # an open process/heal first-train.
        if not process_open or _is_process_arm(extras_by_slug.get(predecessor_priority)):
            return predecessor_priority
        print(
            f"PROCESS_ARM_OUTRANKS_PREDECESSOR process={process_open[0]} "
            f"predecessor={predecessor_priority}",
            flush=True,
        )
    return select_recommended_slug_for_regime(
        decision=decision,
        cycle=cycle,
        skip=skip,
        bank_slugs=bank_slugs,
        isolate_selector=_isolate,
    )




def _preflight_screening_slug(
    rec_slug: str,
    *,
    steps: int,
    endpoint_metric: str,
    minimum_effect: float | None,
    skip: set[str],
    reselect: Any,
) -> tuple[str, dict[str, Any] | None]:
    """Preflight-gate the selected screening arm (fail-soft, WP-3 seam).

    Runs ``slm_training.autoresearch.preflight.run_preflight`` on the chosen
    arm's candidate dict. A ``"block"`` verdict skips the arm and reselects; if
    every open arm blocks, the original pick runs anyway with the override
    recorded (the loop must never die on a preflight). Returns the slug to run
    and a ``preflight`` payload (or ``None`` when the seam is unavailable) that
    :func:`_phase_a_delivery` persists into the delivery record.
    """
    try:
        from slm_training.autoresearch.preflight import has_block, run_preflight
    except Exception as exc:  # noqa: BLE001 — seam optional, never fatal
        print(f"PREFLIGHT_WARN import err={exc!r}", flush=True)
        return rec_slug, None
    bank = {slug: (hyp, extras) for slug, hyp, extras in _all_screening_arm_bank()}
    # Cumulative per-arm seed count (ledger n_complete), so the power gate
    # evaluates decidability against the arm's accumulated evidence across
    # cycles rather than this cycle's single-seed marginal contribution — a
    # literal n_seeds=1 is undecidable by construction and would block every
    # screening cycle forever (see power_check.py's seeds-policy note).
    try:
        from slm_training.autoresearch import evidence_ledger as _ev

        _ledger_arms = _ev.load_ledger().get("arms", {})
        if not isinstance(_ledger_arms, dict):
            _ledger_arms = {}
    except Exception:  # noqa: BLE001 — cumulative-n lookup is best-effort
        _ledger_arms = {}
    verdicts_by_slug: dict[str, list[dict[str, Any]]] = {}
    blocked: list[str] = []
    current = rec_slug
    try:
        for _ in range(len(bank) + 1):
            row = bank.get(current)
            if row is None:
                break
            hypothesis, extras = row
            extras = dict(extras)
            process_arm = _is_process_arm(extras) or current == _HEAL_RESUME_SLUG
            levers = _apply_arm_extras(steps, extras)
            arm_stats = _ledger_arms.get(current)
            cumulative_n = int((arm_stats or {}).get("n_complete") or 0)
            candidate = {
                "hypothesis_text": str(hypothesis),
                "lever_keys": sorted(levers),
                "config_fingerprint": _knobs_fingerprint(levers),
                "n_seeds": cumulative_n + 1,
                "steps": steps,
                "minimum_effect": minimum_effect,
                "endpoint_metric": endpoint_metric,
                "slug": current,
                "levers": levers,
            }
            if process_arm:
                candidate["process_arm"] = True
                candidate["heal_resume"] = True
                candidate["claim_class"] = str(
                    extras.get("process_role") or extras.get("claim_class") or "process"
                )
            verdicts = run_preflight(candidate)
            verdicts_by_slug[current] = [v.model_dump() for v in verdicts]
            if not has_block(verdicts):
                return current, {
                    "schema": "autotrain_preflight/v1",
                    "selected_slug": current,
                    "blocked_slugs": blocked,
                    "verdicts": verdicts_by_slug,
                }
            reasons = [r for v in verdicts if v.verdict == "block" for r in v.reasons]
            print(
                f"PREFLIGHT_BLOCK slug={current} reasons={reasons[:2]}",
                flush=True,
            )
            if process_arm:
                print(
                    f"PROCESS_ARM_PREFLIGHT_CONTINUE slug={current} "
                    "reason=not_confirmatory_design",
                    flush=True,
                )
                return current, {
                    "schema": "autotrain_preflight/v1",
                    "selected_slug": current,
                    "blocked_slugs": blocked,
                    "override": "process_arm_not_confirmatory",
                    "verdicts": verdicts_by_slug,
                }
            blocked.append(current)
            skip = skip | {current}
            current = reselect(skip)
            if current is None or current in blocked:
                break
    except Exception as exc:  # noqa: BLE001 — fail-soft: run original pick
        print(f"PREFLIGHT_WARN err={exc!r}", flush=True)
        return rec_slug, None
    if not verdicts_by_slug:
        # rec_slug is not a known bank arm (no candidate to build): un-gated.
        return rec_slug, None
    # Fail-soft floor: a preflight can never exhaust the loop on its own
    # authority. Prefer an un-gated reselection (slug outside the bank map)
    # over re-running a blocked pick; otherwise run the original pick with the
    # override recorded (arm closure stays with multi-seed null evidence).
    if current and current not in blocked and current != rec_slug:
        fallback, override = current, "reselected_slug_not_in_bank_ran_ungated"
    else:
        fallback, override = rec_slug, "all_open_arms_blocked_ran_original_pick"
    print(f"PREFLIGHT_OVERRIDE slug={fallback} reason={override}", flush=True)
    return fallback, {
        "schema": "autotrain_preflight/v1",
        "selected_slug": fallback,
        "blocked_slugs": blocked,
        "override": override,
        "verdicts": verdicts_by_slug,
    }








def _lean_floor_n() -> int | None:
    """Certified screening n when auto mode is feasible; else None."""
    try:
        n, report = _screening_n_report()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(report, dict) and report.get("verdict") == "feasible":
        chosen = int(report.get("chosen_n") or n or 0)
        return chosen if chosen > 0 else None
    return None


def _lean_floor_measurement(delivery: Mapping[str, Any]) -> bool:
    """True when both arms completed at the certified screening n."""
    floor = _lean_floor_n()
    if not floor:
        return False
    control = delivery.get("control_metrics") or {}
    candidate = delivery.get("candidate_metrics") or {}
    cn, tn = _arm_completed_n(control), _arm_completed_n(candidate)
    if cn is not None and tn is not None:
        return cn >= floor and tn >= floor
    suite = _screening_suite_records()
    return bool(
        delivery.get("measurement_complete") is True
        and suite is not None
        and int(suite) >= floor
    )
























def _is_champion_lever(knobs: dict[str, Any], *, candidate_id: str = "") -> bool:
    """True when knobs encode a thrash arm (not pure matched control)."""
    if _arm_slug_from_knobs(knobs, candidate_id=candidate_id) is not None:
        return True
    # Heal / snapshot first-trains may leave the thrash bank after retirement
    # but still carry a distinct train_version worth confirming.
    train_version = str(knobs.get("train_version") or "")
    if train_version and train_version != _default_screening_train_version():
        return True
    return False




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
            "rejected",
        }:
            # Already open / confirmed / pending promote / rejected — do not
            # re-queue thrash. ponytail: rejected stays exhausted until knobs
            # change (rebuild / train_version changes the fingerprint).
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
        "fixture_volume_confirm_candidate": any(
            str(r).startswith("fixture_insufficient_n")
            for r in (delivery.get("reasons") or [])
        ),
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
    updated = _update_champion_status(
        root=root,
        loop_id=loop_id,
        entry_id=str(entry["entry_id"]),
        status=status,
        confirm_campaign_id=campaign_id,
        confirm_cycle_index=cycle_index,
        resolve_reasons=reasons,
    )
    if ok:
        cand_id = str(delivery.get("candidate_id") or "")
        ckpt = _checkpoint_path_for_candidate(root, campaign_id, cand_id)
        summary: dict[str, Any] = {}
        if cand_id:
            summary_path = (
                root / campaign_id / "runs" / cand_id / "train_summary.json"
            )
            if summary_path.is_file():
                try:
                    loaded = _read_json(summary_path)
                    if isinstance(loaded, dict):
                        summary = loaded
                except Exception:  # noqa: BLE001
                    summary = {}
        knobs = dict(entry.get("knobs") or {})
        try:
            extra_steps = int(knobs.get("steps") or summary.get("steps") or 0)
        except (TypeError, ValueError):
            extra_steps = 0
        try:
            record_count = int(summary.get("record_count") or 1)
        except (TypeError, ValueError):
            record_count = 1
        train_sha = str(
            summary.get("train_data_manifest_sha")
            or summary.get("data_manifest_sha")
            or ""
        )
        params = summary.get("trainable_params")
        if params is None:
            params = (summary.get("track") or {}).get("trainable_params")
        try:
            trainable = int(params) if params is not None else None
        except (TypeError, ValueError):
            trainable = None
        maybe_advance_climb_champion(
            _loop_champion_dir(root, loop_id),
            confirmed=True,
            checkpoint=ckpt,
            source_campaign=campaign_id,
            extra_steps=extra_steps,
            train_data_manifest_sha=train_sha,
            record_count=record_count,
            knobs=knobs,
            trainable_params=trainable,
        )
    return updated




def locked_promote_expectations_sha256() -> str:
    """SHA-256 of the locked continuous promote expectation manifest."""
    path = promote_expectations_path()
    return hashlib.sha256(path.read_bytes()).hexdigest()




def locked_screening_expectations_sha256() -> str:
    """SHA-256 of the locked continuous screening expectation manifest."""
    return hashlib.sha256(screening_expectations_path().read_bytes()).hexdigest()




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
    power_feasibility: dict[str, Any] | None = None,
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

    if power_feasibility is not None and not bool(power_feasibility.get("decisive")):
        # Power admission (adds a refusal, weakens nothing): a locked plan
        # whose exact sign test cannot reject at the policy alpha must not be
        # measured into a promotion decision.
        reasons.append(
            "promotion_infeasible_by_design:"
            f"n={power_feasibility.get('n')}:"
            f"alpha={power_feasibility.get('alpha')}:"
            f"required_n={power_feasibility.get('required_n')}"
        )
        return _base(status="promotion_failed")

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
    raw_resource_candidates: Sequence[Mapping[str, Any]] | None = None,
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

    # LeverProof v2 selects over raw resource samples. The continuous evaluator
    # does not own cold/warm, energy, or cost measurements, so it must never
    # synthesize those axes from aggregate latency. A canonical measurement
    # owner supplies every candidate row or promotion fails closed.
    if raw_resource_candidates is None:
        return None, "promote_evidence_build_failed:raw_resource_evidence_missing"
    candidates = [dict(row) for row in raw_resource_candidates]
    if {str(row.get("id") or "") for row in candidates} != {
        "control",
        "candidate",
    }:
        return None, "promote_evidence_build_failed:raw_resource_candidate_ids"
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

    raw_observations, observation_source = _raw_metric_observations(
        camp_dir, candidate_id
    )
    if raw_observations is None or observation_source is None:
        return None, "promote_evidence_build_failed:raw_metric_observations_missing"
    observations = {
        "schema": "metric_observations/v1",
        "metrics": raw_observations,
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
                "raw_metric_observations_source": {
                    "path": str(observation_source),
                    "sha256": hashlib.sha256(
                        observation_source.read_bytes()
                    ).hexdigest(),
                },
                "raw_resource_candidates": candidates,
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
            formal_preflight_payload,
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
        payload = formal_preflight_payload(preflight)
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




# Reasons that mean "process/infra incomplete" — never a model reject and never
# permanent approach death. After a harness fix (new integration commit), retry.






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


def _PHASE_A_TAGS(phase_a_positive: bool, phase_a_quality_held: bool) -> list[str]:
    tags: list[str] = []
    if phase_a_positive:
        tags.append("phase_a_positive")
    if phase_a_quality_held:
        tags.append("phase_a_quality_held")
    return tags


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
        incomplete_reasons = _promotion_measurement_incomplete_reasons(
            camp,
            control_id=control_id,
            candidate_id=candidate_id,
            delivery=delivery,
        )
        if incomplete_reasons:
            # Partial scoreboard or spent chunk budget: the measurement is
            # incomplete evidence, never a model verdict (retryable, refunded).
            disposition = {
                "status": "promotion_inconclusive",
                "reasons": [*incomplete_reasons, *_PHASE_A_TAGS(phase_a_positive, phase_a_quality)],
                "cert_policy": None,
                "diagnosis_lanes": [],
                "emit_five_lane_matrix": False,
                "breaches": [],
                "primary_improvement": None,
                "promotion_primary_met": None,
                "inconclusive": True,
                "measurement_incomplete": True,
            }
        else:
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
                power_feasibility=_merged_promotion_power_feasibility(
                    camp,
                    control_id=control_id,
                    candidate_id=candidate_id,
                    locked=_campaign_power_feasibility(camp, candidate_id),
                    primary_metric=str(
                        (climb.promotion_primary or {}).get("metric")
                        or "held_out.structural_similarity"
                    ),
                ),
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
                # Stamp promote authority so future harness/policy updates force
                # automatic re-certification under current dispose rules.
                if status in _PROMOTE_AUTHORITY_STATUSES:
                    try:
                        _stamp_promote_authority(row, current_promote_authority())
                    except Exception as exc:  # noqa: BLE001 — never skip promote write
                        row["promote_authority_stamp_error"] = str(exc)[:200]
                    row.pop("recert_required", None)
                    row.pop("recert_required_at", None)
                    row.pop("recert_from_status", None)
                # Incomplete measurement is never a spent promote attempt.
                # Formal timeouts and harness failures (deadline_reserve,
                # missing_promote_run, cert incomplete because arms never ran)
                # refund so the approach stays valid and can be retried after a
                # harness fix — never permanently invalidated as a model reject.
                if (
                    status in {"promotion_inconclusive", "harness_failure"}
                    or disposition.get("timeout")
                    or disposition.get("harness_failure")
                ):
                    attempts = int(row.get("promote_attempts") or 0)
                    row["promote_attempts"] = max(0, attempts - 1)
                    if disposition.get("measurement_incomplete"):
                        row["last_measurement_incomplete"] = True
                        row["last_measurement_incomplete_reasons"] = [
                            str(reason)
                            for reason in (disposition.get("reasons") or [])
                            if str(reason).startswith("measurement_incomplete:")
                        ][:8]
                    elif status == "promotion_inconclusive" or disposition.get("timeout"):
                        row["last_formal_timeout"] = True
                        row["last_formal_timeout_wall_s"] = _PROMOTE_FORMAL_TIMEOUT_S
                    if status == "harness_failure" or disposition.get(
                        "harness_failure"
                    ):
                        row["last_harness_failure"] = True
                        # Stamp integration so a later code/harness fix reopens.
                        camp_meta = _read_json(camp_dir / "campaign.json")
                        tip = (
                            camp_meta.get("integration_commit")
                            or delivery.get("integration_commit")
                            or row.get("source_integration_commit")
                        )
                        if tip:
                            row["harness_failure_integration_commit"] = str(tip)
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
    observed_sd_path: Path | None = None,
) -> dict[str, Any]:
    """Classify cycle for SDLC Phase A stack-layer gate.

    Combines versioned climb policy (role primary, EG_params, fixture rules)
    with quality-aware latency/meaning tradeoffs: pure latency blips with empty
    meaning are not positive; quality may spend a bounded latency budget.
    Fixture insufficient_n / missing metrics / null deltas / uncharged capacity
    growth → non-positive.

    Screening NLL primaries are decided by the policy paired test on the
    per-record ``eval_nll_records.json`` deltas of both arms. The fixture
    ``insufficient_n`` clamp keeps nullifying decoded-quality wins and every
    promotion verdict; it does not nullify a paired NLL win decided on at
    least ``SCREENING_NLL_PAIRED_DECIDABILITY_FLOOR`` pairs — the quality
    probe's volume is then recorded as ``fixture_insufficient_n:quality_probe``.
    Screening positives remain ``claim_class: diagnostic``. I6 is untouched:
    any measured ``parse_rate < 1`` still yields ``invalid_grammar``.

    Each screening classification with paired deltas appends the measured
    paired-delta SD to ``observed_paired_sd_by_metric`` in the screening
    expectations file (``observed_sd_path`` overrides the repo file).
    """
    from slm_training.autoresearch.climb_policy import (
        FIXTURE_INSUFFICIENT_N_QUALITY_PROBE,
        PAIRED_PRIMARY_LEAVES,
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
    if role == "screening":
        _attach_screening_eval_nll(camp_dir / "runs" / control_id)
        _attach_screening_eval_nll(camp_dir / "runs" / candidate_id)
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
    # Per-record NLL pairs (teacher-forced, whole smoke suite) for the paired
    # screening verdict. Arms scored under different NLL definitions never pair.
    paired_records: dict[str, dict[str, float]] | None = None
    paired_records_info: dict[str, Any] = {}
    if (
        role == "screening"
        and effective_metric.rsplit(".", 1)[-1] in PAIRED_PRIMARY_LEAVES
    ):
        c_records, c_digest = _read_eval_nll_records(camp_dir / "runs" / control_id)
        t_records, t_digest = _read_eval_nll_records(camp_dir / "runs" / candidate_id)
        paired_records_info = {
            "control_n": len(c_records),
            "candidate_n": len(t_records),
            "control_definition_hash": c_digest,
            "candidate_definition_hash": t_digest,
        }
        if c_records and t_records:
            if c_digest and t_digest and c_digest != t_digest:
                reasons_pre.append(
                    f"paired_records_definition_mismatch:{control_id}:{candidate_id}"
                )
            else:
                paired_records = {"control": c_records, "candidate": t_records}
    outcomes = list((camp_dir / "artifacts" / "outcomes").glob("*.json"))
    # Latency pre-check verdicts per arm (probe timeout / over-budget skip) so
    # a probe-skipped eval reads as its typed cause, not a bare missing file.
    preflight_by_run: dict[str, str] = {}
    for path in outcomes:
        out = _read_json(path)
        eid = str(out.get("experiment_id") or path.stem)
        for stage in out.get("stage_telemetry") or []:
            if not isinstance(stage, dict):
                continue
            preflight = stage.get("latency_preflight")
            if (
                isinstance(preflight, dict)
                and str(preflight.get("verdict") or "")
                == "latency_preflight_infeasible"
            ):
                preflight_by_run[eid] = "latency_preflight_infeasible"
            elif stage.get("latency_probe") and stage.get("timed_out"):
                preflight_by_run[eid] = "latency_preflight_probe_timeout"
    for run_id in (control_id, candidate_id):
        scoreboard_path = camp_dir / "runs" / run_id / "scoreboard.json"
        if not scoreboard_path.is_file():
            reasons_pre.append(
                f"measurement_incomplete:{run_id}:"
                f"{preflight_by_run.get(run_id) or 'missing_scoreboard'}"
            )
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

    # Ship-gate evidence volume (default_min_suite_n) is always short on a
    # screening smoke suite; it stays a hard clamp for decoded quality wins and
    # promotion, and an information reason for a decided paired NLL win.
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
        paired_records=paired_records,
    )
    # Paired NLL win decided on >= the decidability floor with legal grammar.
    paired_nll_decided = bool(decision.get("fixture_clamp_exempt"))

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
            and candidate_latency > control_latency * (1.0 + _LATENCY_REGRESSION_BUDGET)
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
            (
                "measurement_incomplete:",
                "wall_timeout:",
                "empty_metrics:",
                "invalid_grammar:",
            )
        )
        for reason in reasons
    ):
        decision["positive"] = False
        decision["stack_layer"] = False
    delivery_view = {
        "control_metrics": control if isinstance(control, dict) else {},
        "candidate_metrics": candidate if isinstance(candidate, dict) else {},
        "measurement_complete": True,
        "reasons": reasons,
    }
    if _quality_metrics_identical(delivery_view) and not any(
        str(reason).startswith("mechanism_no_effect:") for reason in reasons
    ):
        if paired_nll_decided:
            # Identical decoded numbers on a fixture-n probe are not a
            # no-effect verdict against a paired NLL test decided per record.
            reasons.append("quality_probe_identical:decoded_quality_metrics_identical")
        else:
            reasons.append("mechanism_no_effect:quality_metrics_identical")
            decision["positive"] = False
            decision["stack_layer"] = False
    if fixture_only_fails or any(
        str(reason).startswith("fixture_insufficient_n") for reason in reasons
    ):
        if paired_nll_decided:
            # Quality-probe volume is information here, never the verdict.
            if FIXTURE_INSUFFICIENT_N_QUALITY_PROBE not in reasons:
                reasons.append(FIXTURE_INSUFFICIENT_N_QUALITY_PROBE)
        else:
            # Tradeoff / primary_metric_win must not re-green fixture ship
            # volume on decoded quality metrics, latency/efficiency paths, or
            # any promotion verdict.
            decision["positive"] = False
            decision["stack_layer"] = False
            lean_ok = _lean_floor_measurement(delivery_view)
            if lean_ok:
                if not any(
                    str(reason).startswith("fixture_volume_gate_ship_only")
                    for reason in reasons
                ):
                    reasons.append("fixture_volume_gate_ship_only")
            elif not any(
                str(reason).startswith("fixture_insufficient_n_alone")
                for reason in reasons
            ):
                reasons.append("fixture_insufficient_n_alone")

    paired = decision.get("paired_test")
    if (
        role == "screening"
        and isinstance(paired, dict)
        and isinstance(paired.get("paired_sd"), (int, float))
        and int(paired.get("n_pairs") or 0) >= 2
    ):
        # Measured paired-delta SD feeds the screening power calibration.
        try:
            _record_observed_paired_sd(
                observed_sd_path or screening_expectations_path(),
                metric_leaf=leaf,
                sd=float(paired["paired_sd"]),
                n=int(paired["n_pairs"]),
                campaign_id=camp_dir.name,
                control_id=control_id,
                candidate_id=candidate_id,
            )
        except Exception as exc:  # noqa: BLE001 — calibration is never a verdict
            print(f"OBSERVED_PAIRED_SD_SKIP metric={leaf} err={exc!r}", flush=True)

    decision["reasons"] = reasons
    decision["paired_records"] = paired_records_info
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
    control_id: str | None = None,
    candidate_id: str | None = None,
    arm_exits: Mapping[str, int] | None = None,
    arm_skipped: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record SDLC Phase A decision; never open stacked PR for non-positive."""
    from slm_training.autoresearch.climb_policy import (
        cycle_role_for_index,
        load_climb_policy,
    )

    policy = load_climb_policy()
    camp_dir = root / campaign_id
    # Prefer the exact matrix identities supplied by the driver.  Filename
    # heuristics are retained only for legacy callers and must never relabel a
    # dynamic successor arm.
    man_dir = camp_dir / "manifests"
    control_run = control_id
    candidate_run = candidate_id
    if man_dir.exists():
        for path in sorted(man_dir.glob("*.json")):
            eid = path.stem
            if control_run is None and (
                eid.endswith("-control") or eid.endswith("_control")
            ):
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
        "arm_exits": dict(arm_exits or {}),
        "arm_skipped": dict(arm_skipped or {}),
        **{k: v for k, v in decision.items() if k not in {"positive", "stack_layer"}},
    }
    # WP-3: persist the cycle's preflight verdicts (written at selection time
    # by _preflight_screening_slug) into the delivery record + ledger entry.
    try:
        preflight_path = camp_dir / "preflight.json"
        if preflight_path.is_file():
            preflight = _read_json(preflight_path)
            if isinstance(preflight, dict) and preflight:
                record["preflight"] = preflight
    except Exception as exc:  # noqa: BLE001 — telemetry only, never fatal
        print(f"PREFLIGHT_WARN delivery err={exc!r}", flush=True)
    out_path = camp_dir / "sdlc_delivery.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    ledger = root / "sdlc_delivery_ledger.jsonl"
    # Dangling symlinks (e.g. prior /tmp continuous worktree) raise FileNotFoundError
    # on open("a"); replace with a real ledger so Phase A closeout never hard-fails.
    if ledger.is_symlink() and not ledger.exists():
        ledger.unlink()
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")

    # Cheap thrash residual ledger (no retrain): mine interesting residuals.
    try:
        cycle_idx = None
        match = re.search(r"-c(\d+)$", campaign_id)
        if match:
            cycle_idx = int(match.group(1))
        _append_interesting_residual(
            root,
            loop_id,
            record,
            campaign_id=campaign_id,
            cycle_index=cycle_idx,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"THRASH_RESIDUAL_WARN err={exc!r}", flush=True)

    # Pareto thrash timing: durable completeness + decode-fit snapshot.
    try:
        from slm_training.autoresearch.climb_policy import stage_wall_minutes_for_role

        arm_s = _arm_wall_seconds(
            policy_minutes=float(stage_wall_minutes_for_role(policy, role)),
            formal_required=str(cycle_intent or role) == "promote",
        )
        decode_fit = None
        if role == "screening":
            # Fit from this campaign's own scoreboards: the timing record is
            # the budget its successor cycle inherits (decode_floor_source
            # measured_p95 once eval_smoke.json exists, else policy_default).
            _fitted, decode_fit = _fit_screening_decode_timeout_seconds(
                policy,
                arm_wall_seconds=arm_s,
                telemetry_root=root,
                predecessor_campaign_id=campaign_id,
            )
        matrix_regime = None
        matrix_path = camp_dir / "matrix-proposal.json"
        if matrix_path.is_file():
            try:
                matrix_regime = _read_json(matrix_path).get("thrash_regime")
            except Exception:  # noqa: BLE001 — telemetry only
                matrix_regime = None
        if matrix_regime is not None:
            record["thrash_regime"] = matrix_regime
            out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _write_thrash_timing(
            camp_dir,
            loop_id=loop_id,
            campaign_id=campaign_id,
            cycle_index=cycle_index,
            role=str(role),
            measurement_complete=bool(record.get("measurement_complete")),
            arm_wall_seconds=arm_s,
            decode_fit=decode_fit,
            reasons=list(record.get("reasons") or []),
            control_metrics=record.get("control_metrics")
            if isinstance(record.get("control_metrics"), dict)
            else None,
            candidate_metrics=record.get("candidate_metrics")
            if isinstance(record.get("candidate_metrics"), dict)
            else None,
            thrash_regime=matrix_regime if isinstance(matrix_regime, dict) else None,
        )
    except Exception as exc:  # noqa: BLE001 — never fail Phase A on telemetry
        print(f"THRASH_TIMING_WRITE_SKIP err={exc}", flush=True)

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
    hill = _persist_hillclimb_cycle_outputs(
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        delivery=record,
        cycle_index=cycle_index,
        primary_metric=primary_metric,
    )
    record["hillclimb"] = hill
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
        f"- candidate: `{candidate_run}` metrics={record.get('candidate_metrics')}\n"
        f"- skipped arms: `{record.get('arm_skipped')}`\n\n"
        "## Hill-climb this cycle\n\n"
        f"- went well: {', '.join(hill.get('went_well') or []) or '—'}\n"
        f"- went wrong: {', '.join(hill.get('went_wrong') or []) or '—'}\n"
        f"- speculate: {', '.join(hill.get('speculate') or []) or '—'}\n"
        f"- deltas: `{hill.get('deltas')}`\n\n"
        "Non-positive cycles do not open stacked PRs "
        "(sdlc autotrain-iteration-delivery).\n"
    )
    note_path.write_text(note, encoding="utf-8")
    return record














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
        "slot_contract_in_context",
        "constraint_graph_mode",
    }

    def has_quality_objective(knobs: dict[str, Any]) -> bool:
        def active(value: Any) -> bool:
            if value is None or value is False:
                return False
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value > 0
            return True

        return (
            any(active(knobs.get(key)) for key in quality_keys)
            or (float(knobs.get("fidelity_loss_weight") or 0.5) != 0.5)
            or str(knobs.get("mask_pattern") or "random") != "random"
        )

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
            # Include loop-local self-heal compose arms; bare next() on the
            # static bank alone raised StopIteration and hard-blocked the loop
            # after dynamic thrash successors were selected.
            successor = next(
                (row for row in _all_screening_arm_bank() if row[0] == successor_slug),
                None,
            )
            if successor is not None:
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




def _completed_retry_priorities(
    matrix: dict[str, Any], candidate_id: str
) -> tuple[NextRunPriorityV1, ...]:
    """Compatibility wrapper for completed frozen-replay steering."""

    return _completed_candidate_priorities(
        matrix, candidate_id, resolved_infrastructure=True
    )




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
            for slug, _hypothesis, _extras in _all_screening_arm_bank():
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
    for _ in range(_recent_exhaustion_cycle_window()):
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
            _read_json(ancestor_dir / "campaign.json").get("predecessor_campaign_id")
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
        for slug, _hypothesis, _extras in _all_screening_arm_bank():
            if (
                experiment_id.endswith(f"-{slug}")
                and slug not in closed
                and (slug not in skip or evidence_directed)
            ):
                return slug
    return None


def _handoff_should_route_exhausted_bank(
    *,
    root: Path,
    campaign_id: str,
    loop_id: str,
    matrix: Mapping[str, Any],
    priorities: Sequence[Any],
) -> bool:
    """Park to rebuild_data when the isolate bank is done, even if ranks remain.

    Ranked ``experiment_next`` slugs used to suppress the I10 off-ramp, so the
    loop rematched exhausted decoder arms forever. ``park_on_exhaust`` plus an
    empty isolate bank — including when the only leftovers are snapshot
    train_version clones — is terminal.
    """
    _load_dynamic_thrash_arms(root, loop_id)
    saturation = (
        (matrix.get("thrash_regime") or {}).get("screening_saturation")
        if isinstance(matrix.get("thrash_regime"), dict)
        else None
    )
    if isinstance(saturation, dict) and not saturation.get("pending_regimes", []):
        return True
    closed = _recent_completed_nonpositive_slugs(root, campaign_id)
    if _terminal_park_on_exhaust():
        open_slugs = _thrash_bank_open_slugs(closed)
        if not open_slugs or _open_slugs_are_snapshot_leftovers(open_slugs):
            return True
        return False
    return not any(
        getattr(priority, "disposition", None) == "experiment_next"
        and getattr(priority, "proposed_experiment_id", None)
        for priority in priorities
    )


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
    cwd: Path | None = None,
) -> AutotrainCycleHandoffV1:
    terminal_verdict: dict[str, Any] | None = None
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
        control_decode_timeout and not finalized_decode_timeout
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
        thrash_timeout_residual = _delivery_is_thrash_timeout_residual(delivery)
        if runtime_arm_rejected or (
            thrash_timeout_residual
            and cycle_intent in {"screening", "retry_measurement"}
        ):
            # Continuous thrash must not hard-block on wall/decode residuals.
            # Retire the residual arm and keep rotating; real harness crashes
            # still take the repair_harness path below.
            actions.append(
                AutotrainActionV1(
                    kind="next_experiment",
                    owner="autotrain",
                    reason=(
                        "retire thrash decode/wall-timeout residual and consume "
                        "the next distinct ranked hypothesis"
                        if thrash_timeout_residual
                        else (
                            "retire the candidate-only runtime rejection and "
                            "consume the next distinct ranked hypothesis"
                        )
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
            if _delivery_is_thrash_timeout_residual(delivery) and cycle_intent in {
                "screening",
                "retry_measurement",
            }:
                actions.append(
                    AutotrainActionV1(
                        kind="next_experiment",
                        owner="autotrain",
                        reason=(
                            "incomplete thrash replay budget exhausted "
                            f"({replay_count}/{replay_limit}); retire residual "
                            "and consume the next distinct hypothesis"
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
        and _handoff_should_route_exhausted_bank(
            root=root,
            campaign_id=campaign_id,
            loop_id=loop_id,
            matrix=matrix,
            priorities=priorities,
        )
    ):
        saturation = (
            (matrix.get("thrash_regime") or {}).get("screening_saturation")
            if isinstance(matrix.get("thrash_regime"), dict)
            else None
        )
        saturation_closed = bool(
            isinstance(saturation, dict) and not saturation.get("pending_regimes", [])
        )
        closed = _recent_completed_nonpositive_slugs(root, campaign_id)
        actions = list(
            _capability_objective_refresh_actions(
                root=root,
                campaign_id=campaign_id,
                preserved_actions=tuple(actions),
            )
        )
        from slm_training.autoresearch import evidence_ledger as _ev
        from slm_training.autoresearch.climb_policy import load_climb_policy

        policy_sha = load_climb_policy().sha256
        terminal_verdict = _ev.build_regime_exhausted_verdict(
            campaign_id=campaign_id,
            loop_id=loop_id,
            cycle_index=cycle_index,
            binding_constraint=(
                "screening_objective_saturated"
                if saturation_closed
                else "quality_arm_bank_exhausted"
            ),
            closed_slugs=sorted(closed),
            policy_sha256=policy_sha,
            resume_predicate=(
                f"a feedback-grounded current-rung ({_current_rung_label()}) "
                "data and capability objective is preregistered under "
                "unchanged I10 rung gates"
            ),
            bank_fingerprint=_screening_bank_fingerprint(policy_sha256=policy_sha),
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
    # A metric-positive fixture with no tracked delta has nothing reviewable to
    # deliver. Do not emit a blocking deliver_stack action in that case; the
    # continuous loop must be able to consume its next-experiment action.
    if delivery.get("positive") and delivery.get("stack_layer"):
        actions.append(
            AutotrainActionV1(
                kind="deliver_stack",
                owner="sdlc",
                reason="document the positive result, then deliver its reviewable delta",
                evidence_ids=(evidence_id,),
            )
        )

    thrash_regime_payload = matrix.get("thrash_regime")
    if thrash_regime_payload is not None and not isinstance(
        thrash_regime_payload, dict
    ):
        thrash_regime_payload = None
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
        thrash_regime=thrash_regime_payload,
        terminal_verdict=terminal_verdict,
    )
    (camp_dir / "cycle_handoff.json").write_text(
        handoff.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if terminal_verdict is not None and _terminal_park_on_exhaust():
        # Park under the typed conclusion: persist the verdict beside the loop
        # state so the successor cycle's deterministic resume predicate
        # (bank-fingerprint change) governs re-entry, not a human re-prompt.
        verdict_path = _terminal_verdict_path(root, loop_id)
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        verdict_path.write_text(
            json.dumps(terminal_verdict, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_loop_state(
            root,
            AutotrainLoopStateV1(
                loop_id=loop_id,
                state="BLOCKED",
                phase="blocked",
                active_campaign_id=None,
                last_completed_campaign_id=campaign_id,
                cycle_index=cycle_index,
                next_action=actions[0].kind,
                blocker_fingerprint=str(
                    terminal_verdict.get("binding_constraint") or ""
                )
                or None,
                blocker_count=1,
                pid=os.getpid(),
            ),
        )
        print(
            f"REGIME_PARK loop={loop_id} campaign={campaign_id} "
            f"constraint={terminal_verdict.get('binding_constraint')}",
            flush=True,
        )
    else:
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
    # Driver-owned document closeout so the successor never waits on a human
    # re-prompt for ordinary thrash screening notes. Only when cwd is explicit
    # (live continuous worktree) — unit tests call this helper without cwd and
    # must not write into the caller's real docs/design tree.
    if cwd is not None:
        try:
            _self_heal_document_actions(
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                campaign_id=campaign_id,
            )
        except Exception as exc:  # noqa: BLE001 — never fail handoff write on closeout
            print(
                f"SELF_HEAL_DOCUMENT_WARN campaign={campaign_id} err={exc!r}",
                flush=True,
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


_VACUOUS_PASS_LIMIT = 3
#: Driver exit code for the typed loop-stalled park (2 = hard pending).
_STALL_EXIT_CODE = 3






#: How a heal receipt's outcome scores the driver pass that produced it.
#: Total over ``slm_training.autoresearch.heal.schemas.HealOutcome`` -- a new
#: outcome with no entry here would fall to the generic ``heal_attempted`` and
#: hide itself, which is how ``postcondition_failed`` (the data-rebuild
#: playbook's no-op verdict) went uncounted while only ``verify_failed`` was
#: mapped. Totality is enforced by
#: ``tests/test_autoresearch/test_pass_outcome_contract.py``.
_PASS_OUTCOME_BY_HEAL_OUTCOME: dict[str, str] = {
    "healed": "verified_heal",
    # Both failure verdicts mean the same thing to a pass: the heal ran and
    # its proof obligation did not hold. ``verify_failed`` comes from the
    # subprocess verify probe, ``postcondition_failed`` from an in-process
    # playbook's count postcondition.
    "verify_failed": "heal_postcondition_failed",
    "postcondition_failed": "heal_postcondition_failed",
    "unhandled": "escalation_unhandled",
    # An attempt that never reached a proof obligation. Visible in the
    # receipt; counted as an attempt, never as progress.
    "attempted": "heal_attempted",
    "step_failed": "heal_attempted",
    "step_timeout": "heal_attempted",
    "refused_scope": "heal_attempted",
    # The runner declined to retry: no work was done, and the blocker is
    # already escalated on the ledger.
    "budget_exhausted": "escalation_unhandled",
    "cycle_detected": "escalation_unhandled",
}


def _record_pass_outcome(
    *, root: Path, loop_id: str, before_campaign: str | None,
    before_receipts: int, typed_action: bool = False, reason: str | None = None,
) -> str:
    """Classify one driver pass; a clean no-op is a counted hard failure.

    Returns the outcome. ``loop_stalled_no_campaign`` means
    ``_VACUOUS_PASS_LIMIT`` consecutive vacuous passes were observed and the
    typed park (``state=BLOCKED``) has been written: the caller exits non-zero
    without raising. Heal receipts are read by outcome, so a driver heal whose
    postcondition failed scores ``heal_postcondition_failed`` (visible,
    counted) rather than ``vacuous_pass`` or ``verified_heal``.
    """
    _idx, after_campaign = _latest_cycle(root, loop_id)
    receipts_path = root / "loops" / loop_id / "heal_receipts.jsonl"
    after_receipts = len(receipts_path.read_text(encoding="utf-8").splitlines()) if receipts_path.is_file() else 0
    if after_campaign and after_campaign != before_campaign:
        outcome = "campaign_initialized"
    elif after_receipts > before_receipts:
        outcome = _PASS_OUTCOME_BY_HEAL_OUTCOME.get(
            _last_heal_receipt_outcome(receipts_path) or "", "heal_attempted"
        )
    elif typed_action:
        outcome = "typed_park_or_escalation"
    else:
        outcome = "vacuous_pass"
    path = root / "loops" / loop_id / "pass_outcomes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = [r for r in path.read_text(encoding="utf-8").splitlines() if r] if path.is_file() else []
    previous_vacuous = 0
    last_non_vacuous: dict[str, Any] | None = None
    counting = True
    for raw in reversed(prior):
        try:
            previous = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if previous.get("outcome") in {"vacuous_pass", _STALL_FINGERPRINT}:
            if counting:
                previous_vacuous += 1
            continue
        counting = False
        last_non_vacuous = previous
        break
    row = {"schema": "pass_outcome/v1", "loop_id": loop_id, "outcome": outcome,
           "campaign_before": before_campaign, "campaign_after": after_campaign,
           "consecutive_vacuous": previous_vacuous + 1 if outcome == "vacuous_pass" else 0,
           "reason": reason, "recorded_at": utc_now()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    if row["consecutive_vacuous"] >= _VACUOUS_PASS_LIMIT:
        _park_loop_stalled(
            root=root,
            loop_id=loop_id,
            cycle_index=_idx,
            campaign_id=after_campaign,
            consecutive=int(row["consecutive_vacuous"]),
            last_non_vacuous=last_non_vacuous,
            reason=reason,
        )
        return _STALL_FINGERPRINT
    return outcome




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
        control_id=control,
        candidate_id=candidate,
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
        cwd=cwd,
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
            or (claim.policy == "required" and preflight.status != "proved")
            or formal_obligation_id(manifest.campaign_id, manifest.experiment_id, claim)
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
        # Incomplete predecessor (e.g. control arm deadline-skipped) cannot be
        # frozen-replayed. Skip to normal thrash rather than hard-blocking the
        # continuous loop — same class as nonreplayable configuration.
        print(
            "FROZEN_REPLAY_SKIP reason=missing_control_manifest "
            f"control_id={control_id} campaign={predecessor_campaign_id} "
            f"action_index={action_index}",
            flush=True,
        )
        return None
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
        _restore_frozen_formal_claims(camp_dir, item["experiment"], item["manifest"])
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

    # Fail-closed: a source arm that finalized with decode timeouts must not
    # reuse train (force a fresh control/candidate eval on the successor).
    timeout_n = _arm_decode_timeout_count(campaign_dir, manifest.experiment_id)
    if timeout_n > 0:
        print(
            "FROZEN_TRAIN_REUSE_SKIP reason=decode_timeout "
            f"experiment={manifest.experiment_id} decode_timeout_count={timeout_n}",
            flush=True,
        )
        return None

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
                raise RuntimeError(
                    "frozen campaign predecessor lineage contains a cycle"
                )
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
        (item[0] for item in _all_screening_arm_bank()), key=len, reverse=True
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
        frozen_fingerprint = _knobs_fingerprint(_lever_knobs(frozen_knobs))
        source_slug = (
            _arm_slug_from_knobs(frozen_knobs, candidate_id=old_candidate_id)
            if confirmation_replay
            else None
        )
        candidate_target = (
            next(
                (
                    item["experiment"]
                    for item in matrix["hypotheses"]
                    if source_slug
                    and item["experiment"]["experiment_id"].endswith(f"-{source_slug}")
                ),
                None,
            )
            or next(
                (
                    item["experiment"]
                    for item in matrix["hypotheses"]
                    if _knobs_fingerprint(
                        _lever_knobs(item["experiment"].get("knobs") or {})
                    )
                    == frozen_fingerprint
                ),
                None,
            )
            or next(
                (
                    item["experiment"]
                    for item in matrix["hypotheses"]
                    if item["experiment"]["experiment_id"]
                    == matrix["recommended_experiment_id"]
                ),
                None,
            )
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
    thrash_regime: ThrashRegimeDecision | None = None,
    initialize_from: str | None = None,
    telemetry_root: Path | None = None,
    predecessor_campaign_id: str | None = None,
    chunk_plan: Mapping[str, Any] | None = None,
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
    # Distinct seeds are required so multi-seed approach closure can re-test a
    # once-null arm on a new seed before permanent bank close.
    seed = 100_000 + int(cycle)
    pol = policy or load_climb_policy()
    # Thrash: micro-steps + decode fit so n×decode + train floor ≤ arm wall
    # (Pareto: incomplete rate drives recalibration of these, not wall++).
    if role == "screening":
        decode_timeout, decode_fit = _fit_screening_decode_timeout_seconds(
            pol,
            telemetry_root=telemetry_root,
            requested_steps=int(steps),
            predecessor_campaign_id=predecessor_campaign_id,
        )
        steps = int(
            (decode_fit or {}).get("fitted_steps")
            or _screening_thrash_steps(
                pol,
                steps,
                floor_seconds=(decode_fit or {}).get("grown_train_floor_seconds"),
                telemetry_root=telemetry_root,
            )
        )
    else:
        decode_timeout = decode_timeout_seconds_for_role(pol, role)
        decode_fit = None
    # Latency pre-check (policy measurement.latency_probe): a probe eval runs
    # before the full screening eval; over-budget projections skip the full
    # eval with a typed verdict instead of burning n x decode-timeout.
    probe_block = pol.measurement.get("latency_probe")
    probe_block = dict(probe_block) if isinstance(probe_block, Mapping) else {}
    latency_probe_knobs: dict[str, int] = {}
    if role == "screening" and probe_block.get("enabled"):
        latency_probe_knobs = {
            "latency_probe_records": max(1, int(probe_block.get("probe_records") or 1)),
            "latency_probe_planned_n": max(
                1, int((decode_fit or {}).get("smoke_n") or 1)
            ),
        }
    steps = int(steps) + (cycle % 3)  # slight variation avoids knob-signature collision
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
        # Semantic-contrast train_loop requires batch_size >= 3. Matched controls
        # that share contrast pair exposure (dir/margin/fraction) use the same
        # batch for size/dynamics match even when contrast weight is 0.
        # Private bank keys (``_thrash_slug``, ``_steps_factor``, …) are
        # classification-only and must never enter the extra-forbidden knobs
        # schema — OFAT control packages copy bank extras and previously
        # leaked ``_thrash_slug`` into hypotheses[0].
        extra_map = {
            k: v
            for k, v in dict(extra).items()
            if not str(k).startswith("_") and k not in _PROCESS_ARM_KNOB_KEYS
        }
        contrast_weight = float(extra_map.get("semantic_contrast_loss_weight") or 0.0)
        contrast_exposure = bool(
            extra_map.get("semantic_contrast_dir")
            or extra_map.get("semantic_contrast_margin")
            or extra_map.get("semantic_contrast_fraction")
            or contrast_weight > 0
        )
        default_batch = 3 if contrast_exposure else 2
        if contrast_exposure:
            try:
                requested_batch = int(extra_map.get("batch_size") or default_batch)
            except (TypeError, ValueError):
                requested_batch = default_batch
            extra_map["batch_size"] = max(3, requested_batch)
        base: dict[str, object] = {
            "train_version": train_version,
            "eval_version": eval_version,
            "steps": steps,
            "batch_size": default_batch,
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
            "solver_energy_loss_weight": 0.0,
            "solver_energy_decode_weight": 0.0,
            "legal_edit_hazard_loss_weight": 0.0,
            "legal_edit_hazard_decode_weight": 0.0,
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
        if role == "promotion" and chunk_plan:
            # Locked chunked measurement: per-suite n, per-record persistence
            # and the per-run decode cap come from the pre-run chunk plan.
            base["eval_limit"] = int(chunk_plan["suite_n"])
            base["eval_partial_scoreboard"] = True
            base["eval_max_records_this_run"] = int(chunk_plan["records_per_run"])
        if role == "screening":
            # Screening smoke suites are tiny: baked generate_batch_size groups
            # every document into one decode chunk, defeating per-record
            # fair-share timeout redistribution.
            base["generate_batch_size"] = 1
            base.update(latency_probe_knobs)
            # The decode probe is bounded by the fitted n_probe. Without this
            # the eval decodes the whole published suite (96 records) however
            # small the eval share is, so the arm is killed at the wall and
            # reports no document counts at all: the fit computed n_probe and
            # nothing consumed it. NLL still spans the full suite -- it is
            # teacher-forced and cheap -- so the screening verdict keeps its
            # pairs while the decoded quality probe stays inside its budget.
            probe_n = (decode_fit or {}).get("n_probe")
            if isinstance(probe_n, int) and probe_n > 0:
                base["eval_limit"] = int(probe_n)
        base.update(extra_map)
        if initialize_from:
            base["initialize_from"] = initialize_from
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
        promo_extra = {k: v for k, v in promote_levers.items() if k in _LEVER_KNOB_KEYS}
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
            k: v for k, v in confirm_levers.items() if k in _LEVER_KNOB_KEYS
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
        # Dual-regime thrash: isolate (causal OFAT) or climb (residual on sticky
        # champion baseline). Timeout residual prefers decode-cost arms.
        regime = thrash_regime or decide_screening_regime(
            climb_baseline_knobs=None,
            compiler_ms_timeout=False,
        )
        bank_slugs = [slug for slug, _, _ in _all_screening_arm_bank()]
        # Soft residual rank only when recommended_slug already computed by
        # caller with root/loop_id; fallback isolate_selector stays pure rotation
        # (matrix builder may not have loop context in unit tests).
        rec_slug = recommended_slug or select_recommended_slug_for_regime(
            decision=regime,
            cycle=cycle,
            skip=skip_slugs,
            bank_slugs=bank_slugs,
            isolate_selector=lambda c, s: _select_recommended_slug(c, skip=s),
        )
        bank_by_slug = {
            slug: (hyp, extras) for slug, hyp, extras in _all_screening_arm_bank()
        }
        if rec_slug not in bank_by_slug:
            rec_slug = _all_screening_arm_bank()[0][0]
        climb_active = regime.base_regime == REGIME_CLIMB and (
            regime.climb_baseline is not None
        )
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
            "exposure-targeted-compiler-decision-margin": ("mixture_sampling_policy"),
            "exposure-targeted-semantic-exhaustive-compiler-decision-margin": (
                "compiler_alignment_semantic_exhaustive"
            ),
        }.get(rec_slug)
        if not climb_active:
            if treatment_key is not None:
                control_extra = {
                    key: value
                    for key, value in bank_by_slug[rec_slug][1].items()
                    if key != treatment_key and not str(key).startswith("_")
                }
            if rec_slug == "exposure-targeted-compiler-decision-margin":
                control_extra = {
                    key: value
                    for key, value in bank_by_slug[rec_slug][1].items()
                    if not str(key).startswith("_")
                }
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
        if climb_active:
            control_levers = compose_climb_control_levers(regime.climb_baseline or {})
            control_hyp = (
                "Climb baseline control: sticky confirmed/climb_accepted champion "
                "recipe under size-matched residual thrash."
            )
            control_rationale = (
                f"Climb control baseline ({regime.reason}); residual-only treatments."
            )
        else:
            # Isolate: precursor package or zeroed template (existing OFAT).
            isolate_precursor = compose_isolate_control_levers(
                precursor_extras=control_extra
            )
            control_levers = {
                "structural_aux_head_profile": str(
                    bank_by_slug[rec_slug][1].get("structural_aux_head_profile", "none")
                ),
                "compiler_decode_mode": str(
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
                **isolate_precursor,
            }
            control_hyp = (
                "Matched fixture control with both grammar levers off completes "
                "smoke eval under the published suite."
            )
            control_rationale = (
                f"All-family margin control for isolated {treatment_key} attribution."
                if treatment_key is not None
                else "Baseline for size-matched continuous attribution."
            )
        candidates = [
            {
                "experiment": exp(
                    f"{prefix}-control",
                    control_hyp,
                    knobs(**control_levers),
                    control_rationale,
                ),
                "evidence_uses": uses(),
                "novelty": novelty(
                    0,
                    (
                        "climb sticky baseline control"
                        if climb_active
                        else "matched control with published eval"
                    ),
                ),
            }
        ]
        # Track lever signatures so static + dynamic compose arms never collide
        # (HypothesisMatrix requires distinct knob signatures). Climb signs the
        # *materialized* knobs() payload — sparse residual overlay alone is not
        # comparable to full control knobs when the champion already carries that
        # residual (no-op arm would otherwise slip through and fail validation).
        _sig_exclude = frozenset(
            {
                "seed",
                "steps",
                "decode_timeout_seconds",
                "generate_batch_size",
                "eval_suites",
                "eval_limit",
                "eval_partial_scoreboard",
                "eval_max_records_this_run",
            }
        )

        def _materialized_sig(full_knobs: Mapping[str, Any]) -> str:
            return _thrash_lever_signature(
                {k: v for k, v in full_knobs.items() if k not in _sig_exclude}
            )

        seen_lever_sigs: set[str] = set()
        warm_skipped_data_arms: list[str] = []
        control_knobs = candidates[0]["experiment"]["knobs"]
        seen_lever_sigs.add(_materialized_sig(control_knobs))
        for i, (slug, hyp, extras) in enumerate(_all_screening_arm_bank(), start=1):
            if (
                not climb_active
                and treatment_key is not None
                and slug == precursor_slug
            ):
                # The matched control is exactly this precursor arm. Emitting it
                # again would violate the preregistration contract's distinct-
                # knob-signature requirement without adding information.
                continue
            arm_extra = _apply_arm_extras(steps, extras)
            if initialize_from and _arm_swaps_train_corpus(
                extras, control_train_version=train_version
            ):
                # Warm start forks the champion for both arms with identical
                # data (``assert_warm_start_launch``); a data arm changes the
                # corpus, so it is only a legal candidate on a cold-start
                # cycle. Skipped here, never silently rewritten.
                warm_skipped_data_arms.append(slug)
                continue
            if climb_active:
                arm_extra = compose_treatment_levers(
                    control_levers=control_levers,
                    residual_extras=arm_extra,
                )
                hyp = f"Climb residual '{slug}' on sticky champion baseline: {hyp}"
            full_knobs = knobs(**arm_extra)
            if climb_active:
                sig = _materialized_sig(full_knobs)
            else:
                # Isolate keeps sparse extras for thrash-arm identity (steps-only
                # arm differs only on measurement keys excluded from full sig).
                sig = _thrash_lever_signature(arm_extra)
            if sig in seen_lever_sigs:
                # Prefer the static bank slug; drop duplicate dynamic recipes.
                # Climb: also drops no-op residuals already present on baseline.
                continue
            seen_lever_sigs.add(sig)
            candidates.append(
                {
                    "experiment": exp(
                        f"{prefix}-{slug}",
                        hyp,
                        full_knobs,
                        (
                            f"Climb residual thrash arm '{slug}' "
                            f"(regime={regime.regime})."
                            if climb_active
                            else f"Continuous thrash arm '{slug}' "
                            f"(rotated recommendation)."
                        ),
                    ),
                    "evidence_uses": uses(),
                    "novelty": novelty(i, f"thrash arm {slug}"),
                }
            )
        rec = f"{prefix}-{rec_slug}"
        if initialize_from:
            if warm_skipped_data_arms:
                print(
                    "WARM_START_SKIP_DATA_ARMS "
                    f"arms={','.join(warm_skipped_data_arms)} "
                    "reason=warm_start_requires_equal_train_data",
                    flush=True,
                )
            for row in candidates[1:]:
                assert_warm_start_launch(
                    candidates[0]["experiment"]["knobs"],
                    row["experiment"]["knobs"],
                )
        candidate_ids = {
            str((row.get("experiment") or {}).get("experiment_id") or "")
            for row in candidates
        }
        if rec not in candidate_ids:
            # Recommended compose arm may have been dropped as a signature
            # duplicate of a static recipe — retarget to first treatment arm.
            for row in candidates:
                eid = str((row.get("experiment") or {}).get("experiment_id") or "")
                if eid and not eid.endswith("-control"):
                    rec = eid
                    rec_slug = eid.split(f"{prefix}-", 1)[-1]
                    break
        regime_label = regime.regime
        priorities = [
            {
                "rank": 1,
                "area": "model",
                "hypothesis": (
                    f"[{regime_label}] Test thrash arm '{rec_slug}' first under "
                    f"the published eval suite."
                ),
                "evidence_ids": [research, prior],
                "confidence": 0.6,
                "expected_information_gain": (
                    "Attributes residual decode/quality metrics vs climb baseline."
                    if climb_active
                    else "Attributes decode metrics vs matched control."
                ),
                "authority": "speculative",
                "disposition": "experiment_next",
                "proposed_experiment_id": rec,
            },
            {
                "rank": 2,
                "area": "evaluation",
                "hypothesis": (
                    "Keep the climb sticky baseline as the size-matched control."
                    if climb_active
                    else "Keep the matched control as the size-matched baseline every cycle."
                ),
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
                "hypothesis": (
                    "Timeout residual prefers decode-cost arms (bounds/canvas/both/cache)."
                    if regime.timeout_residual
                    else (
                        "Climb residual rotation on sticky champion recipe."
                        if climb_active
                        else "Rotate thrash recommendation across the lever bank (not bounds-only)."
                    )
                ),
                "evidence_ids": [research, prior],
                "confidence": 0.65,
                "expected_information_gain": (
                    "Routes compiler_ms timeouts into decode cost thrash."
                    if regime.timeout_residual
                    else "Avoids single-lever thrash collapse."
                ),
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
                f"[{regime_label}] Size-matched continuous thrash "
                f"(base={regime.base_regime}, timeout_residual="
                f"{regime.timeout_residual}) with recommendation '{rec_slug}' "
                f"(cycle {cycle}; reason={regime.reason})."
            ),
            "thrash_regime": regime.as_dict(),
            "decode_residual_slugs": list(DECODE_RESIDUAL_SLUGS),
            "next_run_priorities": priorities,
        }
    if feedback:
        fb_ids = [
            str(item.get("feedback_id")) for item in feedback if item.get("feedback_id")
        ]
        # Only bind feedback ids that hypothesize will also load. Stale ids from
        # distant ancestors fail AgentHypothesisProvider with
        # "matrix conflicts with supplied feedback ids" and abort thrash.
        if fb_ids:
            payload["feedback_ids"] = fb_ids
            if previous_matrix_id:
                payload["predecessor_matrix_id"] = previous_matrix_id
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
    chunk_plan: Mapping[str, Any] | None = None,
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
        # Power admission: lock the exact sign-test feasibility of the planned
        # promotion measurement before any outcome is visible. Dispose refuses
        # a non-decisive report (promotion_infeasible_by_design).
        from slm_training.autoresearch import evidence_ledger as _ev

        measurement = pol.measurement
        gate = (pol.payload or {}).get("power_gate")
        power_feasibility = _ev.power_feasibility_report(
            max(
                1,
                int(
                    measurement.get("promotion_suite_n")
                    or measurement.get("screening_smoke_n")
                    or 3
                ),
            ),
            _ev.parse_alpha(gate.get("alpha") if isinstance(gate, Mapping) else None),
        )
        # Chunk plan (records per bounded run, locked run budget) stamped
        # before execution so every chunk of the measurement is preregistered.
        measurement_chunk_plan: dict[str, Any] | None = (
            dict(chunk_plan) if chunk_plan else _promotion_chunk_plan(pol)
        )
    else:
        claim_class = str(defaults.get("claim_class_screening") or "diagnostic")
        # Screening does not calibrate empirical quality/latency with Lean, but
        # its decision-bearing exact runtime invariants are still preregistered.
        metric_expectations_sha = locked_screening_expectations_sha256()
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
        power_feasibility = None
        measurement_chunk_plan = None

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
                "solver_energy_loss_weight": 0.0,
                "solver_energy_decode_weight": 0.0,
                "legal_edit_hazard_loss_weight": 0.0,
                "legal_edit_hazard_decode_weight": 0.0,
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
        selection_rule=(
            SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST
            if sum(arm.role == "candidate" for arm in arms) > 1
            else None
        ),
        seeds=seeds,
        budget=CampaignBudget(
            max_experiments=1,
            max_wall_minutes=stage_wall_minutes_for_role(pol, role),
            max_gpu_hours=_screening_max_gpu_hours(role=str(role)),
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
        power_feasibility=power_feasibility,
        measurement_chunk_plan=measurement_chunk_plan,
        source_commit=commit,
        source_dirty=False,
        author="autotrain-continuous-driver",
    )




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
    extra_skip_slugs: frozenset[str] = frozenset(),
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

    _integrate_origin_main(
        cwd=cwd, root=root, loop_id=loop_id, deadline=deadline
    )
    if sync_git and startup_commit is not None:
        integrated = _git(
            "rev-parse",
            "HEAD",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-integration-head",
        )
        if integrated != startup_commit:
            raise _CodeUpdated(
                f"integrated {integrated}; restart stale process from {startup_commit}"
            )
    dirty = _git(
        "status",
        "--porcelain",
        cwd=cwd,
        deadline=deadline,
        root=root,
        loop_id=loop_id,
        stage="sync-clean-status",
    )
    if dirty:
        # Continuous-only closeout dirt is driver-owned; foreign WIP still fails.
        _self_heal_continuous_dirty_tree(cwd=cwd, root=root, loop_id=loop_id)
        dirty = _git(
            "status",
            "--porcelain",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-clean-status-recheck",
        )
    if dirty:
        raise RuntimeError("loop worktree is dirty; continuous requires a clean tree")
    try:
        upstream = _git(
            "rev-parse",
            "origin/main",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-upstream-head",
        )
    except Exception:  # noqa: BLE001 — missing origin/main: continue on HEAD
        upstream = _git(
            "rev-parse",
            "HEAD",
            cwd=cwd,
            deadline=deadline,
            root=root,
            loop_id=loop_id,
            stage="sync-upstream-head-fallback",
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
    upstream = _upstream_commit_for_init(
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        upstream=upstream,
        integration=integration,
        deadline=deadline,
    )

    # Terminal governance: a parked regime verdict short-circuits the cycle
    # until its deterministic resume predicate (bank-identity change) holds.
    parked_status = _check_regime_parked(root=root, loop_id=loop_id, cwd=cwd)
    if parked_status is not None:
        return parked_status

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
    # Single soft-unblock owner before the hard prereq gate.
    if require_action_receipts:
        try:
            report = self_heal_unblock_loop(
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                campaign_id=pred,
            )
            if report.get("hard_pending"):
                detail = ", ".join(
                    f"{h.get('index')}:{h.get('kind')}"
                    for h in report["hard_pending"]
                    if h.get("kind") != "foreign_dirty_tree"
                )
                foreign = [
                    h
                    for h in report["hard_pending"]
                    if h.get("kind") == "foreign_dirty_tree"
                ]
                if foreign:
                    raise RuntimeError(
                        "loop worktree is dirty; continuous requires a clean tree"
                    )
                if detail:
                    raise RuntimeError(
                        f"predecessor {pred} has unacknowledged actions: {detail}"
                    )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"SELF_HEAL_UNBLOCK_WARN predecessor={pred} err={exc!r}", flush=True)
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
    reconciled_replays = _reconcile_completed_confirmation_replays(root, queue_entries)
    recovered = _recover_interrupted_champion_entries(root, queue_entries)
    harness_rearmed = _reopen_harness_blocked_champions(
        root, queue_entries, integration_commit=integration
    )
    revalidated = _revalidate_open_champion_entries(root, queue_entries)
    confirmations_revalidated = _revalidate_confirmed_champion_entries(
        root, queue_entries
    )
    # Harness / climb-policy / locked-expectations changes invalidate climb
    # promotions until re-certified under current dispose rules.
    promotions_recertified = _recertify_promoted_champion_entries(
        root, loop_id, queue_entries
    )
    if (
        recipes_refreshed
        or reconciled_replays
        or recovered
        or harness_rearmed
        or revalidated
        or confirmations_revalidated
        or promotions_recertified
    ):
        _write_champion_queue(queue_path, queue_entries)
    replayed_confirmation = _confirmation_replay_entry(queue_entries, replay)
    _load_dynamic_thrash_arms(root, loop_id)
    recent_exhausted = _recent_completed_nonpositive_slugs(root, pred)
    screening_primary = primary_for_role(policy, "screening")
    screening_claim_class = str(
        policy.defaults.get("claim_class_screening") or "diagnostic"
    )
    current_eval_version = default_eval_version()
    predecessor_generation = None
    if pred:
        pred_dir = root / pred
        pred_delivery = _read_json(pred_dir / "sdlc_delivery.json")
        pred_candidate = str(pred_delivery.get("candidate_id") or "")
        pred_knobs = _load_experiment_knobs(pred_dir, pred_candidate)
        if not pred_knobs:
            pred_knobs = _matrix_experiment_knobs(
                _read_json(pred_dir / "matrix-proposal.json"), pred_candidate
            )
        if pred_knobs:
            predecessor_generation = pred_knobs.get("data_generation")
    timeout_retired, selector_signal_sources = _sync_reproduced_timeout_retirements(
        root,
        loop_id,
        pred,
        policy=policy,
        train_version=train_version,
        eval_version=current_eval_version,
        primary_metric=str(screening_primary["metric"]),
        direction=str(screening_primary["direction"]),
        claim_class=screening_claim_class,
        data_generation=predecessor_generation,
    )
    skip_slugs = (
        _skip_arm_slugs(queue_entries, integration_commit=integration)
        | recent_exhausted
        | timeout_retired
        | extra_skip_slugs
        | _stagnation_skip_slugs(root, loop_id)
    )
    # Causal-family CAP must not hard-kill thrash when multi-seed-open arms
    # remain (confirm rejects on noisy fixture burned literal-close CAP while
    # the arm was still multi-seed-open → bank exhaust with no promote head).
    thrash_open = _thrash_bank_open_slugs(recent_exhausted)
    if thrash_open and not (thrash_open - skip_slugs):
        soft_skip = (
            _skip_arm_slugs(
                queue_entries,
                integration_commit=integration,
                include_causal_cap=False,
            )
            | recent_exhausted
            | timeout_retired
            | extra_skip_slugs
            | _stagnation_skip_slugs(root, loop_id)
        )
        if thrash_open - soft_skip:
            relaxed = sorted(thrash_open - soft_skip)
            print(
                "THRASH_CAUSAL_CAP_RELAX "
                f"reopened={relaxed} "
                "reason=multi_seed_open_arms_remain_after_confirm_cap",
                flush=True,
            )
            skip_slugs = soft_skip
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
    saturation_state: dict[str, Any] | None = None
    if cycle_intent == "screening" and replay is None:
        saturation_state = _screening_saturation_state(
            root,
            loop_id,
            policy=policy,
            excluded_slugs=timeout_retired | set(extra_skip_slugs),
        )
        if saturation_state is not None:
            pending = list(saturation_state["pending_regimes"])
            heal_open = _selectable_process_arm(
                root, loop_id, predecessor_campaign_id=pred
            )
            if not pending and not heal_open:
                parked = _park_screening_saturation(
                    root=root,
                    loop_id=loop_id,
                    campaign_id=str(pred or "screening-saturation"),
                    cycle_index=idx,
                    policy=policy,
                    ranked_regimes=saturation_state["ranked_regimes"],
                    cwd=cwd,
                )
                if parked:
                    return parked
            elif pending:
                selected = pending[0]
                skip_slugs = (
                    {
                        slug
                        for slug, _hypothesis, _extras in _all_screening_arm_bank()
                        if slug != selected
                    }
                    | timeout_retired
                    | set(extra_skip_slugs)
                )
                print(
                    "SCREENING_SATURATION_RECOVERY "
                    f"streak={saturation_state['tie_streak']} "
                    f"trigger_cycle={saturation_state['trigger_cycle']} "
                    f"selected={selected} pending={pending}",
                    flush=True,
                )
            elif heal_open:
                print(
                    "SCREENING_SATURATION_HEAL_OPEN "
                    f"streak={saturation_state['tie_streak']}",
                    flush=True,
                )
    if cycle_intent == "screening" and replay is None:
        smoke_n, ss_report = _screening_n_report(policy)
        if isinstance(ss_report, dict) and (
            ss_report.get("must_generate") or int(smoke_n) <= 0
        ):
            print(
                "SCREENING_N_DEFICIT "
                f"smoke_n={smoke_n} n_min={ss_report.get('n_min')} "
                f"binding={ss_report.get('binding_constraints')}",
                flush=True,
            )
            _self_heal_rebuild_screening_eval(
                cwd=cwd, root=root, loop_id=loop_id, campaign_id=pred
            )
            smoke_n, ss_report = _screening_n_report(policy)
        if int(smoke_n) <= 0:
            if not pred:
                raise RuntimeError(
                    "screening n infeasible (empty certified range); "
                    "generate smoke records before the first cycle"
                )
            return _park_screening_n_deficit(
                root=root,
                loop_id=loop_id,
                campaign_id=pred,
                cycle_index=idx,
                report=ss_report if isinstance(ss_report, dict) else {},
            )
    # When multi-seed thrash bank is empty but a retryable promote head still
    # exists (confirmed / promotion_inconclusive / harness_failure), do not hard
    # block the loop — spend a promote slot. Observed: scaffold-prefix was the
    # only open thrash arm, held in harness_failure by deadline_reserve skips,
    # and every screening cycle raised bank-exhausted.
    if cycle_intent == "screening" and promoting_champion is None:
        leftover = _thrash_bank_open_slugs(recent_exhausted) - skip_slugs
        heal_open = _selectable_process_arm(
            root, loop_id, predecessor_campaign_id=pred
        )
        if (
            _terminal_park_on_exhaust()
            and pred
            and _open_slugs_are_snapshot_leftovers(leftover)
            and not heal_open
        ):
            # Isolate OFAT is done. Do not smoke-screen unused snapshot slugs
            # (c96/c120/c78) as if they were a new hill. Do not tombstone a
            # just-registered unused I10 process arm (empty leftover used to
            # count as snapshot leftovers and killed the resume arm).
            print(
                "SELF_HEAL_BANK_EXHAUST parked reason=snapshot_leftovers_before_select "
                f"leftover={sorted(leftover)}",
                flush=True,
            )
            if leftover:
                _retire_i10_heal_arm(
                    root, loop_id, reason="snapshot_leftovers_before_select"
                )
            return _park_screening_saturation(
                root=root,
                loop_id=loop_id,
                campaign_id=pred,
                cycle_index=idx,
                policy=policy,
                ranked_regimes=sorted(recent_exhausted | leftover),
                cwd=cwd,
            )
        try:
            _select_recommended_slug(cycle, skip=skip_slugs, root=root, loop_id=loop_id)
        except RuntimeError as exc:
            if _BANK_EXHAUST_MSG not in str(exc):
                raise
            retry_head = _queue_head_confirmed(queue_entries)
            if retry_head is not None:
                promoting_champion = retry_head
                cycle_intent = "promote"
                role = "promotion"
                print(
                    "BANK_EXHAUST_PROMOTE_FALLBACK "
                    f"entry_id={retry_head.get('entry_id')} "
                    f"status={retry_head.get('status')} "
                    "reason=retryable_promote_head_while_thrash_bank_empty",
                    flush=True,
                )
            elif _self_heal_thrash_bank_exhaust(
                root,
                loop_id,
                closed=recent_exhausted,
                skip=skip_slugs,
                predecessor_campaign_id=pred,
            ):
                # Dynamic thrash successors now in the process bank — continue
                # screening without a human re-prompt.
                _select_recommended_slug(
                    cycle, skip=skip_slugs, root=root, loop_id=loop_id
                )
            elif _terminal_park_on_exhaust() and pred:
                try:
                    return _park_screening_saturation(
                        root=root,
                        loop_id=loop_id,
                        campaign_id=pred,
                        cycle_index=idx,
                        policy=policy,
                        ranked_regimes=sorted(recent_exhausted),
                        cwd=cwd,
                    )
                except RuntimeError:
                    raise RuntimeError(_BANK_EXHAUST_MSG) from exc
            else:
                raise
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
            # Harness-blocked heads are never model-invalidated by attempt caps.
            # Park as harness_failure (retry after harness fix); only complete
            # measurement rejects may become promotion_failed.
            prior_reasons = list(promoting_champion.get("resolve_reasons") or [])
            harness_blocked = bool(
                promoting_champion.get("last_harness_failure")
                or promoting_champion.get("status") == "harness_failure"
                or _reasons_are_harness_incomplete_only(prior_reasons)
            )
            if harness_blocked:
                _update_champion_status(
                    root=root,
                    loop_id=loop_id,
                    entry_id=str(promoting_champion["entry_id"]),
                    status="harness_failure",
                    confirm_campaign_id=campaign_id,
                    confirm_cycle_index=cycle,
                    resolve_reasons=[
                        "promote_harness_parked:incomplete_not_model_reject",
                        f"promote_attempts_paused:{attempts}>{_MAX_PROMOTE_ATTEMPTS}",
                        *prior_reasons,
                    ],
                )
                # Reset attempt budget so a later integration change can retry.
                path = _champion_queue_path(root, loop_id)
                entries = _load_champion_queue(path)
                for row in entries:
                    if row.get("entry_id") == promoting_champion.get("entry_id"):
                        row["promote_attempts"] = 0
                        row["last_harness_failure"] = True
                        row["harness_failure_integration_commit"] = integration
                        break
                _write_champion_queue(path, entries)
                print(
                    "CHAMPION_PROMOTE_HARNESS_PARK "
                    f"entry_id={promoting_champion.get('entry_id')} "
                    f"attempts={attempts} max={_MAX_PROMOTE_ATTEMPTS} "
                    "(incomplete — not a model reject; retry after harness fix)",
                    flush=True,
                )
            else:
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
    multi_arm_cfg = _multi_arm_measurement(policy)
    multi_arm_constraint: str | None = None
    fitted_candidate_count = 1
    if cycle_intent == "screening" and replay is None:
        fitted_candidate_count, multi_arm_constraint = _fit_screening_candidate_count(
            max_candidates=int(multi_arm_cfg["max_arms_per_cycle"]),
            arm_wall_seconds=float(arm_wall_minutes) * 60.0,
            stage_remaining_seconds=max(0.0, deadline - time.monotonic()),
        )
    campaign_max_experiments = (
        1 + fitted_candidate_count
        if cycle_intent == "screening" and replay is None
        else 3
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
        str(campaign_max_experiments),
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
    eval_version = current_eval_version
    if selector_signal_sources:
        _persist_selector_harness_signal(
            root, campaign_id, loop_id, selector_signal_sources
        )
    # Load predecessor matrix feedback only for confirm/promote successors.
    # Thrash bank rotation is NOT a diagnosis successor of the last handoff
    # campaign: continuous pred is last *handoff* campaign, while hypothesize
    # walks full loop lineage and may bind a different formed matrix (e.g.
    # incomplete next cycle with partial feedback). Binding handoff feedback
    # into thrash matrices causes:
    #   agent hypothesis matrix conflicts with supplied feedback ids
    feedback: list[dict] = []
    previous_matrix_id = None
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
    # Confirm/promote/replay matrices may acknowledge handoff-predecessor
    # feedback. Thrash must not — see comment above (lineage ≠ handoff pred).
    bind_pred_feedback = (
        confirm_levers is not None or promote_levers is not None or replay is not None
    )
    if bind_pred_feedback and pred:
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
        if previous_matrix_id:
            feedback = [
                item for item in feedback if item.get("matrix_id") == previous_matrix_id
            ]
    # Dual-regime thrash decision (screening only). Confirm/promote freeze levers.
    thrash_regime: ThrashRegimeDecision | None = None
    initialize_from: str | None = None
    if confirm_levers is None and promote_levers is None and replay is None:
        loop_dir = _loop_champion_dir(root, loop_id)
        _ensure_climb_champion(
            root=root,
            loop_id=loop_id,
            queue_entries=queue_entries,
            eval_data_manifest_sha=None,
            policy=policy,
        )
        parked_epochs = _park_champion_epochs_if_needed(policy, loop_dir)
        if parked_epochs:
            return parked_epochs
        warm = _warm_start_policy(policy)
        ckpt = climb_champion_checkpoint_path(loop_dir)
        sidecar = load_climb_champion(loop_dir)
        if sidecar is not None and ckpt.is_file() and warm.get("enabled", True):
            initialize_from = str(ckpt)
        thrash_regime = _screening_regime_decision(
            queue_entries=queue_entries,
            compiler_ms_timeout=_predecessor_compiler_ms_timeout(root, pred),
            root=root,
            loop_id=loop_id,
        )
        print(
            "THRASH_REGIME "
            f"regime={thrash_regime.regime} base={thrash_regime.base_regime} "
            f"timeout_residual={thrash_regime.timeout_residual} "
            f"reason={thrash_regime.reason}",
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
        thrash_regime=thrash_regime,
        root=root,
        loop_id=loop_id,
    )
    # WP-3 preflight gate seam: skeptic checks (prior-attempt dedup, power,
    # concluded families) run before a screening cycle is spent; "block"
    # verdicts skip the arm. Confirm/promote/replay carry frozen recipes and
    # are never re-gated here.
    preflight_payload: dict[str, Any] | None = None
    if rec_slug is not None and replay is None:
        screening_primary = primary_for_role(policy, "screening")

        def _preflight_reselect(augmented_skip: set[str]) -> str | None:
            return _select_cycle_slug(
                cycle,
                predecessor_priority=None,
                skip=augmented_skip,
                has_confirm_levers=False,
                has_promote_levers=False,
                thrash_regime=thrash_regime,
                root=root,
                loop_id=loop_id,
            )

        rec_slug, preflight_payload = _preflight_screening_slug(
            rec_slug,
            steps=steps,
            endpoint_metric=str(
                screening_primary.get("metric") or "smoke.structural_similarity"
            ),
            minimum_effect=(
                float(screening_primary["minimum_effect"])
                if screening_primary.get("minimum_effect") is not None
                else None
            ),
            skip=set(skip_slugs),
            reselect=_preflight_reselect,
        )
        if preflight_payload is not None:
            try:
                (camp_dir / "preflight.json").write_text(
                    json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                print(f"PREFLIGHT_WARN persist err={exc!r}", flush=True)
    promotion_chunk_plan: dict[str, Any] | None = (
        _promotion_chunk_plan(policy, root=root) if role == "promotion" else None
    )
    if promotion_chunk_plan is not None:
        print(
            "PROMOTION_CHUNK_PLAN "
            f"suites={','.join(promotion_chunk_plan['suites'])} "
            f"suite_n={promotion_chunk_plan['suite_n']} "
            f"total_record_n={promotion_chunk_plan['total_record_n']} "
            f"per_record_s={promotion_chunk_plan['per_record_seconds']:.3f} "
            f"p95_s={promotion_chunk_plan['measured_decode_p95_seconds']} "
            f"records_per_run={promotion_chunk_plan['records_per_run']} "
            f"run_n={promotion_chunk_plan['run_n']} "
            f"chunk_wall_s={promotion_chunk_plan['chunk_wall_seconds']:.0f}",
            flush=True,
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
        feedback=(feedback or None) if bind_pred_feedback else None,
        previous_matrix_id=previous_matrix_id if bind_pred_feedback else None,
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
        thrash_regime=thrash_regime,
        initialize_from=initialize_from,
        telemetry_root=root,
        predecessor_campaign_id=pred,
        chunk_plan=promotion_chunk_plan,
    )
    if saturation_state is not None:
        regime_payload = matrix.setdefault("thrash_regime", {})
        if isinstance(regime_payload, dict):
            recovery_payload = dict(saturation_state)
            recovery_payload["selected_regime"] = rec_slug
            recovery_payload["pending_regimes"] = [
                slug for slug in saturation_state["pending_regimes"] if slug != rec_slug
            ]
            regime_payload["screening_saturation"] = recovery_payload
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
        regime_tag = (
            thrash_regime.regime if thrash_regime is not None else REGIME_ISOLATE
        )
        print(
            f"THRASH_ROTATE cycle={cycle} regime={regime_tag} "
            f"recommended={rec_slug} skip={sorted(skip_slugs)}",
            flush=True,
        )
    # Defense in depth: thrash matrices never carry feedback/predecessor binds.
    # AgentHypothesisProvider rebinds from live loop lineage on hypothesize.
    if (
        promote_levers is None
        and confirm_levers is None
        and replay is None
        and (matrix.get("feedback_ids") or matrix.get("predecessor_matrix_id"))
    ):
        matrix = dict(matrix)
        stripped_ids = list(matrix.pop("feedback_ids", None) or [])
        matrix.pop("predecessor_matrix_id", None)
        print(
            "THRASH_MATRIX_STRIP_FEEDBACK "
            f"campaign={campaign_id} stripped_ids={stripped_ids}",
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
        bool(item["manifest"].formal_obligations) for item in replay_manifests.values()
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
    multi_arm_skip: list[dict[str, str]] = []
    screening_candidate_ids: list[str] = [candidate_eid]
    selection_rule_locked: str | None = None
    screening_multi = (
        cycle_intent == "screening"
        and replay is None
        and confirm_levers is None
        and promote_levers is None
        and fitted_candidate_count > 1
    )
    if screening_multi:
        screening_candidate_ids, multi_arm_skip = _screening_multi_arm_ids(
            matrix=matrix,
            control_id=control_eid,
            recommended_id=candidate_eid,
            fitted_candidates=fitted_candidate_count,
            by_id=by_id,
        )
        if not screening_candidate_ids:
            screening_candidate_ids = [candidate_eid]
        candidate_eid = screening_candidate_ids[0]
        order = [control_eid, *screening_candidate_ids]
        selection_rule_locked = str(multi_arm_cfg["selection_rule"])
    else:
        order = _counterbalanced_arm_order(
            control_eid,
            candidate_eid,
            cycle_index=cycle,
            seed=arm_seed,
            promotion_replicate_index=promotion_replicate_index,
        )
    scheduled_order = list(order)
    _bind_expected_arms(
        root=root,
        campaign_id=campaign_id,
        matrix_path=matrix_path,
        control_id=control_eid,
        candidate_id=candidate_eid,
        arm_order=scheduled_order,
        candidate_ids=screening_candidate_ids[1:] if screening_multi else None,
        selection_rule=selection_rule_locked,
    )
    if screening_multi:
        swarm_exps = [
            json.loads(by_id[eid].read_text(encoding="utf-8"))
            for eid in (control_eid, *screening_candidate_ids)
            if eid in by_id
        ]
        _lock_screening_multi_arm_campaign(
            root=root,
            campaign_id=campaign_id,
            experiments=swarm_exps,
            control_id=control_eid,
            candidate_ids=screening_candidate_ids,
            seeds=(arm_seed,),
            selection_rule=selection_rule_locked
            or SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
            commit=integration,
            role=role,
            policy=policy,
        )
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
            # Promote arm never runs → no terminal hypothesizer feedback via
            # `run --execute`. Record it so successor hypothesize can chain.
            _run(
                [
                    *ar,
                    "block",
                    "--campaign-id",
                    campaign_id,
                    "--experiment-id",
                    str(matrix["recommended_experiment_id"]),
                    "--reason",
                    f"promote formal preflight blocked: status={promote_formal_status}",
                ],
                cwd=cwd,
            )
    elif replay_formal_required and (
        promote_formal_status != "proved" or not promote_preflight_sha
    ):
        kind = (
            "TIMEOUT_INCONCLUSIVE"
            if _formal_status_is_timeout(promote_formal_status)
            else "BLOCK"
        )
        print(
            f"REPLAY_FORMAL_{kind} skip_execute status={promote_formal_status}",
            flush=True,
        )
        order = []

    arm_count = len({eid for eid in order if eid in by_id})
    if arm_count and not screening_multi:
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
    arm_skipped: dict[str, dict[str, Any]] = {
        row["arm_id"]: {"reason": row["reason"]} for row in multi_arm_skip
    }
    for eid in order:
        if eid in seen or eid not in by_id:
            continue
        pending = []
        pending_seen: set[str] = set()
        for pending_id in order:
            if (
                pending_id in seen
                or pending_id not in by_id
                or pending_id in pending_seen
            ):
                continue
            pending_seen.add(pending_id)
            pending.append(pending_id)
        remaining_seconds = max(0.0, deadline - time.monotonic())
        pending_count = 1 if screening_multi else len(pending)
        required_seconds = (
            pending_count * arm_wall_minutes * 60.0
            + HARNESS_FINALIZATION_RESERVE_SECONDS
        )
        # Epsilon after schedule-margin fit: never skip when remaining is
        # within float/clock noise of the reserved budget.
        if remaining_seconds + 1e-3 < required_seconds:
            reason = "deadline_reserve"
            store = CampaignStore(campaign_id, root)
            for skipped_index, skipped_id in enumerate(pending):
                detail = {
                    "arm_id": skipped_id,
                    "order_index": order.index(skipped_id),
                    "reason": reason,
                    "remaining_seconds": remaining_seconds,
                    "required_seconds": required_seconds,
                    "deadline": deadline,
                    "pending_arm_count": len(pending),
                    "skipped_index": skipped_index,
                }
                store.append_event(
                    "arm_skipped",
                    experiment_id=skipped_id,
                    status="not_started",
                    detail=detail,
                )
                arm_skipped[skipped_id] = detail
            print(
                "ARM_SKIPPED "
                f"reason={reason} pending={len(pending)} "
                f"remaining_s={remaining_seconds:.3f} required_s={required_seconds:.3f}",
                flush=True,
            )
            break
        seen.add(eid)
        exp = json.loads(by_id[eid].read_text(encoding="utf-8"))
        is_promote_arm = cycle_intent == "promote" and (
            eid.endswith("-promote") or "-promote" in eid
        )
        prelocked = camp_dir / "manifests" / f"{eid}.json"
        if eid in replay_manifest_paths:
            man_path = replay_manifest_paths[eid]
        elif screening_multi and eid == candidate_eid and prelocked.is_file():
            man_path = prelocked
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
                chunk_plan=promotion_chunk_plan,
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
        # Teacher-forced NLL is cheap and independent of the decode-heavy
        # quality eval: score it whenever a checkpoint exists, even when the
        # quality eval crashed (2) or timed out (124).
        _attach_screening_eval_nll(camp_dir / "runs" / eid, exit_code=int(code))

    promotion_chunks: dict[str, Any] | None = None
    if promotion_chunk_plan is not None and seen:
        _set_active_stage(root, loop_id, "promotion-chunks")
        chunk_started = time.monotonic()
        promotion_chunks = _run_promotion_eval_chunks(
            cwd=cwd,
            root=root,
            loop_id=loop_id,
            campaign_id=campaign_id,
            camp_dir=camp_dir,
            plan=promotion_chunk_plan,
            experiment_paths={eid: by_id[eid] for eid in seen},
            arm_order=list(dict.fromkeys(eid for eid in order if eid in seen)),
        )
        # Every chunk was its own bounded run (fresh MAX_RUN_SECONDS deadline,
        # eval wall <= MAX_HARNESS_WALL_SECONDS); the cycle clock resumes where
        # it paused so closeout stages keep the budget they had before.
        deadline += time.monotonic() - chunk_started

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
        control_id=control_eid,
        candidate_id=candidate_eid,
        arm_exits=arm_exits,
        arm_skipped=arm_skipped,
    )
    # Keep execution completeness explicit in the handoff/result matrix.  A
    # missing arm is a typed scheduling event, never an inferred model reject.
    delivery = {
        **delivery,
        "arm_exits": arm_exits,
        "arm_skipped": arm_skipped,
    }
    if promotion_chunks is not None:
        delivery = _attach_promotion_chunks(delivery, promotion_chunks)
    if screening_multi:
        direction = str(role_primary.get("direction") or "increase")
        scored: list[tuple[str, float, int]] = []
        per_arm: dict[str, float | None] = {}
        for eid in screening_candidate_ids:
            metrics = _run_metrics(camp_dir, eid)
            leaf = str(effective_primary).rsplit(".", 1)[-1]
            val = metrics.get(effective_primary)
            if val is None:
                val = metrics.get(leaf)
            per_arm[eid] = float(val) if isinstance(val, (int, float)) else None
            if per_arm[eid] is None:
                continue
            scored.append(
                (eid, float(per_arm[eid]), _arm_trainable_params(camp_dir, eid))
            )
        winner = None
        if scored:
            winner = select_best_by_primary_then_smallest(
                scored, direction=direction  # type: ignore[arg-type]
            )
            ctrl_metrics = _run_metrics(camp_dir, control_eid)
            ctrl_val = ctrl_metrics.get(effective_primary)
            if ctrl_val is None:
                ctrl_val = ctrl_metrics.get(str(effective_primary).rsplit(".", 1)[-1])
            win_val = per_arm.get(winner)
            min_eff = float(role_primary.get("minimum_effect") or 0.0)
            beats = False
            if isinstance(ctrl_val, (int, float)) and isinstance(win_val, (int, float)):
                if direction == "decrease":
                    beats = float(ctrl_val) - float(win_val) >= min_eff
                else:
                    beats = float(win_val) - float(ctrl_val) >= min_eff
            if not beats:
                winner = None
        if winner:
            candidate_eid = winner
            delivery["candidate_id"] = winner
        losers = [eid for eid in screening_candidate_ids if eid != winner]
        _exhaust_screening_losers(
            root=root,
            loop_id=loop_id,
            policy=policy,
            matrix=matrix,
            control_id=control_eid,
            loser_ids=losers,
            claim_class=claim_for_role,
            primary_metric=effective_primary,
            direction=direction,
            train_version=train_version,
            eval_version=eval_version,
        )
        delivery["multi_arm"] = {
            "max_arms_per_cycle": int(multi_arm_cfg["max_arms_per_cycle"]),
            "fitted_candidates": fitted_candidate_count,
            "scheduled_candidates": list(screening_candidate_ids),
            "constraint": multi_arm_constraint,
            "selection_rule": selection_rule_locked,
            "winner_id": winner,
            "per_arm_primary": per_arm,
            "size_skipped": multi_arm_skip,
        }
    if (
        replay is not None
        and set(arm_exits) == set(replay_manifests)
        and all(code == 0 for code in arm_exits.values())
        and delivery.get("measurement_complete") is True
    ):
        evidence = (
            str(
                (camp_dir / "campaign.json").relative_to(cwd)
                if (camp_dir / "campaign.json").is_relative_to(cwd)
                else camp_dir / "campaign.json"
            ),
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
        skip_slugs=(
            skip_slugs
            | ({rec_slug} if saturation_state is not None and rec_slug else set())
        ),
        cwd=cwd,
    )
    try:
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
    finally:
        _clear_active_stage(root, loop_id)
    # Measured heal process arms must retire: they outrank OFAT selection and
    # otherwise rematch the same fixture-n incomplete win forever.
    try:
        cand_id = str((delivery or {}).get("candidate_id") or "")
        arm_exits = (delivery or {}).get("arm_exits") or {}
        if (
            _HEAL_RESUME_SLUG in cand_id
            and isinstance(arm_exits, Mapping)
            and arm_exits
            and all(int(v) == 0 for v in arm_exits.values())
        ):
            _retire_i10_heal_arm(
                root,
                loop_id,
                reason=f"complete_measurement:{campaign_id}",
            )
    except Exception as exc:  # noqa: BLE001 — retirement never blocks closeout
        print(f"HEAL_RESUME_RETIRE_WARN err={exc!r}", flush=True)
    print(
        f"CYCLE_COMPLETE {campaign_id} role={role} intent={cycle_intent} "
        f"positive={delivery['positive']}",
        flush=True,
    )
    return campaign_id


def _parse_skip_slugs(raw: str) -> frozenset[str]:
    """Parse a comma-separated ``--skip-slugs`` value into a slug set."""
    return frozenset(slug.strip() for slug in raw.split(",") if slug.strip())


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
    parser.add_argument("--primary-metric", default="smoke.eval_nll")
    parser.add_argument(
        "--skip-slugs",
        default="",
        help=(
            "Comma-separated screening arm slugs to skip in addition to the "
            "local campaign-lineage closure state (e.g. slugs already known "
            "exhausted from prior sessions' committed docs/design results, "
            "since the local lineage under --root resets on a fresh checkout)."
        ),
    )
    args = parser.parse_args(argv)
    extra_skip_slugs = _parse_skip_slugs(args.skip_slugs)
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
    # Startup self-heal: single unblock owner — never chat-prompt for soft thrash.
    try:
        report = self_heal_unblock_loop(
            cwd=cwd,
            root=root,
            loop_id=args.loop_id,
            integration_commit=code_sha,
        )
        if report.get("hard_pending"):
            print(
                f"SELF_HEAL_STARTUP hard_pending={report['hard_pending']}",
                flush=True,
            )
    except Exception as startup_exc:  # noqa: BLE001 — never abort startup heal
        print(f"SELF_HEAL_STARTUP_WARN {startup_exc!r}", flush=True)
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
                _before_cycle, before_campaign = _latest_cycle(root, args.loop_id)
                _receipts_path = root / "loops" / args.loop_id / "heal_receipts.jsonl"
                before_receipts = len(_receipts_path.read_text(encoding="utf-8").splitlines()) if _receipts_path.is_file() else 0
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
                    extra_skip_slugs=extra_skip_slugs,
                )
                pass_outcome = _record_pass_outcome(
                    root=root, loop_id=args.loop_id, before_campaign=before_campaign,
                    before_receipts=before_receipts,
                )
                print(f"PASS_OUTCOME {pass_outcome}", flush=True)
                if pass_outcome == _STALL_FINGERPRINT:
                    # Typed park already written (state=BLOCKED); exit
                    # non-zero so the supervisor's governed backoff applies.
                    return _STALL_EXIT_CODE
            except _CodeUpdated as exc:
                print(f"CODE_UPDATED {exc}; re-executing driver", flush=True)
                os.execv(sys.executable, [sys.executable, *sys.argv])
                raise RuntimeError("driver re-exec unexpectedly returned")
            except Exception as exc:  # noqa: BLE001 - continuous must self-heal next pass
                print(f"CYCLE_ERROR {exc!r}", flush=True)
                cycle_index, _ = _latest_cycle(root, args.loop_id)
                # Single unblock owner — soft thrash never needs a human prompt.
                report = self_heal_unblock_loop(
                    cwd=cwd,
                    root=root,
                    loop_id=args.loop_id,
                    integration_commit=code_sha,
                )
                try:
                    pass_outcome = _record_pass_outcome(
                        root=root, loop_id=args.loop_id,
                        before_campaign=before_campaign if "before_campaign" in locals() else None,
                        before_receipts=before_receipts if "before_receipts" in locals() else 0,
                        typed_action=bool(report.get("hard_pending")),
                        reason=repr(exc)[:300],
                    )
                except Exception as outcome_exc:  # noqa: BLE001 — classifier bugs never mask the cycle error
                    print(f"PASS_OUTCOME_WARN {outcome_exc!r}", flush=True)
                    pass_outcome = "unclassified"
                print(f"PASS_OUTCOME {pass_outcome}", flush=True)
                if pass_outcome == _STALL_FINGERPRINT:
                    return _STALL_EXIT_CODE
                # Legacy string heal for bank/soft identity / document / residual.
                heal_kind = _self_heal_cycle_error(
                    root=root,
                    loop_id=args.loop_id,
                    exc=exc,
                    integration_commit=code_sha,
                    cwd=cwd,
                )
                # Also heal git ancestry drift for supervised continuous worktrees
                # (origin/main advanced while local branch has exclusive commits).
                ancestry_healed = _self_heal_git_ancestry(
                    cwd=cwd, root=root, loop_id=args.loop_id, exc=exc
                )
                actually_healed = bool(
                    heal_kind
                    or ancestry_healed
                    or (
                        _exception_is_soft_continuous(exc)
                        and report.get("soft_healed")
                        and any(
                            k
                            in {
                                "document_closeout",
                                "dirty_tree_closeout",
                                "thrash_timeout_repair_bypass",
                                "thrash_bank_compose",
                                "bank_exhaust_compose",
                            }
                            for k in (report.get("soft_healed") or [])
                        )
                        and (
                            "unacknowledged" in str(exc)
                            or "dirty" in str(exc).lower()
                            or "repair_harness" in str(exc)
                            or "bank" in str(exc).lower()
                            or _BANK_EXHAUST_MSG in str(exc)
                            or any(m in str(exc).lower() for m in _BANK_EXHAUST_MARKERS)
                        )
                    )
                )
                if actually_healed and not report.get("hard_pending"):
                    if heal_kind:
                        _clear_loop_blocker(root, args.loop_id, reason=heal_kind)
                    print(
                        f"SELF_HEAL continue kind={heal_kind or ancestry_healed or report.get('soft_healed')}",
                        flush=True,
                    )
                    time.sleep(1)
                    continue
                count = _record_cycle_failure(
                    root=root,
                    loop_id=args.loop_id,
                    exc=exc,
                    cycle_index=cycle_index,
                )
                if report.get("hard_pending"):
                    return 2
                # Soft-class exceptions that still could not be healed: exit 0 so
                # the supervisor restarts after a fresh unblock, without BLOCKED.
                if _exception_is_soft_continuous(exc):
                    if args.supervised:
                        return 0
                    time.sleep(1)
                    continue
                if args.supervised or count >= 3:
                    return 2
                time.sleep(1)
                continue
        return 0
    finally:
        # WP-3 closeout: refresh the durable evidence store (guarded — the
        # sync script is concurrent work and may not exist; failure only logs).
        try:
            sync_script = cwd / "scripts" / "sync_evidence_store.py"
            if os.path.exists(sync_script):
                proc = subprocess.run(  # noqa: S603 — repo-owned script
                    [sys.executable, str(sync_script)],
                    cwd=cwd,
                    timeout=120,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                print(f"EVIDENCE_STORE_SYNC rc={proc.returncode}", flush=True)
        except Exception as sync_exc:  # noqa: BLE001 — closeout never raises
            print(f"EVIDENCE_STORE_SYNC_WARN {sync_exc!r}", flush=True)
        try:
            owned_kind = _self_heal_loop_owned_generated_dirt(cwd=cwd)
            if owned_kind:
                print(f"SELF_HEAL_LOOP_OWNED_DIRT closeout={owned_kind}", flush=True)
        except Exception as owned_exc:  # noqa: BLE001 — closeout never raises
            print(f"SELF_HEAL_LOOP_OWNED_DIRT_WARN {owned_exc!r}", flush=True)
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
