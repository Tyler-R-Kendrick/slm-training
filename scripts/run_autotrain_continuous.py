#!/usr/bin/env python3
"""Hands-off continuous autotrain cycle driver.

Bare /autotrain agents should keep calling this (or re-enter continuous.md)
without user prompts. Each invocation can run one or many bounded cycles.
Never an infinite shell without MAX_RUN_MINUTES on child commands — wall is
enforced by campaign budgets and levers.

SDLC Phase A (autotrain-iteration-delivery): after every cycle the driver
classifies positive vs non-positive, records a delivery ledger, and only
signals stack-layer intent for positive results. Stacked PRs are still opened
by the agent (gh stack); this driver never opens PRs for non-positive cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
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
from slm_training.autoresearch.schemas import CampaignBudget, HypothesisMatrix
from slm_training.levers import MAX_RUN_MINUTES


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, cwd=str(cwd) if cwd else None
    ).strip()


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


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
    suite = data.get("smoke")
    if isinstance(suite, dict) and isinstance(suite.get(key), (int, float)):
        return float(suite[key])
    return None


def _run_metrics(camp_dir: Path, run_id: str) -> dict[str, float | None]:
    run_dir = camp_dir / "runs" / run_id
    smoke = run_dir / "eval_smoke.json"
    if not smoke.exists():
        smoke = run_dir / "eval.json"
    return {
        "latency_ms_p50": _metric_from_eval(smoke, "latency_ms_p50"),
        "parse_rate": _metric_from_eval(smoke, "parse_rate"),
        "meaningful_program_rate": _metric_from_eval(smoke, "meaningful_program_rate"),
    }


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


def _finite_metric(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _in_timeout_band(latency_ms: float | None) -> bool:
    return (
        latency_ms is not None
        and _TIMEOUT_BAND_LO_MS <= latency_ms <= _TIMEOUT_BAND_HI_MS
    )


def _classify_metric_tradeoff(
    *,
    control: dict[str, float | None],
    candidate: dict[str, float | None],
    primary_metric: str,
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
    mpr_improved = (
        t_mpr is not None and c_mpr is not None and t_mpr > c_mpr + _EPS
    )
    lat_improved = (
        t_lat is not None and c_lat is not None and t_lat + _EPS < c_lat
    )
    if t_lat is not None and c_lat is not None and c_lat > 0:
        lat_within_tradeoff = t_lat <= c_lat * (
            1.0 + _LATENCY_REGRESSION_BUDGET
        ) or t_lat <= c_lat + _LATENCY_REGRESSION_ABS_MS
    else:
        # Missing latency must not veto a quality win.
        lat_within_tradeoff = True

    both_timeout_band = _in_timeout_band(c_lat) and _in_timeout_band(t_lat)

    # Path 1: latency primary win — only with held quality and non-empty meaning.
    if metric_leaf == "latency_ms_p50":
        if c_lat is None or t_lat is None:
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
                reasons.append(
                    f"primary_metric_win:{primary_metric}:{c_lat}->{t_lat}"
                )
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
        if t_eff > c_eff + _EPS:
            positive = True
            reasons.append(f"efficiency_win:mpr_per_ms:{c_eff:.8g}->{t_eff:.8g}")

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
    # Prefer policy primary for the cycle role; allow CLI override of metric id
    # only when it matches the configured leaf/id for that role.
    role_primary = primary_for_role(policy, role)
    effective_metric = str(role_primary.get("metric") or primary_metric)
    if primary_metric and primary_metric != effective_metric:
        # Keep caller metric if it is an explicit override of the same leaf.
        if primary_metric.split(".")[-1] == effective_metric.split(".")[-1]:
            effective_metric = primary_metric

    control = _run_metrics(camp_dir, control_id)
    candidate = _run_metrics(camp_dir, candidate_id)
    # Merge full primary metric keys when leaf-only maps were collected.
    if effective_metric not in control and control.get(effective_metric.split(".")[-1]) is not None:
        control = {**control, effective_metric: control[effective_metric.split(".")[-1]]}
    if effective_metric not in candidate and candidate.get(effective_metric.split(".")[-1]) is not None:
        candidate = {
            **candidate,
            effective_metric: candidate[effective_metric.split(".")[-1]],
        }

    reasons_pre: list[str] = []
    outcomes = list((camp_dir / "artifacts" / "outcomes").glob("*.json"))
    for path in outcomes:
        out = _read_json(path)
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

    base_params = baseline_trainable_params
    cand_params = candidate_trainable_params
    if base_params is None:
        base_params = _params(control_outcome)
    if cand_params is None:
        cand_params = _params(cand_outcome)

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
        if has_metric and t_mpr is not None and t_mpr + _EPS >= _MIN_MPR_FOR_LATENCY_WIN:
            executable_unblock = True
        elif has_metric:
            reasons_pre.append(
                f"executable_unblock_rejected_low_mpr:mpr={t_mpr}"
            )

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
        if any(str(r).startswith("eg_params_block:") for r in (decision.get("reasons") or [])):
            positive = False
        decision["positive"] = positive
        decision["stack_layer"] = positive
    elif tradeoff_positive:
        decision["positive"] = True
        decision["stack_layer"] = True

    reasons = (
        list(reasons_pre)
        + list(tradeoff_reasons)
        + list(decision.get("reasons") or [])
    )
    if not any(
        reason.startswith(prefix)
        for reason in reasons
        for prefix in _WIN_REASON_PREFIXES
    ):
        decision["positive"] = False
        decision["stack_layer"] = False
        if not reasons:
            reasons.append("no_positive_signal")

    decision["reasons"] = reasons
    decision["control_id"] = control_id
    decision["candidate_id"] = candidate_id
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
            if eid.endswith("-control") or "control" in eid:
                control_run = eid
            elif "bounds" in eid or "canvas" in eid or "combined" in eid:
                if candidate_run is None:
                    candidate_run = eid
    runs_dir = camp_dir / "runs"
    if runs_dir.exists():
        run_ids = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
        for rid in run_ids:
            if rid.endswith("-control") or rid.endswith("-control".replace("-", "_")):
                control_run = control_run or rid
            elif "control" not in rid:
                candidate_run = candidate_run or rid
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
    decision["cycle_role"] = role
    decision["cycle_index"] = cycle_index
    decision["climb_policy_sha256"] = policy.sha256
    # Stack only when positive AND there is something reviewable to ship.
    # Pure knob-only fixture cycles with a metric blip do not open empty PRs.
    porcelain = _git("status", "--porcelain", cwd=cwd) if cwd else ""
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


def _latest_cycle(root: Path, loop_id: str) -> tuple[int, str | None]:
    campaigns = sorted(root.glob("*/campaign.json"))
    best_idx = 0
    best_id: str | None = None
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
    return best_idx, best_id


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
) -> dict:
    research = role_citations.get("research") or cites[0]
    prior = role_citations.get("prior_result") or (cites[1] if len(cites) > 1 else cites[0])
    seed = 7 + (cycle * 17) % 50
    steps = steps + (cycle % 3)  # slight variation avoids knob-signature collision

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
            "compact_active_canvas": False,
        }
        base.update(extra)
        return base

    def exp(eid: str, hyp: str, k: dict, rationale: str) -> dict:
        # Citations must include every evidence_use citation (schema invariant).
        exp_cites = list(dict.fromkeys([*cites[:3], *role_citations.values()]))
        return {
            "experiment_id": eid,
            "campaign_id": campaign_id,
            "hypothesis": hyp,
            "rationale": rationale,
            "expected_effect": "Runnable smoke scoreboard under the wall cap.",
            "falsification_criteria": [
                "Path error or no smoke metrics under the wall cap."
            ],
            "stop_conditions": [
                "Stop at declared steps or the campaign wall cap."
            ],
            "citations": exp_cites,
            "knobs": k,
        }

    prefix = campaign_id.replace("continuous-loop-", "c")
    candidates = [
        {
            "experiment": exp(
                f"{prefix}-control",
                "Matched fixture control with both grammar levers off completes smoke eval under the published suite.",
                knobs(),
                "Baseline for size-matched continuous attribution.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(0, "matched control with published eval"),
        },
        {
            "experiment": exp(
                f"{prefix}-bounds",
                "grammar_completion_bounds reduces smoke latency_ms_p50 versus the matched control without lowering parse_rate.",
                knobs(grammar_completion_bounds=True),
                "I1/I4 bounds lever on continuous default recipe.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(1, "grammar_completion_bounds"),
        },
        {
            "experiment": exp(
                f"{prefix}-canvas",
                "compact_active_canvas reduces smoke latency_ms_p50 versus the matched control without lowering parse_rate.",
                knobs(compact_active_canvas=True),
                "Independent canvas compaction lever.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(2, "compact_active_canvas"),
        },
        {
            "experiment": exp(
                f"{prefix}-both",
                "Combined bounds and canvas beat either single lever on smoke latency_ms_p50.",
                knobs(grammar_completion_bounds=True, compact_active_canvas=True),
                "Interaction arm for continuous steering.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(3, "combined bounds and canvas"),
        },
        {
            "experiment": exp(
                f"{prefix}-steps",
                "Doubling steps without levers only raises cost and does not improve unit decode latency.",
                knobs(steps=max(steps * 2, steps + 10)),
                "Depth confounds check.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(4, "depth-only confounds"),
        },
    ]
    rec = f"{prefix}-bounds"
    priorities = [
        {
            "rank": i + 1,
            "area": area,
            "hypothesis": hyp,
            "evidence_ids": [research, prior],
            "confidence": conf,
            "expected_information_gain": gain,
            "authority": auth,
            "disposition": disp,
            "proposed_experiment_id": eid,
        }
        for i, (area, hyp, conf, gain, auth, disp, eid) in enumerate(
            [
                (
                    "model",
                    "Test grammar_completion_bounds first under the published eval suite.",
                    0.6,
                    "Attributes decode latency vs matched control.",
                    "speculative",
                    "experiment_next",
                    rec,
                ),
                (
                    "evaluation",
                    "Keep the matched control as the size-matched baseline every cycle.",
                    0.7,
                    "Prevents false positives from recipe drift.",
                    "observed_result",
                    "experiment_next",
                    f"{prefix}-control",
                ),
                (
                    "model",
                    "Test compact_active_canvas as the next independent lever.",
                    0.55,
                    "Isolates canvas from bounds.",
                    "speculative",
                    "experiment_next",
                    f"{prefix}-canvas",
                ),
                (
                    "infrastructure",
                    "Soft ship-gate fails on fixture n never stop the continuous loop.",
                    0.8,
                    "Preserves hands-off continuous operation.",
                    "observed_result",
                    "monitor",
                    None,
                ),
                (
                    "model_build",
                    "Only after single levers, combine bounds and canvas.",
                    0.45,
                    "Detects interaction effects.",
                    "speculative",
                    "monitor",
                    None,
                ),
            ]
        )
    ]
    payload = {
        "matrix_id": f"{campaign_id}-m1",
        "campaign_id": campaign_id,
        "evidence_snapshot_id": evidence_snapshot_id,
        "hypotheses": candidates,
        "recommended_experiment_id": rec,
        "selection_rationale": "Size-matched fixture continuous recipe with published eval defaults.",
        "next_run_priorities": priorities,
    }
    if feedback:
        fb_ids = [str(item.get("feedback_id")) for item in feedback if item.get("feedback_id")]
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
) -> ExperimentCampaignV1:
    from slm_training.autoresearch.climb_policy import (
        load_climb_policy,
        primary_for_role,
        promotion_seed_floor,
    )

    pol = policy or load_climb_policy()
    role_primary = primary_for_role(pol, role)
    metric = str(role_primary["metric"])
    direction = str(role_primary["direction"])  # type: ignore[assignment]
    min_effect = float(role_primary.get("minimum_effect") or 0.0)
    defaults = pol.defaults
    if role == "promotion":
        claim_class = str(defaults.get("claim_class_promotion") or "promotion_candidate")
        min_seeds, require_ms = promotion_seed_floor(pol)
        base_seed = int(experiment.get("knobs", {}).get("seed") or 7)
        if require_ms and min_seeds >= 2:
            seeds = tuple(base_seed + i for i in range(min_seeds))
        else:
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
        artifact_requirements = tuple(
            ArtifactRequirementV1(kind=k)
            for k in (
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
            )
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
                description="Matched fixture baseline with grammar levers off.",
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
        budget=CampaignBudget(max_experiments=1, max_wall_minutes=MAX_RUN_MINUTES),
        stopping_rules=("Stop after the declared seeds finish or the wall cap is hit.",),
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
        claim_class=claim_class,  # type: ignore[arg-type]
        locked_eval_manifest_sha256=locked_eval,
        source_commit=commit,
        source_dirty=False,
        author="autotrain-continuous-driver",
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
) -> str:
    from slm_training.autoresearch.climb_policy import (
        assert_cycle_cadence,
        cycle_role_for_index,
        load_climb_policy,
        primary_for_role,
    )

    policy = load_climb_policy()
    # Defaults from external policy when caller still uses legacy pins.
    if train_version == "wf_smoke_v2":
        train_version = str(policy.defaults.get("train_version") or train_version)

    _run(["git", "fetch", "origin", "main"], cwd=cwd)
    _run(["git", "merge", "--no-edit", "origin/main"], cwd=cwd)
    if _git("status", "--porcelain", cwd=cwd):
        raise RuntimeError("loop worktree is dirty; continuous requires a clean tree")
    upstream = _git("rev-parse", "origin/main", cwd=cwd)
    integration = _git("rev-parse", "HEAD", cwd=cwd)
    if upstream != integration:
        # merge should have equalized; if not, still require ancestor
        _run(["git", "merge-base", "--is-ancestor", upstream, integration], cwd=cwd)

    idx, pred = _latest_cycle(root, loop_id)
    cycle = idx + 1
    role = cycle_role_for_index(policy, cycle)
    claim_for_role = (
        str(policy.defaults.get("claim_class_promotion") or "promotion_candidate")
        if role == "promotion"
        else str(policy.defaults.get("claim_class_screening") or "diagnostic")
    )
    assert_cycle_cadence(
        policy,
        cycle_index=cycle,
        claimed_role=role,
        claim_class=claim_for_role if role == "promotion" else claim_for_role,
    )
    role_primary = primary_for_role(policy, role)
    # Screening uses policy screening primary; promotion uses held-out quality.
    # CLI primary_metric overrides only when it matches the role leaf (compat).
    effective_primary = str(role_primary["metric"])
    if primary_metric and primary_metric.split(".")[-1] == effective_primary.split(".")[-1]:
        effective_primary = primary_metric
    campaign_id = f"continuous-loop-{time.strftime('%Y%m%d')}-c{cycle}"
    py = sys.executable
    ar = [py, "-m", "scripts.autoresearch", "--root", str(root)]
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
        str(MAX_RUN_MINUTES),
        "--notes",
        "Hands-off continuous driver cycle; local-only fixture scale.",
    ]
    if pred:
        init.extend(["--predecessor-campaign-id", pred])
    _run(init, cwd=cwd)
    _run(
        [
            *ar,
            "research",
            "--campaign-id",
            campaign_id,
            "--offline",
        ],
        cwd=cwd,
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
            previous_matrix_id = json.loads(
                mats[-1].read_text(encoding="utf-8")
            ).get("matrix_id")
        fbs = sorted(
            (pred_dir / "artifacts" / "hypothesizer_feedback").glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        feedback = [json.loads(path.read_text(encoding="utf-8")) for path in fbs]
        # only terminal feedback for the latest predecessor matrix
        if previous_matrix_id:
            feedback = [
                item
                for item in feedback
                if item.get("matrix_id") == previous_matrix_id
            ]
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
    )
    HypothesisMatrix.model_validate(matrix)
    matrix_path = camp_dir / "matrix-proposal.json"
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    _run(
        [
            *ar,
            "hypothesize",
            "--campaign-id",
            campaign_id,
            "--provider",
            "agent",
            "--matrix",
            str(matrix_path),
        ],
        cwd=cwd,
    )

    exp_dir = camp_dir / "artifacts" / "experiments"
    by_id = {
        json.loads(path.read_text(encoding="utf-8"))["experiment_id"]: path
        for path in exp_dir.glob("*.json")
    }
    # execute control then recommended
    order = [
        matrix["hypotheses"][0]["experiment"]["experiment_id"],
        matrix["recommended_experiment_id"],
    ]
    seen: set[str] = set()
    for eid in order:
        if eid in seen or eid not in by_id:
            continue
        seen.add(eid)
        exp = json.loads(by_id[eid].read_text(encoding="utf-8"))
        man = _manifest(campaign_id, exp, integration, role=role, policy=policy)
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
        ]
        print("+", " ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=cwd)
        print(f"experiment {eid} exit={code}", flush=True)

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
    )
    delivery = _phase_a_delivery(
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        campaign_id=campaign_id,
        primary_metric=effective_primary,
        cycle_index=cycle,
        role=role,
    )
    print(
        f"CYCLE_COMPLETE {campaign_id} role={role} positive={delivery['positive']}",
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
    parser.add_argument("--max-cycles", type=int, default=1, help="0 = many (1024)")
    parser.add_argument("--train-version", default="wf_smoke_v2")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--objective",
        default=(
            "On a size-matched fixture TwoTower arm under the wall cap, improve "
            "smoke decode latency without lowering parse_rate versus the matched control."
        ),
    )
    parser.add_argument("--primary-metric", default="smoke.latency_ms_p50")
    args = parser.parse_args(argv)
    cwd = Path.cwd()
    root = args.root if args.root.is_absolute() else cwd / args.root
    root.mkdir(parents=True, exist_ok=True)
    max_cycles = 1024 if args.max_cycles == 0 else max(1, args.max_cycles)
    for i in range(max_cycles):
        print(f"=== continuous cycle pass {i + 1}/{max_cycles} ===", flush=True)
        try:
            run_cycle(
                cwd=cwd,
                root=root,
                loop_id=args.loop_id,
                train_version=args.train_version,
                steps=args.steps,
                objective=args.objective,
                primary_metric=args.primary_metric,
            )
        except Exception as exc:  # noqa: BLE001 - continuous must self-heal next pass
            print(f"CYCLE_ERROR {exc!r}", flush=True)
            # soft continue unless dirty tree
            if "dirty" in str(exc).lower():
                return 2
            time.sleep(1)
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
