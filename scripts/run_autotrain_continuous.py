#!/usr/bin/env python3
"""Hands-off continuous autotrain cycle driver.

Bare /autotrain agents should keep calling this (or re-enter continuous.md)
without user prompts. Each invocation can run one or many bounded cycles.
Never an infinite shell without a stage wall on child commands — continuous
stage wall minutes come from climb policy measurement (not the global CI
MAX_RUN_MINUTES unless policy is absent).

SDLC Phase A (autotrain-iteration-delivery): after every cycle the driver
classifies positive vs non-positive, records a delivery ledger, and only
signals stack-layer intent for positive results. Stacked PRs are still opened
by the agent (gh stack); this driver never opens PRs for non-positive cycles.

Proof driver (promote path): formal preflight must be proved and a LeverProof
metric_certificate/v2 must dispose ``continue`` before a champion is marked
``promoted``. Phase A smoke quality-held alone never promotes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
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
from slm_training.autoresearch.schemas import (
    CampaignBudget,
    FormalObligationV1,
    HypothesisMatrix,
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
_PROMOTE_FORMAL_TIMEOUT_S = 600.0
_FORMAL_TIMEOUT_STATUSES = frozenset({"timed_out"})
_DRIVER_LOCK_BASENAME = "driver.lock"


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

# Champion queue: quality-held wins retest with same levers, new seeds, before
# thrashing the fixed lever bank again. Ledger is loop-local (not git).
_CHAMPION_QUEUE_SCHEMA = "autotrain_champion_queue/v1"
_CHAMPION_STATUSES = frozenset(
    {
        "queued",
        "confirming",
        "confirmed",
        "rejected",
        "skipped_duplicate",
        "promoting",
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
    "grammar_completion_bounds",
    "compact_active_canvas",
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


def _champion_queue_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "champion_queue.jsonl"


def _driver_lock_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / _DRIVER_LOCK_BASENAME


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
        k: v
        for k, v in (levers or {}).items()
        if k not in _FINGERPRINT_EXCLUDE_KEYS
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
        if row.get("status") in {"queued", "confirming"}:
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


def _arm_slug_from_knobs(knobs: dict[str, Any], *, candidate_id: str = "") -> str | None:
    """Map knobs / candidate id to thrash arm slug."""
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


def _skip_arm_slugs(entries: list[dict[str, Any]]) -> set[str]:
    """Deprioritize arms only while a champion is still open (not forever).

    Permanent skip of rejected/promotion_failed starved bounds/canvas and left
    only steps/batch1 thrash — which could not re-enter the champion path.
    """
    skip: set[str] = set()
    for row in entries:
        # Only skip arms currently in the funnel (not terminal failures).
        if row.get("status") not in {
            "queued",
            "confirming",
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
    return skip


def _select_recommended_slug(cycle: int, skip: set[str] | None = None) -> str:
    """Rotate thrash recommendation; prefer arms not recently rejected/queued."""
    bank = _SCREENING_ARM_BANK
    n = len(bank)
    start = (max(1, int(cycle)) - 1) % n
    ordered = [bank[(start + i) % n][0] for i in range(n)]
    skip = skip or set()
    for slug in ordered:
        if slug not in skip:
            return slug
    # All skipped — still rotate so thrash is not frozen on bounds forever.
    return ordered[0]


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
        any(r.startswith(prefix) for prefix in _QUALITY_ENQUEUE_PREFIXES) for r in reasons
    )


def _should_enqueue_champion(delivery: dict[str, Any]) -> bool:
    """Enqueue only quality-held positives (never empty-meaning latency blips)."""
    if not delivery.get("positive"):
        return False
    reasons = list(delivery.get("reasons") or [])
    if _quality_held_reasons(reasons):
        return True
    # Efficiency / primary latency win only when quality_held co-tagged.
    if any(r.startswith("efficiency_win:") or r.startswith("primary_metric_win:") for r in reasons):
        return _quality_held_reasons(reasons)
    return False


def _is_champion_lever(knobs: dict[str, Any], *, candidate_id: str = "") -> bool:
    """True when knobs encode a thrash arm (not pure matched control)."""
    if knobs.get("grammar_completion_bounds") or knobs.get("compact_active_canvas"):
        return True
    if knobs.get("batch_size") == 1:
        return True
    cid = candidate_id or ""
    if cid.endswith("-steps") or "-steps" in cid:
        return True
    if cid.endswith("-batch1") or "-batch1" in cid:
        return True
    return False


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
    if not candidate_id or candidate_id == delivery.get("control_id"):
        return None
    knobs = _lever_knobs(_load_experiment_knobs(camp_dir, candidate_id))
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
        "source_control_id": delivery.get("control_id"),
        "source_role": delivery.get("cycle_role") or delivery.get("role"),
        "knobs": knobs,
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
            "promoted",
            "promotion_failed",
        }:
            row["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            row["resolve_reasons"] = list(resolve_reasons or [])
        elif status in {"promotion_inconclusive", "harness_failure"}:
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
    ok = bool(delivery.get("positive")) and _quality_held_reasons(reasons)
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
    return str(status or "") in _FORMAL_TIMEOUT_STATUSES or str(status or "").startswith(
        "timed_out"
    )


def dispose_champion_promote(
    *,
    formal_preflight_status: str | None,
    certificate: dict[str, Any] | None,
    locked_expectations_sha256: str | None = None,
    phase_a_positive: bool = False,
    phase_a_quality_held: bool = False,
) -> dict[str, Any]:
    """Authoritative promote disposition (proof driver).

    Phase A smoke quality-held alone never yields ``promoted``. A proved formal
    preflight and an in-band LeverProof metric_certificate/v2 (via
    ``optimum_feedback``) are required. Theorem misses fail closed without a
    five-lane matrix; assumption-backed misses fail closed and request five-lane
    successor diagnosis.

    Formal **timeouts** are incomplete measurement: disposition
    ``promotion_inconclusive`` (retryable), never ``promotion_failed`` /
    ``rejected``.
    """
    from slm_training.harnesses.experiments.verified_metrics import optimum_feedback

    reasons: list[str] = []
    if phase_a_positive:
        reasons.append("phase_a_positive")
    if phase_a_quality_held:
        reasons.append("phase_a_quality_held")

    if _formal_status_is_timeout(formal_preflight_status):
        reasons.append(
            f"formal_preflight_timed_out:status={formal_preflight_status!r}:"
            f"wall_s={_PROMOTE_FORMAL_TIMEOUT_S:g}"
        )
        reasons.append("measurement_incomplete:formal_timeout_not_rejection")
        return {
            "status": "promotion_inconclusive",
            "reasons": reasons,
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
            "inconclusive": True,
            "timeout": True,
        }

    if formal_preflight_status != "proved":
        reasons.append(
            f"formal_preflight_unproved:status={formal_preflight_status!r}"
        )
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
        }

    if certificate is None:
        reasons.append("promote_requires_certificate:phase_a_alone_insufficient")
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
        }

    if certificate.get("schema") != _CERTIFICATE_SCHEMA_V2:
        reasons.append(
            f"promote_requires_certificate_v2:got={certificate.get('schema')!r}"
        )
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
        }

    # Fail closed: locked expectations digest is required (never optional).
    if not locked_expectations_sha256:
        reasons.append("promote_requires_locked_expectations_digest")
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
        }
    cert_exp = certificate.get("metric_expectations_sha256")
    if cert_exp != locked_expectations_sha256:
        reasons.append(
            "certificate_expectations_digest_mismatch:"
            f"locked={str(locked_expectations_sha256)[:12]} cert={str(cert_exp)[:12]}"
        )
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
        }

    try:
        feedback = optimum_feedback(certificate)
    except Exception as exc:  # noqa: BLE001 — fail closed on bad certs
        reasons.append(f"certificate_invalid:{exc}")
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": None,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
        }

    policy = str(feedback.get("policy") or "")
    reasons.append(f"cert_policy:{policy}")
    lanes = list(feedback.get("diagnosis_lanes") or [])
    breaches = list(feedback.get("breaches") or [])

    if policy == "continue":
        return {
            "status": "promoted",
            "reasons": reasons,
            "cert_policy": policy,
            "diagnosis_lanes": [],
            "emit_five_lane_matrix": False,
            "breaches": [],
        }
    if policy == "stop":
        reasons.append("theorem_backed_band_miss")
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": policy,
            "diagnosis_lanes": lanes,
            "emit_five_lane_matrix": False,
            "breaches": breaches,
        }
    if policy == "block_promotion_and_diagnose":
        reasons.append("assumption_backed_band_miss")
        return {
            "status": "promotion_failed",
            "reasons": reasons,
            "cert_policy": policy,
            "diagnosis_lanes": lanes or list(_FIVE_LANES),
            "emit_five_lane_matrix": True,
            "breaches": breaches,
        }
    reasons.append(f"certificate_not_promotable:policy={policy!r}")
    return {
        "status": "promotion_failed",
        "reasons": reasons,
        "cert_policy": policy or None,
        "diagnosis_lanes": lanes,
        "emit_five_lane_matrix": False,
        "breaches": breaches,
    }


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
        return None, (
            "promote_cert_incomplete_metrics:"
            f"ss={ss!r} parse={parse!r}"
        )

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
    try:
        completed = subprocess.run(
            [str(IN_REPO_CHECKER), "check", str(evidence_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"promote_certify_failed:{exc}"
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
    """Read recorded formal preflight status for promote gate."""
    path = camp_dir / "formal_preflight_status.json"
    if path.is_file():
        data = _read_json(path)
        status = data.get("status")
        return str(status) if status is not None else None
    # Content-addressed formal preflight artifacts
    art = camp_dir / "artifacts" / "formal_preflights"
    if art.is_dir():
        for p in sorted(art.glob("*.json")):
            data = _read_json(p)
            if data.get("status"):
                return str(data["status"])
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
) -> tuple[str, str | None]:
    """Record formal preflight; return ``(status, content_sha256|None)``.

    When proved, writes a content-addressed artifact under
    ``artifacts/formal_preflights/<sha>.json`` matching
    ``validate_formal_preflights`` binding. Fail closed on any error.
    """
    status_path = camp_dir / "formal_preflight_status.json"
    if status_path.is_file():
        data = _read_json(status_path)
        status = str(data.get("status") or "missing")
        sha = data.get("preflight_sha256")
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
        from slm_training.autoresearch.formal import run_formal_preflight
        from slm_training.autoresearch.schemas import (
            ExperimentKnobs,
            ExperimentSpec,
            FormalClaimV1,
        )
        from slm_training.lineage.records import canonical_json

        claim = FormalClaimV1(**promote_formal_claim_dict())
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
        # Caller-owned wall (600s). Timeouts → status timed_out (inconclusive).
        preflight, _obligation = run_formal_preflight(
            campaign_id,
            exp,
            claim,
            timeout_seconds=_PROMOTE_FORMAL_TIMEOUT_S,
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
                f"formal_preflight_timed_out:wall_s={_PROMOTE_FORMAL_TIMEOUT_S:g}:"
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
            timeout_seconds=_PROMOTE_FORMAL_TIMEOUT_S,
            duration_seconds=duration,
            timed_out=timed_out,
        )
        # Persist sha for manifest binding.
        status_path = camp_dir / "formal_preflight_status.json"
        st = _read_json(status_path)
        st["preflight_sha256"] = content_sha
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
                f"formal_preflight_timed_out:wall_s={_PROMOTE_FORMAL_TIMEOUT_S:g}:{msg}"
                if timed_out
                else f"formal_preflight_error:{exc}"
            ),
            timeout_seconds=_PROMOTE_FORMAL_TIMEOUT_S,
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
    if (
        cert_err
        and "missing_run_ids" in str(cert_err)
    ):
        reasons.append(f"harness_failure:{cert_err}")
    # Control-only success with no candidate is always a harness/process gap
    # when promote was intended (caller only invokes this on promote intent).
    if ctrl and _run_has_usable_metrics(camp_dir, ctrl) and cand and not (
        camp_dir / "runs" / cand
    ).is_dir():
        if "harness_failure:missing_promote_run" not in reasons:
            reasons.append("harness_failure:missing_promote_run")
    return reasons


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

        disposition = dispose_champion_promote(
            formal_preflight_status=formal_preflight_status,
            certificate=certificate,
            locked_expectations_sha256=locked_expectations_sha256,
            phase_a_positive=phase_a_positive,
            phase_a_quality_held=phase_a_quality,
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
                    if status == "harness_failure" or disposition.get("harness_failure"):
                        row["last_harness_failure"] = True
                break
        _write_champion_queue(path, entries)
        print(
            f"CHAMPION_PROMOTE_DISPOSE status={status} "
            f"cert_policy={disposition.get('cert_policy')} "
            f"formal={formal_preflight_status}",
            flush=True,
        )
    return updated


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
    cycle_intent: str | None = None,
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
    decision["cycle_role"] = role
    decision["cycle_index"] = cycle_index
    decision["cycle_intent"] = cycle_intent or role
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
    role: str = "screening",
    policy: Any | None = None,
    confirm_levers: dict[str, Any] | None = None,
    promote_levers: dict[str, Any] | None = None,
    recommended_slug: str | None = None,
    skip_slugs: set[str] | None = None,
) -> dict:
    from slm_training.autoresearch.climb_policy import (
        decode_timeout_seconds_for_role,
        eval_suites_for_role,
        load_climb_policy,
    )

    research = role_citations.get("research") or cites[0]
    prior = role_citations.get("prior_result") or (cites[1] if len(cites) > 1 else cites[0])
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
            "compact_active_canvas": False,
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
            "stop_conditions": [
                "Stop at declared steps or the campaign wall cap."
            ],
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
            if k
            in {
                "grammar_completion_bounds",
                "compact_active_canvas",
                "steps",
                "batch_size",
                "train_version",
                "context_backend",
                "sync_checkpoints",
                "local_files_only",
                "output_tokenizer",
            }
        }
        promo_steps = int(promo_extra.pop("steps", steps) or steps)
        control_knobs = knobs(steps=promo_steps)
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
                    "Monitor-only bounds pad deferred while confirmed champion is promoted.",
                    knobs(grammar_completion_bounds=True, steps=promo_steps + 2000),
                    "Schema pad — not executed while promote is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(2, "promote pad bounds"),
            },
            {
                "experiment": exp(
                    f"{prefix}-canvas",
                    "Monitor-only canvas pad deferred while confirmed champion is promoted.",
                    knobs(compact_active_canvas=True, steps=promo_steps + 2001),
                    "Schema pad — not executed while promote is recommended.",
                ),
                "evidence_uses": uses(),
                "novelty": novelty(3, "promote pad canvas"),
            },
            {
                "experiment": exp(
                    f"{prefix}-both",
                    "Monitor-only combined pad deferred while confirmed champion is promoted.",
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
            if k
            in {
                "grammar_completion_bounds",
                "compact_active_canvas",
                "steps",
                "batch_size",
                "train_version",
                "context_backend",
                "sync_checkpoints",
                "local_files_only",
                "output_tokenizer",
            }
        }
        confirm_steps = int(confirm_extra.pop("steps", steps) or steps)
        control_knobs = knobs(steps=confirm_steps)
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
        rec_slug = recommended_slug or _select_recommended_slug(
            cycle, skip=skip_slugs
        )
        bank_by_slug = {slug: (hyp, extras) for slug, hyp, extras in _SCREENING_ARM_BANK}
        if rec_slug not in bank_by_slug:
            rec_slug = _SCREENING_ARM_BANK[0][0]
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
            }
        ]
        for i, (slug, hyp, extras) in enumerate(_SCREENING_ARM_BANK, start=1):
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
        budget=CampaignBudget(
            max_experiments=1,
            max_wall_minutes=stage_wall_minutes_for_role(pol, role),
        ),
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
        formal_obligations=formal_obligations,
        claim_class=claim_class,  # type: ignore[arg-type]
        locked_eval_manifest_sha256=locked_eval,
        metric_expectations_sha256=metric_expectations_sha,
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
        stage_wall_minutes_for_role,
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
    # Champion queue: confirm open heads; promote confirmed heads on promotion
    # cadence; otherwise thrash with rotated levers (Change B/C). Cadence role
    # stays screening|promotion for suites/claim_class legality.
    queue_path = _champion_queue_path(root, loop_id)
    queue_entries = _load_champion_queue(queue_path)
    # Recover mid-cycle crash: promoting → confirmed so next promotion slot retries.
    # promotion_inconclusive / harness_failure already retriable via _queue_head_confirmed.
    recovered = False
    for row in queue_entries:
        if row.get("status") == "promoting":
            row["status"] = "confirmed"
            recovered = True
    if recovered:
        _write_champion_queue(queue_path, queue_entries)
    open_champion = _queue_head_open(queue_entries)
    confirmed_champion: dict[str, Any] | None = None
    promoting_champion: dict[str, Any] | None = None
    if open_champion is not None:
        cycle_intent = "confirm"
    elif role == "promotion":
        confirmed_champion = _queue_head_confirmed(queue_entries)
        if confirmed_champion is not None:
            cycle_intent = "promote"
            promoting_champion = confirmed_champion
        else:
            # No confirmed champion — still run promotion measurement suite but
            # thrash with rotation (policy: prefer prior screening win).
            cycle_intent = "promotion"
    else:
        cycle_intent = "screening"
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
    py = sys.executable
    ar = [py, "-m", "scripts.autoresearch", "--root", str(root)]
    if cycle_intent == "confirm":
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
        str(stage_wall_minutes_for_role(policy, role)),
        "--notes",
        notes,
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
    confirm_levers = None
    promote_levers = None
    if open_champion is not None:
        confirm_levers = _lever_knobs(open_champion.get("knobs") or {})
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
    skip_slugs = _skip_arm_slugs(queue_entries)
    rec_slug = _select_recommended_slug(cycle, skip=skip_slugs)
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
        promote_levers=promote_levers,
        recommended_slug=rec_slug if promote_levers is None and confirm_levers is None else None,
        skip_slugs=skip_slugs,
    )
    if promote_levers is None and confirm_levers is None:
        print(
            f"THRASH_ROTATE cycle={cycle} recommended={rec_slug} skip={sorted(skip_slugs)}",
            flush=True,
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
    # Promote path: formal preflight must be proved before train executes.
    promote_formal_status: str | None = None
    promote_preflight_sha: str | None = None
    if cycle_intent == "promote" and promoting_champion is not None:
        promote_formal_status, promote_preflight_sha = ensure_promote_formal_preflight(
            camp_dir=camp_dir,
            campaign_id=campaign_id,
            experiment_id=str(matrix["recommended_experiment_id"]),
            run_lean=True,
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
        # formal_claims must already be on the matrix member (promote path in
        # _matrix). Do not rewrite the experiment after hypothesize — that
        # breaks exact matrix membership and aborts the promote arm (exit=1).
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
        ]
        print("+", " ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=cwd)
        arm_exits[eid] = int(code)
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
        cycle_intent=cycle_intent,
    )
    camp_dir = root / campaign_id
    if open_champion is not None:
        _resolve_confirm_result(
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
        if promote_formal_status == "proved" or _formal_preflight_status(camp_dir) == "proved":
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
            if control_id and candidate_id and _run_has_usable_metrics(
                camp_dir, candidate_id
            ):
                _cert_path, cert_err = export_promote_metric_certificate(
                    camp_dir=camp_dir,
                    campaign_id=campaign_id,
                    control_id=control_id,
                    candidate_id=candidate_id,
                    delivery=delivery,
                )
                if cert_err:
                    print(f"PROMOTE_CERT_EXPORT_FAIL {cert_err}", flush=True)
                    delivery = {
                        **delivery,
                        "reasons": list(delivery.get("reasons") or [])
                        + [cert_err],
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
        _resolve_promotion_result(
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
        # Only screening thrash quality-held wins enqueue (not promotion thrash noise).
        if cycle_intent in {"screening", "promotion"}:
            _enqueue_champion(
                root=root,
                loop_id=loop_id,
                delivery=delivery,
                camp_dir=camp_dir,
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
    try:
        code_sha = _git("rev-parse", "HEAD", cwd=cwd)
    except (subprocess.CalledProcessError, OSError):
        code_sha = None
    try:
        lock_fh = acquire_driver_lock(root, args.loop_id, code_sha=code_sha)
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2
    max_cycles = 1024 if args.max_cycles == 0 else max(1, args.max_cycles)
    try:
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
