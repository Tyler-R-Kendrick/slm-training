"""Naming a cycle's failure.

One responsibility: classifying an outcome the loop cannot simply score --
whether a terminal is causally decisive, whether an exception is soft enough to
continue on, and whether a hold reflects a real quality result or merely a gap
in the harness. The metric tradeoff rule lives in ``autotrain_tradeoff``.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.autotrain_io import (
    read_json,
)


class CodeUpdated(RuntimeError):
    """The long-lived driver integrated code newer than its Python process."""


QUALITY_ENQUEUE_PREFIXES = (
    "quality_held:",
    "quality_metric_win:",
)

BANK_EXHAUST_MSG = (
    "registered screening arm bank exhausted; add a distinct preregistered "
    "quality objective instead of recycling a rejected approach"
)

BANK_EXHAUST_MARKERS = (
    "quality-arm bank exhausted",
    "screening arm bank exhausted",
    "arm bank exhausted",
    "registered quality-arm bank is exhausted",
)


def exception_is_soft_continuous(exc: BaseException) -> bool:
    """True when the failure class is thrash-soft and must not create BLOCKED."""
    if isinstance(exc, (subprocess.TimeoutExpired, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, subprocess.CalledProcessError):
        cmd = [str(x) for x in (exc.cmd or ())]
        if cmd[:2] == ["git", "fetch"]:
            return True
        if "merge-base" in cmd or "is-ancestor" in cmd:
            return True
    message = str(exc)
    # Document / dirty / thrash residual / soft identity are soft.
    if "unacknowledged actions" in message and ":document" in message:
        return True
    if "loop worktree is dirty" in message:
        return True
    if "unacknowledged actions" in message and "repair_harness" in message:
        # Soft until proven hard: thrash residual / bank-exhaust heal will clear;
        # true AgentV hard cases re-raise after unblock reports hard_pending.
        return True
    if any(m in message.lower() for m in BANK_EXHAUST_MARKERS):
        return True
    if any(
        m in message
        for m in (
            "campaign already exists with different spec",
            "conflicts with supplied feedback",
            "screening arm bank exhausted",
            BANK_EXHAUST_MSG,
            "FROZEN_REPLAY_SKIP",
            "frozen replay control manifest is missing",
            "missing_control_manifest",
        )
    ):
        return True
    return False


def is_decisive_causal_terminal(row: dict[str, Any]) -> bool:
    """Whether a terminal champion burns the thrash family for this integration.

    Fixture-noise confirms (fixture_insufficient_n_alone without a quality
    non-regression or null primary) and harness incompletes do not burn CAP.
    """
    status = str(row.get("status") or "")
    if status not in {
        "rejected",
        "promotion_failed",
        "climb_accepted",
        "promoted",
    }:
        return False
    reasons = [str(r) for r in (row.get("resolve_reasons") or [])]
    if reasons_are_harness_incomplete_only(reasons):
        return False
    decisive_markers = (
        "non_regression_fail:",
        "primary_metric_null_or_worse:",
        "primary_quality_win_rejected",
        "confirmation_rejected:primary_quality",
        "promotion_primary",
        "eg_params_block",
        "primary_metric_win_rejected",
        "mechanism_no_effect:",
    )
    if any(any(marker in r for marker in decisive_markers) for r in reasons):
        return True
    # Pure fixture-volume rejects without quality signal — do not CAP-burn.
    if any("fixture_insufficient_n_alone" in r for r in reasons) and not any(
        "primary_metric_null" in r or "non_regression" in r for r in reasons
    ):
        return False
    return status in {"rejected", "promotion_failed", "climb_accepted", "promoted"}


def quality_held_reasons(reasons: list[str] | None) -> bool:
    """True when Phase A reasons include a quality hold/win (not pure latency)."""
    if not reasons:
        return False
    return any(
        any(r.startswith(prefix) for prefix in QUALITY_ENQUEUE_PREFIXES)
        for r in reasons
    )


HARNESS_INCOMPLETE_REASON_PREFIXES: tuple[str, ...] = (
    "harness_failure:",
    "measurement_incomplete:",
    "promote_cert_incomplete_metrics:",
    "promote_cert_missing_run_ids",
    "formal_preflight_timed_out:",
    "measurement_incomplete:formal_timeout",
    "promote_harness_parked:",
    "promote_attempts_paused:",
    "harness_retry_after_integration_change",
    "promote_attempts_exceeded:",  # only counted as non-model when paired w/ harness
)


def reason_is_harness_incomplete(reason: object) -> bool:
    text = str(reason or "")
    if not text:
        return False
    return any(text.startswith(prefix) for prefix in HARNESS_INCOMPLETE_REASON_PREFIXES)


def reasons_are_harness_incomplete_only(reasons: object) -> bool:
    """True when every reason is harness/infra incomplete (no model reject signal)."""
    if not isinstance(reasons, (list, tuple)) or not reasons:
        return False
    texts = [str(item) for item in reasons if str(item or "").strip()]
    if not texts:
        return False
    # Attempt-cap alone is only non-model when the rest are harness incomplete.
    non_cap = [t for t in texts if not t.startswith("promote_attempts_exceeded:")]
    if not non_cap:
        return True
    return all(reason_is_harness_incomplete(t) for t in non_cap)


OPEN_NUMERIC_LITERAL_RE = re.compile(r"(?:^|[,(])\s*-?\d{6,}$")


def has_numeric_literal_close_starvation(camp_dir: Path, candidate_id: str) -> bool:
    """Detect repeated legal numeric bytes where the model never selects close.

    This is a diagnostic steering signal only. It cannot alter the legal domain
    or certify output; the next arm remains grammar constrained and size matched.
    """

    run_dir = camp_dir / "runs" / candidate_id
    stalled_records: set[str] = set()
    timeout_records = 0
    for eval_path in sorted(run_dir.glob("eval_*.json")):
        payload = read_json(eval_path)
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
                and OPEN_NUMERIC_LITERAL_RE.search(prefix) is not None
            )
            streaks[record_id] = (
                streaks.get(record_id, 0) + 1 if repeated_numeric else 0
            )
            if streaks[record_id] >= 4:
                stalled_records.add(record_id)
    return timeout_records > 0 and bool(stalled_records)


def diagnosis_target(camp_dir: Path) -> str | None:
    targets: list[str] = []
    for path in sorted((camp_dir / "artifacts" / "diagnoses").glob("*.json")):
        target = str(read_json(path).get("target") or "")
        if target:
            targets.append(target)
    for preferred in ("harness", "data", "infrastructure", "model", "researcher"):
        if preferred in targets:
            return preferred
    return targets[-1] if targets else None
