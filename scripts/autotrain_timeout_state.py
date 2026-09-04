"""Whether a decode timeout is live, reproduced, or finalised.

One responsibility: the timeout state machine -- has a timeout been finalised,
was a retirement reproduced, is a formal status a timeout, and what does the
predecessor's state require this cycle to do first.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_io import (
    read_json,
)
from slm_training.autoresearch.schemas import (
    AutotrainCycleHandoffV1,
)
from slm_training.autoresearch.storage import (
    pending_autotrain_actions,
)

FORMAL_TIMEOUT_STATUSES = frozenset({"timed_out"})


def delivery_is_thrash_timeout_residual(
    delivery: Mapping[str, Any] | None,
    handoff: AutotrainCycleHandoffV1 | None = None,
) -> bool:
    """True when incomplete measurement is thrash wall/decode residual, not a hard harness bug.

    Residual requires **explicit timeout evidence**: some arm exit of 124 or a
    ``wall_timeout`` / ``decode_timeout`` style marker in the reasons. Bare
    ``measurement_incomplete`` / ``missing_scoreboard`` /
    ``primary_metric_unavailable`` never suffice — those also appear when an
    arm process crashed (``harness_failure:<arm>:experiment_failed`` with exit
    2), and reading that crash as a soft residual dropped the
    ``repair_harness`` blocker for cycles c536..c543. A crash exit (non-zero,
    non-124) next to the harness-failure marker is always a harness failure.
    """
    # Single source of truth shared with the heal-playbook classifier so the
    # emission-time markers and the crash/residual split can never drift.
    from slm_training.autoresearch.heal.classify import (
        HARD_HARNESS_MARKERS,
        HARNESS_CRASH_REASON_RE,
        TIMEOUT_RESIDUAL_MARKERS,
        crash_arm_exits,
        timeout_arm_exits,
    )

    reasons: list[str] = []
    exits: Mapping[str, Any] = {}
    if delivery:
        reasons.extend(str(r) for r in (delivery.get("reasons") or []))
        raw_exits = delivery.get("arm_exits") or {}
        if isinstance(raw_exits, Mapping):
            exits = raw_exits
    if handoff is not None:
        reasons.extend(str(r) for r in (handoff.reasons or ()))
        for action in handoff.actions:
            reasons.append(str(action.reason or ""))
    joined = " ".join(reasons).lower()
    if any(m in joined for m in HARD_HARNESS_MARKERS):
        return False
    timeout_exit = timeout_arm_exits(exits)
    crash_exit = crash_arm_exits(exits)
    explicit_timeout = any(m in joined for m in TIMEOUT_RESIDUAL_MARKERS)
    if crash_exit and HARNESS_CRASH_REASON_RE.search(joined):
        # An arm process died (exit != 124) without a scoreboard: a harness
        # failure regardless of any wall exit elsewhere in the delivery.
        return False
    return timeout_exit or explicit_timeout


def is_reproduced_timeout_retirement(
    handoff: Mapping[str, Any], delivery: Mapping[str, Any]
) -> bool:
    reasons = [str(item) for item in handoff.get("reasons") or ()]
    if any(
        item.startswith(
            (
                "candidate_runtime_rejected_after_frozen_replay:",
                "candidate_runtime_unblock_reproduced:",
            )
        )
        for item in reasons
    ):
        return True
    action_reasons = " ".join(
        str(item.get("reason") or "")
        for item in handoff.get("actions") or ()
        if isinstance(item, dict)
    ).lower()
    return bool(
        handoff.get("cycle_intent") == "retry_measurement"
        and delivery_is_thrash_timeout_residual(delivery)
        and (
            "retire thrash decode/wall-timeout residual" in action_reasons
            or "incomplete thrash replay budget exhausted" in action_reasons
        )
    )


def formal_status_is_timeout(status: str | None) -> bool:
    return str(status or "") in FORMAL_TIMEOUT_STATUSES or str(status or "").startswith(
        "timed_out"
    )


def has_finalized_decode_timeout(camp_dir: Path, candidate_id: str) -> bool:
    """True when AgentV finalized every record and typed at least one timeout."""

    scoreboard = read_json(camp_dir / "runs" / candidate_id / "scoreboard.json")
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


def arm_decode_timeout_count(camp_dir: Path, experiment_id: str) -> int:
    """Max decode_timeout_count for a finalized arm run (0 if unknown/clean)."""

    timeout_n = 0
    run_dir = camp_dir / "runs" / experiment_id
    for name in ("eval_smoke.json", "eval.json", "scoreboard.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except Exception:  # noqa: BLE001 — best-effort signal only
            continue
        try:
            timeout_n = max(
                timeout_n,
                int(payload.get("decode_timeout_count") or 0),
                int(payload.get("decode_timeout_document_count") or 0),
            )
        except (TypeError, ValueError):
            pass
        suites = payload.get("suites")
        if isinstance(suites, dict):
            suite_rows: list[Any] = list(suites.values())
        elif isinstance(suites, list):
            suite_rows = list(suites)
        else:
            suite_rows = []
        for suite in suite_rows:
            if not isinstance(suite, dict):
                continue
            try:
                suite_timeout = suite.get("decode_timeout_document_count")
                if type(suite_timeout) is not int:
                    suite_timeout = suite.get("decode_timeout_count")
                timeout_n = max(timeout_n, int(suite_timeout or 0))
            except (TypeError, ValueError):
                pass
    if timeout_n > 0:
        return timeout_n
    delivery_path = camp_dir / "sdlc_delivery.json"
    if not delivery_path.is_file():
        return 0
    try:
        delivery = read_json(delivery_path)
    except Exception:  # noqa: BLE001 — best-effort signal only
        return 0
    marker = f"measurement_incomplete:{experiment_id}:"
    for reason in delivery.get("reasons") or []:
        text = str(reason)
        if marker not in text or "decode_timeout_count=" not in text:
            continue
        try:
            raw = text.rsplit("decode_timeout_count=", 1)[1]
            digits = "".join(itertools.takewhile(str.isdigit, raw))
            if digits:
                timeout_n = max(timeout_n, int(digits))
            else:
                timeout_n = max(timeout_n, 1)
        except (TypeError, ValueError, IndexError):
            timeout_n = max(timeout_n, 1)
    return timeout_n


def require_predecessor_actions(
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
    delivery = read_json(root / predecessor_campaign_id / "sdlc_delivery.json")
    if delivery.get("stack_layer") is False:
        # Historical positive fixture handoffs emitted deliver_stack even when
        # Phase A had no tracked delta. That action is not executable and must
        # not block the next experiment; the handoff writer now omits it.
        pending = tuple(
            (index, action)
            for index, action in pending
            if action.kind != "deliver_stack"
        )
    if pending:
        detail = ", ".join(f"{index}:{action.kind}" for index, action in pending)
        raise RuntimeError(
            f"predecessor {predecessor_campaign_id} has unacknowledged actions: {detail}"
        )
