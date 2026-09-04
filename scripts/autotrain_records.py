"""The loop's durable record of what each cycle did.

One responsibility: writing loop state and the cycle's outcome records --
failure, recovery, pass outcome, observed paired standard deviation -- and
reading back the last failure or heal receipt.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.autotrain_diagnosis import exception_is_soft_continuous
from scripts.autotrain_io import read_json
from scripts.autotrain_paths import loop_state_path
from slm_training.autoresearch.schemas import (
    AutotrainLoopStateV1,
    utc_now,
)
from slm_training.autoresearch.storage import (
    autotrain_loop_state_lock,
)

OBSERVED_PAIRED_SD_SCHEMA = "observed_paired_sd/v1"

OBSERVED_PAIRED_SD_SOURCE = "continuous_driver"


def record_observed_paired_sd(
    path: Path,
    *,
    metric_leaf: str,
    sd: float,
    n: int,
    campaign_id: str,
    control_id: str,
    candidate_id: str,
    date: str | None = None,
) -> bool:
    """Append one measured paired-delta SD to ``observed_paired_sd_by_metric``.

    Shape written under ``observed_paired_sd_by_metric[<leaf>]`` (the
    ``{sd, n_deltas, source}`` object form read by
    ``screening_sample_size.lookup_paired_sd_for_metric``)::

        {"schema": "observed_paired_sd/v1", "sd": <latest>,
         "n_deltas": <pairs>, "source": "continuous_driver",
         "date": "YYYY-MM-DD", "history": [{sd, n_deltas, date, campaign_id,
         control_id, candidate_id}, ...]}

    Backward compatible: a missing slot is created, a bare number or a
    history-less mapping is folded into ``history`` as the prior snapshot.
    Idempotent per (campaign, control, candidate). Returns ``True`` when the
    file changed.
    """
    from datetime import datetime, timezone

    path = Path(path)
    data = read_json(path)
    if not data:
        raise ValueError(f"screening expectations missing or invalid: {path}")
    slot = data.get("observed_paired_sd_by_metric")
    if not isinstance(slot, dict):
        slot = {}
        data["observed_paired_sd_by_metric"] = slot
    entry = slot.get(metric_leaf)
    history: list[dict[str, Any]] = []
    if isinstance(entry, dict):
        if isinstance(entry.get("history"), list):
            history = [row for row in entry["history"] if isinstance(row, dict)]
        elif entry.get("sd") is not None:
            history = [{k: v for k, v in entry.items() if k != "schema"}]
    elif isinstance(entry, (int, float, str)) and not isinstance(entry, bool):
        history = [{"sd": entry, "n_deltas": None, "date": None, "source": "prior"}]
    key = (str(campaign_id), str(control_id), str(candidate_id))
    for row in history:
        if (
            str(row.get("campaign_id")),
            str(row.get("control_id")),
            str(row.get("candidate_id")),
        ) == key:
            return False
    stamp = date or datetime.now(timezone.utc).date().isoformat()
    history.append(
        {
            "sd": float(sd),
            "n_deltas": int(n),
            "date": stamp,
            "campaign_id": str(campaign_id),
            "control_id": str(control_id),
            "candidate_id": str(candidate_id),
        }
    )
    slot[metric_leaf] = {
        "schema": OBSERVED_PAIRED_SD_SCHEMA,
        "sd": float(sd),
        "n_deltas": int(n),
        "source": OBSERVED_PAIRED_SD_SOURCE,
        "date": stamp,
        "history": history,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def last_cycle_failure_message(root: Path, loop_id: str) -> str | None:
    """Return the most recent blocking cycle failure message for this loop."""
    path = root / "loops" / loop_id / "cycle_failures.jsonl"
    if not path.is_file():
        return None
    last: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("loop_id") != loop_id:
            continue
        if row.get("blocking") is False:
            continue
        msg = str(row.get("message") or "").strip()
        if msg:
            last = msg
    return last


def record_cycle_recovery(
    *,
    root: Path,
    loop_id: str,
    soft_healed: Sequence[str],
    predecessor_campaign_id: str | None,
) -> None:
    """Append a non-blocking recovery event so soft failures do not accumulate."""
    path = root / "loops" / loop_id / "cycle_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "autotrain_cycle_recovery/v1",
        "loop_id": loop_id,
        "predecessor_campaign_id": predecessor_campaign_id,
        "soft_healed": list(soft_healed),
        "blocking": False,
        "recovered": True,
        "consecutive_count": 0,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def write_loop_state_unlocked(root: Path, state: AutotrainLoopStateV1) -> Path:
    path = loop_state_path(root, state.loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def write_loop_state(root: Path, state: AutotrainLoopStateV1) -> Path:
    with autotrain_loop_state_lock(root, state.loop_id):
        return write_loop_state_unlocked(root, state)


def record_cycle_failure(
    *, root: Path, loop_id: str, exc: Exception, cycle_index: int
) -> int:
    """Persist a stable blocker fingerprint and return its consecutive count.

    Soft continuous failures (document, thrash timeout residual, dirty closeout,
    bank exhaust, soft identity) never accumulate to ``state=BLOCKED``.
    """
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
    soft = exception_is_soft_continuous(exc)
    count = 0 if soft else 1
    if not soft:
        for row in reversed(previous):
            if (
                not row.get("blocking", True)
                or row.get("recovered")
                or row.get("fingerprint") != fingerprint
            ):
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
        "soft": soft,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    blocked = not soft and count >= 3
    write_loop_state(
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
            blocker_fingerprint=fingerprint if not soft else None,
            blocker_count=count if not soft else 0,
            pid=os.getpid(),
            heartbeat_at=utc_now(),
        ),
    )
    return count


def last_heal_receipt_outcome(path: Path) -> str | None:
    if not path.is_file():
        return None
    for raw in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return str(row.get("outcome") or "") if isinstance(row, dict) else None
    return None
