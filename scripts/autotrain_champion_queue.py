"""Reading and writing the champion queue file.

One responsibility: the queue as a data structure -- what is at its head,
whether a candidate may be enqueued, and clearing the blocker that stops it.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scripts.autotrain_campaign import warm_start_policy
from scripts.autotrain_candidate_state import (
    confirmation_quality_reheld,
    is_confirm_candidate_win,
)
from scripts.autotrain_diagnosis import quality_held_reasons
from scripts.autotrain_paths import loop_state_path
from scripts.autotrain_records import write_loop_state
from slm_training.autoresearch.hillclimb import (
    CHAMPION_EPOCHS_EXHAUSTED,
    DEFAULT_MAX_CUMULATIVE_EPOCHS,
    champion_cumulative_epochs,
    champion_epoch_park_reason,
    load_climb_champion,
    train_manifest_record_count,
)
from slm_training.autoresearch.schemas import (
    AutotrainLoopStateV1,
    utc_now,
)

RETRYABLE_PROMOTE_STATUSES = frozenset(
    {"confirmed", "promotion_inconclusive", "harness_failure"}
)

REGIME_PARKED_STATUS = "regime-parked"


def clear_loop_blocker(root: Path, loop_id: str, *, reason: str) -> None:
    """Return loop state to runnable after a successful self-heal."""
    path = loop_state_path(root, loop_id)
    cycle_index = 0
    if path.is_file():
        try:
            prev = AutotrainLoopStateV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            cycle_index = int(prev.cycle_index or 0)
        except (OSError, ValueError):
            pass
    write_loop_state(
        root,
        AutotrainLoopStateV1(
            loop_id=loop_id,
            state="IDLE",
            phase="between_cycles",
            cycle_index=cycle_index,
            next_action=f"continue_after_self_heal:{reason}",
            blocker_fingerprint=None,
            blocker_count=0,
            pid=os.getpid(),
            heartbeat_at=utc_now(),
        ),
    )
    print(f"SELF_HEAL_CLEAR_BLOCKER reason={reason}", flush=True)


def write_champion_queue(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in entries:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def queue_head_open(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First entry still awaiting confirmatory retest (queued or in-flight)."""
    for row in entries:
        if row.get("status") in {
            "queued",
            "confirming",
            "confirmation_inconclusive",
        }:
            return row
    return None


def queue_head_confirmed(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First confirmed / inconclusive / harness-blocked champion for promote.

    ``promotion_inconclusive`` (formal wall) and ``harness_failure`` (execute /
    matrix / missing run) are retryable — not model rejects — so they re-enter
    the promote slot on the next promotion cadence.
    """
    for row in entries:
        if row.get("status") in RETRYABLE_PROMOTE_STATUSES:
            return row
    return None


def park_champion_epochs_if_needed(policy: Any, loop_dir: Path) -> str | None:
    """Park when the champion exceeds ``max_cumulative_epochs`` on the corpus.

    Epochs are recomputed against the current train corpus record count from
    the train manifest (``<train_dir>/manifest.json``) recorded on the sidecar;
    the sidecar's accumulated value is the fallback when unresolvable.
    """

    warm = warm_start_policy(policy)
    sidecar = load_climb_champion(loop_dir)
    record_count: int | None = None
    if sidecar is not None:
        record_count = train_manifest_record_count(sidecar.train_dir)
        if record_count is None:
            record_count = sidecar.record_count
    reason = champion_epoch_park_reason(
        sidecar,
        max_cumulative_epochs=float(
            warm.get("max_cumulative_epochs") or DEFAULT_MAX_CUMULATIVE_EPOCHS
        ),
        record_count=record_count,
    )
    if reason:
        epochs = (
            champion_cumulative_epochs(sidecar, record_count=record_count)
            if sidecar is not None
            else None
        )
        print(
            f"REGIME_PARKED reason={CHAMPION_EPOCHS_EXHAUSTED} "
            f"epochs={epochs} record_count={record_count} "
            f"cumulative_steps={sidecar.cumulative_steps if sidecar else None}",
            flush=True,
        )
        return REGIME_PARKED_STATUS
    return None


def should_enqueue_champion(delivery: dict[str, Any]) -> bool:
    """Enqueue confirm candidates on quality primary wins (never latency-only blips).

    Fixture insufficient_n forces ``positive=False`` so screening cannot stack
    or ship; it must not block champion enqueue — confirm/promote raise n.
    """
    reasons = [str(reason) for reason in delivery.get("reasons") or []]
    if is_confirm_candidate_win(delivery):
        return True
    if not delivery.get("positive"):
        return False
    if any(reason.startswith("fixture_insufficient_n") for reason in reasons):
        # Positive+fixture_n should not happen under current classify; fail closed.
        return False
    primary_leaf = str(delivery.get("primary_metric") or "").rsplit(".", 1)[-1]
    if primary_leaf != "latency_ms_p50" and confirmation_quality_reheld(delivery):
        return True
    if quality_held_reasons(reasons):
        return True
    if any(
        reason.startswith("efficiency_win:") or reason.startswith("primary_metric_win:")
        for reason in reasons
    ):
        return quality_held_reasons(reasons)
    return False
