"""Reading and writing the champion queue file.

One responsibility: the queue as a data structure -- what is at its head,
whether a candidate may be enqueued, and clearing the blocker that stops it.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
import os
import fcntl
import hashlib
import time
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
CHAMPION_STATUSES = frozenset({
    "queued", "confirming", "confirmation_inconclusive", "confirmed", "rejected",
    "skipped_duplicate", "promoting", "climb_accepted", "promotion_failed",
    "promotion_inconclusive", "harness_failure",
    "promoted",  # Read-only compatibility for pre-v30 queue ledgers.
})


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


def _result_digest(value):
    from slm_training.lineage.records import canonical_json

    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _promotion_row(before, record, changes, refund):
    row = {**before, **changes}
    if row["status"] in {"promoted", "climb_accepted"}:
        for key in ("recert_required", "recert_required_at", "recert_from_status"):
            row.pop(key, None)
    campaign = record["campaign_id"]
    refunded = set(before.get("refunded_promotion_campaigns", []))
    attempts = int(before.get("promote_attempts") or 0)
    if refund and campaign not in refunded:
        row["promote_attempts"] = max(0, attempts - 1)
        refunded.add(campaign)
    elif not refund and campaign in refunded:
        row["promote_attempts"] = attempts + 1
        refunded.remove(campaign)
    row["refunded_promotion_campaigns"] = sorted(refunded)
    stamp = record["finished_at"]
    if row["status"] in {"promotion_inconclusive", "harness_failure"}:
        key = "last_inconclusive_at" if row["status"] == "promotion_inconclusive" else "last_harness_failure_at"
        row[key] = stamp
        row.pop("resolved_at", None)
    else:
        row["resolved_at"] = stamp
    return row


def _promotion_artifact(store, before, record, changes, refund):
    identity = {"record": {k: v for k, v in record.items() if k != "finished_at"},
                "changes": changes, "refund": refund}
    key = "promotion-disposition:" + _result_digest(identity)
    events = [row for row in store.verify_event_chain() if row["event_type"] == "promotion_disposition_published"]
    previous = next((row for row in events if row.get("idempotency_key") == key), None)
    if previous is not None:
        if previous != events[-1]:
            raise ValueError("stale promotion disposition retry")
        sha = previous["artifact_sha256"]
        payload = json.loads((store.root / "artifacts/promotion_dispositions" / f"{sha}.json").read_text())
        if _result_digest(payload) != sha:
            raise ValueError("promotion disposition artifact changed")
        return payload
    record = {**record, "publication_id": key,
              "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload = {"before": before, "after": _promotion_row(before, record, changes, refund),
               "record": record, "identity": identity}
    artifact = store.write_artifact("promotion_dispositions", payload)
    store.append_event("promotion_disposition_published", artifact_sha256=artifact.stem,
                       detail={"entry_id": record["entry_id"]}, idempotency_key=key)
    return payload


def _reconcile_promotion(store, path, rows, index):
    events = [row for row in store.verify_event_chain() if row["event_type"] == "promotion_disposition_published"]
    if not events:
        return
    sha = events[-1]["artifact_sha256"]
    payload = json.loads((store.root / "artifacts/promotion_dispositions" / f"{sha}.json").read_text())
    if _result_digest(payload) != sha:
        raise ValueError("promotion disposition artifact changed")
    _project_promotion(store, path, rows, index, payload)


def _project_promotion(store, path, rows, index, payload):
    if rows[index] not in (payload["before"], payload["after"]):
        raise ValueError("promotion queue changed since committed disposition")
    ledger = path.parent / "learning_certificate_ledger.jsonl"
    old = ledger.read_text() if ledger.exists() else ""
    records = [json.loads(line) for line in old.splitlines() if line.strip()]
    record = payload["record"]
    matching = [row for row in records if row.get("publication_id") == record["publication_id"]]
    if matching and matching != [record]:
        raise ValueError("promotion learning projection differs from journal")
    if not matching:
        prefix = old + ("\n" if old and not old.endswith("\n") else "")
        store._replace_durable(ledger, prefix + json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    rows[index] = payload["after"]
    store._replace_durable(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    return payload["after"]


def publish_promotion_disposition(root, loop_id, *, record, changes, refund):
    """Publish the existing verdict, learning event and attempt accounting once.

    The campaign event owns the decision; queue and learning ledger are local
    recoverable projections. This function does not decide promotion eligibility.
    """
    from scripts.autotrain_paths import champion_queue_path
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.checkpoint_publication import controller_artifact_publication

    path = champion_queue_path(root, loop_id)
    store = CampaignStore(record["campaign_id"], root)
    if (record.get("loop_id") != loop_id or record.get("schema") != "autotrain_learning_event/v1"
        or type(refund) is not bool or changes.get("status") != record.get("outcome")
        or changes.get("status") not in CHAMPION_STATUSES
        or changes.get("entry_id", record.get("entry_id")) != record.get("entry_id")):
        raise ValueError("invalid promotion publication contract")
    json.dumps({"record": record, "changes": changes}, allow_nan=False)
    if not path.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink() for p in (path, path.parent, path.parent / "learning_certificate_ledger.jsonl")
    ):
        raise ValueError("promotion publication path is not confined")
    with controller_artifact_publication(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path.parent / ".promotion-publication.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            indices = [i for i, row in enumerate(rows) if row.get("entry_id") == record["entry_id"]]
            if len(indices) != 1:
                raise ValueError("promotion requires one existing queue entry")
            index = indices[0]
            _reconcile_promotion(store, path, rows, index)
            payload = _promotion_artifact(store, rows[index], record, changes, refund)
            return _project_promotion(store, path, rows, index, payload)
        finally:
            os.close(fd)


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
