"""Append-only ledgers and queues written by the continuous autotrain loop.

Extracted from ``scripts/run_autotrain_continuous.py``. One responsibility:
appending a cycle's observations to the loop's durable ledgers -- hillclimb
iterations, the residual eval queue, interesting residuals, historical
reclassifications and the slug-stats ledger -- plus the paths those ledgers
live at and the reader that loads residual observations back.

Deliberately excluded: the dynamic-thrash-arm bank. That bank is a process
cache the runner rebinds with ``global``, and a rebinding does not cross a
module boundary -- an alias import would silently read a stale list. It stays
with ``_load_dynamic_thrash_arms`` in the runner.

Layered above ``autotrain_provenance``; it imports no runner state, so nothing
here can reach back into the loop. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
import fcntl
import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_provenance import checkpoint_path_for_candidate
from slm_training.autoresearch.thrash_residuals import (
    ResidualObservation,
    build_slug_stats_payload,
    classify_delivery_residual,
    residual_boosts_from_observations,
)


def _delivery_artifact(store, record):
    from slm_training.lineage.records import canonical_json

    # Time of a retry is not another comparison. All decision-bearing fields
    # remain part of the key; this does not authenticate worker-supplied output.
    identity = {key: value for key, value in record.items() if key not in {"finished_at", "publication_id"}}
    key = "delivery:" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    if record.get("publication_id", key) != key:
        raise ValueError("delivery publication identity mismatch")
    events = store.verify_event_chain()
    previous = next((row for row in reversed(events) if row.get("idempotency_key") == key), None)
    if previous is not None:
        latest = next(row for row in reversed(events) if row["event_type"] == "cycle_delivery_published")
        if latest != previous:
            raise ValueError("stale delivery cannot replace a newer committed outcome")
        artifact = store.root / "artifacts/cycle_deliveries" / f"{previous['artifact_sha256']}.json"
        stored = json.loads(artifact.read_text())
        if hashlib.sha256(canonical_json(stored).encode()).hexdigest() != artifact.stem:
            raise ValueError("committed delivery artifact changed")
        return stored
    stored = {**record, "publication_id": key}
    artifact = store.write_artifact("cycle_deliveries", stored)
    projection = store.root / "sdlc_delivery.json"
    before = store.write_artifact("cycle_deliveries", json.loads(projection.read_text())) if projection.exists() else None
    store.append_event("cycle_delivery_published", artifact_sha256=artifact.stem,
                       detail={"publication_id": key, "previous_delivery_sha256": before.stem if before else None},
                       idempotency_key=key)
    return stored


def _project_delivery(root, store, record):
    ledger = root / "sdlc_delivery_ledger.jsonl"
    projection = store.root / "sdlc_delivery.json"
    if ledger.is_symlink() or projection.is_symlink():
        raise ValueError("delivery projection cannot follow a symlink")
    old = ledger.read_text() if ledger.exists() else ""
    rows = [json.loads(line) for line in old.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("malformed legacy delivery ledger")
    matching = [row for row in rows if row.get("publication_id") == record["publication_id"]]
    if any(row != record for row in matching):
        raise ValueError("delivery ledger differs from committed artifact")
    if not matching:
        prefix = old + ("\n" if old and not old.endswith("\n") else "")
        store._replace_durable(ledger, prefix + json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    store._replace_durable(projection, json.dumps(record, indent=2, allow_nan=False) + "\n")


def validate_cycle_delivery(record, *, campaign_id=None, loop_id=None):
    """Shared publication/consumption contract; validation never publishes."""
    if (not isinstance(record, dict)
        or record.get("schema") != "autotrain_sdlc_delivery/v1"
        or type(record.get("positive")) is not bool
        or type(record.get("measurement_complete")) is not bool
        or not isinstance(record.get("loop_id"), str) or not record["loop_id"]
        or not isinstance(record.get("campaign_id"), str) or not record["campaign_id"]
        or (campaign_id is not None and record["campaign_id"] != campaign_id)
        or (loop_id is not None and record["loop_id"] != loop_id)):
        raise ValueError("invalid controller delivery contract")
    json.dumps(record, allow_nan=False)


def publish_cycle_delivery(root: Path, record: dict) -> dict:
    """Journal-first delivery with idempotent local projections, not a verdict.

    Existing ledger lines remain historical. A crash after either projection
    reconciles from the same committed artifact, without rerunning measurement.
    """
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.checkpoint_publication import controller_artifact_publication

    validate_cycle_delivery(record)
    store = CampaignStore(record["campaign_id"], root)
    with controller_artifact_publication(store.root):
        for path in (root, store.root, root / "sdlc_delivery_ledger.jsonl", store.root / "sdlc_delivery.json"):
            if path.is_symlink():
                raise ValueError("delivery publication cannot follow a symlink")
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(root / ".sdlc-delivery.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            stored = _delivery_artifact(store, record)
            _project_delivery(root, store, stored)
            return stored
        finally:
            os.close(fd)


def dynamic_thrash_arms_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "dynamic_thrash_arms.jsonl"


def hillclimb_iteration_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "hillclimb_iterations.jsonl"


def append_hillclimb_iteration(
    root: Path,
    loop_id: str,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Append one cycle report; return the loaded ledger (including this row)."""
    path = hillclimb_iteration_path(root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(report), sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def append_historical_reclassification(
    root: Path,
    loop_id: str,
    event: dict[str, Any],
) -> None:
    path = root / "loops" / loop_id / "historical_reclassification.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def interesting_residuals_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "interesting_residuals.jsonl"


def slug_stats_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "slug_stats.json"


def load_residual_observations(root: Path, loop_id: str) -> list[ResidualObservation]:
    path = interesting_residuals_path(root, loop_id)
    if not path.is_file():
        return []
    out: list[ResidualObservation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                ResidualObservation(
                    campaign_id=str(row.get("campaign_id") or ""),
                    cycle_index=(
                        int(row["cycle_index"])
                        if row.get("cycle_index") is not None
                        else None
                    ),
                    slug=row.get("slug") if isinstance(row.get("slug"), str) else None,
                    residual_class=(
                        str(row["residual_class"])
                        if row.get("residual_class") is not None
                        else None
                    ),
                    score=float(row.get("score") or 0.0),
                    ss_control=row.get("ss_control"),
                    ss_cand=row.get("ss_cand"),
                    mpr_control=row.get("mpr_control"),
                    mpr_cand=row.get("mpr_cand"),
                    binder_control=row.get("binder_control"),
                    binder_cand=row.get("binder_cand"),
                    positive=row.get("positive")
                    if isinstance(row.get("positive"), bool)
                    else None,
                    measurement_complete=row.get("measurement_complete")
                    if isinstance(row.get("measurement_complete"), bool)
                    else None,
                    reasons=tuple(str(r) for r in (row.get("reasons") or ())),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def iter_loop_deliveries(
    root: Path, loop_id: str, *, limit: int = 120
) -> list[dict[str, Any]]:
    """Load recent continuous deliveries for this loop (newest first, capped)."""
    rows: list[tuple[int, dict[str, Any]]] = []
    for path in root.glob("continuous-loop-*/sdlc_delivery.json"):
        parent = path.parent.name
        if loop_id not in parent:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        match = re.search(r"-c(\d+)$", parent)
        cycle = int(match.group(1)) if match else 0
        data = {**data, "campaign_id": data.get("campaign_id") or parent}
        rows.append((cycle, data))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [data for _, data in rows[: max(1, int(limit))]]


def refresh_slug_stats_ledger(root: Path, loop_id: str) -> None:
    """Regenerate loops/<id>/slug_stats.json from recent deliveries + residuals."""
    deliveries = iter_loop_deliveries(root, loop_id, limit=120)
    observations = load_residual_observations(root, loop_id)
    boosts = residual_boosts_from_observations(observations)
    payload = build_slug_stats_payload(
        loop_id=loop_id,
        deliveries=deliveries,
        residuals=observations,
        residual_boosts=boosts,
    )
    path = slug_stats_path(root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_residual_eval_queue(
    root: Path,
    loop_id: str,
    obs: ResidualObservation,
    *,
    checkpoint: Path | None,
) -> None:
    """Queue optional eval-only follow-up for actionable residuals (no auto-train)."""
    if obs.residual_class in {None, "control_spike_shared"}:
        return
    if obs.residual_class not in {
        "primary_up_binder_down",
        "efficiency_win_quality_held",
        "high_band_absolute",
    }:
        return
    path = root / "loops" / loop_id / "residual_eval_queue.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "residual_eval_queue/v1",
        "campaign_id": obs.campaign_id,
        "cycle_index": obs.cycle_index,
        "slug": obs.slug,
        "residual_class": obs.residual_class,
        "score": obs.score,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "status": "queued" if checkpoint else "no_checkpoint",
        "note": (
            "eval-only confirm-lite candidate; run evaluate_model --checkpoint "
            "when wall budget allows — does not auto-promote"
        ),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def append_interesting_residual(
    root: Path,
    loop_id: str,
    delivery: Mapping[str, Any],
    *,
    campaign_id: str,
    cycle_index: int | None,
) -> ResidualObservation | None:
    """Classify delivery and append interesting residual to loop ledger."""
    obs = classify_delivery_residual(
        delivery,
        campaign_id=campaign_id,
        cycle_index=cycle_index,
    )
    if obs is None or not obs.residual_class:
        return None
    path = interesting_residuals_path(root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obs.as_dict(), sort_keys=True) + "\n")
    cand_id = str(delivery.get("candidate_id") or "") or None
    ckpt = checkpoint_path_for_candidate(root, campaign_id, cand_id)
    try:
        append_residual_eval_queue(root, loop_id, obs, checkpoint=ckpt)
    except Exception as exc:  # noqa: BLE001
        print(f"THRASH_RESIDUAL_EVAL_QUEUE_WARN err={exc!r}", flush=True)
    try:
        refresh_slug_stats_ledger(root, loop_id)
    except Exception as exc:  # noqa: BLE001
        print(f"THRASH_SLUG_STATS_WARN err={exc!r}", flush=True)
    print(
        f"THRASH_RESIDUAL class={obs.residual_class} slug={obs.slug} "
        f"score={obs.score:.3f} campaign={campaign_id}"
        + (f" ckpt={ckpt.name}" if ckpt else ""),
        flush=True,
    )
    return obs
