"""Repairing champion entries the loop left in a bad state.

One responsibility: reopening entries blocked by a harness gap rather than a
real result, revalidating confirmed entries, and recovering entries interrupted
mid-cycle.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from scripts.autotrain_campaign import campaign_started_experiment
from scripts.autotrain_candidate_state import confirmation_quality_reheld
from scripts.autotrain_diagnosis import reasons_are_harness_incomplete_only
from scripts.autotrain_io import read_json


def reopen_harness_blocked_champions(
    root: Path,
    entries: list[dict[str, Any]],
    *,
    integration_commit: str | None,
) -> bool:
    """Rearm harness-blocked champions after a harness/code fix.

    General pattern: harness/infra incompletes never permanently invalidate an
    experiment. When the integrated tip changes, retry the same champion
    recipe under the fixed harness. Model rejects stay terminal.
    """
    if not integration_commit:
        return False
    changed = False
    for row in entries:
        status = str(row.get("status") or "")
        reasons = row.get("resolve_reasons") or []
        harness_blocked = status == "harness_failure" or (
            status == "promotion_failed"
            and (
                row.get("last_harness_failure")
                or reasons_are_harness_incomplete_only(reasons)
            )
        )
        if not harness_blocked:
            continue
        failed_on = str(row.get("harness_failure_integration_commit") or "")
        # First time we see a harness park without a stamp: treat as blocked on
        # current tip until a later tip arrives.
        if not failed_on:
            row["harness_failure_integration_commit"] = integration_commit
            changed = True
            continue
        if failed_on == integration_commit:
            continue
        prior = status
        row["status"] = "confirmed"
        row["promote_attempts"] = 0
        row["last_harness_failure"] = False
        row.pop("last_harness_failure_at", None)
        row.pop("resolved_at", None)
        row["resolve_reasons"] = [
            "harness_retry_after_integration_change",
            f"prior_status:{prior}",
            f"failed_on:{failed_on}",
            f"retry_on:{integration_commit}",
            *[str(r) for r in reasons if str(r).startswith("harness_failure:")],
        ]
        row["harness_failure_integration_commit"] = integration_commit
        changed = True
        print(
            "CHAMPION_HARNESS_RETRY "
            f"entry_id={row.get('entry_id')} prior={prior} "
            f"failed_on={failed_on[:12]} retry_on={integration_commit[:12]}",
            flush=True,
        )
    return changed


def recover_interrupted_champion_entries(
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
            delivery = read_json(delivery_path)
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
        started = campaign_started_experiment(root, campaign_id)
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


def revalidate_confirmed_champion_entries(
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
        delivery = read_json(delivery_path)
        if delivery.get("measurement_complete") is not True:
            continue
        if confirmation_quality_reheld(delivery):
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
