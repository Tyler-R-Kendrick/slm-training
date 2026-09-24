"""Cumulative accounting for explicit source-repair grant successors."""

from __future__ import annotations

import json
import re

from slm_training.autoresearch.heal.repair_contracts import RepairGrant


def repair_grant_budget(events, grant, journal):
    event_grants, request_grants = _stored_grant_bindings(events, journal)
    _reject_grant_id_reuse(events, grant, event_grants, request_grants)
    known = {}
    for stored in (*event_grants.values(), *request_grants.values()):
        previous = known.get(stored.grant_id)
        if previous and previous.accounting_digest() != stored.accounting_digest():
            raise ValueError("stored repair grant identity changed")
        known[stored.grant_id] = stored
    known.setdefault(grant.grant_id, grant)
    lineage, seen = [], set()
    current = grant
    while current is not None:
        if current.grant_id in seen:
            raise ValueError("repair grant successor cycle")
        seen.add(current.grant_id)
        lineage.append(current)
        current = known.get(current.successor_of) if current.successor_of else None
        if lineage[-1].successor_of and current is None:
            raise ValueError("repair grant predecessor is not recorded")
    grant_ids = {item.grant_id for item in lineage}
    reserved = sum(
        _event_reservation(event, grant_ids, event_grants, request_grants)
        for event in events
    )
    return (
        reserved,
        sum(item.total_seconds for item in lineage),
        min(10, sum(item.max_attempts for item in lineage)),
        known,
        event_grants,
        request_grants,
    )


def validate_blocker_grant_successor(
    events, grant, fingerprint, known, event_grants, request_grants
):
    used = _used_blocker_grants(events, fingerprint, event_grants, request_grants)
    if not used:
        if grant.successor_of:
            raise ValueError("repair grant successor has no blocker predecessor")
        return
    if grant.grant_id in used:
        if used[-1] != grant.grant_id:
            raise ValueError("repair grant predecessor already has a successor")
        return
    _validate_successor_link(grant, used, known)


def _used_blocker_grants(events, fingerprint, event_grants, request_grants):
    starts = [
        event
        for event in events
        if event["event_type"] == "repair_started"
        and event.get("detail", {}).get("fingerprint") == fingerprint
    ]
    used = []
    for event in starts:
        stored = _stored_event_grant(event, event_grants, request_grants)
        grant_id = event.get("detail", {}).get("grant_id") or (
            stored.grant_id if stored else None
        )
        if grant_id is None:
            raise ValueError("repair grant predecessor identity is missing")
        if not used or used[-1] != grant_id:
            used.append(grant_id)
    return used


def _validate_successor_link(grant, used, known):
    if not used:
        raise ValueError("repair grant successor has no blocker predecessor")
    if grant.successor_of != used[-1] or grant.successor_of not in known:
        raise ValueError("repair grant successor must extend the latest blocker grant")
    if any(
        other.grant_id != grant.grant_id and other.successor_of == grant.successor_of
        for other in known.values()
    ):
        raise ValueError("repair grant successor would fork the grant chain")
    if any(
        other.grant_id != grant.grant_id and other.successor_of == used[-1]
        for other in known.values()
    ):
        raise ValueError("repair grant predecessor already has a successor")


def _stored_grant_bindings(events, journal):
    event_grants, request_grants = {}, {}
    root = journal.root / "artifacts" / "repair_requests"
    for event in events:
        sha = event.get("artifact_sha256")
        if event["event_type"] != "repair_started" or not isinstance(sha, str):
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            continue
        try:
            value = json.loads((root / f"{sha}.json").read_text())
            stored = RepairGrant.model_validate_json(json.dumps(value["grant"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        event_grants[id(event)] = stored
        request_digest = event.get("detail", {}).get("request_digest")
        if isinstance(request_digest, str):
            request_grants[request_digest] = stored
    return event_grants, request_grants


def _stored_event_grant(event, event_grants, request_grants):
    return event_grants.get(id(event)) or request_grants.get(
        event.get("detail", {}).get("request_digest")
    )


def _reject_grant_id_reuse(events, grant, event_grants, request_grants):
    for event in events:
        detail = event.get("detail", {})
        stored = _stored_event_grant(event, event_grants, request_grants)
        grant_id = stored.grant_id if stored else detail.get("grant_id")
        accounting_digest = (
            stored.accounting_digest()
            if stored
            else detail.get("grant_accounting_digest")
        )
        if grant_id == grant.grant_id and accounting_digest != grant.accounting_digest():
            raise ValueError("repair grant ID reused with changed or unknown budget scope")


def _event_reservation(event, grant_ids, event_grants, request_grants):
    if event["event_type"] not in {
        "repair_started",
        "repair_verification_started",
        "operation_diagnosis_started",
    }:
        return 0.0
    detail = event.get("detail", {})
    stored = _stored_event_grant(event, event_grants, request_grants)
    grant_id = detail.get("grant_id") or (stored.grant_id if stored else None)
    return float(detail["reserved_seconds"]) if grant_id is None or grant_id in grant_ids else 0.0
