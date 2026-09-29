"""Immutable driver lock loading shared by physical execution and logical cursors."""

import json
import math
from slm_training.autoresearch.storage import _sha

def read_artifact(store, kind, digest):
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("invalid cycle artifact digest")
    data = json.loads((store.root / "artifacts" / kind / f"{digest}.json").read_text())
    if _sha(data) != digest:
        raise ValueError("driver continuation artifact changed")
    return data


def load_context(store, digest=None):
    events = [
        e
        for e in store.verify_event_chain()
        if e["event_type"] == "driver_cycle_locked"
    ]
    if not events and digest is None:
        return None
    expected = events[-1]["artifact_sha256"] if events else digest
    if digest is not None and digest != expected:
        raise ValueError("driver reference differs from campaign lock")
    value = read_artifact(store, "driver_cycle_inputs", expected)
    if (
        value["schema_version"] != "driver_cycle/v1"
        or value["campaign_id"] != store.campaign_id
        or value["total_seconds"] != store.load_campaign().budget.logical_seconds
        or not math.isfinite(value["initial_spent_seconds"])
        or value["initial_spent_seconds"] < 0
        or not value["order"]
        or len(value["order"]) != len(set(value["order"]))
        or set(value["order"]) != set(value["arms"])
    ):
        raise ValueError("invalid driver cycle input contract")
    if not events:
        store.append_event(
            "driver_cycle_locked",
            artifact_sha256=expected,
            idempotency_key="driver-cycle-locked",
        )
    return value


def active_reference(runtime):
    active = {}
    for event in runtime.verify_event_chain():
        if event["event_type"] == "driver_cycle_registered":
            active[event["detail"]["input_digest"]] = event["detail"]
        elif event["event_type"] == "driver_cycle_retired":
            active.pop(event["detail"]["input_digest"], None)
    if len(active) > 1:
        raise ValueError("multiple active driver contexts require reconciliation")
    return next(iter(active.values()), None)
