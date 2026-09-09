"""Pinned driver tail and recoverable index in the existing campaign history.

The loop journal indexes one campaign; it never issues scientific verdicts.
Local writer exclusion supplements, not replaces, supervisor leases/isolation.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from scripts.autoresearch_command_cursor import resolved_continuation_grant
from slm_training.autoresearch.storage import CampaignStore, _sha


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


@contextmanager
def writer(root, loop_id):
    runtime = CampaignStore("runtime", Path(root) / "loops" / loop_id)
    runtime.root.mkdir(parents=True, exist_ok=True)
    with (runtime.root / ".cycle-continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield runtime
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


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


def retire(runtime, digest):
    runtime.append_event(
        "driver_cycle_retired",
        detail={"input_digest": digest},
        idempotency_key=f"driver-cycle-retired:{digest}",
    )


def register(store, runtime, payload):
    existing = active_reference(runtime)
    path = store.write_artifact("driver_cycle_inputs", payload)
    ref = {"campaign_id": store.campaign_id, "input_digest": path.stem}
    if existing is not None and existing != ref:
        raise ValueError("a different driver campaign is still pending")
    # Artifact -> recoverable loop reference -> campaign lock. If the last
    # append is interrupted, load_context validates the artifact and repairs it.
    runtime.append_event(
        "driver_cycle_registered",
        detail=ref,
        idempotency_key=f"driver-cycle-register:{path.stem}",
    )
    store.append_event(
        "driver_cycle_locked",
        artifact_sha256=path.stem,
        idempotency_key="driver-cycle-locked",
    )
    return path.stem


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


def verify_inputs(store, cwd, value):
    from slm_training.autoresearch.climb_policy import load_climb_policy

    current_identity = resolved_continuation_grant(cwd, value["total_seconds"]).execution_identity
    if value["execution_identity"] != current_identity:
        raise ValueError(
            "driver continuation release/environment/policy changed: "
            f"expected={value['execution_identity']} current={current_identity}"
        )
    if value["policy_sha256"] != load_climb_policy().sha256:
        raise ValueError("driver continuation release/environment/policy changed: policy")
    for filename, expected in value["files"].items():
        path = Path(filename)
        if not path.is_absolute():
            path = Path(cwd) / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"driver continuation input changed: {filename}")
    for eid, arm in value["arms"].items():
        if (
            store.load_experiment_campaign(eid).manifest_sha256
            != arm["manifest_digest"]
        ):
            raise ValueError("driver arm manifest lock changed")


def assert_locked_cycle_plan(store, experiment_id, commands, manifest):
    """cmd_run consumes the exact full plan locked before any sibling starts."""
    value = load_context(store)
    if value is None:
        return  # Historical/standalone cmd_run callers retain their own cursor.
    arm = value["arms"].get(experiment_id)
    if arm is None or arm["commands"] != commands or arm["manifest_digest"] != manifest:
        raise ValueError("compiled command plan differs from locked driver inputs")


class CycleJournal:
    """Executing/finalizing projection with real wall charges and crash reserve."""

    def __init__(self, store, value, *, started=None):
        self.store, self.value, self.digest = store, value, _sha(value)
        self.state = {
            "input_digest": self.digest,
            "phase": "arms",
            "index": 0,
            "arm_exits": {},
            "seen": [],
            "final_index": 0,
            "spent_seconds": value["initial_spent_seconds"],
            "attempt": 0,
            "inflight": None,
            "delivery": None,
            "resolution": None,
            "promotion_chunks": None,
        }
        for event in store.verify_event_chain():
            if event["event_type"] == "driver_cycle_checkpoint":
                data = read_artifact(
                    store, "driver_cycle_state", event["artifact_sha256"]
                )
                if data["input_digest"] != self.digest:
                    raise ValueError("driver state belongs to another context")
                self.state = data
        self._validate()
        self._recover_completed_write()
        self.started = time.monotonic() if started is None else started
        self.initial_spent = self.state["spent_seconds"]

    def _validate(self):
        state = self.state
        if (
            state["phase"] not in {"arms", "finalizing", "completed"}
            or type(state["index"]) is not int
            or not 0 <= state["index"] <= len(self.value["order"])
            or not math.isfinite(state["spent_seconds"])
            or state["spent_seconds"] < 0
            or state["seen"] != self.value["order"][: state["index"]]
            or set(state["arm_exits"]) != set(state["seen"])
            or any(type(code) is not int for code in state["arm_exits"].values())
            or type(state["final_index"]) is not int
            or state["final_index"] < 0
            or type(state["attempt"]) is not int
            or state["attempt"] < 0
            or (state["phase"] != "arms" and state["index"] != len(self.value["order"]))
        ):
            raise ValueError("invalid driver cursor or charge")

    def _recover_completed_write(self):
        if self.state["inflight"] is None:
            return
        matches = []
        for path in (self.store.root / "artifacts/driver_cycle_state").glob("*.json"):
            data = read_artifact(self.store, "driver_cycle_state", path.stem)
            if (
                data["input_digest"] == self.digest
                and data["attempt"] == self.state["attempt"]
                and data["inflight"] is None
                and data.get("settled_attempt") is True
            ):
                matches.append(data)
        if len(matches) > 1:
            raise ValueError("ambiguous driver attempt reconciliation")
        if matches:
            self.state = matches[0]
            self.save()

    @property
    def remaining(self):
        # Whole invocation overhead is charged, not just child-reported times.
        return (
            self.value["total_seconds"]
            - self.initial_spent
            - (time.monotonic() - self.started)
        )

    def save(self):
        self._validate()
        path = self.store.write_artifact("driver_cycle_state", self.state)
        self.store.append_event(
            "driver_cycle_checkpoint",
            artifact_sha256=path.stem,
            detail={"input_digest": self.digest},
            idempotency_key=f"driver-state:{path.stem}",
        )
        return path.stem

    def start(self, operation, reserve):
        self.state.update(
            attempt=self.state["attempt"] + 1,
            inflight={"operation": operation, "reserved_seconds": reserve},
            spent_seconds=self.initial_spent
            + time.monotonic()
            - self.started
            + reserve,
            settled_attempt=False,
        )
        self.save()

    def settle(self):
        self.state.update(
            inflight=None,
            settled_attempt=True,
            spent_seconds=self.initial_spent + time.monotonic() - self.started,
        )
        return self.save()

    def pending(self, reason, *, capability=False):
        if self.state["inflight"] is None:
            self.state["spent_seconds"] = (
                self.initial_spent + time.monotonic() - self.started
            )
        digest = self.save()
        result = {
            "schema_version": "driver_pending/v1",
            "measurement_complete": False,
            "outcome": "capability" if capability else "yielded",
            "reason": reason,
            "campaign_id": self.store.campaign_id,
            "input_digest": self.digest,
            "wake": {
                "predicate": "resume the locked driver command/finalization cursor",
                "source": "driver_cycle_checkpoint",
                "identity_digest": digest,
            },
        }
        if capability:
            result["blocker"] = {
                "kind": "repair_harness",
                "blocker_code": reason,
                "required_capability": "driver_continuation_reconciliation",
                "input_digest": self.digest,
                "original_outcome": self.state.get("last_yield"),
            }
        return result
