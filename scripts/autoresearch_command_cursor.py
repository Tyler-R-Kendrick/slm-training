"""Command checkpoints projected from CampaignStore, not a second scheduler.

Local writer exclusion does not replace the controller's leases or sandbox.
"""

from __future__ import annotations

import fcntl
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

from slm_training.autoresearch.schemas import ExperimentOutcome
from slm_training.autoresearch.storage import _sha


@dataclass(frozen=True)
class ContinuationGrant:
    """Controller-resolved identity and finite logical wall allowance."""

    execution_identity: str
    total_wall_seconds: float
    max_attempts: int | None = None
    interrupt_seconds: float | None = None
    finalization_reserve_seconds: float | None = None


def resolved_continuation_grant(
    root: Path,
    total_seconds: float,
    max_attempts=None,
    *,
    interrupt_seconds: float | None = None,
    finalization_reserve_seconds: float | None = None,
) -> ContinuationGrant:
    """Reuse the release/environment identity owners, excluding incidental clocks."""
    from scripts.merge_verification_evidence import (
        environment_identity,
        source_identity,
    )
    from slm_training.harness_core.activity_contract import contract_digest
    from slm_training.harness_core.execution_release import runtime_source_identity

    identity = contract_digest(
        {
            "source": runtime_source_identity(root) or source_identity(root),
            "environment": environment_identity(),
        }
    )
    return ContinuationGrant(
        identity, total_seconds, max_attempts,
        interrupt_seconds, finalization_reserve_seconds,
    )


def record_execution_outcome(store, outcome, manifest, *, pending):
    """A bounded yield is durable work, not finished experiment/feedback evidence."""
    path = store.write_artifact("outcomes", outcome)
    store.append_event(
        "experiment_yielded" if pending else "experiment_finished",
        experiment_id=outcome.experiment_id,
        status=outcome.status,
        artifact_sha256=path.stem,
        detail={"exit_code": outcome.exit_code, "campaign_manifest_sha256": manifest},
        idempotency_key=f"execution-outcome:{outcome.experiment_id}:{path.stem}",
    )
    return pending


def yielded_outcome_since(store, event_ids, experiment_id, manifest):
    """Read one fresh identity-bound yield; callers also classify its pending type.

    Exit 10/stdout alone never authorizes a driver continuation. Capture event
    IDs before launch, then check this result with is_continuation_pending.
    """
    rows = [
        row
        for row in store.verify_event_chain()
        if row["event_type"] == "experiment_yielded"
        and row["experiment_id"] == experiment_id
        and row["event_id"] not in event_ids
    ]
    if len(rows) != 1:
        raise ValueError("missing or ambiguous current experiment yield")
    row = rows[0]
    digest = row["artifact_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("invalid yielded outcome digest")
    data = json.loads(
        (store.root / "artifacts/outcomes" / f"{digest}.json").read_text()
    )
    outcome = ExperimentOutcome.model_validate(data)
    if (
        _sha(data) != digest
        or outcome.campaign_id != store.campaign_id
        or outcome.experiment_id != experiment_id
        or outcome.status != "stopped"
        or outcome.campaign_manifest_sha256 != manifest
        or row["detail"].get("campaign_manifest_sha256") != manifest
    ):
        raise ValueError("yielded outcome identity or content mismatch")
    return outcome


class CommandCursor:
    def __init__(self, store, experiment, commands, manifest, identity, total, *, cwd, max_attempts=None):
        if type(total) not in (int, float) or not math.isfinite(total) or total <= 0:
            raise ValueError("continuation total allowance must be positive and finite")
        if store is not None and (
            store.campaign_id != experiment.campaign_id
            or not isinstance(identity, str)
            or not identity
        ):
            raise ValueError(
                "durable continuation requires matching store and execution identity"
            )
        self.store, self.experiment = store, experiment
        self.inputs = dict(
            schema="command_cursor/v1",
            experiment=experiment.model_dump(mode="json"),
            commands=commands,
            manifest=manifest,
            identity=identity,
            total=float(total),
            cwd=str(Path(cwd).resolve()),
        )
        self.digest = _sha(self.inputs)
        if max_attempts is not None:
            if type(max_attempts) is not int or max_attempts < 1:
                raise ValueError("continuation attempts must be a positive integer")
            self.inputs["max_attempts"] = max_attempts
            self.digest = _sha(self.inputs)
        self.position, self.spent, self.attempt = 0, 0.0, 0
        self.outcome = None
        self.unresolved = False
        self.reserved = 0.0
        self._lock = None
        self._invocation = None
        self._observed = 0.0
        self._checkpoint_count = 0
        self._invocation_id = uuid.uuid4().hex
        self._launched = False
        self._unaccounted_reserve = 0.0

    @property
    def remaining(self):
        return max(0.0, self.inputs["total"] - self.spent)

    @property
    def observed_seconds(self):
        return self._observed

    def __enter__(self):
        if self.store is None:
            return self
        self.store.root.mkdir(parents=True, exist_ok=True)
        lock_name = _sha(self.experiment.experiment_id)
        self._lock = (self.store.root / f".command-cursor-{lock_name}.lock").open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._replay()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        try:
            if self._invocation is not None and not self.unresolved and self._launched:
                started, clock = self._invocation
                overhead = max(0.0, clock() - started - self._observed)
                self._event("overhead", spent_seconds=overhead)
                self.spent += overhead
        finally:
            if self._lock is not None:
                self._lock.close()

    def track_invocation(self, started, clock):
        """Charge observed controller overhead through final accounting emission."""
        self._invocation = started, clock

    def _replay(self):
        found = False
        for event in self.store.verify_event_chain():
            if event["experiment_id"] != self.experiment.experiment_id or not event[
                "event_type"
            ].startswith("command_cursor_"):
                continue
            detail = event["detail"]
            if detail["input_digest"] != self.digest:
                raise ValueError(
                    "command cursor immutable inputs changed; successor identity required"
                )
            found = True
            self._apply_event(event)
        if not found:
            artifact = self.store.write_artifact("command_cursor_inputs", self.inputs)
            self._event("locked", artifact=artifact.stem)
        elif self.unresolved:
            self._recover_commit()
        if not self.unresolved and self._unaccounted_reserve:
            self._event(
                "overhead",
                spent_seconds=self._unaccounted_reserve,
                charge_kind="unobserved_finalization_reservation",
            )
            self.spent += self._unaccounted_reserve
            self._unaccounted_reserve = 0.0

    def _recover_commit(self):
        """Reconcile a controller-written result whose terminal append crashed."""
        matches = []
        for path in (self.store.root / "artifacts/command_cursors").glob("*.json"):
            payload = json.loads(path.read_text())
            if (
                payload.get("input_digest") == self.digest
                and payload.get("attempt") == self.attempt
                and payload.get("record_type") == "committed"
            ):
                matches.append(path)
        if len(matches) > 1:
            raise ValueError("command cursor ambiguous uncommitted results")
        if matches:
            event = {"artifact_sha256": matches[0].stem}
            self._restore(event, committed=True)
            self._event("committed", artifact=matches[0].stem)

    def _apply_event(self, event):
        kind, detail = event["event_type"], event["detail"]
        if kind == "command_cursor_started":
            if self.unresolved or detail["attempt"] != self.attempt + 1:
                raise ValueError("command cursor invalid attempt sequence")
            self.attempt = detail["attempt"]
            self.reserved = detail["reserved_seconds"]
            if (
                type(self.reserved) not in (int, float)
                or not math.isfinite(self.reserved)
                or self.reserved <= 0
            ):
                raise ValueError("command cursor invalid reservation")
            self.spent += self.reserved
            self.unresolved = True
        elif kind in ("command_cursor_committed", "command_cursor_checkpoint"):
            self._restore(event, committed=kind == "command_cursor_committed")
        elif kind == "command_cursor_overhead":
            charge = detail["spent_seconds"]
            if (
                type(charge) not in (int, float)
                or not math.isfinite(charge)
                or charge < 0
            ):
                raise ValueError("command cursor invalid overhead")
            self.spent += charge
            self._unaccounted_reserve = 0.0
        elif kind == "command_cursor_locked":
            path = (
                self.store.root
                / "artifacts/command_cursor_inputs"
                / f"{event['artifact_sha256']}.json"
            )
            if (
                json.loads(path.read_text()) != self.inputs
                or event["artifact_sha256"] != self.digest
            ):
                raise ValueError("command cursor locked artifact mismatch")
        else:
            raise ValueError("unknown command cursor event")

    def _restore(self, event, *, committed):
        if not self.unresolved:
            raise ValueError("command cursor result without a started attempt")
        path = (
            self.store.root
            / "artifacts/command_cursors"
            / f"{event['artifact_sha256']}.json"
        )
        payload = json.loads(path.read_text())
        if (
            _sha(payload) != event["artifact_sha256"]
            or payload["input_digest"] != self.digest
            or set(payload)
            != {
                "input_digest",
                "outcome",
                "position",
                "spent_seconds",
                "attempt",
                "record_type",
            }
            or payload["attempt"] != self.attempt
            or payload["record_type"] != ("committed" if committed else "checkpoint")
        ):
            raise ValueError("command cursor artifact digest mismatch")
        position, spent = payload["position"], payload["spent_seconds"]
        if (
            type(position) is not int
            or not self.position <= position <= len(self.inputs["commands"])
            or type(spent) not in (int, float)
            or not math.isfinite(spent)
            or spent < 0
        ):
            raise ValueError("command cursor invalid position or charge")
        restored = ExperimentOutcome.model_validate(payload["outcome"])
        self._validate_outcome(restored)
        self.outcome, self.position = restored, position
        if committed:
            self.spent += spent - self.reserved
            self._unaccounted_reserve = max(0.0, self.reserved - spent)
            self.unresolved = False

    def _validate_outcome(self, outcome):
        if (
            outcome.experiment_id != self.experiment.experiment_id
            or outcome.campaign_id != self.experiment.campaign_id
            or outcome.campaign_manifest_sha256 != self.inputs["manifest"]
        ):
            raise ValueError("command cursor outcome identity mismatch")

    def _event(self, operation, *, artifact="", **detail):
        if self.store is not None:
            suffix = str(detail["checkpoint"]) if operation == "checkpoint" else ""
            if operation == "overhead":
                suffix = self._invocation_id
            self.store.append_event(
                "command_cursor_" + operation,
                experiment_id=self.experiment.experiment_id,
                artifact_sha256=artifact,
                detail={"input_digest": self.digest, **detail},
                idempotency_key=f"command-cursor:{self.digest}:{self.attempt}:{operation}:{suffix}",
            )

    def start(self, allowance):
        self._launched = True
        self.attempt += 1
        self.reserved = allowance
        self.spent += allowance  # A lost result retains the entire reservation.
        self.unresolved = True
        self._event("started", attempt=self.attempt, reserved_seconds=allowance)

    def checkpoint(self, outcome, position):
        self._validate_outcome(outcome)
        payload = dict(
            input_digest=self.digest,
            outcome=outcome.model_dump(mode="json"),
            position=position,
            spent_seconds=0.0,
            attempt=self.attempt,
            record_type="checkpoint",
        )
        self._checkpoint_count += 1
        artifact = self.store.write_artifact("command_cursors", payload)
        self._event(
            "checkpoint", artifact=artifact.stem, checkpoint=self._checkpoint_count
        )

    def commit(self, outcome, position, spent):
        self._validate_outcome(outcome)
        payload = dict(
            input_digest=self.digest,
            outcome=outcome.model_dump(mode="json"),
            position=position,
            spent_seconds=spent,
            attempt=self.attempt,
            record_type="committed",
        )
        artifact = (
            self.store.write_artifact("command_cursors", payload)
            if self.store
            else None
        )
        self._event("committed", artifact=artifact.stem if artifact else "")
        self.outcome, self.position = outcome, position
        self.spent += spent - self.reserved
        self._observed += spent
        self.unresolved = False

    def result(self, error=None):
        if self.unresolved:
            error = "continuation_reconciliation_required:attempt_result_missing"
        base = self.outcome or ExperimentOutcome(
            experiment_id=self.experiment.experiment_id,
            campaign_id=self.experiment.campaign_id,
            status="stopped",
            campaign_manifest_sha256=self.inputs["manifest"],
        )
        return (
            base.model_copy(update={"status": "stopped", "error": error})
            if error
            else base
        )
