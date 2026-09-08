"""CampaignStore local event transactions and recoverable projections.

No independent verdict or index authority lives here; callers retain the
canonical store API and hash-chain validation.
"""

from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from slm_training.lineage.records import canonical_json
from .schemas import AutotrainCycleHandoffV1
from .experiment_campaign import CampaignLockV1, CampaignDeviationV1
from .storage import _sha, utc_now


def verify_event_chain(self) -> list[dict[str, Any]]:
    """Fail closed on edited, reordered, deleted-link, or forked events."""
    path = self.root / "events.jsonl"
    if not path.exists():
        return []
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    previous = ""
    seen: set[str] = set()
    for event in events:
        event_id = str(event.get("event_id", ""))
        payload = dict(event)
        payload.pop("event_id", None)
        if event_id != _sha(payload):
            raise RuntimeError("event_id digest mismatch")
        if event_id in seen:
            raise RuntimeError("duplicate event_id in campaign chain")
        if str(event.get("previous_event_sha256", "")) != previous:
            raise RuntimeError("campaign event chain is broken or forked")
        seen.add(event_id)
        previous = event_id
        event_type = str(event.get("event_type", ""))
        if event_type in {
            "experiment_campaign_locked",
            "campaign_deviation_appended",
        }:
            kind = (
                "experiment_campaigns"
                if event_type == "experiment_campaign_locked"
                else "campaign_deviations"
            )
            artifact_sha = str(event.get("artifact_sha256", ""))
            artifact_path = self.root / "artifacts" / kind / f"{artifact_sha}.json"
            try:
                artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"missing or invalid governed artifact: {artifact_path}"
                ) from exc
            if _sha(artifact_payload) != artifact_sha:
                raise RuntimeError("governed artifact content digest mismatch")
            if event_type == "experiment_campaign_locked":
                CampaignLockV1.model_validate(artifact_payload)
            else:
                CampaignDeviationV1.model_validate(artifact_payload)
    return events


def lock_learning_trial(store, run_id: str, inputs: dict, *, resumable: bool) -> Path:
    """Append release changes, never rewrite a locked recipe or claim a resume.

    The caller holds the trial lock; the trainer independently validates the
    full-state compatibility migration before consuming any further examples.
    """
    history = store.verify_event_chain()
    key = f"learning-trial:{run_id}"
    prior = [
        row
        for row in history
        if row.get("idempotency_key") == key
        or (
            row["event_type"] == "learning_trial_release_successor"
            and row["detail"].get("run_id") == run_id
        )
    ]
    artifact = store.write_artifact("learning_trial_input", inputs)
    detail = {}
    event_type = "learning_trial_locked"
    if prior:
        old_sha = prior[-1]["artifact_sha256"]
        old = json.loads((artifact.parent / f"{old_sha}.json").read_text())
        if _sha(old) != old_sha:
            raise ValueError("locked trial input digest mismatch")
        if old == inputs:
            return artifact
        if not resumable or {**old, "versions": inputs["versions"]} != inputs:
            raise ValueError("trial idempotency conflict: immutable inputs changed")
        event_type = "learning_trial_release_successor"
        detail = {
            "run_id": run_id,
            "previous_input_sha256": old_sha,
            "previous_event_id": prior[-1]["event_id"],
            "resume_verified": False,
        }
        key = f"{key}:release:{artifact.stem}"
    store.append_event(
        event_type,
        artifact_sha256=artifact.stem,
        detail=detail,
        idempotency_key=key,
        expected_parent=history[-1]["event_id"] if history else "",
    )
    return artifact


def publish_cycle_handoff(self, handoff: AutotrainCycleHandoffV1) -> Path:
    """Commit a producer's handoff, then reconcile its single display pointer.

    This is publication, not another scientific decision engine. Successors
    preserve the exact previous handoff artifact instead of rewriting history.
    """
    from slm_training.harness_core.checkpoint_publication import controller_artifact_publication

    handoff = AutotrainCycleHandoffV1.model_validate(handoff.model_dump(mode="json"))
    if handoff.campaign_id != self.campaign_id:
        raise ValueError("handoff belongs to another campaign")
    with controller_artifact_publication(self.root) as fence:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".handoff.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._reconcile_handoff_revision()
            path = self.root / "cycle_handoff.json"
            before = self.write_artifact("handoff_revisions", json.loads(path.read_text())) if path.exists() else None
            after = self.write_artifact("handoff_revisions", handoff)
            published = any(row["event_type"] in {"cycle_handoff_published", "handoff_actions_revised"}
                            for row in self.verify_event_chain())
            if not published or before is None or before.stem != after.stem:
                self.append_event(
                    "cycle_handoff_published", artifact_sha256=after.stem,
                    detail={"previous_handoff_sha256": before.stem if before else None,
                            "fence": fence, "lease_verified": fence is not None},
                    idempotency_key=f"handoff-published:{before.stem if before else 'initial'}:{after.stem}",
                )
            self._reconcile_handoff_revision()
            return path


def revise_handoff_actions(self, handoff: AutotrainCycleHandoffV1) -> Path:
    """Append an operational remedy revision, retaining the original handoff.

    Scientific fields are immutable. The named JSON is a recoverable display
    projection of the revision event, not a replacement historical verdict.
    """
    handoff = AutotrainCycleHandoffV1.model_validate(handoff.model_dump(mode="json"))
    self.root.mkdir(parents=True, exist_ok=True)
    fd = os.open(self.root / ".handoff.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        self._reconcile_handoff_revision()
        path = self.root / "cycle_handoff.json"
        original = json.loads(path.read_text())
        previous = AutotrainCycleHandoffV1.model_validate(original)
        if handoff.campaign_id != self.campaign_id:
            raise ValueError("handoff belongs to another campaign")
        if previous.model_dump(exclude={"actions"}) != handoff.model_dump(
            exclude={"actions"}
        ):
            raise ValueError("operational revision may only change remedy actions")
        if previous == handoff:
            return path
        before = self.write_artifact("handoff_revisions", original)
        after = self.write_artifact("handoff_revisions", handoff)
        self.append_event(
            "handoff_actions_revised",
            artifact_sha256=after.stem,
            detail={"previous_handoff_sha256": before.stem},
            idempotency_key=f"handoff:{before.stem}:{after.stem}",
        )
        self._reconcile_handoff_revision()
        return path
    finally:
        os.close(fd)


def _reconcile_handoff_revision(self) -> None:
    revisions = [
        row
        for row in self.verify_event_chain()
        if row["event_type"] in {"handoff_actions_revised", "cycle_handoff_published"}
    ]
    if not revisions:
        return
    event = revisions[-1]
    path = self.root / "cycle_handoff.json"
    current = _sha(json.loads(path.read_text())) if path.exists() else None
    target = event["artifact_sha256"]
    if current not in {None, target, event["detail"]["previous_handoff_sha256"]}:
        raise RuntimeError("handoff projection differs from committed revision")
    artifact = self.root / "artifacts" / "handoff_revisions" / f"{target}.json"
    payload = json.loads(artifact.read_text())
    if _sha(payload) != target:
        raise RuntimeError("handoff revision content digest mismatch")
    AutotrainCycleHandoffV1.model_validate(payload)
    if current != target:
        self._replace_durable(path, artifact.read_text())


def append_event(
    self,
    event_type: str,
    *,
    experiment_id: str = "",
    status: str = "",
    artifact_sha256: str = "",
    detail: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    expected_parent: str | None = None,
) -> dict[str, Any]:
    events = self.root / "events.jsonl"
    lock_path = self.root / ".events.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        history = self.verify_event_chain()
        previous = history[-1]["event_id"] if history else ""
        payload = {
            "event_type": event_type,
            "experiment_id": experiment_id,
            "status": status,
            "artifact_sha256": artifact_sha256,
            "detail": detail or {},
        }
        if idempotency_key is not None:
            if not idempotency_key:
                raise ValueError("empty event idempotency key")
            for prior in history:
                if prior.get("idempotency_key") == idempotency_key:
                    if any(prior.get(key) != value for key, value in payload.items()):
                        raise ValueError(
                            "event idempotency key reused with different input"
                        )
                    self._reconcile_event_projection(history)
                    return prior
        if expected_parent is not None and previous != expected_parent:
            raise ValueError("stale campaign event parent")
        event = {
            "timestamp": utc_now(),
            "campaign_id": self.campaign_id,
            **payload,
            "previous_event_sha256": previous,
        }
        if idempotency_key is not None:
            event["idempotency_key"] = idempotency_key
        event["event_id"] = _sha(event)
        # ponytail: whole-journal replacement; segment after measured replay latency exhausts finalization reserve.
        # Logical append-only history is replaced atomically: a failed write
        # cannot leave a truncated final JSON row that blocks all recovery.
        self._replace_durable(
            events,
            "".join(canonical_json(item) + "\n" for item in [*history, event]),
        )
        self._reconcile_event_projection([*history, event])
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return event


def _replace_durable(path: Path, data: str) -> None:
    """Atomic local-filesystem publication; not a multi-host guarantee."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def _reconcile_event_projection(self, history: list[dict[str, Any]]) -> None:
    """results.tsv is a disposable projection, never a second event authority."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=self.EVENT_COLUMNS, delimiter="\t")
    writer.writeheader()
    for event in history:
        row = {key: event.get(key, "") for key in self.EVENT_COLUMNS}
        row["detail"] = canonical_json(event.get("detail") or {})
        writer.writerow(row)
    self._replace_durable(self.root / "results.tsv", stream.getvalue())


def reconcile_event_projection(self) -> None:
    """Recover a missing/stale display after committed event publication."""
    self.root.mkdir(parents=True, exist_ok=True)
    with (self.root / ".events.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            self._reconcile_event_projection(self.verify_event_chain())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_new(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise
    finally:
        tmp.unlink(missing_ok=True)


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        pending = memoryview(line.encode("utf-8"))
        while pending:
            count = os.write(fd, pending)
            if count <= 0:
                raise OSError("short write made no progress")
            pending = pending[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
