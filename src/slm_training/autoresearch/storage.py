"""Content-addressed campaign artifacts and append-only telemetry."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slm_training.autoresearch.schemas import CampaignSpec, utc_now
from slm_training.autoresearch.experiment_campaign import (
    CampaignDeviationV1,
    CampaignLockV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.lineage.records import canonical_json


def _payload(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class CampaignStore:
    """One durable local source of truth; remote sinks mirror this directory."""

    EVENT_COLUMNS = (
        "timestamp",
        "event_id",
        "previous_event_sha256",
        "event_type",
        "campaign_id",
        "experiment_id",
        "status",
        "artifact_sha256",
        "detail",
    )

    def __init__(
        self,
        campaign_id: str,
        root: Path | str = Path("outputs/autoresearch"),
    ) -> None:
        self.campaign_id = campaign_id
        self.root = Path(root) / campaign_id

    def initialize(self, campaign: CampaignSpec) -> Path:
        if campaign.campaign_id != self.campaign_id:
            raise ValueError("campaign id does not match store")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "campaign.json"
        data = json.dumps(campaign.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != data:
                raise FileExistsError(f"campaign already exists with different spec: {path}")
            return path
        self._atomic_new(path, data)
        artifact = self.write_artifact("campaign", campaign)
        self.append_event("campaign_initialized", artifact_sha256=artifact.stem)
        return path

    def load_campaign(self) -> CampaignSpec:
        return CampaignSpec.model_validate_json(
            (self.root / "campaign.json").read_text(encoding="utf-8")
        )

    def lock_experiment_campaign(
        self, manifest: ExperimentCampaignV1
    ) -> CampaignLockV1:
        """Record the one authoritative manifest before execution starts."""
        manifest = ExperimentCampaignV1.model_validate(
            manifest.model_dump(mode="json")
        )
        if manifest.campaign_id != self.campaign_id:
            raise ValueError("campaign manifest belongs to a different campaign")
        events = self.verify_event_chain()
        experiment_events = [
            row
            for row in events
            if row.get("experiment_id") == manifest.experiment_id
        ]
        digest = campaign_manifest_sha256(manifest)
        locks = [
            row
            for row in experiment_events
            if row.get("event_type") == "experiment_campaign_locked"
        ]
        if locks:
            locked_digest = str(locks[0].get("detail", {}).get("manifest_sha256", ""))
            if locked_digest != digest:
                raise FileExistsError(
                    "experiment campaign is already locked with different content"
                )
            return self.load_experiment_campaign(manifest.experiment_id)
        if any(row.get("event_type") == "experiment_started" for row in experiment_events):
            raise RuntimeError("cannot lock a campaign after experiment start")
        lock = CampaignLockV1(manifest_sha256=digest, manifest=manifest)
        path = self.write_artifact("experiment_campaigns", lock)
        self.append_event(
            "experiment_campaign_locked",
            experiment_id=manifest.experiment_id,
            status="locked",
            artifact_sha256=path.stem,
            detail={"manifest_sha256": digest},
        )
        return lock

    def load_experiment_campaign(self, experiment_id: str) -> CampaignLockV1:
        """Load and verify the earliest pre-start campaign lock."""
        events = self.verify_event_chain()
        first_start = next(
            (
                index
                for index, row in enumerate(events)
                if row.get("experiment_id") == experiment_id
                and row.get("event_type") == "experiment_started"
            ),
            len(events),
        )
        lock_rows = [
            (index, row)
            for index, row in enumerate(events)
            if row.get("experiment_id") == experiment_id
            and row.get("event_type") == "experiment_campaign_locked"
        ]
        if not lock_rows:
            raise FileNotFoundError(
                f"no locked experiment campaign for {experiment_id}"
            )
        index, row = lock_rows[0]
        if index >= first_start:
            raise RuntimeError("campaign lock was not recorded before experiment start")
        if len(lock_rows) != 1:
            raise RuntimeError("multiple campaign locks recorded for experiment")
        artifact_sha = str(row.get("artifact_sha256", ""))
        path = self.root / "artifacts" / "experiment_campaigns" / f"{artifact_sha}.json"
        lock = CampaignLockV1.model_validate_json(path.read_text(encoding="utf-8"))
        event_sha = str(row.get("detail", {}).get("manifest_sha256", ""))
        if lock.manifest_sha256 != event_sha:
            raise RuntimeError("campaign lock event digest mismatch")
        return lock

    def append_campaign_deviation(
        self, deviation: CampaignDeviationV1
    ) -> Path:
        """Append an exploratory deviation without replacing the locked plan."""
        lock = self.load_experiment_campaign(deviation.experiment_id)
        if deviation.campaign_id != self.campaign_id:
            raise ValueError("deviation belongs to a different campaign")
        if deviation.manifest_sha256 != lock.manifest_sha256:
            raise ValueError("deviation manifest digest does not match lock")
        path = self.write_artifact("campaign_deviations", deviation)
        self.append_event(
            "campaign_deviation_appended",
            experiment_id=deviation.experiment_id,
            status="exploratory",
            artifact_sha256=path.stem,
            detail={"manifest_sha256": lock.manifest_sha256},
        )
        return path

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
                artifact_path = (
                    self.root / "artifacts" / kind / f"{artifact_sha}.json"
                )
                try:
                    artifact_payload = json.loads(
                        artifact_path.read_text(encoding="utf-8")
                    )
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

    def write_artifact(
        self, kind: str, value: BaseModel | dict[str, Any]
    ) -> Path:
        payload = _payload(value)
        digest = _sha(payload)
        path = self.root / "artifacts" / kind / f"{digest}.json"
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != data:
                raise RuntimeError(f"content-address collision: {path}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_new(path, data)
        checksum = {
            "path": str(path.relative_to(self.root)),
            "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
            "content_sha256": digest,
            "written_at": utc_now(),
        }
        self._append_line(self.root / "checksums.jsonl", canonical_json(checksum) + "\n")
        return path

    def append_event(
        self,
        event_type: str,
        *,
        experiment_id: str = "",
        status: str = "",
        artifact_sha256: str = "",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        events = self.root / "events.jsonl"
        lock_path = self.root / ".events.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            previous = self._last_event_sha(events)
            event = {
                "timestamp": utc_now(),
                "event_type": event_type,
                "campaign_id": self.campaign_id,
                "experiment_id": experiment_id,
                "status": status,
                "artifact_sha256": artifact_sha256,
                "detail": detail or {},
                "previous_event_sha256": previous,
            }
            event["event_id"] = _sha(event)
            self._append_line(events, canonical_json(event) + "\n")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        self._append_tsv(event)
        return event

    def status(self) -> dict[str, Any]:
        events_path = self.root / "events.jsonl"
        events = []
        if events_path.exists():
            events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        return {
            "campaign_id": self.campaign_id,
            "root": str(self.root),
            "event_count": len(events),
            "last_event": events[-1] if events else None,
            "artifacts": sum(1 for _ in (self.root / "artifacts").rglob("*.json"))
            if (self.root / "artifacts").exists()
            else 0,
        }

    @staticmethod
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

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _last_event_sha(path: Path) -> str:
        if not path.exists():
            return ""
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        return str(json.loads(lines[-1])["event_id"]) if lines else ""

    def _append_tsv(self, event: dict[str, Any]) -> None:
        path = self.root / "results.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.EVENT_COLUMNS, delimiter="\t")
            if new:
                writer.writeheader()
            row = {key: event.get(key, "") for key in self.EVENT_COLUMNS}
            row["detail"] = canonical_json(event.get("detail") or {})
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
