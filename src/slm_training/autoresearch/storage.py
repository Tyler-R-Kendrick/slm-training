"""Content-addressed campaign artifacts and append-only telemetry."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slm_training.autoresearch.schemas import CampaignSpec, utc_now
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
