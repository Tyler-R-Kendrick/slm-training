"""Append-only escalation ledger + backoff governor (advisory-only).

Replaces the supervisor's blind fixed-interval spin on ``hard_pending`` with
typed, deduplicated, attempt-counted state at
``outputs/autoresearch/loops/<loop_id>/escalations.jsonl``. Event-sourced like
``action_receipts.jsonl``: append rows, latest row per fingerprint wins.

Authority boundary (skeptic O7, binding): this ledger schedules backoff and
names the owner skill / needed authority for agents and ``evidence-brief`` to
read. It has **no** API that acknowledges, softens, or expires a hard action —
downgrade authority would be RC4's never-stop bypass rebuilt with better
paperwork.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from slm_training.autoresearch.heal.schemas import (
    EscalationRecordV1,
    escalation_fingerprint,
)
from slm_training.autoresearch.schemas import utc_now

__all__ = [
    "ESCALATIONS_FILENAME",
    "EscalationLedger",
    "blocker_fingerprint",
    "next_backoff_seconds",
]

ESCALATIONS_FILENAME = "escalations.jsonl"

_BACKOFF_BASE_SECONDS = 30
_BACKOFF_CAP_SECONDS = 3600

_OWNER_BY_CLASS = {
    "environment": "run_autotrain_supervisor",
    "code": "improve-openui-harnesses",
    "formal_infra": "improve-lean-optimums",
    "formal_contradiction": "improve-lean-optimums",
    "delivery": "sdlc",
    "dirty_tree": "run_autotrain_supervisor",
    "data": "synthesis-feedback",
    "authority": "human",
    "unknown": "autotrain",
}

_AUTHORITY_BY_CLASS = {
    "environment": "none",
    "dirty_tree": "none",
    "data": "none",
    "code": "agent_session",
    "formal_infra": "agent_session",
    "formal_contradiction": "agent_session",
    "delivery": "agent_session",
    "authority": "human_decision",
    "unknown": "agent_session",
}


def blocker_fingerprint(kind: str, reason: str) -> str:
    """Cross-campaign identity: campaign ids are payload, never key."""
    return escalation_fingerprint(kind, reason)


def next_backoff_seconds(seen_count: int) -> int:
    """Exponential governor: 30s doubling to a 1h cap."""
    # Clamp before the shift: seen_count is unbounded in a standing loop and
    # 2**7 already reaches the cap.
    exponent = min(max(0, int(seen_count) - 1), 8)
    return int(min(_BACKOFF_BASE_SECONDS * (2**exponent), _BACKOFF_CAP_SECONDS))


@dataclass
class EscalationLedger:
    """Fold of the append-only per-loop escalation jsonl."""

    root: Path
    loop_id: str
    records: dict[str, EscalationRecordV1] = field(default_factory=dict)
    _dirty: list[EscalationRecordV1] = field(default_factory=list)

    @classmethod
    def path_for(cls, root: Path, loop_id: str) -> Path:
        return Path(root) / "loops" / loop_id / ESCALATIONS_FILENAME

    @classmethod
    def load(cls, root: Path, loop_id: str) -> "EscalationLedger":
        ledger = cls(root=Path(root), loop_id=loop_id)
        path = cls.path_for(root, loop_id)
        if not path.is_file():
            return ledger
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = EscalationRecordV1.model_validate_json(line)
            except Exception:  # noqa: BLE001 — malformed rows never block the fold
                continue
            ledger.records[record.fingerprint] = record
        return ledger

    def observe(
        self,
        *,
        kind: str,
        reason: str,
        blocker_class: str,
        campaign_id: str,
        owner_skill: str = "",
    ) -> EscalationRecordV1:
        """Record one sighting of a blocker; dedups on the stable fingerprint."""
        fingerprint = blocker_fingerprint(kind, reason)
        now = utc_now()
        existing = self.records.get(fingerprint)
        if existing is None:
            record = EscalationRecordV1(
                loop_id=self.loop_id,
                fingerprint=fingerprint,
                kind=str(kind),
                blocker_class=str(blocker_class),
                reason=str(reason)[:400],
                campaign_ids=(str(campaign_id),) if campaign_id else (),
                owner_skill=owner_skill
                or _OWNER_BY_CLASS.get(str(blocker_class), "autotrain"),
                needed_authority=_AUTHORITY_BY_CLASS.get(  # type: ignore[arg-type]
                    str(blocker_class), "agent_session"
                ),
                first_seen_at=now,
                last_seen_at=now,
                seen_count=1,
                next_backoff_seconds=next_backoff_seconds(1),
            )
        else:
            campaigns = (
                tuple(dict.fromkeys((*existing.campaign_ids, str(campaign_id))))
                if campaign_id
                else existing.campaign_ids
            )
            seen = existing.seen_count + 1
            record = existing.model_copy(
                update={
                    "last_seen_at": now,
                    "seen_count": seen,
                    "campaign_ids": campaigns[-16:],
                    "status": (
                        "open" if existing.status == "resolved" else existing.status
                    ),
                    "next_backoff_seconds": next_backoff_seconds(seen),
                }
            )
        self._put(record)
        return record

    def record_attempt(self, fingerprint: str, playbook_id: str) -> None:
        record = self.records.get(fingerprint)
        if record is None:
            return
        tried = tuple(dict.fromkeys((*record.playbooks_tried, playbook_id)))
        self._put(
            record.model_copy(
                update={
                    "attempts": record.attempts + 1,
                    "playbooks_tried": tried,
                    "status": "healing",
                    "last_seen_at": utc_now(),
                }
            )
        )

    def resolve(self, fingerprint: str, *, note: str = "") -> None:
        record = self.records.get(fingerprint)
        if record is None:
            return
        self._put(
            record.model_copy(
                update={
                    "status": "resolved",
                    "note": note[:400],
                    "last_seen_at": utc_now(),
                }
            )
        )

    def escalate(self, fingerprint: str, *, note: str = "") -> None:
        record = self.records.get(fingerprint)
        if record is None:
            return
        self._put(
            record.model_copy(
                update={
                    "status": "escalated",
                    "note": note[:400],
                    "last_seen_at": utc_now(),
                }
            )
        )

    def open_records(self) -> tuple[EscalationRecordV1, ...]:
        return tuple(r for r in self.records.values() if r.status != "resolved")

    def sleep_seconds(self, *, default: float = 30.0) -> float:
        """Governed supervisor backoff: the max open-record backoff, else default."""
        open_backoffs = [r.next_backoff_seconds for r in self.open_records()]
        if not open_backoffs:
            return float(default)
        return float(max(open_backoffs))

    def _put(self, record: EscalationRecordV1) -> None:
        self.records[record.fingerprint] = record
        self._dirty.append(record)

    def save(self) -> Path:
        path = self.path_for(self.root, self.loop_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._dirty:
            return path
        # One row per fingerprint: resightings update last_seen_at/count in
        # place instead of appending a duplicate escalation line.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            for record in self.records.values():
                stream.write(record.model_dump_json() + "\n")
        tmp.replace(path)
        self._dirty.clear()
        return path


def wait_for_backoff(seconds: float) -> None:  # pragma: no cover — trivial
    time.sleep(max(0.0, float(seconds)))
