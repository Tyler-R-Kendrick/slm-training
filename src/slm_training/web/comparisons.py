"""Blinded champion-versus-candidate pairwise annotation events."""

from __future__ import annotations

import json
import os
import random
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Side = Literal["left", "right", "tie"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class BlindedComparisonStore:
    def __init__(self, path: Path | str = Path("outputs/annotations/comparisons.jsonl")) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def create(
        self,
        *,
        prompt: str,
        champion_run_id: str,
        candidate_run_id: str,
        champion_openui: str,
        candidate_openui: str,
        seed: int | None = None,
    ) -> dict[str, str]:
        pair_id = f"cmp_{uuid.uuid4().hex}"
        candidate_side = "left" if random.Random(seed).randrange(2) == 0 else "right"
        left = candidate_openui if candidate_side == "left" else champion_openui
        right = champion_openui if candidate_side == "left" else candidate_openui
        self._append(
            {
                "kind": "pair",
                "id": pair_id,
                "ts": _now(),
                "prompt": prompt,
                "left_openui": left,
                "right_openui": right,
                "candidate_side": candidate_side,
                "candidate_run_id": candidate_run_id,
                "champion_run_id": champion_run_id,
            }
        )
        # Identity mapping stays server-side; this is the only create response.
        return {"id": pair_id, "prompt": prompt, "left_openui": left, "right_openui": right}

    def vote(self, pair_id: str, winner: Side, *, reviewer_id: str) -> dict[str, Any]:
        pair = next(
            (row for row in reversed(self._rows()) if row.get("kind") == "pair" and row.get("id") == pair_id),
            None,
        )
        if pair is None:
            raise KeyError(pair_id)
        if any(row.get("kind") == "vote" and row.get("pair_id") == pair_id for row in self._rows()):
            raise ValueError("comparison already voted")
        outcome = "tie" if winner == "tie" else (
            "candidate" if winner == pair["candidate_side"] else "champion"
        )
        event = {
            "kind": "vote",
            "id": f"vote_{uuid.uuid4().hex}",
            "pair_id": pair_id,
            "ts": _now(),
            "winner_side": winner,
            "outcome": outcome,
            "candidate_run_id": pair["candidate_run_id"],
            "champion_run_id": pair["champion_run_id"],
            "reviewer_id": reviewer_id,
            "blinded": True,
        }
        self._append(event)
        return event

    def metrics(self, candidate_run_id: str) -> dict[str, int]:
        votes = [
            row
            for row in self._rows()
            if row.get("kind") == "vote" and row.get("candidate_run_id") == candidate_run_id
        ]
        return {
            "total": len(votes),
            "candidate_wins": sum(row.get("outcome") == "candidate" for row in votes),
            "champion_wins": sum(row.get("outcome") == "champion" for row in votes),
            "ties": sum(row.get("outcome") == "tie" for row in votes),
        }

    def _append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock, self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
