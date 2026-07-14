"""Read-only web view of atomic lineage deployment pointers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DeploymentRegistry:
    def __init__(self, root: Path | str = Path("outputs/lineage/deployments")) -> None:
        self.root = Path(root)

    def selected(self) -> dict[str, Any] | None:
        selected = self.root / "selected.json"
        if not selected.exists():
            return None
        pointer = json.loads(selected.read_text(encoding="utf-8"))
        record = self.root.parent / str(pointer["record"])
        return json.loads(record.read_text(encoding="utf-8"))

    def tracks(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for track in ("twotower", "causal_lm"):
            track_dir = self.root / track
            current = track_dir / "current.json"
            if not current.exists():
                continue
            pointer = json.loads(current.read_text(encoding="utf-8"))
            result[track] = json.loads(
                (track_dir / str(pointer["record"])).read_text(encoding="utf-8")
            )
        return result

    def payload(self) -> dict[str, Any]:
        return {"selected": self.selected(), "tracks": self.tracks()}
