"""Frozen frontier artifact: committed, versioned skill output the build re-validates.

The agent-skill (P5 `frontier-describe`) writes one JSON bundle per train gold to
``fixtures/frontier/<gold_id>.<gold_hash8>.json``. The deterministic Python build
never calls a model — it only reads these committed bundles, binds each to the
exact gold (content hash + structural fingerprint), and re-validates every row it
emits. See ``fixtures/frontier/SCHEMA.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FRONTIER_DIR = Path("fixtures/frontier")


@dataclass(frozen=True)
class FrozenArtifact:
    """One skill-authored bundle describing a single gold program (skeleton-only)."""

    gold_id: str
    gold_content_hash: str
    skeleton_openui: str
    provenance: dict[str, Any] = field(default_factory=dict)
    paraphrases: tuple[str, ...] = ()
    ladder: dict[str, str] = field(default_factory=dict)
    edits: tuple[dict[str, Any], ...] = ()
    vision: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrozenArtifact:
        return cls(
            gold_id=str(data["gold_id"]),
            gold_content_hash=str(data["gold_content_hash"]),
            skeleton_openui=str(data["skeleton_openui"]),
            provenance=dict(data.get("provenance") or {}),
            paraphrases=tuple(str(p) for p in (data.get("paraphrases") or [])),
            ladder={str(k): str(v) for k, v in (data.get("ladder") or {}).items()},
            edits=tuple(dict(e) for e in (data.get("edits") or [])),
            vision=dict(data.get("vision") or {}),
        )

    @classmethod
    def from_path(cls, path: Path) -> FrozenArtifact:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def artifact_path(gold_id: str, gold_hash8: str, *, root: Path | None = None) -> Path:
    return (root or FRONTIER_DIR) / f"{gold_id}.{gold_hash8}.json"


def load_artifact(
    gold_id: str, gold_hash8: str, *, root: Path | None = None
) -> FrozenArtifact | None:
    """Load the bundle for ``gold_id`` at content hash ``gold_hash8``.

    Returns ``None`` on a miss (no artifact yet, or a stale hash) so a changed gold
    silently drops its old artifact until the skill regenerates it.
    """
    path = artifact_path(gold_id, gold_hash8, root=root)
    if not path.exists():
        return None
    return FrozenArtifact.from_path(path)
