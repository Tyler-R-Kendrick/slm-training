"""Version-registry and checkpoint-doc bookkeeping for the continuous runner.

Extracted from ``scripts/run_autotrain_continuous.py``. One responsibility:
recording what a cycle touched -- appending the provenance note to a
checkpoint's design doc, and auto-appending a ``no-bump:`` history entry to the
component registry for the files a cycle changed.

Standard library only, so it is the bottom layer of the runner's extracted
modules and holds no loop state. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

VERSION_REGISTRY_REL = "src/slm_training/resources/versions.json"


def auto_no_bump_version_registry(
    cwd: Path,
    *,
    touched_rel_paths: Sequence[str],
    loop_id: str,
    campaign_id: str,
) -> Path | None:
    """Append a same-version no-bump history entry for components that claim
    ``touched_rel_paths``, so the honesty-stub checkpoint note (which edits
    only doc prose, never harness behavior) does not trip
    ``scripts/verify_version_stamps.py`` every single continuous cycle.

    Never bumps ``version`` and never touches an unrelated component — only
    components whose registered ``paths`` intersect the files this cycle
    actually wrote. Behavior-neutral by construction: doc prose only.
    """
    registry_path = cwd / VERSION_REGISTRY_REL
    if not registry_path.is_file() or not touched_rel_paths:
        return None
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    components = registry.get("components")
    if not isinstance(components, dict):
        return None
    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    touched_set = set(touched_rel_paths)
    changed = False
    for component_id, entry in components.items():
        paths = entry.get("paths") or []
        if not touched_set.intersection(paths):
            continue
        history = entry.get("history")
        if not isinstance(history, list):
            continue
        marker = f"loop {loop_id} {campaign_id}'s"
        if history and marker in str(history[0].get("note") or ""):
            continue  # already recorded for this exact campaign
        history.insert(
            0,
            {
                "version": entry.get("version"),
                "date": stamp,
                "note": (
                    f"no-bump: record scheduled loop {loop_id} {campaign_id}'s "
                    "checkpoint-note-only doc update (fixture/scratch continuous "
                    "cycle honesty stub); behavior unchanged."
                ),
            },
        )
        changed = True
    if not changed:
        return None
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return registry_path


def append_checkpoint_doc_notes(
    cwd: Path,
    *,
    campaign_id: str,
    checkpoint_paths: Sequence[str],
    loop_id: str | None = None,
) -> list[Path]:
    """Minimal honesty-labeled checkpoint notes when handoff requires them."""
    touched: list[Path] = []
    if not checkpoint_paths:
        return touched
    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    note = (
        f"\n## Continuous autotrain note ({stamp}, {campaign_id})\n\n"
        f"- campaign: `{campaign_id}`\n"
        f"- checkpoints: {', '.join(f'`{p}`' for p in checkpoint_paths)}\n"
        "- honesty: fixture/scratch continuous cycle — **not** a ship promotion.\n"
    )
    touched_rel: list[str] = []
    for rel in ("docs/MODEL_CARD.md", "README.md"):
        path = cwd / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        marker = f"campaign: `{campaign_id}`"
        if marker in text:
            continue
        path.write_text(text.rstrip() + "\n" + note, encoding="utf-8")
        touched.append(path)
        touched_rel.append(rel)
    if touched_rel and loop_id:
        registry_path = auto_no_bump_version_registry(
            cwd,
            touched_rel_paths=touched_rel,
            loop_id=loop_id,
            campaign_id=campaign_id,
        )
        if registry_path is not None:
            touched.append(registry_path)
    return touched


def checkpoint_path_for_candidate(
    root: Path, campaign_id: str, candidate_id: str | None
) -> Path | None:
    """Return candidate last.pt if present (for residual eval-lite notes)."""
    if not candidate_id:
        return None
    ckpt = root / campaign_id / "runs" / str(candidate_id) / "checkpoints" / "last.pt"
    return ckpt if ckpt.is_file() else None
