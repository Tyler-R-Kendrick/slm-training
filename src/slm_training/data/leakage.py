"""Shared leakage fingerprints for train/test disjointness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, load_jsonl


def norm_text(value: str) -> str:
    return " ".join(value.strip().split())


def fingerprint_prompt(prompt: str) -> str:
    return hashlib.sha256(norm_text(prompt).encode("utf-8")).hexdigest()


def fingerprint_openui(openui: str) -> str:
    return hashlib.sha256(norm_text(openui).encode("utf-8")).hexdigest()


def fingerprint_pair(prompt: str, openui: str) -> str:
    payload = norm_text(prompt) + "\n" + norm_text(openui)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_train_fingerprints(manifest_path: Path | None) -> dict[str, set[str]]:
    """Load id / prompt / openui / pair fingerprints from a train manifest."""
    empty = {"ids": set(), "prompts": set(), "openuis": set(), "pairs": set()}
    if manifest_path is None:
        return empty
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"train manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = set(data.get("ids") or [])
    prompts = set(data.get("prompt_fingerprints") or [])
    openuis = set(data.get("openui_fingerprints") or [])
    pairs = set(data.get("pair_fingerprints") or [])

    # Backfill from records when older manifests lack fingerprint fields.
    records_path = data.get("records")
    if records_path and (not prompts or not openuis or not pairs):
        for record in load_jsonl(records_path):
            ids.add(record.id)
            prompts.add(fingerprint_prompt(record.prompt))
            openuis.add(fingerprint_openui(record.openui))
            pairs.add(fingerprint_pair(record.prompt, record.openui))

    return {"ids": ids, "prompts": prompts, "openuis": openuis, "pairs": pairs}


def find_leakage(
    record: ExampleRecord,
    train_fps: dict[str, set[str]],
) -> list[str]:
    """Return human-readable leakage reasons (empty if clean)."""
    reasons: list[str] = []
    if record.id in train_fps["ids"]:
        reasons.append("id")
    if fingerprint_prompt(record.prompt) in train_fps["prompts"]:
        reasons.append("prompt")
    if fingerprint_openui(record.openui) in train_fps["openuis"]:
        reasons.append("openui")
    if fingerprint_pair(record.prompt, record.openui) in train_fps["pairs"]:
        reasons.append("pair")
    return reasons
