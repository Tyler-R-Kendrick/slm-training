"""Shared leakage fingerprints for train/test disjointness."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, load_jsonl

_PLACEHOLDER_RE = re.compile(r":[A-Za-z0-9_.]+")
_BINDER_RE = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)(\s*=)")
# Lowercase / snake idents (variable refs), not PascalCase components.
_VAR_REF_RE = re.compile(r"\b([a-z_][A-Za-z0-9_]*)\b")


def norm_text(value: str) -> str:
    return " ".join(value.strip().split())


def fingerprint_prompt(prompt: str) -> str:
    return hashlib.sha256(norm_text(prompt).encode("utf-8")).hexdigest()


def fingerprint_openui(openui: str) -> str:
    return hashlib.sha256(norm_text(openui).encode("utf-8")).hexdigest()


def normalize_openui_structure(openui: str) -> str:
    """
    Collapse placeholder namespaces and binder/var names so isomorphic trees
    (e.g. :hero.title vs :smoke.hero.title) share one structural fingerprint.

    Quoted string literals (direction, gaps, enums) are preserved so
    `"column"` ≠ `"row"` and `"m"` ≠ `"2xl"`.
    """
    text = _PLACEHOLDER_RE.sub(":ph", openui or "")
    text = _BINDER_RE.sub(r"v\2", text)

    def _var_sub(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in {"true", "false", "null"}:
            return token
        return "id"

    parts: list[str] = []
    last = 0
    for quoted in re.finditer(r'"[^"]*"', text):
        chunk = text[last : quoted.start()]
        parts.append(_VAR_REF_RE.sub(_var_sub, chunk))
        parts.append(quoted.group(0))
        last = quoted.end()
    parts.append(_VAR_REF_RE.sub(_var_sub, text[last:]))
    return norm_text("".join(parts))


def fingerprint_openui_structure(openui: str) -> str:
    return hashlib.sha256(normalize_openui_structure(openui).encode("utf-8")).hexdigest()


def fingerprint_pair(prompt: str, openui: str) -> str:
    payload = norm_text(prompt) + "\n" + norm_text(openui)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_design_md(design_md: str | None) -> str | None:
    if not design_md:
        return None
    return hashlib.sha256(norm_text(design_md).encode("utf-8")).hexdigest()


def load_train_fingerprints(manifest_path: Path | None) -> dict[str, set[str]]:
    """Load id / prompt / openui / structure / pair / design_md fingerprints."""
    empty = {
        "ids": set(),
        "prompts": set(),
        "openuis": set(),
        "structures": set(),
        "pairs": set(),
        "design_mds": set(),
    }
    if manifest_path is None:
        return empty
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"train manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = set(data.get("ids") or [])
    prompts = set(data.get("prompt_fingerprints") or [])
    openuis = set(data.get("openui_fingerprints") or [])
    structures = set(data.get("structure_fingerprints") or [])
    pairs = set(data.get("pair_fingerprints") or [])
    design_mds = set(data.get("design_md_fingerprints") or [])

    records_path = data.get("records")
    need_backfill = (
        records_path
        and (
            not prompts
            or not openuis
            or not pairs
            or not design_mds
            or not structures
        )
    )
    if need_backfill:
        for record in load_jsonl(records_path):
            ids.add(record.id)
            prompts.add(fingerprint_prompt(record.prompt))
            openuis.add(fingerprint_openui(record.openui))
            structures.add(fingerprint_openui_structure(record.openui))
            pairs.add(fingerprint_pair(record.prompt, record.openui))
            dm = fingerprint_design_md(record.design_md)
            if dm:
                design_mds.add(dm)

    return {
        "ids": ids,
        "prompts": prompts,
        "openuis": openuis,
        "structures": structures,
        "pairs": pairs,
        "design_mds": design_mds,
    }


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
    if fingerprint_openui_structure(record.openui) in train_fps.get("structures", set()):
        reasons.append("openui_structure")
    if fingerprint_pair(record.prompt, record.openui) in train_fps["pairs"]:
        reasons.append("pair")
    dm = fingerprint_design_md(record.design_md)
    if dm and dm in train_fps.get("design_mds", set()):
        reasons.append("design_md")
    return reasons
