#!/usr/bin/env python3
"""Generate the committed OpenFeature history from experiment evidence.

Historical documents predate flag snapshots. This tool captures only values
that their recipe/config explicitly records; omitted values stay unknown when
the dashboard renders the run matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.feature_flags import SNAPSHOT_SCHEMA, catalog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src/slm_training/resources/experiment_feature_flag_history.json"
ALIASES = {
    "backend": "context_backend",
    "decode_steps": "gen_steps",
    "hf_model": "hf_model_name",
    "learning_rate": "lr",
    "max_attempts": "generate_max_attempts",
    "steps_requested": "steps",
    "steps_target": "steps",
    "target_tokens": "target_token_budget",
}
INVERTED_ALIASES = {
    "no_design_md_context": "design_md_in_context",
    "no_fuse_ltr": "fuse_ltr_loss",
    "no_unconstrained_fallback": "allow_unconstrained_fallback",
}


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _recipe(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("recipe", "config", "training", "evaluation", "train"):
        value = payload.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _records(path: Path, payload: dict[str, Any]):
    base = _recipe(payload)
    if isinstance(payload.get("run_id"), str):
        yield str(payload["run_id"]), base, "root"
    for key in ("training", "train", "train_result", "evaluation"):
        nested = payload.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("run_id"), str):
            yield str(nested["run_id"]), {**base, **_recipe(nested)}, key
    for group in ("matched_runs", "results"):
        values = payload.get(group)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
                continue
            yield str(item["run_id"]), {**base, **_recipe(item)}, f"{group}[{index}]"


def _normalize(recipe: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in recipe.items():
        name = ALIASES.get(key, key)
        if key in INVERTED_ALIASES:
            name, value = INVERTED_ALIASES[key], not bool(value)
        if name in known:
            values[name] = value
        elif key not in {"honesty_mode", "record_count", "n", "suite", "suites"}:
            unknown.append(key)
    return values, sorted(unknown)


def build() -> dict[str, Any]:
    registry = catalog()
    known = {row["field"] for row in registry["flags"]}
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "docs/design").glob("*.json")):
        payload = _read(path)
        if payload is None:
            continue
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for run_id, recipe, location in _records(path, payload):
            values, unknown = _normalize(recipe, known)
            entry = entries.setdefault(
                run_id,
                {"run_id": run_id, "values": {}, "conflicts": {}, "sources": []},
            )
            for name, value in values.items():
                if name not in entry["values"]:
                    entry["values"][name] = value
                    continue
                prior = entry["values"][name]
                if prior == value:
                    continue
                candidates = entry["conflicts"].setdefault(name, [prior])
                if all(existing != value for existing in candidates):
                    candidates.append(value)
                    candidates.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
            entry["sources"].append(
                {
                    "path": f"docs/design/{path.name}",
                    "location": location,
                    "sha256": source_sha,
                    "unrecognized": unknown,
                }
            )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "registry_revision": registry["revision"],
        "generated_from": "docs/design/*.json",
        "runs": [entries[run_id] for run_id in sorted(entries)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
    if args.check:
        if current != rendered:
            raise SystemExit(f"stale experiment feature-flag history: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
