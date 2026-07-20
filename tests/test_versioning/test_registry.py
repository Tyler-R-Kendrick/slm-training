"""Structural invariants of the committed component-version registry."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from slm_training.versioning import (
    REGISTRY_REPO_PATH,
    REGISTRY_SCHEMA,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[2]

_COMPONENT_ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def test_registry_schema_and_location() -> None:
    registry = load_registry()
    assert registry["schema"] == REGISTRY_SCHEMA
    assert (ROOT / REGISTRY_REPO_PATH).is_file()
    assert registry["components"], "registry must declare at least one component"


def test_component_entries_are_well_formed() -> None:
    for component_id, entry in load_registry()["components"].items():
        assert _COMPONENT_ID_RE.match(component_id), component_id
        assert _VERSION_RE.match(entry["version"]), component_id
        assert entry["kind"] in {"harness", "metric", "gate", "matrix", "data_builder", "model"}
        history = entry["history"]
        assert history, f"{component_id}: history must be non-empty"
        assert history[0]["version"] == entry["version"], (
            f"{component_id}: history[0] must match the current version"
        )
        for row in history:
            assert _VERSION_RE.match(row["version"]), component_id
            date.fromisoformat(row["date"])
            assert row["note"].strip(), f"{component_id}: history notes must be non-empty"


def test_watched_paths_exist_and_are_unique() -> None:
    seen: dict[str, str] = {}
    for component_id, entry in load_registry()["components"].items():
        assert entry["paths"], f"{component_id}: paths must be non-empty"
        for path in entry["paths"]:
            assert not path.startswith("/"), f"{component_id}: {path} must be repo-relative"
            previous = seen.setdefault(path, component_id)
            assert previous == component_id, (
                f"path {path!r} claimed by both {previous} and {component_id}"
            )
            target = ROOT / path
            if path.endswith("/"):
                assert target.is_dir(), f"{component_id}: missing directory {path}"
            else:
                assert target.is_file(), f"{component_id}: missing file {path}"
