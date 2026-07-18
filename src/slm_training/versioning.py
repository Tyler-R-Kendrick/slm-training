"""Normalized component versioning for evals, smoke runs, and checkpoints.

The harnesses in this repository are self-improving: metric definitions, gate
thresholds, harness implementations, and eval-suite builders all change over
time. This module is the single normalization point for recording *which
revision of those constraints* produced a result, so that historical numbers
stay comparable and experiments that ran under since-changed constraints stay
discoverable for re-testing.

Two committed artifacts define the contract (see
``docs/design/version-stamp-contract.md``):

- the **registry** ``src/slm_training/resources/versions.json`` — the canonical
  ``component id -> version`` map with per-component watched ``paths`` and an
  append-only ``history``; and
- the **stamp** — the ``version_stamp`` envelope built here and embedded in
  every result payload (eval/scoreboard/gates JSON, matrix summaries, bench
  reports, train summaries).

Stamping is provenance, not a gate: every helper degrades to the explicit
:data:`UNKNOWN` sentinel (never raises) on environmental failure — a missing
git binary, a non-repo install, an unreadable registry — because a provenance
failure must never kill a training or evaluation run. Passing a component id
that is absent from a *loadable* registry, however, raises ``KeyError``: that
is a repository bug (writers and registry are committed together) and the unit
tests catch it long before any run.

Bump enforcement lives in ``scripts/verify_version_stamps.py``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "REGISTRY_SCHEMA",
    "STAMP_SCHEMA",
    "UNKNOWN",
    "REGISTRY_REPO_PATH",
    "load_registry",
    "component_version",
    "git_commit",
    "git_dirty",
    "build_version_stamp",
]

REGISTRY_SCHEMA = "version_registry/v1"
STAMP_SCHEMA = "version_stamp/v1"
#: Explicit sentinel for unresolvable provenance, matching the
#: ``checkpoint_reference`` convention: unknown is stated, never inferred.
UNKNOWN = "UNKNOWN"
#: Registry location relative to the repository root.
REGISTRY_REPO_PATH = "src/slm_training/resources/versions.json"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "versions.json"


@lru_cache(maxsize=1)
def load_registry() -> Mapping[str, Any]:
    """Return the parsed version registry.

    Loads the packaged ``resources/versions.json`` (shipped as package data, so
    wheel installs resolve it too). Raises on a missing or malformed registry —
    callers that must never fail go through :func:`build_version_stamp`, which
    degrades instead.
    """
    with _registry_path().open(encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(
            f"version registry schema mismatch: expected {REGISTRY_SCHEMA!r}, "
            f"got {registry.get('schema')!r}"
        )
    if not isinstance(registry.get("components"), dict):
        raise ValueError("version registry is missing a 'components' mapping")
    return registry


def component_version(component_id: str) -> str:
    """Return the current version for ``component_id`` (KeyError if absent)."""
    components = load_registry()["components"]
    entry = components[component_id]
    return str(entry["version"])


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Current HEAD commit, or :data:`UNKNOWN` outside a usable git checkout."""
    return _git_output(["rev-parse", "HEAD"]) or UNKNOWN


@lru_cache(maxsize=1)
def git_dirty() -> bool | None:
    """True when the worktree has uncommitted changes; None when unknowable."""
    out = _git_output(["status", "--porcelain"])
    if out is None:
        return None
    return bool(out)


def build_version_stamp(*component_ids: str) -> dict[str, Any]:
    """Build the normalized ``version_stamp`` envelope for a result payload.

    Component versions come from the registry; an id missing from a loadable
    registry raises ``KeyError`` (repository bug). If the registry itself
    cannot be loaded, every requested component degrades to :data:`UNKNOWN`
    rather than failing the run.
    """
    try:
        components = {cid: component_version(cid) for cid in component_ids}
    except (OSError, ValueError, json.JSONDecodeError):
        components = {cid: UNKNOWN for cid in component_ids}
    return {
        "stamp_schema": STAMP_SCHEMA,
        "code_commit": git_commit(),
        "code_dirty": git_dirty(),
        "components": components,
        "stamped_at": datetime.now(timezone.utc).isoformat(),
    }
