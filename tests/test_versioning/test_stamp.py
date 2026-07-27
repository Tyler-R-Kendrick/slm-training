"""Behavior of the normalized ``version_stamp`` envelope builder."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from slm_training import versioning
from slm_training.versioning import STAMP_SCHEMA, UNKNOWN, build_version_stamp


@pytest.fixture(autouse=True)
def _fresh_caches():
    versioning.git_commit.cache_clear()
    versioning.git_dirty.cache_clear()
    yield
    versioning.git_commit.cache_clear()
    versioning.git_dirty.cache_clear()


def test_stamp_envelope_shape() -> None:
    stamp = build_version_stamp("gates.ship", "evals.meaningful_program")
    assert stamp["stamp_schema"] == STAMP_SCHEMA
    assert stamp["code_commit"] == UNKNOWN or re.fullmatch(
        r"[0-9a-f]{40}", stamp["code_commit"]
    )
    assert stamp["code_dirty"] in (True, False, None)
    assert stamp["components"] == {
        "gates.ship": versioning.component_version("gates.ship"),
        "evals.meaningful_program": versioning.component_version(
            "evals.meaningful_program"
        ),
    }
    datetime.fromisoformat(stamp["stamped_at"])


def test_unknown_component_id_raises() -> None:
    # A writer citing an unregistered component is a repo bug, not an
    # environmental failure — it must fail tests loudly, never stamp garbage.
    with pytest.raises(KeyError):
        build_version_stamp("harness.does_not_exist")


def test_unreadable_registry_degrades_to_unknown(monkeypatch) -> None:
    def _boom() -> dict:
        raise OSError("registry unreadable")

    monkeypatch.setattr(versioning, "load_registry", _boom)
    stamp = build_version_stamp("gates.ship")
    assert stamp["components"] == {"gates.ship": UNKNOWN}


def test_git_failure_degrades_to_unknown(monkeypatch) -> None:
    monkeypatch.setattr(versioning, "_git_output", lambda args: None)
    stamp = build_version_stamp("gates.ship")
    assert stamp["code_commit"] == UNKNOWN
    assert stamp["code_dirty"] is None


def test_git_output_prefers_the_callers_worktree(monkeypatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    def _check_output(_args, *, text, cwd, stderr):
        assert text is True
        assert stderr is subprocess.DEVNULL
        calls.append(cwd)
        if cwd == tmp_path:
            return "worktree-head\n"
        raise AssertionError("the fallback worktree should not be consulted")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(versioning.subprocess, "check_output", _check_output)

    assert versioning._git_output(["rev-parse", "HEAD"]) == "worktree-head"
    assert calls == [tmp_path]


def test_git_introspection_is_cached(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake(args: list[str]) -> str:
        calls.append(args)
        return "deadbeef" * 5 if args[0] == "rev-parse" else ""

    monkeypatch.setattr(versioning, "_git_output", _fake)
    build_version_stamp("gates.ship")
    build_version_stamp("gates.ship")
    assert len(calls) == 2  # one rev-parse + one status, cached thereafter
