"""Bump-enforcement and staleness-report behavior of verify_version_stamps."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_version_stamps as vvs


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _entry(
    version: str,
    paths: list[str],
    history: list[dict[str, str]] | None = None,
    kind: str = "harness",
) -> dict[str, Any]:
    return {
        "version": version,
        "kind": kind,
        "paths": paths,
        "history": history
        or [{"version": version, "date": "2026-07-18", "note": "initial registration"}],
    }


def _write_registry(repo: Path, components: dict[str, Any]) -> None:
    path = repo / vvs.REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": vvs.REGISTRY_SCHEMA, "components": components}, indent=2)
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "docs" / "design").mkdir(parents=True)
    (repo / "src" / "comp_a.py").write_text("A = 1\n", encoding="utf-8")
    (repo / "src" / "pkg" / "mod.py").write_text("B = 1\n", encoding="utf-8")
    (repo / "src" / "pkg" / "other.py").write_text("C = 1\n", encoding="utf-8")
    _write_registry(
        repo,
        {
            "harness.comp_a": _entry("v1", ["src/comp_a.py"]),
            "harness.pkg": _entry("v1", ["src/pkg/"]),
            "harness.pkg_special": _entry("v1", ["src/pkg/mod.py"]),
        },
    )
    (repo / "docs" / "design" / "legacy-results.json").write_text(
        json.dumps({"results": [1]}) + "\n", encoding="utf-8"
    )
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(vvs, "ROOT", repo)
    return repo


def _bump(repo: Path, component_id: str, new_version: str | None, note: str) -> None:
    registry = json.loads((repo / vvs.REGISTRY_PATH).read_text(encoding="utf-8"))
    entry = registry["components"][component_id]
    version = new_version or entry["version"]
    entry["version"] = version
    entry["history"].insert(0, {"version": version, "date": "2026-07-19", "note": note})
    (repo / vvs.REGISTRY_PATH).write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )


def test_clean_worktree_passes(repo: Path) -> None:
    assert vvs.run_check(base_arg=None, staged=False) == 0


def test_change_without_bump_fails(repo: Path, capsys: pytest.CaptureFixture) -> None:
    (repo / "src" / "comp_a.py").write_text("A = 2\n", encoding="utf-8")
    assert vvs.run_check(base_arg=None, staged=False) == 1
    out = capsys.readouterr().out
    assert "harness.comp_a" in out
    assert "no-bump:" in out  # the remedy is spelled out


def test_version_bump_passes(repo: Path) -> None:
    (repo / "src" / "comp_a.py").write_text("A = 2\n", encoding="utf-8")
    _bump(repo, "harness.comp_a", "v2", "tighten metric")
    assert vvs.run_check(base_arg=None, staged=False) == 0


def test_no_bump_note_passes(repo: Path) -> None:
    (repo / "src" / "comp_a.py").write_text("A = 1  # comment\n", encoding="utf-8")
    _bump(repo, "harness.comp_a", None, "no-bump: comment-only change")
    assert vvs.run_check(base_arg=None, staged=False) == 0


def test_same_version_without_no_bump_prefix_fails(repo: Path) -> None:
    (repo / "src" / "comp_a.py").write_text("A = 3\n", encoding="utf-8")
    _bump(repo, "harness.comp_a", None, "cosmetic")
    assert vvs.run_check(base_arg=None, staged=False) == 1


def test_history_rewrite_fails(repo: Path, capsys: pytest.CaptureFixture) -> None:
    (repo / "src" / "comp_a.py").write_text("A = 4\n", encoding="utf-8")
    registry = json.loads((repo / vvs.REGISTRY_PATH).read_text(encoding="utf-8"))
    registry["components"]["harness.comp_a"]["version"] = "v2"
    registry["components"]["harness.comp_a"]["history"] = [
        {"version": "v2", "date": "2026-07-19", "note": "rewrote history"}
    ]
    (repo / vvs.REGISTRY_PATH).write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    assert vvs.run_check(base_arg=None, staged=False) == 1
    assert "append-only" in capsys.readouterr().out


def test_new_result_doc_requires_stamp(repo: Path, capsys: pytest.CaptureFixture) -> None:
    doc = repo / "docs" / "design" / "new-results.json"
    doc.write_text(json.dumps({"results": []}) + "\n", encoding="utf-8")
    assert vvs.run_check(base_arg=None, staged=False) == 1
    assert "version_stamp" in capsys.readouterr().out
    doc.write_text(
        json.dumps(
            {
                "results": [],
                "version_stamp": {
                    "stamp_schema": vvs.STAMP_SCHEMA,
                    "code_commit": "0" * 40,
                    "code_dirty": False,
                    "components": {"harness.comp_a": "v1"},
                    "stamped_at": "2026-07-18T00:00:00+00:00",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert vvs.run_check(base_arg=None, staged=False) == 0


def test_modified_legacy_result_only_warns(
    repo: Path, capsys: pytest.CaptureFixture
) -> None:
    (repo / "docs" / "design" / "legacy-results.json").write_text(
        json.dumps({"results": [1, 2]}) + "\n", encoding="utf-8"
    )
    assert vvs.run_check(base_arg=None, staged=False) == 0
    assert "grandfathered" in capsys.readouterr().out


def test_staged_mode(repo: Path) -> None:
    (repo / "src" / "comp_a.py").write_text("A = 5\n", encoding="utf-8")
    _git(repo, "add", "src/comp_a.py")
    assert vvs.run_check(base_arg=None, staged=True) == 1
    _bump(repo, "harness.comp_a", "v2", "staged bump")
    _git(repo, "add", vvs.REGISTRY_PATH)
    assert vvs.run_check(base_arg=None, staged=True) == 0


def test_base_mode_covers_committed_changes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "src" / "comp_a.py").write_text("A = 6\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change without bump")
    assert vvs.run_check(base_arg=base, staged=False) == 1
    _bump(repo, "harness.comp_a", "v2", "bump after the fact")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bump")
    assert vvs.run_check(base_arg=base, staged=False) == 0


def test_unresolvable_base_fails(repo: Path, capsys: pytest.CaptureFixture) -> None:
    assert vvs.run_check(base_arg="does-not-exist", staged=False) == 1
    assert "cannot resolve" in capsys.readouterr().out


def test_longest_prefix_mapping(repo: Path) -> None:
    registry = vvs.registry_snapshot("WORKTREE")
    assert registry is not None
    claims = vvs.path_claims(registry)
    assert vvs.component_for_path("src/pkg/mod.py", claims) == "harness.pkg_special"
    assert vvs.component_for_path("src/pkg/other.py", claims) == "harness.pkg"
    assert vvs.component_for_path("src/comp_a.py", claims) == "harness.comp_a"
    assert vvs.component_for_path("README.md", claims) is None


def test_registry_lint_rejects_duplicates_and_bad_rows(repo: Path) -> None:
    registry = {
        "schema": vvs.REGISTRY_SCHEMA,
        "components": {
            "a.one": _entry("v1", ["src/comp_a.py"]),
            "a.two": _entry(
                "v1",
                ["src/comp_a.py"],
                history=[{"version": "v1", "date": "not-a-date", "note": ""}],
            ),
        },
    }
    errors = "\n".join(vvs.lint_registry(registry))
    assert "claimed by both" in errors
    assert "invalid date" in errors
    assert "notes must be non-empty" in errors


def test_stale_report_orders_by_history_index(
    repo: Path, capsys: pytest.CaptureFixture
) -> None:
    # Mixed grammars: ordering must come from history position, never parsing.
    _write_registry(
        repo,
        {
            "gates.ship": _entry(
                "openui_ship_gates_v2",
                ["src/comp_a.py"],
                history=[
                    {"version": "openui_ship_gates_v2", "date": "2026-07-19", "note": "recalibrated"},
                    {"version": "openui_ship_gates_v1", "date": "2026-07-18", "note": "initial"},
                ],
                kind="gate",
            ),
        },
    )

    def _doc(name: str, payload: dict[str, Any]) -> None:
        (repo / "docs" / "design" / name).write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )

    def _stamp(version: str) -> dict[str, Any]:
        return {
            "stamp_schema": vvs.STAMP_SCHEMA,
            "components": {"gates.ship": version},
            "stamped_at": "2026-07-18T00:00:00+00:00",
        }

    _doc("stale-results.json", {"results": [], "version_stamp": _stamp("openui_ship_gates_v1")})
    _doc("fresh-results.json", {"results": [], "version_stamp": _stamp("openui_ship_gates_v2")})
    _doc("odd-results.json", {"results": [], "version_stamp": _stamp("v99")})

    assert (
        vvs.run_stale(
            component_filter=None, include_outputs=False, as_json=True, fail_on_stale=False
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    rows = {row["file"]: row for row in report["stale"]["gates.ship"]}
    assert rows["docs/design/stale-results.json"]["behind_by"] == 1
    assert rows["docs/design/stale-results.json"]["status"] == "stale"
    assert rows["docs/design/odd-results.json"]["status"] == "unrecognized_version"
    assert "docs/design/fresh-results.json" not in rows
    assert "docs/design/legacy-results.json" in report["legacy_unstamped"]
    assert report["fresh_count"] >= 1

    assert (
        vvs.run_stale(
            component_filter=None, include_outputs=False, as_json=True, fail_on_stale=True
        )
        == 1
    )


def test_post_tool_use_nudges_until_registry_touched(repo: Path) -> None:
    (repo / "src" / "comp_a.py").write_text("A = 7\n", encoding="utf-8")
    payload = {"tool_input": {"file_path": str(repo / "src" / "comp_a.py")}}
    nudge = vvs.post_tool_use_nudge(payload)
    assert nudge is not None and "harness.comp_a" in nudge
    _bump(repo, "harness.comp_a", "v2", "bumped")
    assert vvs.post_tool_use_nudge(payload) is None
    assert vvs.post_tool_use_nudge({"tool_input": {"file_path": "/elsewhere/x.py"}}) is None
    assert (
        vvs.post_tool_use_nudge({"tool_input": {"file_path": str(repo / "README.md")}})
        is None
    )


def test_constants_mirror_slm_training_versioning() -> None:
    from slm_training import versioning

    assert vvs.REGISTRY_SCHEMA == versioning.REGISTRY_SCHEMA
    assert vvs.STAMP_SCHEMA == versioning.STAMP_SCHEMA
    assert vvs.REGISTRY_PATH == versioning.REGISTRY_REPO_PATH
