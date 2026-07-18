"""Verify normalized version stamps and component version bumps.

Contract: docs/design/version-stamp-contract.md. The registry
(``src/slm_training/resources/versions.json``) maps component ids for the
self-improving eval/smoke/checkpoint stack to their current version, watched
``paths``, and an append-only ``history``. Result writers embed a
``version_stamp`` envelope (built by ``slm_training.versioning``) into every
payload.

Modes:

- ``--check`` (blocking; used by CI, the pre-commit changed-file check, and
  agents directly): lints the registry, then for every changed file under a
  component's watched paths requires that component's registry entry to have
  gained a history entry in the same diff — a version bump, or a same-version
  entry whose note starts with ``no-bump:`` for behavior-neutral changes.
  Newly **added** ``docs/design/*.json`` experiment results must carry a
  ``version_stamp``; modified legacy files only warn (grandfathered).
- ``--stale`` (report; the re-test discovery query): scans the committed
  ``docs/design`` ledger (and ``--include-outputs`` run artifacts) and lists
  results whose stamped component versions are behind the current registry —
  the candidates worth re-running after a constraint change.
- ``--post-tool-use`` (advisory agent hook): reads Claude-style hook JSON from
  stdin and prints a one-line nudge when the edited file belongs to a
  component whose registry entry has not been touched yet. Always exits 0.

This script is deliberately stdlib-only and never imports ``slm_training``:
agent hooks may run it under an interpreter without the package installed.
``tests/test_scripts/test_verify_version_stamps.py`` pins its constants to
``slm_training.versioning``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = "src/slm_training/resources/versions.json"
# Mirrored from slm_training.versioning (kept import-free on purpose).
REGISTRY_SCHEMA = "version_registry/v1"
STAMP_SCHEMA = "version_stamp/v1"
NO_BUMP_PREFIX = "no-bump:"
COMPONENT_KINDS = {"harness", "metric", "gate", "matrix", "data_builder"}
# Top-level keys that structurally mark a docs/design JSON as an experiment
# result record (mirrors the sniffing style of verify_checkpoint_references —
# never filename-only).
RESULT_SHAPE_KEYS = {
    "results",
    "registered_rows",
    "suites",
    "matrix",
    "matrix_set",
    "gate_policy",
    "gates",
    "scoreboard",
}

_COMPONENT_ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------- #
# Registry parsing and lint
# --------------------------------------------------------------------------- #
def parse_registry(text: str) -> dict[str, Any]:
    registry = json.loads(text)
    if not isinstance(registry, dict) or not isinstance(
        registry.get("components"), dict
    ):
        raise ValueError("registry must be an object with a 'components' mapping")
    return registry


def lint_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("schema") != REGISTRY_SCHEMA:
        errors.append(
            f"registry schema must be {REGISTRY_SCHEMA!r}, got {registry.get('schema')!r}"
        )
    claims: dict[str, str] = {}
    for component_id, entry in registry.get("components", {}).items():
        where = f"components.{component_id}"
        if not _COMPONENT_ID_RE.match(component_id):
            errors.append(f"{where}: invalid component id")
        if not isinstance(entry, dict):
            errors.append(f"{where}: entry must be an object")
            continue
        version = entry.get("version")
        if not isinstance(version, str) or not _VERSION_RE.match(version):
            errors.append(f"{where}: invalid version {version!r}")
        if entry.get("kind") not in COMPONENT_KINDS:
            errors.append(f"{where}: kind must be one of {sorted(COMPONENT_KINDS)}")
        paths = entry.get("paths")
        if not isinstance(paths, list) or not paths:
            errors.append(f"{where}: paths must be a non-empty list")
            paths = []
        for path in paths:
            if not isinstance(path, str) or path.startswith(("/", "..")):
                errors.append(f"{where}: path {path!r} must be repo-relative")
                continue
            other = claims.setdefault(path, component_id)
            if other != component_id:
                errors.append(f"path {path!r} claimed by both {other} and {component_id}")
            target = ROOT / path
            if path.endswith("/"):
                if not target.is_dir():
                    errors.append(f"{where}: watched directory missing: {path}")
            elif not target.is_file():
                errors.append(f"{where}: watched file missing: {path}")
        history = entry.get("history")
        if not isinstance(history, list) or not history:
            errors.append(f"{where}: history must be a non-empty list (newest first)")
            continue
        if history[0].get("version") != version:
            errors.append(f"{where}: history[0].version must equal version")
        for row in history:
            if not isinstance(row, dict):
                errors.append(f"{where}: history rows must be objects")
                continue
            if not isinstance(row.get("version"), str) or not _VERSION_RE.match(
                row.get("version") or ""
            ):
                errors.append(f"{where}: history row has invalid version")
            try:
                date.fromisoformat(row.get("date") or "")
            except ValueError:
                errors.append(f"{where}: history row has invalid date {row.get('date')!r}")
            if not (row.get("note") or "").strip():
                errors.append(f"{where}: history notes must be non-empty")
    return errors


def path_claims(registry: dict[str, Any]) -> list[tuple[str, str]]:
    claims = [
        (path, component_id)
        for component_id, entry in registry.get("components", {}).items()
        for path in entry.get("paths", [])
        if isinstance(path, str)
    ]
    # Longest claim first so the first match wins for a given file.
    return sorted(claims, key=lambda item: len(item[0]), reverse=True)


def component_for_path(path: str, claims: list[tuple[str, str]]) -> str | None:
    for claim, component_id in claims:
        if claim.endswith("/"):
            if path.startswith(claim):
                return component_id
        elif path == claim:
            return component_id
    return None


# --------------------------------------------------------------------------- #
# Change sets and registry snapshots
# --------------------------------------------------------------------------- #
def resolve_base(explicit: str) -> str | None:
    out = _git(["rev-parse", "--verify", f"{explicit}^{{commit}}"])
    return out.strip() if out else None


def _parse_name_status(raw: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        status = parts[0][0]
        path = parts[-1]  # renames list old\tnew; the new path is authoritative
        if path:
            entries.append((status, path))
    return entries


def changed_entries(*, base: str, staged: bool) -> list[tuple[str, str]]:
    """Changed (status, path) pairs.

    ``--staged`` diffs the index against HEAD (pre-commit parity). Otherwise
    the worktree (plus untracked files) is diffed against ``base`` — HEAD for
    the local default, the PR base commit in CI.
    """
    if staged:
        raw = _git(["diff", "--cached", "--name-status", "--diff-filter=ACMRD", "--"])
        return _parse_name_status(raw or "")
    raw = _git(["diff", base, "--name-status", "--diff-filter=ACMRD", "--"])
    entries = _parse_name_status(raw or "")
    untracked = _git(["ls-files", "--others", "--exclude-standard"]) or ""
    seen = {path for _, path in entries}
    entries.extend(
        ("A", path) for path in untracked.splitlines() if path and path not in seen
    )
    return sorted(entries, key=lambda item: item[1])


def registry_snapshot(rev: str) -> dict[str, Any] | None:
    """Registry parsed from ``rev``: a commit-ish, ``:0`` (index), or ``WORKTREE``."""
    if rev == "WORKTREE":
        path = ROOT / REGISTRY_PATH
        if not path.is_file():
            return None
        return parse_registry(path.read_text(encoding="utf-8"))
    spec = f"{rev}:{REGISTRY_PATH}" if rev != ":0" else f":0:{REGISTRY_PATH}"
    out = _git(["show", spec])
    if out is None:
        return None
    return parse_registry(out)


def read_json_at(rev: str, path: str) -> Any | None:
    if rev == "WORKTREE":
        target = ROOT / path
        if not target.is_file():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    out = _git(["show", f"{rev}:{path}" if rev != ":0" else f":0:{path}"])
    if out is None:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Bump enforcement (--check)
# --------------------------------------------------------------------------- #
def _history_versions(entry: dict[str, Any]) -> list[str]:
    return [str(row.get("version")) for row in entry.get("history", [])]


def _is_suffix(old: list[Any], new: list[Any]) -> bool:
    return len(new) >= len(old) and (not old or new[-len(old) :] == old)


def _looks_like_result(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(RESULT_SHAPE_KEYS & set(obj))


def _has_valid_stamp(obj: dict[str, Any]) -> bool:
    stamp = obj.get("version_stamp")
    return (
        isinstance(stamp, dict)
        and stamp.get("stamp_schema") == STAMP_SCHEMA
        and isinstance(stamp.get("components"), dict)
        and bool(stamp["components"])
    )


def run_check(*, base_arg: str | None, staged: bool) -> int:
    new_rev = ":0" if staged else "WORKTREE"
    errors: list[str] = []
    warnings: list[str] = []

    try:
        new_registry = registry_snapshot(new_rev)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"version-stamps: failed\n- registry unparsable: {exc}")
        return 1
    if new_registry is None:
        print(f"version-stamps: failed\n- registry missing at {REGISTRY_PATH}")
        return 1
    errors.extend(lint_registry(new_registry))
    if errors:
        print("version-stamps: failed")
        for error in errors:
            print(f"- {error}")
        return 1

    if staged or not base_arg:
        base = "HEAD"
    else:
        resolved = resolve_base(base_arg)
        if resolved is None:
            print(f"version-stamps: failed\n- cannot resolve --base {base_arg!r}")
            return 1
        base = resolved

    try:
        old_registry = registry_snapshot(base)
    except (ValueError, json.JSONDecodeError):
        old_registry = None  # unparsable at base: treat all components as new

    changes = changed_entries(base=base, staged=staged)
    claims = path_claims(new_registry)

    touched: dict[str, list[str]] = {}
    for status, path in changes:
        if path == REGISTRY_PATH:
            continue
        component_id = component_for_path(path, claims)
        if component_id is not None and status in "ACMRD":
            touched.setdefault(component_id, []).append(path)

    old_components = (old_registry or {}).get("components", {})
    new_components = new_registry.get("components", {})

    # Append-only history for every component present on both sides.
    for component_id, old_entry in old_components.items():
        new_entry = new_components.get(component_id)
        if new_entry is None:
            continue
        if not _is_suffix(old_entry.get("history", []), new_entry.get("history", [])):
            errors.append(
                f"{component_id}: history must be append-only (new entries are "
                "prepended; existing entries are never edited or dropped)"
            )

    for component_id, files in sorted(touched.items()):
        old_entry = old_components.get(component_id)
        if old_entry is None:
            continue  # newly registered (or base predates the registry)
        new_entry = new_components.get(component_id)
        listing = ", ".join(sorted(files))
        if new_entry is None:
            errors.append(
                f"{component_id}: files changed ({listing}) but the component was "
                "removed from the registry"
            )
            continue
        old_history = old_entry.get("history", [])
        new_history = new_entry.get("history", [])
        if len(new_history) <= len(old_history):
            errors.append(
                f"{component_id}: files changed ({listing}) without a registry "
                f"update. Either bump 'version' in {REGISTRY_PATH} with a new "
                "history entry (newest first), or append a same-version history "
                f"entry whose note starts with {NO_BUMP_PREFIX!r} explaining why "
                "the change is behavior-neutral."
            )
            continue
        if new_entry.get("version") == old_entry.get("version"):
            note = str(new_history[0].get("note") or "")
            if not note.startswith(NO_BUMP_PREFIX):
                errors.append(
                    f"{component_id}: files changed ({listing}) and history grew, "
                    f"but the version did not change and the newest note does not "
                    f"start with {NO_BUMP_PREFIX!r}. Bump the version or mark the "
                    "entry as an explicit no-bump."
                )

    # Newly added docs/design result JSONs must be stamped; modified legacy
    # files are grandfathered with a warning.
    for status, path in changes:
        if not path.startswith("docs/design/") or not path.endswith(".json"):
            continue
        payload = read_json_at(new_rev, path)
        if not _looks_like_result(payload):
            continue
        if _has_valid_stamp(payload):
            continue
        if status == "A":
            errors.append(
                f"{path}: new experiment result JSON lacks a 'version_stamp' "
                f"(schema {STAMP_SCHEMA!r} with non-empty components). Emit it "
                "via slm_training.versioning.build_version_stamp from the "
                "writing script."
            )
        elif status in "MR":
            warnings.append(
                f"{path}: legacy result JSON still has no version_stamp "
                "(grandfathered; stamp it when the producing script next runs)"
            )

    for warning in warnings:
        print(f"version-stamps: warning: {warning}")
    if errors:
        print("version-stamps: failed")
        for error in errors:
            print(f"- {error}")
        return 1
    scope = "staged" if staged else f"vs {base[:12]}"
    print(
        f"version-stamps: ok ({scope}; {len(changes)} changed file(s), "
        f"{len(touched)} component(s) touched)"
    )
    return 0


# --------------------------------------------------------------------------- #
# Staleness report (--stale)
# --------------------------------------------------------------------------- #
def _stale_targets(include_outputs: bool) -> list[Path]:
    targets = sorted((ROOT / "docs" / "design").glob("*.json"))
    if include_outputs:
        targets.extend(sorted((ROOT / "outputs" / "runs").glob("*/*.json")))
        targets.extend(
            sorted((ROOT / "outputs" / "autoresearch").glob("*/runs/*/*.json"))
        )
    return [path for path in targets if path.is_file()]


def run_stale(
    *, component_filter: str | None, include_outputs: bool, as_json: bool,
    fail_on_stale: bool,
) -> int:
    registry = registry_snapshot("WORKTREE")
    if registry is None:
        print("version-stamps: failed\n- registry missing")
        return 1
    components = registry.get("components", {})

    stale: dict[str, list[dict[str, Any]]] = {}
    legacy_unstamped: list[str] = []
    fresh = 0
    for path in _stale_targets(include_outputs):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rel = str(path.relative_to(ROOT))
        if not isinstance(payload, dict):
            continue
        stamp = payload.get("version_stamp")
        if not isinstance(stamp, dict) or not isinstance(stamp.get("components"), dict):
            if _looks_like_result(payload):
                legacy_unstamped.append(rel)
            continue
        file_is_stale = False
        for component_id, stamped_version in stamp["components"].items():
            if component_filter and component_id != component_filter:
                continue
            entry = components.get(component_id)
            if entry is None:
                status = "component_retired"
                current = None
                behind = None
            else:
                current = entry.get("version")
                versions = _history_versions(entry)
                if stamped_version == current:
                    continue
                behind = (
                    versions.index(stamped_version)
                    if stamped_version in versions
                    else None
                )
                status = "stale" if behind is not None else "unrecognized_version"
            file_is_stale = True
            stale.setdefault(component_id, []).append(
                {
                    "file": rel,
                    "stamped_version": stamped_version,
                    "current_version": current,
                    "behind_by": behind,
                    "status": status,
                    "stamped_at": stamp.get("stamped_at"),
                    "code_commit": stamp.get("code_commit"),
                }
            )
        if not file_is_stale:
            fresh += 1

    report = {
        "stale": stale,
        "legacy_unstamped": legacy_unstamped,
        "fresh_count": fresh,
    }
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if stale:
            print("Retest candidates (stamped version behind current registry):")
            for component_id in sorted(stale):
                entry = components.get(component_id, {})
                print(f"  {component_id} (current: {entry.get('version')}):")
                for row in stale[component_id]:
                    print(
                        f"    - {row['file']} stamped {row['stamped_version']} "
                        f"({row['status']})"
                    )
        else:
            print("No stale stamped results found.")
        print(f"Fresh stamped results: {fresh}")
        print(f"Legacy result files without a stamp: {len(legacy_unstamped)}")
    if fail_on_stale and stale:
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Advisory agent hook (--post-tool-use)
# --------------------------------------------------------------------------- #
def post_tool_use_nudge(payload: dict[str, Any]) -> str | None:
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path:
        return None
    try:
        rel = str(Path(file_path).resolve().relative_to(ROOT))
    except ValueError:
        return None
    if rel == REGISTRY_PATH:
        return None
    worktree = registry_snapshot("WORKTREE")
    if worktree is None:
        return None
    component_id = component_for_path(rel, path_claims(worktree))
    if component_id is None:
        return None
    head = registry_snapshot("HEAD")
    if head is None:
        return None
    if worktree.get("components", {}).get(component_id) != head.get(
        "components", {}
    ).get(component_id):
        return None  # registry entry already touched
    return (
        f"version-stamps: {rel} belongs to component {component_id!r}; if this "
        f"change affects behavior, bump it in {REGISTRY_PATH} (or append a "
        f"'{NO_BUMP_PREFIX} <reason>' history entry) before finishing."
    )


def run_post_tool_use() -> int:
    try:
        payload = json.load(sys.stdin)
        nudge = post_tool_use_nudge(payload)
        if nudge:
            print(nudge)
    except Exception:  # noqa: BLE001 — advisory hook must never block or crash
        pass
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="blocking bump/stamp check")
    mode.add_argument("--stale", action="store_true", help="report retest candidates")
    mode.add_argument(
        "--post-tool-use",
        action="store_true",
        help="read agent hook JSON from stdin and print an advisory nudge",
    )
    parser.add_argument("--base", help="diff base commit-ish for --check")
    parser.add_argument(
        "--staged", action="store_true", help="check the staged index (pre-commit)"
    )
    parser.add_argument("--component", help="filter --stale to one component id")
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="also scan outputs/runs and outputs/autoresearch in --stale",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable --stale")
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="exit 1 when --stale finds retest candidates",
    )
    args = parser.parse_args(argv)
    if args.post_tool_use:
        return run_post_tool_use()
    if args.stale:
        return run_stale(
            component_filter=args.component,
            include_outputs=args.include_outputs,
            as_json=args.json,
            fail_on_stale=args.fail_on_stale,
        )
    return run_check(base_arg=args.base, staged=args.staged)


if __name__ == "__main__":
    raise SystemExit(main())
