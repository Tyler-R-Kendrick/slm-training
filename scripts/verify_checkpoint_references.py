#!/usr/bin/env python3
"""Fail-closed audit of committed checkpoint references.

A ``frontier`` / ``ship_candidate`` checkpoint reference must be fully
provenanced and resolvable/verified from a fresh clone; a ``fixture`` /
``diagnostic`` reference may stay local-only as long as it is honestly
classified. This audit enforces that contract over the structured
``CheckpointReferenceV1`` objects embedded in (or committed alongside)
``docs/design`` result JSON — it never guesses provenance from prose, and it
never treats an honestly-labeled local scratch row as a frontier claim.

Checks performed per durable reference:

* every provenance field required for publication is present (fail closed);
* the artifact resolves — a tracked local file is byte/SHA-256 verified, an
  ``hf://`` remote is accepted only when a sync-time verification is recorded;
* duplicate ``(run_id, role)`` references never map to different SHA-256;
* a ``(run_id, role)`` never carries two conflicting training source commits.

Run ``python -m scripts.verify_checkpoint_references`` (scans ``docs/design``)
or ``--paths a.json b.json`` for specific files. Exit code is ``0`` when the
audit passes and ``1`` when any durable reference is unresolvable, unverified,
mismatched, or under-provenanced. See ``docs/design/checkpoint-provenance.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from slm_training.harnesses.model_build.checkpoint_reference import (
    DURABLE_CLAIM_CLASSES,
    SCHEMA_VERSION,
    UNKNOWN,
    CheckpointReferenceV1,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

# Directories scanned by default for committed references.
DEFAULT_SCAN_DIRS: tuple[str, ...] = ("docs/design",)

# The only checkpoint tree that survives a fresh clone (everything else is
# gitignored under ``outputs/``); frontier references verified here are byte-checked.
TRACKED_CHECKPOINT_ROOT = "src/slm_training/resources/checkpoints"


def _looks_like_reference(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("schema_version") == SCHEMA_VERSION
        and "claim_class" in obj
        and "checkpoint_role" in obj
    )


def _extract_references(data: Any) -> list[tuple[dict[str, Any], str]]:
    """Structurally find every reference dict in parsed JSON (no regex)."""
    found: list[tuple[dict[str, Any], str]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            refs = obj.get("references")
            if isinstance(refs, list) and obj.get("schema_version") == SCHEMA_VERSION:
                for entry in refs:
                    if _looks_like_reference(entry):
                        found.append((entry, "manifest"))
            if _looks_like_reference(obj):
                found.append((obj, "reference"))
            for key, value in obj.items():
                if key == "references":
                    continue
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return found


def iter_reference_sources(
    root: Path,
    scan_dirs: Iterable[str],
    extra_paths: Iterable[str] = (),
) -> Iterator[tuple[Path, dict[str, Any], str]]:
    """Yield ``(json_path, reference_dict, origin)`` for every committed reference."""
    files: list[Path] = []
    for rel in scan_dirs:
        base = root / rel
        if base.exists():
            files.extend(sorted(base.rglob("*.json")))
    for extra in extra_paths:
        candidate = Path(extra)
        files.append(candidate if candidate.is_absolute() else root / candidate)

    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for ref_dict, origin in _extract_references(data):
            yield path, ref_dict, origin


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_reference(ref: CheckpointReferenceV1, root: Path) -> dict[str, Any]:
    """Locate + byte-verify the artifact when possible.

    Tracked local files are SHA-256 verified. ``hf://`` remotes cannot be
    fetched here, so they are accepted only when the reference carries a
    sync-time verification stamp; otherwise they resolve as ``remote_unverified``.
    """
    candidates: list[Path] = []
    tracked = root / TRACKED_CHECKPOINT_ROOT
    if tracked.exists() and ref.checkpoint_filename not in ("", UNKNOWN):
        candidates.extend(sorted(tracked.rglob(ref.checkpoint_filename)))
    for key, value in ref.metadata:
        if key in {"local_path", "checkpoint_path"} and value:
            hinted = Path(value)
            candidates.append(hinted if hinted.is_absolute() else root / value)

    for candidate in candidates:
        if candidate.is_file():
            actual = sha256_file(candidate)
            if ref.sha256 not in ("", UNKNOWN) and actual != ref.sha256:
                return {
                    "status": "hash_mismatch",
                    "path": _rel(candidate, root),
                    "expected": ref.sha256,
                    "actual": actual,
                }
            return {"status": "verified_local", "path": _rel(candidate, root)}

    if ref.remote_uri.startswith("hf://"):
        if ref.verification_timestamp:
            return {"status": "remote_recorded_verified", "remote_uri": ref.remote_uri}
        return {"status": "remote_unverified", "remote_uri": ref.remote_uri}
    return {"status": "unresolved"}


def audit_reference(
    ref: CheckpointReferenceV1, source: str, root: Path
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(errors, resolution)`` for one reference."""
    resolution = resolve_reference(ref, root)
    errors: list[str] = []
    label = f"{source}: run={ref.run_id!r} role={ref.checkpoint_role!r} [{ref.claim_class}]"

    if resolution["status"] == "hash_mismatch":
        # A positive byte mismatch is always a failure, regardless of claim class.
        errors.append(
            f"{label} SHA-256 mismatch at {resolution['path']}: "
            f"expected {resolution['expected']} got {resolution['actual']}"
        )

    if ref.claim_class in DURABLE_CLAIM_CLASSES:
        for field, reason in ref.blocking_reasons():
            errors.append(f"{label} missing {field}: {reason}")
        if resolution["status"] in {"unresolved", "remote_unverified"}:
            errors.append(
                f"{label} checkpoint not resolvable/verified "
                f"(resolution={resolution['status']}); a durable claim requires a "
                "fresh-clone-resolvable, verified artifact"
            )
    return errors, resolution


def cross_reference_checks(
    entries: list[tuple[CheckpointReferenceV1, str]],
) -> list[str]:
    """Detect duplicate hashes and conflicting commits across references."""
    errors: list[str] = []
    by_key: dict[tuple[str, str], list[tuple[CheckpointReferenceV1, str]]] = {}
    for ref, source in entries:
        by_key.setdefault((ref.run_id, ref.checkpoint_role), []).append((ref, source))
    for (run_id, role), group in sorted(by_key.items()):
        shas = sorted(
            {ref.sha256 for ref, _ in group if ref.sha256 not in ("", UNKNOWN)}
        )
        if len(shas) > 1:
            errors.append(
                f"duplicate run/role ({run_id!r}, {role!r}) maps to different "
                f"SHA-256: {shas}"
            )
        commits = sorted(
            {
                ref.training_source_commit
                for ref, _ in group
                if ref.training_source_commit not in ("", UNKNOWN)
            }
        )
        if len(commits) > 1:
            errors.append(
                f"run/role ({run_id!r}, {role!r}) has conflicting "
                f"training_source_commit: {commits}"
            )
    return errors


def build_report(
    *,
    root: Path = ROOT,
    scan_dirs: Iterable[str] = DEFAULT_SCAN_DIRS,
    extra_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit every committed checkpoint reference and return a JSON report."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    entries: list[tuple[CheckpointReferenceV1, str]] = []

    for path, ref_dict, origin in iter_reference_sources(root, scan_dirs, extra_paths):
        rel = _rel(path, root)
        try:
            ref = CheckpointReferenceV1.from_dict(ref_dict)
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"{rel}: invalid checkpoint reference ({origin}): {exc}")
            checks.append(
                {"source": rel, "origin": origin, "status": "invalid", "error": str(exc)}
            )
            continue
        ref_errors, resolution = audit_reference(ref, rel, root)
        entries.append((ref, rel))
        checks.append(
            {
                "source": rel,
                "origin": origin,
                "run_id": ref.run_id,
                "role": ref.checkpoint_role,
                "claim_class": ref.claim_class,
                "status": "fail" if ref_errors else "pass",
                "resolution": resolution,
                "errors": ref_errors,
            }
        )
        errors.extend(ref_errors)

    cross = cross_reference_checks(entries)
    errors.extend(cross)

    by_class: dict[str, int] = {}
    for ref, _ in entries:
        by_class[ref.claim_class] = by_class.get(ref.claim_class, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "pass": not errors,
        "reference_count": len(entries),
        "by_claim_class": by_class,
        "checks": checks,
        "cross_reference_errors": cross,
        "errors": errors,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Checkpoint reference audit", ""]
    verdict = "PASS" if report["pass"] else "FAIL"
    lines.append(f"- verdict: **{verdict}**")
    lines.append(f"- references audited: {report['reference_count']}")
    if report["by_claim_class"]:
        breakdown = ", ".join(
            f"{cls}={n}" for cls, n in sorted(report["by_claim_class"].items())
        )
        lines.append(f"- by claim class: {breakdown}")
    if report["errors"]:
        lines.append("")
        lines.append("## Errors")
        for err in report["errors"]:
            lines.append(f"- {err}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        default=(),
        help="Explicit JSON/reference files to audit (in addition to the scan).",
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        default=None,
        help=f"Directory to scan for references (default: {', '.join(DEFAULT_SCAN_DIRS)}).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: inferred from this file).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report here.")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="Write a Markdown report here."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero on any failure (default behavior; kept for CI symmetry).",
    )
    args = parser.parse_args(argv)

    scan_dirs = tuple(args.scan_dir) if args.scan_dir else DEFAULT_SCAN_DIRS
    report = build_report(root=args.root, scan_dirs=scan_dirs, extra_paths=args.paths)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "pass": report["pass"],
                "reference_count": report["reference_count"],
                "errors": report["errors"][:20],
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
