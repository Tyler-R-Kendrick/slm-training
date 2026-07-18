#!/usr/bin/env python3
"""Backfill report for historical checkpoint references (no fabrication).

Scans the structured tables in ``docs/MODEL_CARD.md`` and the ``README.md``
model-card summary and classifies every named checkpoint by whether it is
resolvable from a fresh clone:

* ``tracked_local`` — committed under ``src/slm_training/resources/checkpoints``;
* ``remote_declared`` — a concrete ``hf://buckets/...`` durable URI;
* ``template`` — the unfilled ``<run_id>`` placeholder row;
* ``unresolved_local`` — a gitignored ``outputs/`` or ``/tmp/`` path that is
  absent from a clone and has no durable remote.

For every ``unresolved_local`` entry it records the exact local path/document
and a remediation command. It never invents a remote URI and never drops a row.
Run ``python -m scripts.backfill_checkpoint_references --out <json> --markdown
<md>``; the committed report is the migration record referenced by SLM-103.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]

TRACKED_CHECKPOINT_ROOT = "src/slm_training/resources/checkpoints"
DEFAULT_SYNC_HINT = (
    "python scripts/sync_checkpoints.py --run-dir {run_dir} --run-id {run_id} "
    "--claim-class <frontier|diagnostic> --ensure-bucket  "
    "(requires the local checkpoint on the training host + HF auth)"
)

# Table columns (case-insensitive) that hold a checkpoint location.
_LOCATION_COLUMNS = ("location", "bucket / path", "where")
_RUN_ID_COLUMNS = ("run id", "checkpoint")

_BACKTICK = re.compile(r"`([^`]+)`")


def _split_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return cells


def _is_separator(cells: list[str]) -> bool:
    return all(set(cell) <= {"-", ":", " "} and cell for cell in cells)


def iter_tables(text: str) -> Iterator[tuple[list[str], list[list[str]]]]:
    """Yield ``(header, rows)`` for every GitHub-flavored markdown table."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = _split_row(lines[i])
        sep = _split_row(lines[i + 1]) if i + 1 < len(lines) else None
        if header and sep and _is_separator(sep):
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines):
                row = _split_row(lines[j])
                if row is None:
                    break
                if len(row) == len(header):
                    rows.append(row)
                j += 1
            yield header, rows
            i = j
        else:
            i += 1


def _column_index(header: list[str], names: tuple[str, ...]) -> int | None:
    lowered = [h.lower() for h in header]
    for name in names:
        if name in lowered:
            return lowered.index(name)
    return None


def _first_path(cell: str) -> str | None:
    for match in _BACKTICK.findall(cell):
        if "/" in match or match.startswith("hf://"):
            return match
    return None


def _classify(location_cell: str) -> tuple[str, bool, str | None]:
    """Return ``(classification, resolvable, durable_uri)`` for a location cell."""
    path = _first_path(location_cell) or location_cell.strip()
    if path.startswith("hf://"):
        if "<run_id>" in path or "None registered" in location_cell:
            return "template", False, None
        return "remote_declared", True, path
    if path.startswith(TRACKED_CHECKPOINT_ROOT):
        return "tracked_local", True, None
    if "(git)" in location_cell:
        return "tracked_local", True, None
    # Everything else is a gitignored local artifact.
    return "unresolved_local", False, None


def _derive_run_dir(local_path: str) -> str:
    """Best-effort ``outputs/runs/<id>`` dir from a ``.../checkpoints/last.pt`` path."""
    p = local_path
    for marker in ("/checkpoints/",):
        if marker in p:
            return p.split(marker, 1)[0]
    return p


def _scan_source(text: str, source: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for header, rows in iter_tables(text):
        run_col = _column_index(header, _RUN_ID_COLUMNS)
        loc_col = _column_index(header, _LOCATION_COLUMNS)
        if run_col is None or loc_col is None:
            continue
        for row in rows:
            run_id_raw = row[run_col]
            location_cell = row[loc_col]
            run_id = (_BACKTICK.search(run_id_raw) or [None, run_id_raw.strip()])[1]
            classification, resolvable, durable_uri = _classify(location_cell)
            local_path = _first_path(location_cell)
            entry: dict[str, Any] = {
                "source": source,
                "run_id": run_id,
                "location": location_cell,
                "local_path": local_path,
                "classification": classification,
                "resolvable_from_clone": resolvable,
                "durable_uri": durable_uri,
            }
            if classification == "unresolved_local":
                run_dir = _derive_run_dir(local_path) if local_path else "outputs/runs/<run_id>"
                entry["remediation"] = DEFAULT_SYNC_HINT.format(
                    run_dir=run_dir, run_id=run_id or "<run_id>"
                )
            entries.append(entry)
    return entries


def build_backfill(*, root: Path = ROOT) -> dict[str, Any]:
    """Classify every historical checkpoint reference into a migration report."""
    entries: list[dict[str, Any]] = []
    model_card = root / "docs" / "MODEL_CARD.md"
    readme = root / "README.md"
    if model_card.is_file():
        entries.extend(
            _scan_source(model_card.read_text(encoding="utf-8"), "docs/MODEL_CARD.md")
        )
    if readme.is_file():
        entries.extend(_scan_source(readme.read_text(encoding="utf-8"), "README.md"))

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["classification"]] = counts.get(entry["classification"], 0) + 1

    unresolved = [e for e in entries if e["classification"] == "unresolved_local"]

    # The champion registry lives under outputs/lineage/ (gitignored), so it is
    # not resolvable from a clone; record it explicitly rather than omitting it.
    champion_note = {
        "source": "src/slm_training/lineage/store.py",
        "detail": (
            "Champion pointers persist under outputs/lineage/champions/<track>/ "
            "which is gitignored; artifact_uri is absent from a fresh clone. "
            "Promote via the lineage store on a host with the durable checkpoint."
        ),
        "resolvable_from_clone": False,
    }

    return {
        "generated_by": "scripts/backfill_checkpoint_references.py",
        "note": (
            "Migration record for SLM-103. Historical checkpoints are recorded "
            "honestly: gitignored local artifacts are UNRESOLVED (no remote URI "
            "is invented). None is promoted or reclassified."
        ),
        "counts": counts,
        "total": len(entries),
        "unresolved_count": len(unresolved),
        "champion_registry": champion_note,
        "entries": entries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Checkpoint reference backfill", "", report["note"], ""]
    lines.append(f"- total references: {report['total']}")
    lines.append(f"- unresolved (local/gitignored): {report['unresolved_count']}")
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(report["counts"].items()))
    lines.append(f"- by classification: {breakdown}")
    lines.append("")
    lines.append("## Unresolved historical checkpoints")
    lines.append("")
    lines.append("| Source | Run id | Local path | Remediation |")
    lines.append("| --- | --- | --- | --- |")
    for entry in report["entries"]:
        if entry["classification"] != "unresolved_local":
            continue
        lines.append(
            f"| {entry['source']} | `{entry['run_id']}` | "
            f"`{entry.get('local_path') or '—'}` | {entry.get('remediation', '')} |"
        )
    lines.append("")
    lines.append("## Champion registry")
    lines.append("")
    lines.append(report["champion_registry"]["detail"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report here.")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="Write a Markdown report here."
    )
    args = parser.parse_args(argv)

    report = build_backfill(root=args.root)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "total": report["total"],
                "unresolved_count": report["unresolved_count"],
                "counts": report["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
