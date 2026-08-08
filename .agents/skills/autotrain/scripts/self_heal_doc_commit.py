#!/usr/bin/env python3
"""Land a continuous-autotrain doc-only closeout commit.

The continuous driver stages README.md / docs/MODEL_CARD.md / a
docs/design/<campaign>-results.{md,json} pair and expects the agent to
finish the commit. verify_version_stamps requires a matching no-bump
history entry for every touched component whenever README.md/MODEL_CARD.md
change. This helper inserts that entry (surgical text insert, not a
json.dump rewrite, to avoid reformatting/escaping the whole registry) and
commits the staged + newly-touched files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VERSIONS = ROOT / "src/slm_training/resources/versions.json"


def _staged_paths() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [p for p in out.stdout.splitlines() if p]


def _bump_component(component_id: str, date: str, note: str) -> bool:
    data = json.loads(VERSIONS.read_text(encoding="utf-8"))
    comp = data["components"][component_id]
    version = comp["version"]
    text = VERSIONS.read_text(encoding="utf-8")
    idx = text.index(f'"{component_id}"')
    hist_idx = text.index('"history": [', idx)
    insert_at = text.index("{", hist_idx)
    entry = (
        "{\n"
        f'          "version": "{version}",\n'
        f'          "date": "{date}",\n'
        f'          "note": "{note}"\n'
        "        },\n        "
    )
    new_text = text[:insert_at] + entry + text[insert_at:]
    VERSIONS.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", default="harness.experiments.slm228_spectral_disposition")
    parser.add_argument("--date", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    staged = _staged_paths()
    if not staged:
        print("no staged paths; nothing to commit", file=sys.stderr)
        return 1

    _bump_component(args.component, args.date, args.note)
    subprocess.run(["git", "add", str(VERSIONS.relative_to(ROOT))], cwd=ROOT, check=True)

    check = subprocess.run(
        [str(ROOT / ".venv/bin/python3"), "-m", "scripts.verify_version_stamps", "--check", "--staged"],
        cwd=ROOT,
    )
    if check.returncode != 0:
        print("version-stamps check still failing after bump", file=sys.stderr)
        return check.returncode

    commit = subprocess.run(
        ["git", "commit", "-m", args.message],
        cwd=ROOT,
    )
    return commit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
