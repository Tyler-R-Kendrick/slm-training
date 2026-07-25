#!/usr/bin/env python3
"""Publish or verify the SLM-228 SpectralDispositionV1 report."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from slm_training.harnesses.experiments.slm228_spectral_disposition import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    build_report,
    render_markdown,
)


def _without_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_volatile(child)
            for key, child in value.items()
            if key not in {"generated_at", "report_hash", "stamped_at"}
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile(child) for child in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path(DEFAULT_JSON))
    parser.add_argument("--markdown", type=Path, default=Path(DEFAULT_MARKDOWN))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    json_path = args.json if args.json.is_absolute() else root / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else root / args.markdown
    report = build_report(root)
    if args.check:
        if not json_path.is_file() or not markdown_path.is_file():
            print("missing committed SpectralDispositionV1 report")
            return 1
        committed = json.loads(json_path.read_text(encoding="utf-8"))
        stamp = dict(report.version_stamp)
        committed_stamp = committed.get("version_stamp", {})
        stamp["code_commit"] = committed_stamp.get("code_commit")
        stamp["code_dirty"] = committed_stamp.get("code_dirty")
        report = replace(
            report,
            evidence_cutoff_commit=str(
                committed.get("evidence_cutoff_commit", "UNKNOWN")
            ),
            version_stamp=stamp,
        )
        if _without_volatile(committed) != _without_volatile(report.to_dict()):
            print(f"stale disposition JSON: {json_path}")
            return 1
        if markdown_path.read_text(encoding="utf-8") != render_markdown(report):
            print(f"stale disposition Markdown: {markdown_path}")
            return 1
        print(f"{report.schema} {report.report_hash}")
        return 0

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"{report.schema} {report.report_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
