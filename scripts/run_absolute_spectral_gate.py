#!/usr/bin/env python3
"""Run or verify the bounded SLM-226 absolute spectral target gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.slm226_absolute_spectral_gate import (
    DEFAULT_NULL_DRAWS,
    DEFAULT_REPORT_PATH,
    render_markdown,
    run_absolute_spectral_boundary,
)

DEFAULT_MARKDOWN = "docs/design/iter-slm226-absolute-spectral-gate-20260723.md"


def _without_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_volatile(child)
            for key, child in value.items()
            if key
            not in {
                "elapsed_ms",
                "generated_at",
                "report_hash",
                "stamped_at",
                "timestamp",
                "total_elapsed_ms",
            }
        }
    if isinstance(value, list):
        return [_without_volatile(child) for child in value]
    if isinstance(value, tuple):
        return [_without_volatile(child) for child in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path(DEFAULT_REPORT_PATH))
    parser.add_argument("--markdown", type=Path, default=Path(DEFAULT_MARKDOWN))
    parser.add_argument("--null-draws", type=int, default=DEFAULT_NULL_DRAWS)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    json_path = args.json if args.json.is_absolute() else root / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else root / args.markdown
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    report = run_absolute_spectral_boundary(
        repo_root=root,
        seeds=seeds,
        null_draws=args.null_draws,
    )
    expected_json = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    expected_markdown = render_markdown(report)
    if args.check:
        if not json_path.is_file() or not markdown_path.is_file():
            print("missing committed SLM-226 report")
            return 1
        committed = json.loads(json_path.read_text(encoding="utf-8"))
        if _without_volatile(committed) != _without_volatile(report.to_dict()):
            print(f"stale result JSON: {json_path}")
            return 1
        if markdown_path.read_text(encoding="utf-8") != expected_markdown:
            print(f"stale result Markdown: {markdown_path}")
            return 1
        print(f"{report.gate.verdict} {report.report_hash}")
        return 0
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(expected_json, encoding="utf-8")
    markdown_path.write_text(expected_markdown, encoding="utf-8")
    print(f"{report.gate.verdict} {report.report_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
