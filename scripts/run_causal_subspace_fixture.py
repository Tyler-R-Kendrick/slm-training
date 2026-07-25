#!/usr/bin/env python3
"""Run or verify the bounded SLM-220 causal-subspace fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.slm220_causal_subspace import (
    render_markdown,
    run_fixture_retrospective,
)

DEFAULT_JSON = "docs/design/iter-slm220-causal-subspace-20260723.json"
DEFAULT_MARKDOWN = "docs/design/iter-slm220-causal-subspace-20260723.md"


def _without_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_volatile(child)
            for key, child in value.items()
            if key not in {"stamped_at", "timestamp", "source_commit", "code_commit"}
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
    markdown_path = (
        args.markdown if args.markdown.is_absolute() else root / args.markdown
    )
    report = run_fixture_retrospective(root)
    payload = report.to_dict()
    expected_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    expected_markdown = render_markdown(report)
    if args.check:
        if not json_path.is_file() or not markdown_path.is_file():
            print("missing committed SLM-220 result")
            return 1
        if _without_volatile(json.loads(json_path.read_text())) != _without_volatile(
            payload
        ):
            print(f"stale result JSON: {json_path}")
            return 1
        if markdown_path.read_text() != expected_markdown:
            print(f"stale result Markdown: {markdown_path}")
            return 1
        print(f"{report.verdict} {report.report_hash}")
        return 0
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(expected_json)
    markdown_path.write_text(expected_markdown)
    print(f"{report.verdict} {report.report_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
