#!/usr/bin/env python3
"""Run or verify the bounded SLM-216 fixed-token spectral-regime matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.slm216_spectral_regime import (
    DEFAULT_GATE_PATH,
    render_markdown,
    run_spectral_regime_matrix,
)

DEFAULT_MARKDOWN = "docs/design/iter-slm216-spectral-regime-20260723.md"


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
    parser.add_argument("--json", type=Path, default=Path(DEFAULT_GATE_PATH))
    parser.add_argument("--markdown", type=Path, default=Path(DEFAULT_MARKDOWN))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/slm216-spectral-regime-20260723"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--null-draws", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    json_path = args.json if args.json.is_absolute() else root / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else root / args.markdown
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    report = run_spectral_regime_matrix(
        repo_root=root,
        seeds=seeds,
        null_draws=args.null_draws,
    )
    expected_json = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    expected_markdown = render_markdown(report)
    if args.check:
        if not json_path.is_file():
            print(f"missing result JSON: {json_path}")
            return 1
        committed = json.loads(json_path.read_text(encoding="utf-8"))
        if _without_volatile(committed) != _without_volatile(report.to_dict()):
            print(f"stale result JSON: {json_path}")
            return 1
        if not markdown_path.is_file():
            print(f"missing result Markdown: {markdown_path}")
            return 1
        if markdown_path.read_text(encoding="utf-8") != expected_markdown:
            print(f"stale result Markdown: {markdown_path}")
            return 1
        print(f"{report.gate.verdict} {report.report_hash}")
        return 0

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(expected_json, encoding="utf-8")
    markdown_path.write_text(expected_markdown, encoding="utf-8")
    (args.output_dir / "spectral_regime_report.json").write_text(
        expected_json,
        encoding="utf-8",
    )
    print(f"{report.gate.verdict} {report.report_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
