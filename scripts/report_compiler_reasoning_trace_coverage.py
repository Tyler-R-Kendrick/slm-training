"""CLI for the LOT0-02 (SLM-249) bounded CompilerReasoningTraceV1 K/c coverage probe.

Extracts a trace for each of the 16 hand-authored records in
``src/slm_training/resources/test_seeds.jsonl`` (the repo's existing bounded
fixture set) and reports per-stage length distributions plus a provisional
K/c budget. This is a ``bounded_probe`` -- explicitly not a corpus-scale
statistic (see the LOT0-01 authorization for why corpus generation is out of
scope for this issue).

Example::

    python -m scripts.report_compiler_reasoning_trace_coverage \\
        --out-dir docs/design/compiler-reasoning-trace-coverage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.compiler_reasoning_trace_coverage import (
    build_bounded_coverage_report,
    render_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write coverage_report.json/.md into (stdout only if omitted).",
    )
    args = parser.parse_args()

    report = build_bounded_coverage_report()
    payload = report.to_dict()

    if args.out_dir is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "coverage_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "coverage_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.out_dir / 'coverage_report.json'} and coverage_report.md")


if __name__ == "__main__":
    main()
