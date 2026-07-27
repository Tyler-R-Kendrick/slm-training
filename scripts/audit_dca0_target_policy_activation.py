#!/usr/bin/env python3
"""Fail-closed activation audit for SLM-269 (DCA0-01).

No training, corpus build, or evaluation is launched. Emits the
machine-readable activation disposition.

Example:
  python -m scripts.audit_dca0_target_policy_activation \
      --out outputs/experiments/slm269-activation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.slm269_dca0_activation import (
    build_activation_report,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-269 DCA0-01 target-policy activation audit (no-spend)"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing docs/design artifacts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/experiments/slm269-activation"),
    )
    args = parser.parse_args(argv)

    report = build_activation_report(args.root)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "activation_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (args.out / "activation_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
