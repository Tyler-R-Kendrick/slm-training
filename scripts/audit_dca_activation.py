#!/usr/bin/env python3
"""Evaluate a DCA-chain issue's activation gate (fail-closed, no-spend).

No training, surgery, curriculum, or tournament is launched. Reads the
real committed prerequisite artifacts and emits a per-issue
``DcaActivationGateV1`` disposition.

Example:
  python -m scripts.audit_dca_activation --issue SLM-270 \
      --out outputs/experiments/slm270-activation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.dca_activation_gate import (
    DCA_ISSUES,
    evaluate_dca_gate,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DCA-chain activation-gate evaluator (fail-closed, no-spend)"
    )
    parser.add_argument(
        "--issue",
        required=True,
        choices=sorted(DCA_ISSUES),
        help="Linear issue id to evaluate (registry: DCA_ISSUES)",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default outputs/experiments/<issue>-activation)",
    )
    args = parser.parse_args(argv)

    spec = DCA_ISSUES[args.issue]
    out = args.out or Path("outputs/experiments") / f"{args.issue.lower()}-activation"

    contract = evaluate_dca_gate(spec, args.root)

    out.mkdir(parents=True, exist_ok=True)
    stem = spec.slug.replace("-", "_")
    (out / f"{stem}_gate.json").write_text(
        json.dumps(contract.to_dict(), indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(contract)
    (out / f"{stem}_gate.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
