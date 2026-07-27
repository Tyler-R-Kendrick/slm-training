#!/usr/bin/env python3
"""Publish SLM-276 (DCA2-03)'s CausalDispositionBundleV1 (fail-closed).

Reads the real committed prerequisite disposition artifacts and emits the
bundle JSON + Markdown. No tournament, generation, or promotion spend.

Example:
  python -m scripts.build_causal_disposition --out outputs/disposition/slm276
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.dca_causal_disposition import (
    build_bundle,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-276 DCA2-03 CausalDispositionBundleV1 publisher (fail-closed)"
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/disposition/slm276"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero when evidence gaps exist",
    )
    args = parser.parse_args(argv)

    bundle = build_bundle(args.root)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "causal_disposition_bundle.json").write_text(
        json.dumps(bundle.to_dict(), indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(bundle)
    (args.out / "causal_disposition_bundle.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if args.check and bundle.evidence_gaps:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
