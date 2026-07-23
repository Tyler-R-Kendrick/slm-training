"""Replay committed ship scoreboards and emit deterministic census evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build.evidence_census import (
    build_census,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prior-census",
        type=Path,
        help="Verified earlier census whose adjudication ledger must remain a prefix",
    )
    args = parser.parse_args()
    prior = []
    if args.prior_census is not None:
        prior_payload = json.loads(args.prior_census.read_text(encoding="utf-8"))
        prior = prior_payload["adjudications"]
    census = build_census(args.revision, prior_adjudications=prior)
    text = (
        json.dumps(census, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(census)
    )
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
