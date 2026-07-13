#!/usr/bin/env python3
"""Export the official OpenUI system prompt for teacher / synth pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.dsl import generate_system_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/openui_system_prompt.txt"),
    )
    parser.add_argument(
        "--preamble",
        default=None,
        help="Optional preamble override for library.prompt()",
    )
    args = parser.parse_args(argv)

    options = {}
    if args.preamble:
        options["preamble"] = args.preamble
    text = generate_system_prompt(**options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
