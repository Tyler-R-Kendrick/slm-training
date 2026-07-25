#!/usr/bin/env python3
"""Publish SLM-258 (LOT4-02)'s LotusOpenUIDispositionV1 (fail-closed).

Reads the real committed upstream gate artifacts and emits the
disposition JSON + Markdown. No training, deployment, or default change.

Example:
  python -m scripts.publish_lotus_openui_disposition \
      --out outputs/runs/slm258_disposition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.lotus_openui_disposition import (
    build_disposition,
    load_project_evidence,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-258 LOT4-02 LotusOpenUIDispositionV1 publisher (fail-closed)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/runs/slm258_disposition"),
    )
    args = parser.parse_args(argv)

    evidence = load_project_evidence()
    disposition = build_disposition(evidence)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lotus_openui_disposition.json").write_text(
        json.dumps(disposition.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(disposition)
    (args.out / "lotus_openui_disposition.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
