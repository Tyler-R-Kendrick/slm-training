#!/usr/bin/env python3
"""SLM-492: honest external-blocked prepared-package closeout for EXP-SR-11.

Example:
  python -m scripts.run_slm492_external_blocked_closeout
  python -m scripts.run_slm492_external_blocked_closeout --output-dir outputs/runs/slm492
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.harnesses.experiments.slm492_external_blocked_closeout import (
    DESIGN_STEM,
    run_closeout,
    render_markdown,
)

_DESIGN_JSON = Path(f"{DESIGN_STEM}.json")
_DESIGN_MD = Path(f"{DESIGN_STEM}.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm492_external_blocked_closeout"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_closeout(root=args.output_dir, seed=args.seed)

    run_json = args.output_dir / "slm492_external_blocked_closeout.json"
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    docs_json = root / _DESIGN_JSON
    docs_md = root / _DESIGN_MD
    docs_json.parent.mkdir(parents=True, exist_ok=True)
    docs_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    docs_md.write_text(
        render_markdown(payload)
        + "\nCommand: `python -m scripts.run_slm492_external_blocked_closeout`\n",
        encoding="utf-8",
    )

    print(str(run_json))
    print(f"wrote {docs_json}")
    print(f"wrote {docs_md}")
    print(f"status={payload.get('status')} gap={payload.get('srbench_matched_score_gap')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
