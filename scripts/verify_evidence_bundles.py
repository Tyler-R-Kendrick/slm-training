"""Verify evidence bundles and census model-card checkpoint references (SLM-291).

Deterministic and no-write: exits 1 when any committed ``evidence_bundle/v1``
payload fails verification or any model-card roster row with a durable claim
references a non-durable location.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.harnesses.model_build.evidence_bundle_census import (
    build_census,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-card", type=Path, default=None)
    parser.add_argument("--design-dir", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    census = build_census(
        model_card=args.model_card,
        design_dir=args.design_dir,
        artifact_root=args.artifact_root,
    )
    if args.format == "markdown":
        rendered = render_markdown(census)
    else:
        rendered = json.dumps(census, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 1 if census["summary"]["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
