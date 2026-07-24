#!/usr/bin/env python3
"""Measure exact compiler legal-action entropy for a locked manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.data.locked_eval_manifest import measure_stratified_legal_entropy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=24)
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    measured = measure_stratified_legal_entropy(payload, max_records=args.max_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(measured["records"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
