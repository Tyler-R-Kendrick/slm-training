#!/usr/bin/env python3
"""CI entrypoint: KERN-12 computability classification fixtures (SLM-532)."""

from __future__ import annotations

import json
import sys

from slm_training.formal import computability_classification as cc


def main() -> int:
    report = cc.evaluate_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except cc.ComputabilityClassificationError as exc:
        print(f"computability classification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
