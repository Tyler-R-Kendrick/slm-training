#!/usr/bin/env python3
"""Explain cross-split artifact-graph overlap candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.train_data.artifact_graph import ArtifactGraphStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    findings = ArtifactGraphStore(args.dataset_root).explain_overlaps()
    if args.json:
        print(json.dumps([item.to_dict() for item in findings], indent=2))
    elif not findings:
        print("no cross-split overlap candidates")
    else:
        for item in findings:
            print(
                f"{item.code.value}: {item.left_id} ({item.left_split}) <> "
                f"{item.right_id} ({item.right_split}); {item.evidence}"
            )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
