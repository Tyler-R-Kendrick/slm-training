"""Verify the typed G0/CAP0 foundation disposition against this checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.staged import StagedHarnessFoundationDispositionV1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disposition", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    disposition = StagedHarnessFoundationDispositionV1.load(args.disposition)
    disposition.require_supported(args.repo_root.resolve())
    print(
        json.dumps(
            {
                "decision": "supported",
                "disposition_sha256": disposition.sha,
                "claim_count": len(disposition.claims),
                "frozen_identity_count": len(disposition.frozen_identities),
                "next_work_item": disposition.next_work_item,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
