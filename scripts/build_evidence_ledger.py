"""Build or inspect the committed autotrain evidence ledger.

Mines the durable experiment record under ``docs/design`` into the committed
cross-version arm-evidence index consumed by the continuous hill-climb
selector. Deterministic: re-running over the same tree reproduces the same
artifact byte-for-byte.

Usage:
    python -m scripts.build_evidence_ledger            # dry run, print summary
    python -m scripts.build_evidence_ledger --write    # rewrite the resource
    python -m scripts.build_evidence_ledger --status   # summarize the ledger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.autoresearch import evidence_ledger as ev

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_DIR = REPO_ROOT / "docs" / "design"


def _summarize(arms: dict) -> list[str]:
    rows = []
    ranked = sorted(
        arms.items(),
        key=lambda item: (-item[1].get("n_obs", 0), item[0]),
    )
    for slug, stats in ranked:
        rows.append(
            f"{slug:44s} obs={stats.get('n_obs', 0):4d} "
            f"null={stats.get('n_null', 0):4d} pos={stats.get('n_positive', 0):3d} "
            f"mean_delta={stats.get('mean_delta', 0.0):+.4f}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--out", type=Path, default=ev.DEFAULT_LEDGER_PATH)
    parser.add_argument("--write", action="store_true", help="write the ledger")
    parser.add_argument(
        "--status", action="store_true", help="summarize the committed ledger"
    )
    args = parser.parse_args(argv)

    if args.status:
        arms = ev.load_ledger(args.out)
        if not arms:
            print(f"no ledger at {args.out}")
            return 1
        print(f"ledger: {args.out}")
        for row in _summarize(arms):
            print(row)
        return 0

    payload = ev.build_ledger(args.design_dir)
    counts = payload["file_counts"]
    print(
        f"scanned={counts['scanned']} unreadable={counts['unreadable']} "
        f"observations={counts['observations']} arms={len(payload['arms'])}"
    )
    for row in _summarize(payload["arms"])[:20]:
        print(row)
    if args.write:
        ev.write_ledger(payload, args.out)
        print(f"wrote {args.out}")
    else:
        print("dry run (pass --write to persist)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
