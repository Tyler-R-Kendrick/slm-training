"""Query the durable evidence store for prior attempts.

Answers "has this been tried before?" from the committed local index (always)
or Supabase (when ``SUPABASE_URL`` / ``SUPABASE_SERVICE_ROLE_KEY`` are set).
Never requires network access — fail-soft degradation to the local index.

Usage:
    python -m scripts.query_evidence --hypothesis "macro tokens improve parse"
    python -m scripts.query_evidence --lever bounds --lever binder-topology
    python -m scripts.query_evidence --fingerprint <sha256> --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from slm_training.evidence_store.client import find_prior_attempts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis", help="free-text hypothesis to search for")
    parser.add_argument(
        "--lever",
        action="append",
        default=None,
        help="lever key to match (repeatable)",
    )
    parser.add_argument("--fingerprint", help="exact config_fingerprint (sha256)")
    parser.add_argument(
        "--json", action="store_true", help="emit records as a JSON array"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="max records to print (default 20)"
    )
    args = parser.parse_args(argv)

    if not (args.hypothesis or args.lever or args.fingerprint):
        parser.error("pass at least one of --hypothesis, --lever, --fingerprint")

    records = find_prior_attempts(
        hypothesis_text=args.hypothesis,
        lever_keys=args.lever,
        config_fingerprint=args.fingerprint,
    )[: max(0, args.limit)]

    if args.json:
        print(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not records:
        print("no prior attempts found")
        return 0
    for record in records:
        effect = (
            f"{record.effect_size:+.4f}" if record.effect_size is not None else "n/a"
        )
        blocker = f" blocker={record.blocker}" if record.blocker else ""
        print(
            f"{record.outcome:15s} {record.endpoint_metric:40s} "
            f"effect={effect:>8s} n={record.n_seeds:<3d} "
            f"{record.experiment_id}{blocker}"
        )
        print(f"  {record.hypothesis_text}")
        print(f"  fingerprint={record.config_fingerprint} src={record.source_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
