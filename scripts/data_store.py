#!/usr/bin/env python3
"""List, resolve, verify, publish, and migrate canonical model data."""

from __future__ import annotations

import argparse
import json

from slm_training.data.store import DATA_KINDS, DataStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--kind", choices=DATA_KINDS)
    for name in ("resolve", "verify", "publish"):
        command = sub.add_parser(name)
        command.add_argument("kind", choices=DATA_KINDS)
        command.add_argument("dataset_id")
    migrate = sub.add_parser("migrate")
    migrate.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    store = DataStore(args.root)

    if args.command == "list":
        kinds = [args.kind] if args.kind else DATA_KINDS
        result = [ref.to_dict() for kind in kinds for ref in store.versions(kind)]
    elif args.command == "resolve":
        result = store.resolve(args.kind, args.dataset_id).to_dict()
    elif args.command == "verify":
        result = store.verify(args.kind, args.dataset_id).to_dict()
    elif args.command == "publish":
        result = store.publish(args.kind, args.dataset_id).to_dict()
    else:
        plan = store.migration_plan()
        if args.apply:
            plan = store.migrate()
        result = {
            "applied": bool(args.apply),
            "moves": [{"source": str(src), "destination": str(dst)} for src, dst in plan],
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
