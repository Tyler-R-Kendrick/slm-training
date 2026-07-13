#!/usr/bin/env python3
"""Build preference pairs and/or run DPO-style preference training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.preference import (
    collect_pairs_with_generator,
    write_pairs,
)
from slm_training.preference.train import train_preference_from_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-pairs", help="Build preference pairs from train records")
    build.add_argument("--train-records", type=Path, required=True)
    build.add_argument("--out", type=Path, default=Path("outputs/preferences/pairs.jsonl"))
    build.add_argument(
        "--corrupt",
        action="store_true",
        help="Synthesize a rejected candidate by breaking the gold OpenUI.",
    )
    build.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help="Generate rejected candidates from a TwoTower checkpoint (model samples).",
    )
    build.add_argument("--limit", type=int, default=None, help="Optional record cap.")
    build.add_argument("--device", default="cpu")

    train = sub.add_parser("train", help="Run preference training from a checkpoint")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--pairs", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, default=Path("outputs/runs/preference"))
    train.add_argument("--steps", type=int, default=50)
    train.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    if args.cmd == "build-pairs":
        records = load_jsonl(args.train_records)
        if args.limit is not None:
            records = records[: max(0, int(args.limit))]

        if args.from_checkpoint is not None:
            from slm_training.models.twotower import TwoTowerModel

            model = TwoTowerModel.from_checkpoint(
                args.from_checkpoint, device=args.device
            )
            model.config.grammar_ltr_primary = True
            model.config.design_md_in_context = False

            def gen(record):
                pred = model.generate(record.prompt, gold=None)
                # Gold + model sample (+ optional corruption) as candidates.
                cands = [record.openui, pred]
                if args.corrupt:
                    bad = record.openui.replace("TextContent", "BrokenText", 1)
                    if bad == record.openui:
                        bad = "root = Broken()"
                    cands.append(bad)
                return cands
        else:

            def gen(record):
                bad = record.openui.replace("TextContent", "BrokenText", 1)
                if bad == record.openui:
                    bad = "root = Broken()"
                return [record.openui, bad]

        pairs = collect_pairs_with_generator(records, gen)
        n = write_pairs(args.out, pairs)
        print(json.dumps({"pairs": n, "out": str(args.out)}, indent=2))
        return 0

    summary = train_preference_from_paths(
        args.checkpoint,
        args.pairs,
        out_dir=args.out_dir,
        steps=args.steps,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
