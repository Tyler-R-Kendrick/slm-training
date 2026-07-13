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
        "--soft-corrupt",
        action="store_true",
        default=True,
        help="Synthesize valid-but-worse rejects (default on; prefer over BrokenText).",
    )
    build.add_argument(
        "--no-soft-corrupt",
        action="store_true",
        help="Disable soft-corrupt rejects.",
    )
    build.add_argument(
        "--corrupt",
        action="store_true",
        help="Also synthesize BrokenText-style rejects (discouraged).",
    )
    build.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help="Generate candidates from a TwoTower checkpoint (model samples).",
    )
    build.add_argument("--limit", type=int, default=None, help="Optional record cap.")
    build.add_argument("--device", default="cpu")
    build.add_argument(
        "--samples-per-prompt",
        type=int,
        default=2,
        help="When using --from-checkpoint, generate this many samples per prompt.",
    )
    build.add_argument(
        "--allow-invalid-rejects",
        action="store_true",
        help="Allow grammar-invalid rejects when ranking pairs (default: prefer valid).",
    )

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

        from slm_training.quality import soft_corrupt_openui

        use_soft = bool(args.soft_corrupt) and not bool(args.no_soft_corrupt)
        prefer_valid = not bool(args.allow_invalid_rejects)

        if args.from_checkpoint is not None:
            from slm_training.models.twotower import TwoTowerModel

            model = TwoTowerModel.from_checkpoint(
                args.from_checkpoint, device=args.device
            )
            model.config.grammar_ltr_primary = True
            model.config.design_md_in_context = False
            n_samp = max(1, int(args.samples_per_prompt))

            def gen(record):
                cands = [record.openui]
                for _ in range(n_samp):
                    cands.append(model.generate(record.prompt, gold=None))
                if use_soft:
                    cands.append(soft_corrupt_openui(record.openui))
                if args.corrupt:
                    bad = record.openui.replace("TextContent", "BrokenText", 1)
                    if bad == record.openui:
                        bad = "root = Broken()"
                    cands.append(bad)
                return cands
        else:

            def gen(record):
                cands = [record.openui]
                if use_soft:
                    cands.append(soft_corrupt_openui(record.openui))
                if args.corrupt or not use_soft:
                    bad = record.openui.replace("TextContent", "BrokenText", 1)
                    if bad == record.openui:
                        bad = "root = Broken()"
                    cands.append(bad)
                return cands

        pairs = collect_pairs_with_generator(
            records,
            gen,
            prefer_valid_rejects=prefer_valid,
            structure_only=True,
        )
        n = write_pairs(args.out, pairs)
        print(
            json.dumps(
                {
                    "pairs": n,
                    "out": str(args.out),
                    "soft_corrupt": use_soft,
                    "prefer_valid_rejects": prefer_valid,
                    "structure_only": True,
                },
                indent=2,
            )
        )
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
