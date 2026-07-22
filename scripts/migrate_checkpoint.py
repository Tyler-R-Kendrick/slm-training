#!/usr/bin/env python3
"""Migrate legacy TwoTower or fixed-canvas grammar checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.levers import DEFAULT_TRAIN_DATA_DIR

import torch

from slm_training.models.checkpoint_migrate import (
    migrate_grammar_diffusion_checkpoint,
    migrate_twotower_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Source checkpoint (.pt) with legacy tokenizer sidecar.",
    )
    parser.add_argument(
        "--train-records",
        type=Path,
        default=DEFAULT_TRAIN_DATA_DIR / "records.jsonl",
        help="Train records used to rebuild compositional vocabulary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination checkpoint path (writes .tokenizer.json + .meta.json).",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("kind") == "grammar_diffusion":
        report = migrate_grammar_diffusion_checkpoint(
            source_checkpoint=args.checkpoint,
            output_checkpoint=args.output,
            device=args.device,
        )
    else:
        report = migrate_twotower_checkpoint(
            source_checkpoint=args.checkpoint,
            train_records_path=args.train_records,
            output_checkpoint=args.output,
            device=args.device,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
