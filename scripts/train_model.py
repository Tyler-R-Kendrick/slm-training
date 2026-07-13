#!/usr/bin/env python3
"""Train a ModelPlugin (default: TwoTower) on train-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build import ModelBuildConfig, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train_data/v0"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument(
        "--model",
        choices=("twotower", "stub"),
        default="twotower",
        help="Model plug-in to train (default: twotower).",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--denoiser-layers", type=int, default=4)
    parser.add_argument("--mask-min", type=float, default=0.15)
    parser.add_argument("--mask-max", type=float, default=0.85)
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument(
        "--freeze-context",
        action="store_true",
        help="Freeze context tower (for pretrained encoders). Default: train both towers.",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.0,
        help="Stub-only: rate of intentional broken generations.",
    )
    args = parser.parse_args(argv)

    summary = train(
        ModelBuildConfig(
            train_dir=args.train_dir,
            run_root=args.run_root,
            run_id=args.run_id,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            model_name=args.model,
            d_model=args.d_model,
            n_heads=args.n_heads,
            context_layers=args.context_layers,
            denoiser_layers=args.denoiser_layers,
            mask_min=args.mask_min,
            mask_max=args.mask_max,
            gen_steps=args.gen_steps,
            freeze_context=args.freeze_context,
            noise_rate=args.noise_rate,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
