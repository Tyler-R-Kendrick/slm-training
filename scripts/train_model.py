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
        default=Path("outputs/train_data/v1"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional test dir for periodic eval (--eval-every).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run smoke eval every N steps (0 disables).",
    )
    parser.add_argument("--eval-suite", default="smoke")
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
        "--context-backend",
        choices=("scratch", "hf"),
        default="hf",
        help="Context tower backend (default: hf; use scratch for offline CI).",
    )
    parser.add_argument(
        "--hf-model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HF model id when --context-backend hf.",
    )
    parser.add_argument(
        "--freeze-context",
        action="store_true",
        help="Freeze context tower weights.",
    )
    parser.add_argument(
        "--no-freeze-context",
        action="store_true",
        help="Allow context tower gradients (overrides HF default freeze).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load HF weights only from local cache.",
    )
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="Disable streaming/grammar-constrained decode at generate time.",
    )
    parser.add_argument("--grammar-top-k", type=int, default=16)
    parser.add_argument("--structural-bias", type=float, default=1.25)
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.0,
        help="Stub-only: rate of intentional broken generations.",
    )
    args = parser.parse_args(argv)

    freeze = args.freeze_context
    if args.context_backend == "hf" and not args.no_freeze_context:
        freeze = True
    if args.no_freeze_context:
        freeze = False

    summary = train(
        ModelBuildConfig(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            suite=args.eval_suite,
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
            context_backend=args.context_backend,
            hf_model_name=args.hf_model,
            freeze_context=freeze,
            local_files_only=args.local_files_only,
            grammar_constrained=not args.no_grammar,
            grammar_top_k=args.grammar_top_k,
            structural_bias=args.structural_bias,
            noise_rate=args.noise_rate,
            eval_every=args.eval_every,
            eval_suite=args.eval_suite,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
