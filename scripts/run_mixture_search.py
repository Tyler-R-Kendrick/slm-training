#!/usr/bin/env python3
"""Probe mixture weights and propose finalists (P1b).

By default runs a *dry* search: emit local/global probes + regression-proposed
mixtures to ``--out``. Pass ``--train-dir`` + ``--score`` to score probes on
weighted denoising NLL with a small TwoTower (slow; optional).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/mixtures"))
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Optional base mixture JSON (default: design defaults).",
    )
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Train tiny probes and score weighted NLL (requires train/test dirs).",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-probes", type=int, default=12)
    args = parser.parse_args(argv)

    from slm_training.data.mixture import (
        default_base_weights,
        fit_weight_regression,
        global_probe_candidates,
        load_mixture_manifest,
        local_probe_candidates,
        propose_from_fit,
        write_mixture_manifest,
    )

    base = (
        load_mixture_manifest(args.base).weights
        if args.base
        else default_base_weights()
    )
    probes = local_probe_candidates(base) + global_probe_candidates(base)
    probes = probes[: max(1, int(args.limit_probes))]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for probe in probes:
        write_mixture_manifest(out / f"{probe.mixture_id}.json", probe)

    scored: list[dict] = []
    if args.score:
        if args.train_dir is None or args.test_dir is None:
            parser.error("--score requires --train-dir and --test-dir")
        from slm_training.harnesses.model_build import ModelBuildConfig, train

        for probe in probes:
            write_mixture_manifest(out / f"{probe.mixture_id}.json", probe)
            cfg = ModelBuildConfig(
                train_dir=args.train_dir,
                test_dir=args.test_dir,
                run_root=out / "runs",
                run_id=probe.mixture_id,
                steps=args.steps,
                seed=args.seed,
                device=args.device,
                context_backend="scratch",
                d_model=64,
                n_heads=4,
                context_layers=1,
                denoiser_layers=2,
                mixture_manifest=out / f"{probe.mixture_id}.json",
                loss_eval_every=max(1, args.steps // 2),
                target_token_budget=5_000,
            )
            summary = train(cfg)
            nll = summary.get("best_weighted_nll")
            scored.append(
                {
                    "mixture_id": probe.mixture_id,
                    "weights": probe.weights,
                    "weighted_nll": nll,
                    "run_id": probe.mixture_id,
                }
            )

        fit = fit_weight_regression(scored)
        proposals = propose_from_fit(fit, base=base, n=3)
        for prop in proposals:
            write_mixture_manifest(out / f"{prop.mixture_id}.json", prop)
        (out / "search_summary.json").write_text(
            json.dumps(
                {"probes": scored, "fit": fit, "proposals": [p.mixture_id for p in proposals]},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (out / "search_summary.json").write_text(
            json.dumps(
                {
                    "probes": [p.mixture_id for p in probes],
                    "scored": False,
                    "note": "Pass --score to train and regress.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"out": str(out), "n_probes": len(probes), "scored": bool(args.score)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
