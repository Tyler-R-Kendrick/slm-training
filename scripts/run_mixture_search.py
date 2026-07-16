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
    parser.add_argument("--out", type=Path, default=Path("outputs/data/mixture"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional durable copy of search_summary.json.",
    )
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
    parser.add_argument(
        "--limit-probes",
        type=int,
        default=0,
        help="Optional bounded probe count; 0 keeps all family probes.",
    )
    args = parser.parse_args(argv)

    from slm_training.data.mixture import (
        DEFAULT_TASK_WEIGHTS,
        MixtureManifest,
        corpus_diagnostics,
        default_base_weights,
        fit_weight_regression,
        global_probe_candidates,
        load_mixture_manifest,
        local_probe_candidates,
        propose_from_fit,
        write_mixture_manifest,
    )

    base_manifest = (
        load_mixture_manifest(args.base)
        if args.base
        else MixtureManifest(
            mixture_id="task_balanced_v2",
            weights=default_base_weights(),
            task_weights=DEFAULT_TASK_WEIGHTS,
            notes="equal task-group targets; family weights are within-task priors",
        ).normalized()
    )
    base = base_manifest.weights
    local = local_probe_candidates(base, task_weights=base_manifest.task_weights)
    scales_per_family = 3
    local = [
        probe
        for offset in range(scales_per_family)
        for probe in local[offset::scales_per_family]
    ]
    probes = local + global_probe_candidates(
        base, task_weights=base_manifest.task_weights
    )
    if args.limit_probes > 0:
        probes = probes[: args.limit_probes]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diagnostics = None
    if args.train_dir is not None:
        from slm_training.harnesses.model_build.data import load_train_records

        diagnostics = corpus_diagnostics(
            load_train_records(args.train_dir), configured_weights=base
        )
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
                    "task_weights": probe.task_weights,
                    "nll_learning_curve": summary.get("nll_history") or [],
                }
            )

        fit = fit_weight_regression(scored)
        proposals = propose_from_fit(
            fit,
            base=base,
            n=3,
            task_weights=base_manifest.task_weights,
        )
        for prop in proposals:
            write_mixture_manifest(out / f"{prop.mixture_id}.json", prop)
        summary_payload = {
            "base": base_manifest.mixture_id,
            "task_weights": base_manifest.task_weights,
            "corpus_diagnostics": diagnostics,
            "probes": scored,
            "fit": fit,
            "fit_stable": len(scored) > len(fit.get("families") or []) + 1,
            "proposals": [p.mixture_id for p in proposals],
            "scored": True,
        }
    else:
        summary_payload = {
            "base": base_manifest.mixture_id,
            "probes": [p.mixture_id for p in probes],
            "task_weights": base_manifest.task_weights,
            "corpus_diagnostics": diagnostics,
            "scored": False,
            "note": "Pass --score to train and regress.",
        }

    summary_payload["run"] = {
        "kind": "scored_mixture_search" if args.score else "dry_fixture_wiring",
        "train_dir": str(args.train_dir) if args.train_dir else None,
        "test_dir": str(args.test_dir) if args.test_dir else None,
        "device": args.device,
        "steps": args.steps if args.score else 0,
        "context_backend": "scratch" if args.score else None,
        "honesty": "weighted_nll" if args.score else "no_quality_claim",
    }
    summary_text = json.dumps(summary_payload, indent=2) + "\n"
    (out / "search_summary.json").write_text(summary_text, encoding="utf-8")
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary_text, encoding="utf-8")

    print(
        json.dumps(
            {"out": str(out), "n_probes": len(probes), "scored": bool(args.score)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
