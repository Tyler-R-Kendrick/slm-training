#!/usr/bin/env python3
"""Deterministic denoising-NLL loss suites for a TwoTower checkpoint.

Candidate-invariant inner-loop signal: fixed held-out records, fixed
hash-derived masks, fixed rates, raw vs grammar-legal-support decomposition.
Use this to compare data / architecture / training candidates cheaply before
running the generated scoreboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report JSON path (default: <checkpoint dir>/loss_suites.json)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-suite", default="held_out")
    parser.add_argument("--ood-suite", default="ood")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument(
        "--rates",
        default="0.15,0.30,0.50,0.70,0.85",
        help="Comma-separated fixed mask rates.",
    )
    parser.add_argument(
        "--no-legal-support",
        action="store_true",
        help="Skip grammar legal-support NLL (raw only).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap suite sizes.")
    parser.add_argument(
        "--grammar-dsl", default="openui", help="Grammar backend for legal support."
    )
    args = parser.parse_args(argv)

    from slm_training.evals.denoising_nll import DenoisingNLLConfig
    from slm_training.evals.loss_suites import (
        LOSS_SUITE_VERSION,
        evaluate_loss_suites,
        write_loss_suite_report,
    )
    from slm_training.models.grammar import set_active_dsl
    from slm_training.models.twotower import TwoTowerModel

    set_active_dsl(args.grammar_dsl)
    model = TwoTowerModel.from_checkpoint(args.checkpoint, device=args.device)

    rates = tuple(float(r) for r in str(args.rates).split(",") if r.strip())
    nll_cfg = DenoisingNLLConfig(
        suite_version=LOSS_SUITE_VERSION,
        mask_rates=rates,
        mask_seed=int(args.mask_seed),
        compute_legal_support=not bool(args.no_legal_support),
    )
    report = evaluate_loss_suites(
        model,
        args.test_dir,
        nll_config=nll_cfg,
        base_suite=args.base_suite,
        ood_suite=args.ood_suite,
        limit=args.limit,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["test_dir"] = str(args.test_dir)

    out = args.out or (Path(args.checkpoint).parent / "loss_suites.json")
    write_loss_suite_report(out, report)
    print(
        json.dumps(
            {
                "out": str(out),
                "aggregate": report["aggregate"],
                "bits_per_char": (report["categories"].get("broad") or {}).get(
                    "bits_per_char"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
