#!/usr/bin/env python3
"""Three-seed honest baseline reproduction (X0 corrected baseline).

Runs gx_x0_baseline with seeds 0/1/2 on scratch CPU and writes a summary JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build import ModelBuildConfig, train
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)

SUITES = ["smoke", "held_out", "adversarial", "ood", "rico_held"]
DEFAULT_SEEDS = [0, 1, 2]
RUN_ID = "gx_x0_baseline"


def _baseline_cfg(args: argparse.Namespace, seed: int) -> ModelBuildConfig:
    return ModelBuildConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        suite="smoke",
        run_root=args.run_root,
        run_id=f"{RUN_ID}_s{seed}",
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=seed,
        device=args.device,
        model_name="twotower",
        d_model=128,
        n_heads=4,
        context_layers=2,
        denoiser_layers=4,
        context_backend=args.context_backend,
        local_files_only=args.local_files_only,
        grammar_constrained=True,
        grammar_ltr_primary=True,
        grammar_ltr_repair=True,
        grammar_ltr_max_tokens=96,
        design_md_in_context=True,
        ltr_loss_weight=1.0,
        structural_bias=2.5,
        gen_steps=args.gen_steps,
        rico_eval_limit=args.rico_limit,
        telemetry=True,
    )


def _summarize(board: dict[str, Any]) -> dict[str, Any]:
    suites = board.get("suites") or {}
    gates = evaluate_ship_gates(suites)
    return {
        "pass": gates.get("pass"),
        "failures": gates.get("failures"),
        "suites": {
            name: {
                "parse_rate": m.get("parse_rate"),
                "placeholder_fidelity": m.get("placeholder_fidelity"),
                "structural_similarity": m.get("structural_similarity"),
                "reward_score": m.get("reward_score"),
                "n": m.get("n"),
            }
            for name, m in suites.items()
        },
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    cfg = _baseline_cfg(args, seed)
    summary = train(cfg)
    ckpt = Path(summary["checkpoint"])
    board = evaluate_suites(cfg, SUITES, checkpoint=ckpt, write_gates=True)
    result = {
        "run_id": RUN_ID,
        "seed": seed,
        "checkpoint": str(ckpt),
        "steps": summary.get("steps"),
        "last_loss": summary.get("last_loss"),
        **_summarize(board),
    }
    out_dir = args.run_root / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"baseline_s{seed}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("outputs/train_data/v1"))
    parser.add_argument("--test-dir", type=Path, default=Path("outputs/test_data/v1"))
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-backend", choices=("scratch", "hf"), default="scratch")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument("--rico-limit", type=int, default=32)
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds (default 0,1,2).",
    )
    args = parser.parse_args(argv)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = list(DEFAULT_SEEDS)

    results = []
    for seed in seeds:
        print(json.dumps({"status": "start", "seed": seed}))
        results.append(run_seed(args, seed))
        print(json.dumps({"status": "done", "seed": seed, "pass": results[-1]["pass"]}))

    out = {
        "baseline": RUN_ID,
        "description": "Corrected twotower baseline with honest DESIGN.md eval (3 seeds)",
        "reference": "docs/design/quality-experiment-matrix.md",
        "gate_policy": {k: v for k, v in DEFAULT_SHIP_GATES.items()},
        "seeds": seeds,
        "steps": args.steps,
        "context_backend": args.context_backend,
        "results": results,
    }
    out_path = args.run_root / "baseline_reproduction_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    docs_out = Path("docs/design/baseline-reproduction-results.json")
    docs_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_path), "n": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
