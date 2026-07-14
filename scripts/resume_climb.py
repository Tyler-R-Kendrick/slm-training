#!/usr/bin/env python3
"""Climb tranche: rollouts → trajectory RL → ship gates → promote (P3)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/runs/climb"))
    parser.add_argument("--suite", default="held_out")
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--rl-steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Harvest traces only (ablation arm: no update).",
    )
    parser.add_argument(
        "--record-support",
        action="store_true",
        help="Persist grammar allowed_id_set on commits (E64 support match).",
    )
    args = parser.parse_args(argv)

    from scripts.collect_trajectories import main as collect_main
    from slm_training.harnesses.distill.trace_store import TraceStore, checkpoint_sha
    from slm_training.harnesses.experiments.promotion import register_promoted_checkpoint
    from slm_training.harnesses.model_build import ModelBuildConfig, evaluate_suites
    from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.harnesses.rl.trajectory import TrajectoryRLConfig, train_trajectory_rl

    out = Path(args.out)
    traces_dir = out / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    collect_argv = [
        "--checkpoint",
        str(args.checkpoint),
        "--test-dir",
        str(args.test_dir),
        "--suite",
        args.suite,
        "--out",
        str(traces_dir),
        "--samples-per-prompt",
        str(args.samples_per_prompt),
        "--limit",
        str(args.limit),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.record_support:
        collect_argv.append("--record-support")
    collect_main(collect_argv)

    store = TraceStore(traces_dir)
    traces = list(store.iter_traces())
    policy_sha = checkpoint_sha(args.checkpoint)

    rl_summary = None
    model_out = out / "model.pt"
    if not args.skip_rl and traces:
        model = TwoTowerModel.from_checkpoint(args.checkpoint, device=args.device)
        try:
            rl_summary = train_trajectory_rl(
                model,
                traces,
                config=TrajectoryRLConfig(
                    steps=args.rl_steps,
                    group_size=min(4, max(2, args.samples_per_prompt)),
                    seed=args.seed,
                ),
                out_dir=out / "rl",
                base_policy_sha=policy_sha,
            )
            shutil.copy2(out / "rl" / "model.pt", model_out)
        except ValueError as exc:
            rl_summary = {"skipped": True, "reason": str(exc)}
            shutil.copy2(args.checkpoint, model_out)
    else:
        shutil.copy2(args.checkpoint, model_out)
        rl_summary = {"skipped": True, "reason": "skip_rl or empty traces"}

    eval_cfg = ModelBuildConfig(
        train_dir=args.test_dir,  # unused for eval
        test_dir=args.test_dir,
        suite=args.suite,
        run_root=out,
        run_id="climb_eval",
        device=args.device,
        context_backend="scratch",
    )
    board = evaluate_suites(
        eval_cfg,
        [args.suite, "smoke"] if args.suite != "smoke" else ["smoke"],
        checkpoint=model_out,
        write_gates=True,
    )
    gates = evaluate_ship_gates(board)
    if gates.get("pass"):
        register_promoted_checkpoint(
            out / "checkpoints",
            source=model_out,
            meta={"policy_sha": policy_sha, "gates": gates},
        )

    summary = {
        "traces": len(traces),
        "rl": rl_summary,
        "gates": gates,
        "checkpoint": str(model_out),
    }
    (out / "climb_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if gates.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
