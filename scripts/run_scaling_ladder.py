#!/usr/bin/env python3
"""Run scaling-ladder points and compute EG / promotion checks (P1c)."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path


def _wall_minutes(value: str) -> float:
    minutes = float(value)
    if not 0 < minutes <= 3:
        raise argparse.ArgumentTypeError("must be positive and at most 3")
    return minutes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("scaling", "discrete-bottleneck"),
        default="scaling",
        help="Experiment family to run (default: scaling).",
    )
    parser.add_argument("--train-dir", type=Path, help="Required for scaling family.")
    parser.add_argument("--test-dir", type=Path, help="Required for scaling family.")
    parser.add_argument("--out", type=Path, default=Path("outputs/ladders"))
    parser.add_argument("--track", choices=("scratch", "hf"), default="scratch")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--widths",
        default="64",
        help="Comma-separated d_model widths (default: 64 for cheap CI).",
    )
    parser.add_argument("--horizons", default="1.0", help="Comma-separated horizon multipliers.")
    parser.add_argument(
        "--representation",
        choices=("compositional", "lexer", "choice"),
        default="compositional",
        help="Output representation for all ladder points (B3 axis).",
    )
    parser.add_argument("--base-token-budget", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-wall-minutes",
        type=_wall_minutes,
        default=3.0,
        help="Cumulative ladder-arm wall budget (default and maximum: 3).",
    )
    parser.add_argument(
        "--capacity-arm",
        choices=("lexer", "choice"),
        default=None,
        help=(
            "B3 capacity ladder (SLM-23): run one output-representation arm "
            "(scratch track). Overrides --track; pairs with the other arm at "
            "matched widths for the quality-vs-d_model comparison."
        ),
    )
    parser.add_argument(
        "--dry-fit",
        action="store_true",
        help="Skip training; only exercise fit/EG helpers on synthetic points.",
    )
    # CAP2-01 discrete-bottleneck family options.
    parser.add_argument("--state-report", type=Path, help="CAP1-01/CAP0-03 state report")
    parser.add_argument(
        "--state-count",
        type=int,
        default=41,
        help="Fixture state count for discrete-bottleneck when no report is given.",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated CAP2 arm ids (default: full fixture matrix).",
    )
    args = parser.parse_args(argv)

    if args.family == "discrete-bottleneck":
        from scripts.run_cap2_bottleneck import main as cap2_main

        cap2_argv = [
            "--out-dir",
            str(args.out),
            "--seeds",
            args.seeds,
            "--state-count",
            str(args.state_count),
        ]
        if args.state_report:
            cap2_argv.extend(["--state-report", str(args.state_report)])
        if args.arms.strip():
            cap2_argv.extend(["--arms", args.arms])
        return cap2_main(cap2_argv)

    if args.train_dir is None or args.test_dir is None:
        parser.error("--train-dir and --test-dir are required for the scaling family")

    from slm_training.harnesses.experiments.efficiency_gain import efficiency_gain, efficiency_gain_lcb
    from slm_training.harnesses.experiments.ladder import (
        capacity_ladder,
        hf_ladder_default,
        model_build_config_for_point,
        scratch_ladder_default,
    )
    from slm_training.harnesses.experiments.promotion import (
        check_data_integrity,
        evaluate_promotion,
        register_promoted_checkpoint,
    )
    from slm_training.harnesses.experiments.scaling_fit import (
        ScalingObservation,
        fit_power_law,
        observation_from_summary,
    )

    widths = tuple(int(x) for x in args.widths.split(",") if x.strip())
    horizons = tuple(float(x) for x in args.horizons.split(",") if x.strip())
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.capacity_arm is not None:
        ladder = capacity_ladder(
            args.capacity_arm,
            base_token_budget=args.base_token_budget,
            widths=widths,
            horizons=horizons,
        )
    elif args.track == "scratch":
        ladder = scratch_ladder_default(
            base_token_budget=args.base_token_budget,
            widths=widths,
            horizons=horizons,
            representation=args.representation,
        )
    else:
        ladder = hf_ladder_default(
            base_token_budget=args.base_token_budget,
            widths=widths,
            horizons=horizons,
            representation=args.representation,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wall_started = time.monotonic()

    def stop_on_budget(_signum, _frame) -> None:
        stopped = {
            "status": "stopped",
            "reason": "wall_time_budget",
            "max_wall_minutes": args.max_wall_minutes,
            "elapsed_wall_seconds": time.monotonic() - wall_started,
            "capacity_arm": args.capacity_arm,
        }
        (out / "ladder_stopped_wall_budget.json").write_text(
            json.dumps(stopped, indent=2) + "\n", encoding="utf-8"
        )
        raise SystemExit(2)

    previous_alarm = signal.signal(signal.SIGALRM, stop_on_budget)
    signal.setitimer(signal.ITIMER_REAL, args.max_wall_minutes * 60)
    integrity = check_data_integrity(args.train_dir, args.test_dir)

    observations: list[ScalingObservation] = []
    summaries: list[dict] = []

    if args.dry_fit:
        # Synthetic decreasing loss vs cost for unit/CI smoke.
        for i, point in enumerate(ladder.points[:4]):
            observations.append(
                ScalingObservation(
                    track=args.track,
                    candidate_id="dry",
                    point_id=point.point_id,
                    seed=0,
                    loss=2.0 / (i + 1) + 0.5,
                    cost_time_s=float(10 * (i + 1)),
                )
            )
    else:
        from slm_training.harnesses.model_build import train

        for point in ladder.points:
            for seed in seeds:
                cfg = model_build_config_for_point(
                    point,
                    ladder,
                    train_dir=args.train_dir,
                    test_dir=args.test_dir,
                    run_root=out / "runs",
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                )
                cfg.device = args.device
                cfg.max_wall_minutes = args.max_wall_minutes
                cfg.eval_every = args.steps
                cfg.eval_suites = "smoke,held_out,adversarial,ood,rico_held"
                summary = train(cfg)
                summaries.append(summary)
                observations.append(
                    observation_from_summary(
                        summary,
                        candidate_id="ladder",
                        point_id=point.point_id,
                        seed=seed,
                    )
                )

    # B3 (SLM-23): corpus choice bits for this representation + params-per-bit
    # per trained point (E1, evals/semantic_bits.py). Corpus-side only when no
    # training ran; model-side rows need trainable_params from summaries.
    from slm_training.dsl.schema import load_jsonl
    from slm_training.evals.semantic_bits import semantic_bits

    representation = (
        ladder.output_tokenizer
        if ladder.output_tokenizer != "compositional"
        else args.representation
    )
    bits_stream = {
        "compositional": "surface",
        "lexer": "surface",
        "choice": "choice",
    }[representation]
    train_records_path = Path(args.train_dir) / "records.jsonl"
    bits_report = (
        semantic_bits(load_jsonl(train_records_path), stream=bits_stream)
        if train_records_path.is_file()
        else None
    )
    params_per_bit_rows: list[dict] = []
    if bits_report and bits_report.get("total_bits"):
        for summary in summaries:
            trainable = (summary.get("track") or {}).get("trainable_params")
            if not trainable:
                continue
            params_per_bit_rows.append(
                {
                    "run_id": summary.get("run_id"),
                    "trainable_params": int(trainable),
                    "params_per_bit": float(trainable) / bits_report["total_bits"],
                }
            )

    fit = fit_power_law(observations, cost_key="time") if len(observations) >= 2 else None
    eg_vals: list[float] = []
    if fit is not None:
        for obs in observations:
            eg = efficiency_gain(fit, obs, cost_key="time")
            if eg is not None:
                eg_vals.append(eg)
    eg_stats = efficiency_gain_lcb(eg_vals) if eg_vals else None

    # Rank by loss ascending per horizon bucket (point_id encodes horizon).
    rankings: dict[str, list[str]] = {}
    by_point: dict[str, list[ScalingObservation]] = {}
    for obs in observations:
        by_point.setdefault(obs.point_id, []).append(obs)
    for point_id, rows in by_point.items():
        ordered = sorted(rows, key=lambda o: o.loss)
        rankings[point_id] = [o.candidate_id for o in ordered]

    promotion = evaluate_promotion(
        integrity=integrity,
        rankings=rankings if len(rankings) >= 2 else None,
        eg_time_by_seed=eg_vals or None,
    )

    if summaries and promotion.get("promotable"):
        best = min(
            summaries,
            key=lambda s: float(s.get("best_weighted_nll") or 1e9),
        )
        ckpt = Path(best["checkpoint"]).parent
        register_promoted_checkpoint(
            ckpt,
            source=best["checkpoint"],
            meta={"ladder_id": ladder.ladder_id, "track": ladder.track},
        )

    payload = {
        "ladder_id": ladder.ladder_id,
        "track": ladder.track,
        "representation": representation,
        "max_wall_minutes": args.max_wall_minutes,
        "elapsed_wall_seconds": time.monotonic() - wall_started,
        "n_points": len(ladder.points),
        "semantic_bits": bits_report,
        "params_per_bit_rows": params_per_bit_rows,
        "fit": fit,
        "eg_time": (
            {"mean": eg_stats[0], "lcb": eg_stats[1], "ucb": eg_stats[2]}
            if eg_stats
            else None
        ),
        "promotion": promotion,
        "observations": [
            {
                "point_id": o.point_id,
                "seed": o.seed,
                "loss": o.loss,
                "cost_time_s": o.cost_time_s,
            }
            for o in observations
        ],
    }
    payload["output_tokenizer"] = ladder.output_tokenizer
    summary_name = (
        f"ladder_summary_{ladder.ladder_id}.json"
        if args.capacity_arm is not None
        else "ladder_summary.json"
    )
    (out / summary_name).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_alarm)
    print(json.dumps({"out": str(out), "fit": fit, "promotion": promotion}, indent=2))
    return 0 if promotion.get("promotable") or args.dry_fit else 1


if __name__ == "__main__":
    raise SystemExit(main())
