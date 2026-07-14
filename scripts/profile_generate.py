#!/usr/bin/env python3
"""Profile TwoTower generate latency with per-phase DecodeStats breakdown.

Loads the committed playground demo checkpoint by default and prints a
JSON report attributing wall time to denoiser / DFA / stream_check / etc.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from slm_training.models.decode_stats import DecodeStats, aggregate_stats
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.models.twotower import TwoTowerModel


def _apply_profile_flags(model: TwoTowerModel, args: argparse.Namespace) -> None:
    cfg = model.config
    cfg.grammar_constrained = True
    cfg.grammar_ltr_primary = True
    if args.no_incremental:
        cfg.grammar_incremental_state = False
    if args.verify_chosen_only:
        cfg.grammar_verify_chosen_only = True
    if args.multitoken:
        cfg.grammar_multitoken_accept = True
        cfg.grammar_multitoken_max = int(args.multitoken_max)
    if args.lookahead > 0:
        cfg.grammar_canvas_lookahead = int(args.lookahead)
    if args.no_repair:
        cfg.grammar_ltr_repair = False
    if args.no_finalize:
        cfg.grammar_finalize_validate = False
    if args.maskgit:
        cfg.grammar_ltr_primary = False
    if args.quant:
        model.apply_dynamic_quant()
    if args.compile:
        from slm_training.runtime.accel import maybe_compile

        model.denoiser = maybe_compile(model.denoiser, enabled=True, mode="default")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PLAYGROUND_DEMO_CHECKPOINT,
        help="TwoTower checkpoint path (default: playground demo fixture).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to generate (repeatable). Defaults to a short fixture set.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-incremental", action="store_true")
    parser.add_argument("--verify-chosen-only", action="store_true")
    parser.add_argument("--multitoken", action="store_true")
    parser.add_argument("--multitoken-max", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--maskgit", action="store_true")
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/runs/profile_generate.json"),
    )
    args = parser.parse_args(argv)

    prompts = args.prompt or [
        "A hero card with title and subtitle",
        "Login form with email and password",
        "Settings page with a toggle list",
    ]

    model = TwoTowerModel.from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    _apply_profile_flags(model, args)

    for _ in range(max(0, args.warmup)):
        model.generate(prompts[0])

    rows: list[DecodeStats] = []
    texts: list[str] = []
    wall_t0 = time.perf_counter()
    for _ in range(max(1, args.rounds)):
        for prompt in prompts:
            text, stats = model.generate_with_stats(prompt)
            rows.append(stats)
            texts.append(text)
    wall_s = time.perf_counter() - wall_t0

    summary = aggregate_stats(rows)
    report = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "n_generates": len(rows),
        "wall_sec": round(wall_s, 4),
        "sec_per_generate": round(wall_s / max(1, len(rows)), 4),
        "flags": {
            "grammar_incremental_state": bool(model.config.grammar_incremental_state),
            "grammar_verify_chosen_only": bool(model.config.grammar_verify_chosen_only),
            "grammar_multitoken_accept": bool(model.config.grammar_multitoken_accept),
            "grammar_canvas_lookahead": int(model.config.grammar_canvas_lookahead),
            "grammar_ltr_primary": bool(model.config.grammar_ltr_primary),
            "grammar_ltr_repair": bool(model.config.grammar_ltr_repair),
            "grammar_finalize_validate": bool(model.config.grammar_finalize_validate),
            "use_dynamic_quant": bool(model.config.use_dynamic_quant),
        },
        "summary": summary,
        "per_call": [r.as_dict() for r in rows],
        "sample_outputs": texts[: len(prompts)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
