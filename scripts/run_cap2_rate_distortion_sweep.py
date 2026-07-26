#!/usr/bin/env python3
"""Run the CAP2-06 program-factor latent size/width/codec rate-distortion sweep.

Sweeps latent count x encoder width x codec family (uniform scalar,
mixed-radix FSQ, binary LFQ, learned VQ, continuous) over tensorized
SemanticPlanV1 program factors and reports per-factor-family reconstruction
distortion, serialized rate, and which configuration is the smallest to
clear a distortion threshold on every factor family.

Example:

    python -m scripts.run_cap2_rate_distortion_sweep \
      --fixture-count 12 --train-steps 400 \
      --out-dir outputs/runs/cap2_rate_distortion_sweep

This is fixture/wiring evidence only: real meaning-v2, binder/reference F1,
canonical AST, and G0-G8 metrics require a token/tree decoder and the full
compiler/evaluator pipeline, which this issue's own scope gates to a later
stage ("add one token/tree decoder only after factor gates pass"). No such
claim is made here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import (
    RATE_DISTORTION_CODECS,
    RATE_DISTORTION_LATENT_COUNTS,
    RATE_DISTORTION_WIDTHS,
    RateDistortionSweepReport,
    build_sweep,
    run_sweep,
)


def _markdown_report(report: RateDistortionSweepReport) -> str:
    lines = [
        "# CAP2-06 program-factor rate-distortion sweep",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Records:* {report.num_records}  ",
        f"*Distortion threshold (MSE proxy):* {report.threshold}",
        "",
        "## Honest caveat",
        "",
        "This is a fixture/wiring CPU run. It measures per-factor-family",
        "reconstruction distortion as a function of latent count, encoder",
        "width, and codec family -- the stage this issue's own scope covers",
        "before a token/tree decoder exists. Real meaning-v2, binder/reference",
        "F1, canonical AST, and G0-G8 metrics against generated OpenUI",
        "programs are not measured here; they require the full",
        "compiler/evaluator pipeline this issue gates to a later stage.",
        "",
        "## Retained configurations (smallest passing cell per codec, 3 seeds)",
        "",
    ]
    if not report.retained:
        lines.append("No codec family cleared the distortion threshold in this screened grid.")
    else:
        lines.append("| codec | num_latents | width | seed0 max_factor_mse | seed1 | seed2 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for retained in report.retained:
            seeds = [r.max_factor_mse for r in retained.confirmation_seeds]
            seed_cols = " | ".join(f"{v:.4f}" for v in seeds) if seeds else ""
            lines.append(
                f"| {retained.codec} | {retained.num_latents} | {retained.width} | "
                f"{retained.seed0_result.max_factor_mse:.4f} | {seed_cols} |"
            )
    lines.append("")
    lines.append("## Full staged frontier (all cells, including failed)")
    lines.append("")
    lines.append(
        "| cell | codec | n | width | capacity | mse_overall | max_factor_mse | "
        "factors_over_threshold | no_bypass |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | :---: |")
    for cell in report.cells:
        factors = ",".join(cell.factors_over_threshold) or "-"
        lines.append(
            f"| {cell.cell_id} | {cell.codec} | {cell.num_latents} | {cell.width} | "
            f"{cell.capacity:.0f} | {cell.mse_overall:.4f} | {cell.max_factor_mse:.4f} | "
            f"{factors} | {cell.no_bypass_ok} |"
        )
    lines.append("")
    lines.append("## Hard gates")
    lines.append("")
    no_bypass_failures = [c for c in report.cells if not c.no_bypass_ok]
    lines.append(f"- No-bypass audit failures: {len(no_bypass_failures)}")
    if no_bypass_failures:
        lines.append("- **FAIL** " + ", ".join(c.cell_id for c in no_bypass_failures))
    else:
        lines.append("- **PASS** every cell's decoder input is codec-only.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-count", type=int, default=12)
    parser.add_argument("--fixture-seed", type=int, default=0)
    parser.add_argument(
        "--codecs", default="", help="Comma-separated codec families. Empty means the full default set."
    )
    parser.add_argument(
        "--num-latents", default="", help="Comma-separated latent counts. Empty means the default {4,8,16,32}."
    )
    parser.add_argument(
        "--widths", default="", help="Comma-separated widths. Empty means the default {64,128,256}."
    )
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/cap2_rate_distortion_sweep"))
    parser.add_argument("--dry-run", action="store_true", help="Describe the grid without running it.")
    args = parser.parse_args(argv)

    codecs = tuple(x.strip() for x in args.codecs.split(",") if x.strip()) or RATE_DISTORTION_CODECS
    num_latents = tuple(int(x) for x in args.num_latents.split(",") if x.strip()) or RATE_DISTORTION_LATENT_COUNTS
    widths = tuple(int(x) for x in args.widths.split(",") if x.strip()) or RATE_DISTORTION_WIDTHS

    if args.dry_run:
        cells = build_sweep(
            codecs=codecs, num_latents_values=num_latents, widths=widths, train_steps=args.train_steps
        )
        print(f"CAP2-06 dry-run: {len(cells)} screening cell(s)")
        for cell in cells:
            print(f"  {cell.cell_id}: capacity={cell.capacity:.0f}")
        return 0

    report = run_sweep(
        fixture_count=args.fixture_count,
        fixture_seed=args.fixture_seed,
        codecs=codecs,
        num_latents_values=num_latents,
        widths=widths,
        train_steps=args.train_steps,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap2_rate_distortion_sweep_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap2_rate_distortion_sweep_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    no_bypass_failures = [c for c in report.cells if not c.no_bypass_ok]
    print(f"Wrote {json_path} and {md_path}")
    print(f"No-bypass audit failures: {len(no_bypass_failures)}")
    print(f"Retained configurations: {len(report.retained)}/{len(RATE_DISTORTION_CODECS)}")
    return 1 if no_bypass_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
