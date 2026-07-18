#!/usr/bin/env python3
"""Run the CAP2-01 strict K-ary bottleneck phase-boundary fixture matrix.

Example:

    python -m scripts.run_cap2_bottleneck \
      --state-report outputs/runs/arity/bounded_expr_report.json \
      --arms b2d5,b2d6,t3d3,t3d4,k4d3,k8d2 \
      --seeds 0,1,2 \
      --out-dir outputs/runs/cap2_bottleneck

If no state report is supplied the harness falls back to the verified M=41
synthetic fixture used by CAP0-03 robust-code tests.  Fixture CPU runs are
wiring/mathematical evidence only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.cap2_bottleneck import (
    BottleneckMatrixReport,
    load_state_report,
    run_matrix,
    state_count_from_report,
)


def _markdown_report(report: BottleneckMatrixReport) -> str:
    lines = [
        "# CAP2-01 K-ary bottleneck phase-boundary fixture matrix",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*State count:* {report.state_count}  ",
        f"*State report:* {report.state_report_path or 'M=41 synthetic fixture'}",
        "",
        "## Honest caveat",
        "",
        "This is a fixture CPU run.  It verifies the mathematical capacity bound",
        "``K**d >= M`` and the no-bypass wiring invariant; it does not train a",
        "production model or make a ship-quality claim.",
        "",
        "## Results",
        "",
        "| arm | K | d | capacity | states | mode | exact_rate | occupied | collisions | leakage | notes |",
        "| --- | - | - | -------- | ------ | ---- | ---------- | -------- | ---------- | ------- | ----- |",
    ]
    for arm in report.arms:
        notes = " ".join(arm.notes)
        lines.append(
            f"| {arm.arm_id} | {arm.K} | {arm.d} | {arm.capacity} | "
            f"{arm.state_count} | {arm.mode} | {arm.exact_reconstruction_rate:.4f} | "
            f"{arm.occupied_codewords} | {arm.collision_count} | {arm.leakage} | {notes} |"
        )
    lines.append("")
    lines.append("## Hard gates")
    lines.append("")
    below = [a for a in report.arms if a.capacity < a.state_count]
    leaked = [a for a in below if a.exact_reconstruction_rate >= 1.0]
    lines.append(f"- Below-capacity arms: {len(below)}")
    lines.append(f"- Leakage violations (below-capacity exact reconstruction): {len(leaked)}")
    if leaked:
        lines.append("- **FAIL** leakage detected in: " + ", ".join(a.arm_id for a in leaked))
    else:
        lines.append("- **PASS** no below-capacity arm achieved perfect reconstruction.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-report", type=Path, help="CAP1-01/CAP0-03 state report")
    parser.add_argument(
        "--state-count",
        type=int,
        default=41,
        help="Fixture state count when no state report is given (default: 41).",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated arm ids.  Empty means run the default matrix.",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/cap2_bottleneck"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the matrix without running it.",
    )
    args = parser.parse_args(argv)

    state_report_path = args.state_report
    if state_report_path is not None:
        report_data = load_state_report(state_report_path)
        state_count = state_count_from_report(report_data)
    else:
        state_count = args.state_count

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    arms_filter: tuple[str, ...] | None = None
    if args.arms.strip():
        arms_filter = tuple(x.strip() for x in args.arms.split(",") if x.strip())

    if args.dry_run:
        from slm_training.harnesses.experiments.cap2_bottleneck import build_matrix

        arms = build_matrix(state_count, seeds=seeds, arms_filter=arms_filter)
        print(f"CAP2-01 dry-run: {len(arms)} arm(s) for M={state_count}")
        for arm in arms:
            print(
                f"  {arm.arm_id}: K={arm.K} d={arm.d} capacity={arm.capacity} "
                f"mode={arm.mode} seed={arm.seed}"
            )
        return 0

    matrix_report = run_matrix(
        state_count,
        seeds=seeds,
        arms_filter=arms_filter,
        state_report_path=state_report_path,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap2_bottleneck_{matrix_report.run_id}.json"
    json_path.write_text(matrix_report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap2_bottleneck_{matrix_report.run_id}.md"
    md_path.write_text(_markdown_report(matrix_report), encoding="utf-8")

    leaked = [a for a in matrix_report.arms if a.leakage]
    print(f"Wrote {json_path} and {md_path}")
    print(f"Below-capacity arms: {sum(1 for a in matrix_report.arms if a.capacity < a.state_count)}")
    print(f"Leakage violations: {len(leaked)}")
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
