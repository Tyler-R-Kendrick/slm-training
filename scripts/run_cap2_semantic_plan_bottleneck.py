#!/usr/bin/env python3
"""Run the CAP2-05 program-scale SemanticPlanV1 latent-codec bottleneck matrix.

Bridges the CAP2-02 latent-codec families onto tensorized SemanticPlanV1
program factors (inventory, cardinality, topology, binder/reference graph,
property roles/values, style/layout) instead of oracle integer state ids.

Example:

    python -m scripts.run_cap2_semantic_plan_bottleneck \
      --mode fixture --fixture-count 32 --seeds 0 \
      --out-dir outputs/runs/cap2_semantic_plan_bottleneck

Small-corpus and full-corpus modes require an immutable SemanticPlanV1 JSONL
manifest via --corpus-path; this script does not fabricate one. Fixture CPU
runs are wiring/mathematical evidence only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.cap2_semantic_plan_bottleneck import (
    ProgramFactorMatrixReport,
    build_matrix,
    run_matrix,
)


def _markdown_report(report: ProgramFactorMatrixReport) -> str:
    lines = [
        "# CAP2-05 program-scale SemanticPlanV1 latent-codec bottleneck matrix",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Mode:* {report.mode}  ",
        f"*Records:* {report.num_records}  ",
        f"*Corpus:* {report.corpus_path or 'SLM-144 deterministic fixture corpus'}",
        "",
        "## Honest caveat",
        "",
        "This is a fixture/wiring CPU run. It verifies that each CAP2-02 latent",
        "codec family can be bridged onto tensorized SemanticPlanV1 program",
        "factors, that no unencoded gold factor reaches the decoder (no-bypass",
        "audit), and reports actual serialized bits/bytes per example. It does",
        "not train a production model, does not claim which codec family is",
        "best for real OpenUI semantics, and factor-wise distortion is a",
        "code-distinguishability proxy, not a reconstruction-error metric.",
        "",
        "## Results",
        "",
        "| arm | codec | levels | capacity | records | exact_rate | occupied | "
        "no_bypass | leakage | serialized_bytes | nominal_bytes |",
        "| --- | --- | --- | -------- | ------- | ---------- | -------- | "
        "--------- | ------- | ---------------- | -------------- |",
    ]
    for arm in report.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.codec} | {list(arm.codec_levels)} | {arm.capacity} | "
            f"{arm.num_records} | {arm.exact_reconstruction_rate:.4f} | "
            f"{arm.occupied_codewords} | {arm.no_bypass_ok} | {arm.leakage} | "
            f"{arm.serialized_bytes_per_example:.2f} | {arm.nominal_bytes_per_example:.2f} |"
        )
    lines.append("")
    lines.append("## Factor-wise distortion (code-distinguishability proxy)")
    lines.append("")
    lines.append("| arm | " + " | ".join(sorted(report.arms[0].factor_distortion)) + " |" if report.arms else "")
    if report.arms:
        factor_names = sorted(report.arms[0].factor_distortion)
        lines.append("| --- | " + " | ".join("---" for _ in factor_names) + " |")
        for arm in report.arms:
            values = " | ".join(f"{arm.factor_distortion.get(name, float('nan')):.4f}" for name in factor_names)
            lines.append(f"| {arm.arm_id} | {values} |")
    lines.append("")
    lines.append("## Hard gates")
    lines.append("")
    no_bypass_failures = [a for a in report.arms if not a.no_bypass_ok]
    leaked = [a for a in report.arms if a.leakage]
    lines.append(f"- No-bypass audit failures: {len(no_bypass_failures)}")
    lines.append(f"- Leakage violations: {len(leaked)}")
    if no_bypass_failures or leaked:
        lines.append(
            "- **FAIL** "
            + ", ".join(a.arm_id for a in (no_bypass_failures + leaked))
        )
    else:
        lines.append("- **PASS** every arm's decoder input is codec-only and no below-capacity arm leaked.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fixture", "small_corpus", "full_corpus"),
        default="fixture",
        help="Corpus source mode (default: fixture).",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        help="Immutable SemanticPlanV1 JSONL manifest (required for small_corpus/full_corpus).",
    )
    parser.add_argument(
        "--fixture-count",
        type=int,
        default=32,
        help="SLM-144 fixture record count before the 80/20 train/val split (default: 32).",
    )
    parser.add_argument("--fixture-seed", type=int, default=0)
    parser.add_argument(
        "--arms",
        default="",
        help="Comma-separated arm ids. Empty means run the default matrix.",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/runs/cap2_semantic_plan_bottleneck")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the matrix without running it.",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    arms_filter: tuple[str, ...] | None = None
    if args.arms.strip():
        arms_filter = tuple(x.strip() for x in args.arms.split(",") if x.strip())

    if args.dry_run:
        arms = build_matrix(seeds=seeds, arms_filter=arms_filter)
        print(f"CAP2-05 dry-run: {len(arms)} arm(s), mode={args.mode}")
        for arm in arms:
            print(f"  {arm.arm_id}: codec={arm.codec} capacity={arm.capacity} seed={arm.seed}")
        return 0

    matrix_report = run_matrix(
        mode=args.mode,
        corpus_path=args.corpus_path,
        fixture_count=args.fixture_count,
        fixture_seed=args.fixture_seed,
        seeds=seeds,
        arms_filter=arms_filter,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap2_semantic_plan_bottleneck_{matrix_report.run_id}.json"
    json_path.write_text(matrix_report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap2_semantic_plan_bottleneck_{matrix_report.run_id}.md"
    md_path.write_text(_markdown_report(matrix_report), encoding="utf-8")

    no_bypass_failures = [a for a in matrix_report.arms if not a.no_bypass_ok]
    leaked = [a for a in matrix_report.arms if a.leakage]
    print(f"Wrote {json_path} and {md_path}")
    print(f"No-bypass audit failures: {len(no_bypass_failures)}")
    print(f"Leakage violations: {len(leaked)}")
    return 1 if (no_bypass_failures or leaked) else 0


if __name__ == "__main__":
    raise SystemExit(main())
