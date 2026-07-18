#!/usr/bin/env python3
"""Run the CAP2-04 state-ownership ablation fixture matrix.

Example::

    python -m scripts.run_cap2_04_state_ablation --out-dir outputs/runs/cap2_04_state_ablation

Fixture CPU runs are wiring evidence only; no production ship claim is made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.cap2_04_state_ablation import (
    StateAblationReport,
    run_matrix,
)


def _markdown_report(report: StateAblationReport) -> str:
    lines = [
        "# CAP2-04 state-ownership ablation fixture matrix",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Hidden dim:* {report.hidden_dim}  ",
        f"*Semantic dim:* {report.semantic_dim}  ",
        f"*Unseen state ids:* {list(report.unseen_state_ids)}",
        "",
        "## Honest caveat",
        "",
        "This is a fixture CPU run.  It verifies the five state-ownership arms can be",
        "instantiated, trained for a few steps, and evaluated under a matched manifest;",
        "it does not train a production model or make a ship-quality claim.",
        "",
        "## Results",
        "",
        "| arm_id | mode | oracle | random_init | unseen | forced | params | active | capacity | leakage | notes |",
        "| ------ | ---- | ------ | ----------- | ------ | ------ | ------ | ------ | -------- | ------- | ----- |",
    ]
    for r in report.arms:
        notes = " ".join(r.notes)
        capacity = r.capacity if r.capacity is not None else "-"
        lines.append(
            f"| {r.arm_id} | {r.mode} | {r.oracle_accuracy:.4f} | "
            f"{r.random_init_accuracy:.4f} | {r.unseen_state_accuracy:.4f} | "
            f"{r.forced_decisions} | {r.trainable_parameters} | {r.active_parameters} | "
            f"{capacity} | {r.leakage} | {notes} |"
        )
    lines.append("")
    lines.append("## Hard gates")
    lines.append("")
    failed_oracle = [r for r in report.arms if r.oracle_accuracy < 1.0]
    leakage = [r for r in report.arms if r.leakage]
    lines.append(f"- Arms with perfect oracle recovery: {len(report.arms) - len(failed_oracle)}/{len(report.arms)}")
    lines.append(f"- Arms with leakage violations: {len(leakage)}")
    if failed_oracle:
        lines.append("- **FAIL** oracle recovery missing for: " + ", ".join(r.arm_id for r in failed_oracle))
    else:
        lines.append("- **PASS** every arm recovered the oracle-encoded action.")
    if leakage:
        lines.append("- **FAIL** leakage violation for: " + ", ".join(r.arm_id for r in leakage))
    else:
        lines.append("- **PASS** no arm leaked below-capacity state information.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        default="",
        help="Comma-separated arm modes.  Empty means run all five.",
    )
    parser.add_argument(
        "--state-count",
        type=int,
        default=8,
        help="Number of fixture states.",
    )
    parser.add_argument(
        "--action-count",
        type=int,
        default=5,
        help="Number of legal actions per state.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=16,
        help="Hidden dimension for all arms.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/cap2_04_state_ablation"),
    )
    parser.add_argument(
        "--no-match-parameters",
        action="store_true",
        help="Do not equalize active-parameter counts across arms.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the matrix without running it.",
    )
    args = parser.parse_args(argv)

    modes: tuple[str, ...] | None = None
    if args.modes.strip():
        modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    if args.dry_run:
        all_modes = modes or (
            "implicit",
            "explicit_exact",
            "discrete_code",
            "compiler_owned",
            "compiler_owned_no_state",
        )
        print(f"CAP2-04 dry-run: {len(all_modes)} mode/s, seeds={seeds}")
        for mode in all_modes:
            print(f"  {mode}")
        return 0

    report = run_matrix(
        state_count=args.state_count,
        action_count=args.action_count,
        hidden_dim=args.hidden_dim,
        seeds=seeds,
        modes=modes,
        match_parameters=not args.no_match_parameters,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap2_04_state_ablation_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap2_04_state_ablation_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    failed = any(r.oracle_accuracy < 1.0 for r in report.arms) or any(
        r.leakage for r in report.arms
    )
    print(f"Wrote {json_path} and {md_path}")
    print(f"Oracle perfect recovery: {sum(1 for r in report.arms if r.oracle_accuracy == 1.0)}/{len(report.arms)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
