#!/usr/bin/env python3
"""Run the CAP2-03 state-local action-head fixture matrix.

Example::

    python -m scripts.run_cap2_state_local_action --out-dir outputs/runs/cap2_state_local_action

Fixture CPU runs are wiring evidence only; no production ship claim is made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.experiments.cap2_state_local_action import (
    StateLocalActionReport,
    run_fixture,
)


def _markdown_report(report: StateLocalActionReport) -> str:
    lines = [
        "# CAP2-03 state-local action-head fixture matrix",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Timestamp:* {report.timestamp}  ",
        f"*Hidden dim:* {report.hidden_dim}  ",
        "",
        "## Honest caveat",
        "",
        "This is a fixture CPU run.  It verifies the five state-local action-head",
        "families can route legal actions and recover an oracle-encoded choice; it",
        "does not train a production model or make a ship-quality claim.",
        "",
        "## Results",
        "",
        "| head_family | oracle_accuracy | random_init_accuracy | forced | abstain | detected_error | notes |",
        "| ----------- | --------------- | -------------------- | ------ | ------- | -------------- | ----- |",
    ]
    for r in report.results:
        notes = " ".join(r.notes)
        lines.append(
            f"| {r.head_family} | {r.oracle_accuracy:.4f} | "
            f"{r.random_init_accuracy:.4f} | {r.forced_states_recovered} | "
            f"{r.abstain_count} | {r.detected_error_count} | {notes} |"
        )
    lines.append("")
    lines.append("## Hard gates")
    lines.append("")
    failed_oracle = [r for r in report.results if r.oracle_accuracy < 1.0]
    illegal_random = [
        r
        for r in report.results
        if r.random_init_accuracy + (r.abstain_count + r.detected_error_count) / len(report.states) < 1.0
    ]
    lines.append(f"- Head families with perfect oracle recovery: {len(report.results) - len(failed_oracle)}/{len(report.results)}")
    lines.append(f"- Head families with only legal/abstain/detected outputs: {len(report.results) - len(illegal_random)}/{len(report.results)}")
    if failed_oracle:
        lines.append("- **FAIL** oracle recovery missing for: " + ", ".join(r.head_family for r in failed_oracle))
    else:
        lines.append("- **PASS** every head family recovered the oracle-encoded action.")
    if illegal_random:
        lines.append("- **FAIL** illegal output observed for: " + ", ".join(r.head_family for r in illegal_random))
    else:
        lines.append("- **PASS** no head family emitted an illegal action.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-families",
        default="",
        help="Comma-separated head families.  Empty means run all five.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/cap2_state_local_action"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the matrix without running it.",
    )
    args = parser.parse_args(argv)

    head_families: tuple[str, ...] | None = None
    if args.head_families.strip():
        head_families = tuple(x.strip() for x in args.head_families.split(",") if x.strip())

    if args.dry_run:
        families = head_families or (
            "global_masked",
            "local_flat",
            "ternary_digit",
            "ternary_ecoc",
            "grammar_factorized",
        )
        print(f"CAP2-03 dry-run: {len(families)} head family/ies")
        for family in families:
            print(f"  {family}")
        return 0

    report = run_fixture(head_families=head_families)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cap2_state_local_action_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"cap2_state_local_action_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    failed = any(r.oracle_accuracy < 1.0 for r in report.results) or any(
        r.random_init_accuracy + (r.abstain_count + r.detected_error_count) / len(report.states) < 1.0
        for r in report.results
    )
    print(f"Wrote {json_path} and {md_path}")
    print(f"Oracle perfect recovery: {sum(1 for r in report.results if r.oracle_accuracy == 1.0)}/{len(report.results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
