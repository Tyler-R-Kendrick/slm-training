"""CLI for CAP1-05 template-abstraction sufficiency audit.

Example:
    python -m scripts.audit_template_sufficiency \
        --records outputs/data/train/v1/records.jsonl \
        --dsl-pack openui \
        --out outputs/runs/arity/template_sufficiency.json \
        --markdown-out docs/design/cap1-05-template-sufficiency-20260718.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.analysis.arity.template_sufficiency import (
    TemplateSufficiencyReport,
    audit_template_sufficiency,
    load_records,
)


def _write_json(path: Path, report: TemplateSufficiencyReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: TemplateSufficiencyReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CAP1-05: Template-abstraction sufficiency audit",
        "",
        f"**Date:** {datetime.now(timezone.utc).isoformat()}",
        "**Status:** wiring harness / fixture-only evidence. No training run, no checkpoint, no ship claim.",
        "",
        "## Summary",
        "",
        f"- Records audited: {report.metrics.get('records_audited', 0)}",
        f"- Value classes in inventory: {report.metrics.get('value_classes', 0)}",
        f"- Paired variants generated: {report.metrics.get('variants_generated', 0)}",
        f"- Violations (value change altered choice stream): {report.metrics.get('violations', 0)}",
        "",
        "## Value-class inventory",
        "",
    ]
    for vc in report.inventory.value_classes:
        lines.extend(
            [
                f"### `{vc.class_id}` ({vc.value_kind})",
                "",
                f"- Slot representation: `{vc.slot_representation}`",
                f"- Retained: {', '.join(vc.information_retained) or '(none)'}" ,
                f"- Discarded: {', '.join(vc.information_discarded) or '(none)'}" ,
                f"- Structural decisions: {', '.join(vc.structural_decisions) or '(none)'}" ,
                f"- Pack constraints: {', '.join(vc.pack_constraints) or '(none)'}" ,
                f"- Late-realization owner: {vc.late_realization_owner or '(none)'}" ,
                f"- Example fingerprints: {vc.to_dict()['example_fingerprints'][:5]}",
                "",
            ]
        )
    lines.extend(
        [
            "## Refinement candidates",
            "",
        ]
    )
    if report.refinements:
        for r in report.refinements:
            lines.append(
                f"- `{r.value_class_id}`: retain {list(r.retained_attributes)} "
                f"(est. +{r.estimated_added_bits:.2f} bits); removes "
                f"{list(r.removes_violations)}"
            )
    else:
        lines.append("- None proposed.")
    lines.extend(
        [
            "",
            "## Honesty caveats",
            "",
        ]
    )
    for note in report.notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Next steps",
            "",
            "- Feed the violation JSON and aligned-action records into "
            "`scripts.analyze_task_quotient` and `scripts.analyze_conditional_rate` "
            "to measure before/after state and rate costs.",
            "- Validate candidate refinements against held-out examples and update "
            "the template contract version if adopted.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CAP1-05 template-abstraction sufficiency audit"
    )
    parser.add_argument("--records", required=True, type=Path, help="JSONL records")
    parser.add_argument("--dsl-pack", default="openui", help="DSL pack name")
    parser.add_argument("--profile", default="openui-cap-v1", help="Audit profile")
    parser.add_argument(
        "--max-examples-per-class",
        type=int,
        default=10,
        help="Example literals stored per value class",
    )
    parser.add_argument(
        "--max-per-record",
        type=int,
        default=20,
        help="Maximum variants generated per record",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument(
        "--markdown-out", type=Path, default=None, help="Optional Markdown note path"
    )
    args = parser.parse_args(argv)

    if not args.records.exists():
        print(f"Records not found: {args.records}", file=sys.stderr)
        return 1

    records = load_records(args.records)
    report = audit_template_sufficiency(
        records,
        max_examples_per_class=args.max_examples_per_class,
        max_per_record=args.max_per_record,
    )
    _write_json(args.out, report)
    if args.markdown_out:
        _write_markdown(args.markdown_out, report)
    print(
        f"Wrote {report.metrics.get('variants_generated', 0)} variants, "
        f"{report.metrics.get('violations', 0)} violations to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
