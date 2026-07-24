#!/usr/bin/env python3
"""SLM-297: unified final-program re-score of the SLM-155 fixture campaign.

Re-runs the deterministic SLM-155 (SPV3-02) fixture campaign with per-example
final programs retained, then scores every record under one commensurate
semantic metric (``meaningful_program_v1_report`` with ``gold=None``, plus
parse and output-contract facts). Emits paired old-vs-new transition tables,
reason-code prevalence, Wilson 95% intervals, and a pivot disposition for the
SLM-155 headline. Fixture-scale CPU; no training beyond the 20-step fixture
scorer.

Example:
  python -m scripts.run_slm297_unified_rescore
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.power_protocol import wilson_interval
from slm_training.harnesses.experiments.slm155_factorization_comparison import (
    FINAL_PROGRAM_OUTCOME_SCHEMA,
    FactorizationFamily,
    FinalProgramOutcomeV1,
    build_transition_table,
    old_outcome_label,
    paired_validity_table,
    run_fixture_campaign_with_records,
    score_final_program,
)
from slm_training.versioning import build_version_stamp

SCHEMA = "Slm297UnifiedRescoreV1"
EXPERIMENT_ID = "slm297-unified-rescore"
_DESIGN_JSON = "docs/design/iter-slm297-unified-rescore-20260724.json"
_DESIGN_MD = "docs/design/iter-slm297-unified-rescore-20260724.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _family_rates(
    outcomes: list[FinalProgramOutcomeV1],
    family: str,
) -> dict[str, float]:
    """Old process-label rate vs new final-program rate for one family."""
    subset = [o for o in outcomes if o.family == family]
    old = _mean(
        [
            1.0 if old_outcome_label(o) in {"accepted", "recovered"} else 0.0
            for o in subset
        ]
    )
    new = _mean([1.0 if o.v1_verdict else 0.0 for o in subset])
    return {"n": len(subset), "old_process_rate": old, "new_final_program_rate": new}


def _pivot_disposition(outcomes: list[FinalProgramOutcomeV1]) -> dict[str, Any]:
    """Does the corrected metric change the SLM-155 1.0-vs-0.0 headline?"""
    ar = _family_rates(outcomes, FactorizationFamily.AR.value)
    x22 = _family_rates(outcomes, FactorizationFamily.X22.value)
    hybrid = _family_rates(outcomes, FactorizationFamily.HYBRID.value)
    old_gap = ar["old_process_rate"] - x22["old_process_rate"]
    new_gap = ar["new_final_program_rate"] - x22["new_final_program_rate"]
    changed = (old_gap > 0) != (new_gap > 0) or abs(old_gap - new_gap) > 0.05
    if changed:
        narrative = (
            "PIVOT CONFIRMED: the SLM-155 headline was an incommensurate-label "
            "artifact. The old AR 'accepted' rate measured action-id membership "
            "while the old X22 'recovered' rate measured structural slice repair; "
            "under the unified final-program metric the AR-vs-X22 gap moves "
            f"from {old_gap:+.3f} to {new_gap:+.3f}."
        )
    else:
        narrative = (
            "NO PIVOT: the unified final-program metric preserves the AR-vs-X22 "
            f"ordering (old gap {old_gap:+.3f}, new gap {new_gap:+.3f})."
        )
    return {
        "old_gap_ar_minus_x22": round(old_gap, 4),
        "new_gap_ar_minus_x22": round(new_gap, 4),
        "headline_changed": changed,
        "narrative": narrative,
        "family_rates": {"ar": ar, "x22": x22, "hybrid": hybrid},
    }


def build_payload(
    *,
    n_records: int = 16,
    scorer_steps: int = 20,
) -> dict[str, Any]:
    """Run the retained-record campaign and build the full re-score payload."""
    report, records = run_fixture_campaign_with_records(
        run_id="slm297_unified_rescore",
        output_dir=None,
        n_records=n_records,
        scorer_steps=scorer_steps,
    )
    outcomes = [score_final_program(record) for record in records]

    arm_ids = [arm.arm_id for arm in report.manifest.arms]
    by_arm: dict[str, list[FinalProgramOutcomeV1]] = {
        arm_id: [o for o in outcomes if o.arm_id == arm_id] for arm_id in arm_ids
    }

    transition_tables = {
        arm_id: build_transition_table(arm_outcomes)
        for arm_id, arm_outcomes in by_arm.items()
    }

    reason_prevalence: dict[str, dict[str, int]] = {}
    for arm_id, arm_outcomes in by_arm.items():
        counts: dict[str, int] = {}
        for outcome in arm_outcomes:
            for code in outcome.v1_reason_codes:
                counts[code] = counts.get(code, 0) + 1
        reason_prevalence[arm_id] = dict(sorted(counts.items()))

    wilson = {}
    for arm_id, arm_outcomes in by_arm.items():
        successes = sum(1 for o in arm_outcomes if o.v1_verdict)
        interval = wilson_interval(successes, len(arm_outcomes))
        interval["successes"] = successes
        wilson[arm_id] = interval

    ar_arms = [
        a for a in arm_ids if by_arm[a] and by_arm[a][0].family == FactorizationFamily.AR.value
    ]
    hybrid_arms = [
        a
        for a in arm_ids
        if by_arm[a] and by_arm[a][0].family == FactorizationFamily.HYBRID.value
    ]
    paired_ar_vs_hybrid = {
        f"{ar_id} vs {hy_id}": paired_validity_table(by_arm[ar_id], by_arm[hy_id])
        for ar_id in ar_arms
        for hy_id in hybrid_arms
    }

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_id": report.run_id,
        "status": "fixture",
        "claim_class": "wiring",
        "outcome_schema": FINAL_PROGRAM_OUTCOME_SCHEMA,
        "metric": {
            "name": "meaningful_program_v1",
            "gold": None,
            "note": (
                "Unified final-program semantic metric; process metrics "
                "(accepted/recovered/costs) retained for provenance only."
            ),
        },
        "n_records": len(records),
        "outcomes": [o.to_dict() for o in outcomes],
        "transition_tables": transition_tables,
        "reason_code_prevalence": reason_prevalence,
        "wilson_final_program_rate": wilson,
        "paired_ar_vs_hybrid": paired_ar_vs_hybrid,
        "pivot_disposition": _pivot_disposition(outcomes),
        "aggregate_rows": [row.to_dict() for row in report.rows],
        "version_stamp": build_version_stamp(
            "harness.experiments.slm155_factorization_comparison"
        ),
        "timestamp": _now(),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the measured-results markdown for the re-score payload."""
    pivot = payload["pivot_disposition"]
    lines = [
        "# SLM-297: Unified final-program re-score of the SLM-155 fixture campaign",
        "",
        "**Run date:** 2026-07-24",
        "",
        "**Claim class:** wiring / fixture only. Deterministic CPU re-score; no "
        "model training beyond the shared 20-step fixture scorer, no ship-gate "
        "claim.",
        "",
        "**Machine-readable result:** ["
        "`iter-slm297-unified-rescore-20260724.json`](iter-slm297-unified-rescore-20260724.json)",
        "",
        "## Correction note (append-only)",
        "",
        "The SLM-155 (SPV3-02) 2026-07-20 comparison scored AR arms with an "
        "`accepted` process label (chosen action ∈ accepted_action_ids; no "
        "program was ever built) and X22/hybrid arms with a `recovered` label "
        "(conflict-slice structural repair flag). Those labels are "
        "incommensurate. This re-score retains every arm's final program and "
        "scores all arms under one metric (`meaningful_program_v1`, gold-free "
        "lane). The 2026-07-20 iter docs are left untouched; this document is "
        "the corrected comparison.",
        "",
        "## Pivot disposition",
        "",
        pivot["narrative"],
        "",
        "| Family | n | Old process rate | New final-program rate |",
        "| --- | --- | --- | --- |",
    ]
    for family, rates in pivot["family_rates"].items():
        lines.append(
            f"| {family} | {rates['n']} | {rates['old_process_rate']:.3f} | "
            f"{rates['new_final_program_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## New final-program rate per arm (Wilson 95%)",
            "",
            "| Arm | Valid | n | Estimate | Low | High |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm_id, interval in payload["wilson_final_program_rate"].items():
        est = interval["estimate"]
        low = interval["low"]
        high = interval["high"]
        lines.append(
            f"| {arm_id} | {interval['successes']} | {interval['n']} | "
            f"{est if est is None else f'{est:.3f}'} | "
            f"{low if low is None else f'{low:.3f}'} | "
            f"{high if high is None else f'{high:.3f}'} |"
        )

    lines.extend(
        [
            "",
            "## Old × new transition tables (per arm)",
            "",
            "| Arm | Old label | New valid | New invalid |",
            "| --- | --- | --- | --- |",
        ]
    )
    for arm_id, table in payload["transition_tables"].items():
        for label, counts in table.items():
            lines.append(
                f"| {arm_id} | {label} | {counts['valid']} | {counts['invalid']} |"
            )

    lines.extend(
        [
            "",
            "## Paired AR-vs-hybrid final-program validity (by record id + seed)",
            "",
            "| Pair | Both valid | AR valid, hybrid invalid | AR invalid, hybrid valid | Both invalid | Unpaired |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for pair, counts in payload["paired_ar_vs_hybrid"].items():
        lines.append(
            f"| {pair} | {counts['both_valid']} | {counts['a_valid_b_invalid']} | "
            f"{counts['a_invalid_b_valid']} | {counts['both_invalid']} | "
            f"{counts['unpaired']} |"
        )

    lines.extend(
        [
            "",
            "## Reason-code prevalence per arm",
            "",
            "| Arm | Reason code | Count |",
            "| --- | --- | --- |",
        ]
    )
    for arm_id, counts in payload["reason_code_prevalence"].items():
        if not counts:
            lines.append(f"| {arm_id} | (none) | 0 |")
        for code, count in counts.items():
            lines.append(f"| {arm_id} | {code} | {count} |")

    lines.extend(
        [
            "",
            "## Exact command",
            "",
            "```bash\npython -m scripts.run_slm297_unified_rescore\n```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-297 unified final-program re-score of SLM-155 fixture campaign",
        exit_on_error=False,
    )
    parser.add_argument("--n-records", type=int, default=16)
    parser.add_argument("--scorer-steps", type=int, default=20)
    parser.add_argument(
        "--no-write-docs",
        action="store_true",
        help="Print payload to stdout only; do not write docs/design artifacts.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    payload = build_payload(n_records=args.n_records, scorer_steps=args.scorer_steps)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    if args.no_write_docs:
        print(text)
        return 0

    root = Path(__file__).resolve().parents[1]
    json_path = root / _DESIGN_JSON
    md_path = root / _DESIGN_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(text, encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(str(json_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
