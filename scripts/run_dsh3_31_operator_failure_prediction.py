#!/usr/bin/env python3
"""Run DSH3-31's small synthetic operator-decode failure-prediction experiment.

Builds a modest, hand-built set of synthetic ``(features, ReplayOutcomeLabel)``
rows spanning all 7 :class:`ReplayOutcomeLabel` values plus several
``SAFE_DEFER`` rows (the same fixture-construction style as
``tests/test_evals/test_operator_failure_prediction.py``, no real
``DecodeStats`` production traces), scores all four named feature arms
(``all_counters``, ``compact_subset``, ``entropy_margin_baseline``,
``context_free_baseline``) via
``slm_training.evals.operator_failure_prediction.evaluate_failure_prediction_arms``,
and writes the resulting ``OperatorFailurePredictionReportV1`` plus a markdown
summary. This is a fixture-scale wiring demonstration, never a ship claim:
see ``docs/design/dsh3-31-operator-failure-prediction.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.operator_failure_prediction import (
    ALL_COUNTER_FEATURES,
    PREREGISTERED_COMPACT_FEATURES,
    OperatorFailurePredictionReportV1,
    ReplayOutcomeLabel,
    evaluate_failure_prediction_arms,
)
from slm_training.versioning import build_version_stamp

#: Every numeric DecodeStats counter this synthetic fixture actually varies;
#: every other ALL_COUNTER_FEATURES name defaults to 0.0 via
#: score_from_features's dict.get(name, 0.0) fallback.
_VARIED_COUNTERS: tuple[str, ...] = (
    "compiler_lattice_false_hard_eliminations",
    "compiler_lattice_invalid_selected_over_valid",
    "constrained_dead_ends",
    "constrained_last_legal_candidates",
    "compiler_lattice_selector_regret",
    "solver_backtracks",
    "compiler_fallbacks",
)


def _row(
    *,
    label: ReplayOutcomeLabel,
    false_hard_eliminations: float = 0.0,
    invalid_selected_over_valid: float = 0.0,
    dead_ends: float = 0.0,
    last_legal_candidates: float = 0.0,
    selector_regret: float = 0.5,
    solver_backtracks: float = 0.0,
    compiler_fallbacks: float = 0.0,
) -> tuple[dict[str, float], ReplayOutcomeLabel]:
    return (
        {
            "compiler_lattice_false_hard_eliminations": false_hard_eliminations,
            "compiler_lattice_invalid_selected_over_valid": invalid_selected_over_valid,
            "constrained_dead_ends": dead_ends,
            "constrained_last_legal_candidates": last_legal_candidates,
            "compiler_lattice_selector_regret": selector_regret,
            "solver_backtracks": solver_backtracks,
            "compiler_fallbacks": compiler_fallbacks,
        },
        label,
    )


def _build_rows() -> list[tuple[dict[str, float], ReplayOutcomeLabel]]:
    """18 synthetic rows: 2-3 rows per failure label (all 7 labels present)
    with large compact-subset counters but a selector_regret deliberately
    overlapping the SAFE_DEFER rows' regret range, and 6 SAFE_DEFER rows with
    near-zero compact-subset counters and a similarly overlapping regret
    range. Mirrors the unit test's smaller version of this same by-
    construction separation, at a bit more volume."""
    rows: list[tuple[dict[str, float], ReplayOutcomeLabel]] = [
        _row(
            label=ReplayOutcomeLabel.TIMEOUT,
            false_hard_eliminations=4,
            invalid_selected_over_valid=2,
            dead_ends=2,
            last_legal_candidates=5,
            selector_regret=0.52,
            solver_backtracks=3,
        ),
        _row(
            label=ReplayOutcomeLabel.TIMEOUT,
            false_hard_eliminations=5,
            invalid_selected_over_valid=3,
            dead_ends=3,
            last_legal_candidates=4,
            selector_regret=0.48,
            solver_backtracks=2,
        ),
        _row(
            label=ReplayOutcomeLabel.ILLEGAL_SELECTION,
            false_hard_eliminations=6,
            invalid_selected_over_valid=6,
            dead_ends=2,
            last_legal_candidates=3,
            selector_regret=0.44,
            compiler_fallbacks=1,
        ),
        _row(
            label=ReplayOutcomeLabel.ILLEGAL_SELECTION,
            false_hard_eliminations=7,
            invalid_selected_over_valid=5,
            dead_ends=3,
            last_legal_candidates=6,
            selector_regret=0.56,
            compiler_fallbacks=2,
        ),
        _row(
            label=ReplayOutcomeLabel.EXECUTOR_REJECTION,
            false_hard_eliminations=5,
            invalid_selected_over_valid=4,
            dead_ends=4,
            last_legal_candidates=5,
            selector_regret=0.45,
        ),
        _row(
            label=ReplayOutcomeLabel.EXECUTOR_REJECTION,
            false_hard_eliminations=6,
            invalid_selected_over_valid=3,
            dead_ends=3,
            last_legal_candidates=4,
            selector_regret=0.55,
        ),
        _row(
            label=ReplayOutcomeLabel.REPLAY_FAILURE,
            false_hard_eliminations=6,
            invalid_selected_over_valid=5,
            dead_ends=4,
            last_legal_candidates=5,
            selector_regret=0.45,
        ),
        _row(
            label=ReplayOutcomeLabel.REPLAY_FAILURE,
            false_hard_eliminations=7,
            invalid_selected_over_valid=4,
            dead_ends=5,
            last_legal_candidates=6,
            selector_regret=0.50,
        ),
        _row(
            label=ReplayOutcomeLabel.DEAD_END,
            false_hard_eliminations=5,
            invalid_selected_over_valid=4,
            dead_ends=3,
            last_legal_candidates=6,
            selector_regret=0.50,
        ),
        _row(
            label=ReplayOutcomeLabel.DEAD_END,
            false_hard_eliminations=8,
            invalid_selected_over_valid=2,
            dead_ends=6,
            last_legal_candidates=7,
            selector_regret=0.46,
        ),
        _row(
            label=ReplayOutcomeLabel.SEMANTIC_MISS,
            false_hard_eliminations=4,
            invalid_selected_over_valid=3,
            dead_ends=2,
            last_legal_candidates=4,
            selector_regret=0.54,
        ),
        _row(
            label=ReplayOutcomeLabel.SEMANTIC_MISS,
            false_hard_eliminations=5,
            invalid_selected_over_valid=2,
            dead_ends=2,
            last_legal_candidates=5,
            selector_regret=0.42,
        ),
    ]
    for i, regret in enumerate((0.50, 0.40, 0.60, 0.55, 0.45, 0.50)):
        rows.append(
            _row(
                label=ReplayOutcomeLabel.SAFE_DEFER,
                false_hard_eliminations=0 if i % 2 == 0 else 1,
                invalid_selected_over_valid=0,
                dead_ends=0,
                last_legal_candidates=i % 2,
                selector_regret=regret,
            )
        )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DSH3-31 operator-decode failure-prediction feature-arm ablation (SLM-406)",
        "",
        "Status: fixture-scale synthetic demonstration; not a ship claim",
        "",
        f"## Per-arm breakdown ({report['run']['n_rows']} synthetic rows, "
        f"{report['run']['n_positive']} replay-grounded failures, "
        f"{report['run']['n_safe_defer']} SAFE_DEFER)",
        "",
        "| Arm | # features | AUROC | AUPRC | Calibration error | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in report["prediction"]["arms"]:
        auroc_str = "n/a" if arm["auroc"] is None else f"{arm['auroc']:.3f}"
        auprc_str = "n/a" if arm["auprc"] is None else f"{arm['auprc']:.3f}"
        calib_str = (
            "n/a" if arm["calibration_error"] is None else f"{arm['calibration_error']:.3f}"
        )
        lines.append(
            f"| `{arm['arm']}` | {len(arm['feature_names'])} | {auroc_str} | "
            f"{auprc_str} | {calib_str} | {arm['n_examples']} |"
        )
    lines.extend(
        [
            "",
            f"Best arm by AUROC: `{report['prediction']['best_arm_by_auroc']}`.",
            "",
            "## Honesty",
            "",
            "This is a hand-built, 18-row synthetic fixture spanning all 7 "
            "`ReplayOutcomeLabel`s plus 6 SAFE_DEFER rows, not real "
            "`DecodeStats` production/replay traces -- it demonstrates the "
            "feature-arm scoring, AUROC/AUPRC, and calibration wiring end to "
            "end, and shows a fixture-scale positive signal for the compact "
            "counter subset over the entropy/margin-only baseline. It is not "
            "a claim about real held-out generalization, real checkpoint "
            "cross-generalization, or a production early-abort deployment "
            "decision. No time-indexed 'earliest decode fraction reaching "
            "target precision' metric is computed here -- that requires real "
            "time-indexed decode traces which do not exist yet in this repo; "
            "this is explicitly out of scope, not approximated. This module "
            "changes no production decode/abort behavior (shadow-analysis "
            "only).",
            "",
            f"Decision: `{report['decision']['verdict']}` -- {report['decision']['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows()
    arms = {
        "all_counters": ALL_COUNTER_FEATURES,
        "compact_subset": PREREGISTERED_COMPACT_FEATURES,
        "entropy_margin_baseline": ("compiler_lattice_selector_regret",),
        "context_free_baseline": (),
    }
    prediction: OperatorFailurePredictionReportV1 = evaluate_failure_prediction_arms(rows, arms)
    prediction_payload = prediction.to_dict()

    by_arm = {arm["arm"]: arm for arm in prediction_payload["arms"]}
    compact_auroc = by_arm["compact_subset"]["auroc"]
    baseline_auroc = by_arm["entropy_margin_baseline"]["auroc"]

    stamp = build_version_stamp("evals.operator_failure_prediction")

    if (
        compact_auroc is not None
        and baseline_auroc is not None
        and compact_auroc > baseline_auroc
    ):
        decision = {
            "verdict": "fixture-scale-positive-signal",
            "reason": (
                "compact_subset strictly exceeds entropy_margin_baseline's AUROC on this "
                "18-row synthetic fixture -- fixture-scale positive signal for DSH3-31's "
                "hypothesis that a compact DecodeStats compiler-lattice/constrained-decode "
                "counter subset predicts replay-grounded operator-decode failure better than "
                "a selector_regret-only margin baseline. This is not real held-out "
                "generalization evidence, not a real checkpoint cross-generalization claim, "
                "and not a production early-abort deployment decision -- DSH3-31's own stop "
                "rule ('if no feature set beats entropy/margin within confidence bounds, "
                "reject telemetry-based early abort') requires real decode-trace-scale "
                "evidence this fixture cannot provide. The re-test trigger is once real "
                "per-decision DecodeStats rows with replay-grounded outcome labels exist at "
                "volume -- only then should this ablation be re-run against real traces and "
                "confidence bounds computed."
            ),
        }
    else:
        decision = {
            "verdict": "no-benefit-observed",
            "reason": (
                "compact_subset did not strictly exceed entropy_margin_baseline's AUROC on "
                "this fixture; this would contradict DSH3-31's hypothesis and should be "
                "investigated (fixture construction or feature choice) before any further "
                "telemetry-based early-abort work is proposed."
            ),
        }

    report = {
        "schema": "dsh3_31_operator_failure_prediction_report/v1",
        "issue": "SLM-406",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(rows),
            "n_positive": sum(1 for _, label in rows if label != ReplayOutcomeLabel.SAFE_DEFER),
            "n_safe_defer": sum(1 for _, label in rows if label == ReplayOutcomeLabel.SAFE_DEFER),
            "label_counts": {
                label.value: sum(1 for _, row_label in rows if row_label == label)
                for label in ReplayOutcomeLabel
            },
            "ship_claim": False,
        },
        "prediction": prediction_payload,
        "decision": decision,
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "report.json"), "decision": decision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
