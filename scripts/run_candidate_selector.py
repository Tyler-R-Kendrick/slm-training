#!/usr/bin/env python3
"""Run the SLM-127 EFS3-04 candidate-selector wiring/fixture harness.

Examples:
  python -m scripts.run_candidate_selector --fixture --out outputs/runs/slm127/report.json
  python -m scripts.run_candidate_selector --groups groups.jsonl --selector model_score --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.candidate_selector import (
    CandidateSelectionGroupV1,
    EnergyScoreSelector,
    HardThenSimpleSelector,
    LearnedCandidateSelector,
    ModelScoreSelector,
    ThresholdManifestV1,
    ValueScoreSelector,
    evaluate_selector,
    load_selection_groups,
    make_fixture_groups,
    select_threshold_on_validation,
    train_selector_fixture,
)
from slm_training.versioning import build_version_stamp

__all__ = ["main"]

SELECTOR_CHOICES = (
    "model_score",
    "value_score",
    "energy_score",
    "hard_then_simple",
    "learned_no_abstain",
    "learned_abstain",
)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_learned_selector(
    groups: tuple[CandidateSelectionGroupV1, ...],
    *,
    abstain: bool,
    target_risk: float,
) -> tuple[LearnedCandidateSelector, ThresholdManifestV1 | None, dict[str, Any]]:
    """Train a tiny scorer and optionally calibrate an abstention threshold."""
    model, recipe = train_selector_fixture(groups)
    selector_no_threshold = LearnedCandidateSelector(model, threshold_manifest=None)
    threshold_manifest: ThresholdManifestV1 | None = None
    if abstain:
        threshold_manifest = select_threshold_on_validation(
            groups, selector_no_threshold, target_risk=target_risk
        )
        selector = LearnedCandidateSelector(model, threshold_manifest=threshold_manifest)
    else:
        threshold_manifest = ThresholdManifestV1(
            threshold_id=f"{selector_no_threshold.selector_id}_no_abstain",
            selector_id=selector_no_threshold.selector_id,
            calibration_set_hash="",
            metric_label_version="candidate_acceptability_v1",
            threshold=0.0,
            validation_coverage=1.0,
            validation_risk=0.0,
            policy="no_abstain",
            timestamp=_now(),
        )
        selector = LearnedCandidateSelector(model, threshold_manifest=threshold_manifest)
    return selector, threshold_manifest, recipe


def _selector_factory(
    name: str,
    groups: tuple[CandidateSelectionGroupV1, ...],
    *,
    target_risk: float,
) -> tuple[Any, ThresholdManifestV1 | None]:
    """Instantiate the requested selector, training if needed."""
    if name == "model_score":
        return ModelScoreSelector(), None
    if name == "value_score":
        return ValueScoreSelector(), None
    if name == "energy_score":
        return EnergyScoreSelector(), None
    if name == "hard_then_simple":
        return HardThenSimpleSelector(), None
    if name == "learned_no_abstain":
        selector, manifest, _recipe = _build_learned_selector(
            groups, abstain=False, target_risk=target_risk
        )
        return selector, manifest
    if name == "learned_abstain":
        selector, manifest, _recipe = _build_learned_selector(
            groups, abstain=True, target_risk=target_risk
        )
        return selector, manifest
    raise ValueError(f"unknown selector: {name}")


def _run_all_arms(
    groups: tuple[CandidateSelectionGroupV1, ...],
    *,
    target_risk: float,
) -> dict[str, Any]:
    """Run every selector arm and return a comparison report."""
    learned_no_abstain, learned_no_abstain_manifest, recipe = _build_learned_selector(
        groups, abstain=False, target_risk=target_risk
    )
    learned_abstain, learned_abstain_manifest, _recipe = _build_learned_selector(
        groups, abstain=True, target_risk=target_risk
    )
    selectors = [
        ("model_score", ModelScoreSelector(), None),
        ("value_score", ValueScoreSelector(), None),
        ("energy_score", EnergyScoreSelector(), None),
        ("hard_then_simple", HardThenSimpleSelector(), None),
        ("learned_no_abstain", learned_no_abstain, learned_no_abstain_manifest),
        ("learned_abstain", learned_abstain, learned_abstain_manifest),
    ]
    arm_results: dict[str, Any] = {}
    for selector_id, selector, manifest in selectors:
        arm_results[selector_id] = evaluate_selector(
            groups, selector, threshold_manifest=manifest
        )
    return {
        "arm_results": arm_results,
        "threshold_manifest_learned_abstain": (
            learned_abstain_manifest.to_dict() if learned_abstain_manifest else None
        ),
        "training_recipe": recipe,
    }


def _run_single_arm(
    name: str,
    groups: tuple[CandidateSelectionGroupV1, ...],
    *,
    target_risk: float,
) -> dict[str, Any]:
    selector, manifest = _selector_factory(name, groups, target_risk=target_risk)
    metrics = evaluate_selector(groups, selector, threshold_manifest=manifest)
    return {
        "selector": name,
        "arm_results": {name: metrics},
        "threshold_manifest": manifest.to_dict() if manifest else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-127 EFS3-04 candidate selector wiring/fixture harness",
        exit_on_error=False,
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the built-in tiny synthetic fixture and compare all arms",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        help="Path to a JSONL file of CandidateSelectionGroupV1 records",
    )
    parser.add_argument(
        "--selector",
        choices=SELECTOR_CHOICES,
        help="Selector arm to run when --groups is provided",
    )
    parser.add_argument(
        "--calibrate-target-risk",
        type=float,
        default=0.05,
        help="Maximum selected-error risk on the validation split (default: 0.05)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to write the JSON report",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.fixture and args.groups is not None:
        print("error: --fixture and --groups are mutually exclusive", file=sys.stderr)
        return 2
    if not args.fixture and args.groups is None:
        print("error: specify --fixture or --groups", file=sys.stderr)
        return 2
    if args.groups is not None and args.selector is None:
        print("error: --groups requires --selector", file=sys.stderr)
        return 2
    if args.groups is None and args.selector is not None:
        print("error: --selector requires --groups (or use --fixture)", file=sys.stderr)
        return 2

    if args.fixture:
        groups = make_fixture_groups()
        result = _run_all_arms(groups, target_risk=args.calibrate_target_risk)
    else:
        assert args.groups is not None and args.selector is not None
        groups = load_selection_groups(str(args.groups))
        result = _run_single_arm(
            args.selector, groups, target_risk=args.calibrate_target_risk
        )

    report: dict[str, Any] = {
        "schema": "CandidateSelectorReportV1",
        "claim_class": "wiring",
        "status": "fixture",
        "fixture_groups": len(groups),
        "timestamp": _now(),
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.candidate_selector",
        ),
        **result,
    }

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report_text, encoding="utf-8")
    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
