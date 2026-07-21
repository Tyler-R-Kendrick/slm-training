#!/usr/bin/env python3
"""Run the SLM-186 verified-utility audit fixture.

Modes:
  describe         Print the schema and fixture scope.
  fixture          Run the fixture audit and write the report.
  analyze-history  Read an existing iter JSON and synthesize a utility table.
  sensitivity      Run only the sensitivity rank-reversal analysis.

Example:
  python -m scripts.run_verified_utility_audit --mode describe
  python -m scripts.run_verified_utility_audit --mode fixture --write-design-docs
  python -m scripts.run_verified_utility_audit --mode analyze-history \
      --history docs/design/iter-slm186-verified-utility-20260721.json
  python -m scripts.run_verified_utility_audit --mode sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.verified_utility import (
    UtilityWeightManifestV1,
    sensitivity_rank_reversals,
)
from slm_training.harnesses.experiments.slm186_verified_utility import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    VerifiedUtilityAuditReport,
    build_default_weight_manifest,
    build_fixture_candidates,
    render_markdown,
    run_verified_utility_audit,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _design_paths() -> tuple[Path, Path]:
    today = _today_yyyymmdd()
    return (
        Path(f"docs/design/iter-slm186-verified-utility-{today}.json"),
        Path(f"docs/design/iter-slm186-verified-utility-{today}.md"),
    )


def _describe_schema() -> str:
    return """\
SLM-186 verified-utility audit schema

VerifiedUtilityV1 factors:
  hard_valid, support_status, contract_coverage, binding_aware_meaningful_v2,
  component_role_recall, topology_node_f1, topology_edge_f1,
  reference_graph_exactness, behavior_evidence, render_evidence,
  independent_judge_score, human_pair_preference, complexity_cost, inference_cost,
  abstained, failure_reason_codes, availability.

UtilityWeightManifestV1 fields:
  weights, normalization, primary_policy, dev_fit_hash, confirmation_hash,
  permitted_ranges, version.

Audit report fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status, claim_class,
  manifest, weight_manifest, candidates, scalar_ranking, lexicographic_ranking,
  pareto_front_ids, abstention_economics, sensitivity, canary_summary,
  version_stamp, generated_at.

Fixture scope:
  - 7 synthetic candidates covering pareto-dominant, dominated, abstained,
    canary, economy, and partial-data scenarios.
  - Scalar and lexicographic rankings.
  - Pareto frontier and dominance analysis.
  - Abstention economics with a risk threshold.
  - Sensitivity rank-reversal analysis across two weight manifests.
  - Canary summary from the SLM-186 metric_gaming extension.
  - No live judge or GPU calls.

Claim class: wiring / fixture only.
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _run_fixture(
    output_dir: Path,
    *,
    write_design_docs: bool = False,
    design_json: Path | None = None,
    design_md: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, report = run_verified_utility_audit(
        mode="fixture",
        run_id=f"slm186-verified-utility-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seed=seed,
    )

    payload = report.to_dict()
    payload["timestamp"] = _now()
    _write_json(output_dir / "verified_utility_report.json", payload)

    if write_design_docs:
        root = Path(__file__).resolve().parents[1]
        default_json, default_md = _design_paths()
        json_path = design_json or (root / default_json)
        md_path = design_md or (root / default_md)
        _write_json(json_path, payload)
        md_path.write_text(render_markdown(report), encoding="utf-8")

    return payload


def _run_sensitivity(
    output_dir: Path,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        (candidate_id, util)
        for candidate_id, _scenario, util in build_fixture_candidates(seed=seed)
    ]
    weight_manifest = build_default_weight_manifest()
    perturb_manifest = UtilityWeightManifestV1(
        weights={k: v * 0.8 for k, v in weight_manifest.weights.items()},
        normalization=weight_manifest.normalization,
        primary_policy=weight_manifest.primary_policy,
        permitted_ranges=weight_manifest.permitted_ranges,
        version=weight_manifest.version,
    )
    sensitivity = sensitivity_rank_reversals(
        candidates,
        [weight_manifest, perturb_manifest],
        perturbations_per_manifest=50,
        seed=seed,
    )
    payload = {
        "schema": "VerifiedUtilitySensitivityV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"slm186-verified-utility-sensitivity-{_today_yyyymmdd()}",
        "status": "fixture",
        "claim_class": "wiring",
        "sensitivity": sensitivity,
        "timestamp": _now(),
    }
    _write_json(output_dir / "verified_utility_sensitivity.json", payload)
    return payload


def _run_analyze_history(
    history_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    raw = json.loads(history_path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        report = VerifiedUtilityAuditReport.from_dict(raw[0])
    else:
        report = VerifiedUtilityAuditReport.from_dict(raw)

    # Build a small utility table from the report's candidates.
    table: list[dict[str, Any]] = []
    for rec in report.candidates:
        util = rec.utility
        table.append(
            {
                "candidate_id": rec.candidate_id,
                "scenario": rec.scenario,
                "scalar_score": rec.scalar_score,
                "lexicographic_rank": rec.rank_lexicographic,
                "hard_valid": util.hard_valid,
                "binding_aware_meaningful_v2": util.binding_aware_meaningful_v2,
                "component_role_recall": util.component_role_recall,
                "complexity_cost": util.complexity_cost,
                "inference_cost": util.inference_cost,
                "abstained": util.abstained,
            }
        )

    payload = {
        "schema": "VerifiedUtilityHistoryAnalysisV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"slm186-verified-utility-history-{_today_yyyymmdd()}",
        "status": "fixture",
        "claim_class": "wiring",
        "source": str(history_path),
        "n_candidates": len(table),
        "utility_table": table,
        "honest_caveat": (
            "Fixture-only history analysis.  Real eval records are required "
            "before claiming floor-escape."
        ),
        "timestamp": _now(),
    }
    _write_json(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument(
        "--mode",
        choices={"describe", "fixture", "analyze-history", "sensitivity"},
        default="describe",
        help="Audit mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/slm186-verified-utility-{_today_yyyymmdd()}"),
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Path to an existing iter JSON for --mode analyze-history.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="Output path for analyze-history (default: <output-dir>/verified_utility_history_analysis.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for fixture generation (default: 0).",
    )
    parser.add_argument(
        "--write-design-docs",
        action="store_true",
        help="Write design doc pair in fixture mode.",
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        default=None,
        help="Override path for the design JSON (fixture mode).",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown (fixture mode).",
    )

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.mode == "describe":
        print(_describe_schema())
        return 0

    if args.mode == "fixture":
        _run_fixture(
            args.output_dir,
            write_design_docs=args.write_design_docs,
            design_json=args.design_json,
            design_md=args.design_md,
            seed=args.seed,
        )
        print(
            json.dumps(
                {"report": str(args.output_dir / "verified_utility_report.json")},
                indent=2,
            )
        )
        return 0

    if args.mode == "sensitivity":
        payload = _run_sensitivity(args.output_dir, seed=args.seed)
        print(
            json.dumps(
                {
                    "sensitivity": str(
                        args.output_dir / "verified_utility_sensitivity.json"
                    ),
                    "reversal_count": payload["sensitivity"]["reversal_count"],
                    "reversal_rate": payload["sensitivity"]["reversal_rate"],
                },
                indent=2,
            )
        )
        return 0

    if args.mode == "analyze-history":
        if args.history is None:
            print("error: --history is required for analyze-history mode", file=sys.stderr)
            return 2
        output_path = args.analysis_output or (
            args.output_dir / "verified_utility_history_analysis.json"
        )
        payload = _run_analyze_history(args.history, output_path)
        print(
            json.dumps(
                {
                    "analysis": str(output_path),
                    "n_candidates": payload["n_candidates"],
                },
                indent=2,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
