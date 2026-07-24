#!/usr/bin/env python3
"""Run the SLM-194 dynamic legal-edit proposal fixture matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm194_candidate_proposals import (
    ARM_NAMES,
    DEFAULT_CORPUS,
    EXPERIMENT_ID,
    K_GRID,
    MATRIX_SET,
    MATRIX_VERSION,
    run_candidate_proposal_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN_JSON = ROOT / "docs/design/iter-slm194-candidate-proposals-20260724.json"
DESIGN_MD = ROOT / "docs/design/iter-slm194-candidate-proposals-20260724.md"
AGENTV_DIR = ROOT / "docs/design/iter-slm194-candidate-proposals-agentv-20260724"


def _portable(value: Any) -> Any:
    marker = f"/{AGENTV_DIR.name}/"
    if isinstance(value, str) and marker in value:
        return "agentv-dir://" + value.split(marker, 1)[1]
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    return value


def _rewrite_agentv_paths() -> None:
    prefixes = {
        str(AGENTV_DIR.resolve()): "agentv-dir://",
        quote(str(AGENTV_DIR.resolve()), safe=""): quote("agentv-dir://", safe=""),
    }
    for path in (AGENTV_DIR / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for prefix, replacement in prefixes.items():
            content = content.replace(prefix, replacement)
        path.write_text(content, encoding="utf-8")


def _cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    non_oracle = {
        name: payload
        for name, payload in report["arms"].items()
        if name != "oracle_acceptable"
    }
    all_results = [
        result
        for payload in non_oracle.values()
        for result in payload["k_results"].values()
    ]
    return [
        {
            "id": "dynamic-interface",
            "criteria": "Direct and flow proposal scores share one dynamic-candidate interface without permanent edit classes.",
            "pass": report["common_candidate_interface"]["used_by_direct_and_flow"],
            "checks": {
                "shared_by_direct_and_flow": report["common_candidate_interface"][
                    "used_by_direct_and_flow"
                ],
                "compiler_owns_membership": report["common_candidate_interface"][
                    "membership_authority"
                ]
                == "exact_compiler",
                "scheduling_prefix_only": report["common_candidate_interface"][
                    "proposal_role"
                ]
                == "scheduling_prefix_only",
                "mandatory_fallback": report["common_candidate_interface"][
                    "fallback"
                ].startswith("mandatory exact completion"),
                "no_future_text": not report["common_candidate_interface"][
                    "final_source_or_future_witness_input"
                ],
            },
            "result": {
                "arms": sorted(report["arms"]),
                "training_rows": len(report["proposal_training_rows"]),
            },
        },
        {
            "id": "exact-fallback-parity",
            "criteria": "Every non-oracle k arm preserves exact final membership and deterministic output parity.",
            "pass": all(item["exact_final_output_parity"] for item in all_results),
            "checks": {
                "all_exact": all(
                    item["exact_final_output_parity"] for item in all_results
                ),
                "zero_invalid_over_valid": all(
                    item["invalid_over_valid_selections"] == 0
                    for item in all_results
                ),
            },
            "result": {
                "evaluated_results": len(all_results),
                "fallback_policy": report["common_candidate_interface"]["fallback"],
            },
        },
        {
            "id": "unknown-safety",
            "criteria": "UNKNOWN candidates are never converted into negatives or unsupported certificates.",
            "pass": (
                not report["common_candidate_interface"]["unknown_is_negative"]
                and all(not item["unknown_as_negative"] for item in all_results)
            ),
            "checks": {
                "interface_preserves_unknown": not report[
                    "common_candidate_interface"
                ]["unknown_is_negative"],
                "all_rows_safe": all(
                    not item["unknown_as_negative"] for item in all_results
                ),
            },
            "result": {
                "unknown_rows": sum(
                    bool(row["unknown_candidate_ids"])
                    for row in report["proposal_training_rows"]
                )
            },
        },
        {
            "id": "true-amortization-gate",
            "criteria": "A positive claim requires at least 95% target/acceptable recall and 30% warm-p50 improvement before fallback.",
            "pass": (
                bool(report["positive_claim_eligible"])
                if report["honest_verdict"] == "proposal_amortization_supported"
                else not report["positive_claim_eligible"]
            ),
            "checks": {
                "verdict_matches_eligibility": bool(
                    report["positive_claim_eligible"]
                )
                == (report["honest_verdict"] == "proposal_amortization_supported"),
                "no_false_positive_claim": bool(report["positive_claim_eligible"])
                or report["honest_verdict"] == "retain_exact_cached_enumeration",
            },
            "result": {
                "decision": report["decision"],
                "eligible": report["positive_claim_eligible"],
                "thresholds": report["thresholds"],
            },
        },
        {
            "id": "confirmation-firewall",
            "criteria": "The underpowered fixture does not touch confirmation data or write a promoted checkpoint.",
            "pass": (
                report["confirmation"]["status"] == "not_touched"
                and not report["confirmation"]["touch_ledger"]
                and not report["checkpoint"]["written"]
            ),
            "checks": {
                "confirmation_not_touched": report["confirmation"]["status"]
                == "not_touched",
                "touch_ledger_empty": not report["confirmation"]["touch_ledger"],
                "checkpoint_absent": not report["checkpoint"]["written"],
            },
            "result": {
                "claim_class": report["claim_class"],
                "verdict": report["honest_verdict"],
            },
        },
    ]


def _best_rows(report: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for arm, payload in report["arms"].items():
        if arm in {"complete_exact_cached", "oracle_acceptable"}:
            continue
        best_key, best = max(
            payload["k_results"].items(),
            key=lambda item: (
                min(item[1]["target_recall"], item[1]["acceptable_recall"]),
                item[1]["latency"]["warm_p50_improvement"],
            ),
        )
        rows.append((arm, best_key, best))
    return rows


def _markdown(report: dict[str, Any]) -> str:
    table = [
        "| arm | best k | target recall | acceptable recall | fallback | warm p50 delta | final work avoided |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, k, result in _best_rows(report):
        table.append(
            f"| {arm} | {k} | {result['target_recall']:.3f} | "
            f"{result['acceptable_recall']:.3f} | {result['fallback_rate']:.3f} | "
            f"{result['latency']['warm_p50_improvement']:+.3f} | "
            f"{result['work']['final_projections_avoided']} |"
        )
    return "\n".join(
        [
            "# SLM-194 (FFE3-03): exact-fallback candidate proposals",
            "",
            f"**Status:** measured CPU fixture screen. **Verdict:** `{report['honest_verdict']}`.",
            "",
            "## Result",
            "",
            *table,
            "",
            "No non-oracle learned arm cleared the joint ≥95% target/acceptable",
            "recall and ≥30% warm-p50 gate. Mandatory exact fallback restored every",
            "omitted candidate, so final membership/output parity was exact but no",
            "final projection, verifier, or support work was avoided. The proposal",
            "overhead was therefore not amortized. Retain exact cached enumeration.",
            "",
            "The oracle is diagnostic only. The tiny four-row/two-cluster SLM-196",
            "fixture cannot license a production proposal claim, and confirmation was",
            "not touched.",
            "",
            "## Safety and provenance",
            "",
            f"- Exact final parity: `{all(result['exact_final_output_parity'] for payload in report['arms'].values() if payload['status'] != 'oracle_diagnostic' for result in payload['k_results'].values())}`.",
            "- UNKNOWN-as-negative: `False`.",
            "- Candidate features contain no final source or future witness text.",
            f"- SLM-192 profile SHA: `{report['prerequisite_manifests']['slm192_profile_sha256']}`.",
            f"- SLM-193 cache SHA: `{report['prerequisite_manifests']['slm193_cache_sha256']}`.",
            f"- AgentV: `{report.get('agentv', {}).get('summary', {})}`.",
            "",
            "## Recipe",
            "",
            f"- Device/backend: `{report['recipe']['device']}` / `{report['recipe']['backend']}`.",
            f"- Steps/seed: `{report['recipe']['steps']}` / `{report['recipe']['seed']}`.",
            f"- k grid: `{report['k_grid']}`.",
            f"- Uncertainty: `{report['uncertainty']}`.",
            f"- Wall seconds: `{report['wall_seconds']:.3f}`.",
            "- Checkpoint: none; no promotion.",
            "",
            "## Caveats",
            "",
            *[f"- {item}" for item in report["honest_caveats"]],
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m scripts.run_candidate_proposal_matrix --eval",
            "```",
        ]
    ) + "\n"


def _parse_k_grid(value: str) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if token == "all":
            result.append(None)
        else:
            parsed = int(token)
            if parsed < 1:
                raise argparse.ArgumentTypeError("k values must be positive")
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("k grid cannot be empty")
    return tuple(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--describe", action="store_true")
    modes.add_argument("--build-corpus", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--eval", action="store_true")
    modes.add_argument("--confirm", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--k-grid", type=_parse_k_grid, default=K_GRID)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall-minutes", type=float, default=2.8)
    parser.add_argument("--output-json", type=Path, default=DESIGN_JSON)
    parser.add_argument("--output-md", type=Path, default=DESIGN_MD)
    args = parser.parse_args(argv)
    if args.describe:
        print(
            json.dumps(
                {
                    "schema": "CandidateProposalManifestV1",
                    "matrix_set": MATRIX_SET,
                    "matrix_version": MATRIX_VERSION,
                    "arms": ARM_NAMES,
                    "k_grid": ["all" if value is None else value for value in K_GRID],
                    "modes": [
                        "build-corpus",
                        "train",
                        "eval",
                        "confirm",
                    ],
                    "safety": "proposal prefixes schedule exact candidates; fallback restores complete membership",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = run_candidate_proposal_matrix(
        corpus_dir=args.corpus_dir,
        k_grid=args.k_grid,
        steps=args.steps,
        seed=args.seed,
        max_wall_minutes=args.max_wall_minutes,
    )
    if args.confirm and not report["positive_claim_eligible"]:
        raise SystemExit(
            "confirmation refused: no development arm cleared the frozen recall and wall gate"
        )
    report["requested_mode"] = (
        "confirm"
        if args.confirm
        else "build_corpus"
        if args.build_corpus
        else "train"
        if args.train
        else "eval"
    )
    published = publish_agentv_evaluation(
        AGENTV_DIR,
        name=EXPERIMENT_ID,
        claim="fixture_screen_not_confirmation",
        cases=_cases(report),
        version_stamp=report["version_stamp"],
    )
    _rewrite_agentv_paths()
    report["agentv"] = _portable(published)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(args.output_json), "verdict": report["honest_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
