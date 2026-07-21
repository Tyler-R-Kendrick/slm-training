#!/usr/bin/env python3
"""Run the SLM-185 judge resolution audit fixture.

Modes:
  describe         Print the schema and fixture scope.
  build-corpus     Write the deterministic fixture corpus.
  run              Run fixture judges, compute metrics, write the manifest.
  analyze-history  Reclassify endpoint deltas in an existing scoreboard or manifest.

Example:
  python -m scripts.run_judge_resolution_audit --mode describe
  python -m scripts.run_judge_resolution_audit --mode build-corpus
  python -m scripts.run_judge_resolution_audit --mode run --write-design-docs
  python -m scripts.run_judge_resolution_audit --mode analyze-history \
      --history outputs/runs/some-run/quality_matrix_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.judge_resolution import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    JudgeResolutionItemV1,
    SemanticResolutionManifestV1,
    apply_resolution_manifest,
    build_fixture_corpus,
    run_resolution_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm185-judge-resolution-20260720.json"
_DESIGN_MD = "docs/design/iter-slm185-judge-resolution-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-185 judge resolution audit schema

SemanticResolutionManifestV1 fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status, claim_class,
  hypothesis, falsifier, endpoints, global_floor, version_stamp, generated_at,
  provenance.

SemanticResolutionEndpointV1 fields:
  endpoint_label, provider, model, revision, metric_family, measured_flip_rate,
  cohen_kappa, fleiss_kappa, krippendorff_alpha, icc_1_1,
  pairwise_ordering_consistency, equivalence_invariance_error_rate,
  perturbation_detection_rate, brier_score, ece, abstention_rate, required_repeats,
  majority_rule, minimum_resolvable_delta, equivalence_margin,
  claim_language_permit_set, requires_independent_confirmation.

Fixture scope:
  - 6 canonical-equivalent pairs (verified with canonical_equal).
  - 6 minimal semantic-error pairs (source A correct, source B perturbed).
  - 3 historical-delta stubs.
  - 3 deterministic fixture endpoints with 5 repeats each.
  - No live LLM calls.

Claim class: wiring / fixture only.
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _run_build_corpus(
    output_dir: Path,
    *,
    repeats: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items, envelopes = build_fixture_corpus(repeats=repeats)

    corpus_path = output_dir / "judge_resolution_corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")

    summary = {
        "schema": "JudgeResolutionCorpusSummaryV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"slm185-judge-resolution-corpus-{_today_yyyymmdd()}",
        "status": "fixture",
        "claim_class": "wiring",
        "corpus_path": str(corpus_path),
        "item_n": len(items),
        "repeats": repeats,
        "envelopes": [e.to_dict() for e in envelopes],
        "version_stamp": build_version_stamp(
            "evals.judge_resolution",
        ),
        "timestamp": _now(),
    }
    _write_json(output_dir / "judge_resolution_corpus_summary.json", summary)
    return summary


def _load_corpus(corpus_path: Path) -> list[JudgeResolutionItemV1]:
    items: list[JudgeResolutionItemV1] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(JudgeResolutionItemV1.from_dict(json.loads(line)))
    return items


def _run_run(
    output_dir: Path,
    corpus_path: Path | None,
    *,
    repeats: int = 5,
    write_design_docs: bool = False,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if corpus_path is not None:
        items = _load_corpus(corpus_path)
        manifest, _ = run_resolution_fixture(
            repeats=repeats,
            run_id=f"slm185-judge-resolution-{_today_yyyymmdd()}",
            corpus_path=str(corpus_path),
        )
    else:
        manifest, items = run_resolution_fixture(
            repeats=repeats,
            run_id=f"slm185-judge-resolution-{_today_yyyymmdd()}",
        )

    # Ensure the manifest carries the actual item list in provenance summary.
    provenance = dict(manifest.provenance)
    provenance["item_n"] = len(items)

    payload = manifest.to_dict()
    payload["provenance"] = provenance
    payload["timestamp"] = _now()

    _write_json(output_dir / "judge_resolution_report.json", payload)

    if write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = design_json or (root / _DESIGN_JSON)
        md_path = design_md or (root / _DESIGN_MD)
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(payload), encoding="utf-8")

    return payload


def _run_analyze_history(
    history_path: Path,
    manifest_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    raw = json.loads(history_path.read_text(encoding="utf-8"))

    if manifest_path is not None:
        manifest = SemanticResolutionManifestV1.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    else:
        # Build a default fixture manifest for classification.
        _, _ = build_fixture_corpus(repeats=1)
        manifest, _ = run_resolution_fixture(repeats=1)

    if isinstance(raw, list):
        scoreboard = {"suites": {"history": {"rows": raw}}}
    elif isinstance(raw, dict):
        scoreboard = raw
    else:
        raise ValueError("history must be a JSON object or array")

    annotated = apply_resolution_manifest(scoreboard, manifest)

    payload = {
        "schema": "JudgeResolutionHistoryReclassificationV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source": str(history_path),
        "manifest_source": str(manifest_path) if manifest_path else "default_fixture",
        "semantic_resolution": annotated.get("semantic_resolution", {}),
        "version_stamp": build_version_stamp(
            "evals.judge_resolution",
            "matrix.quality",
        ),
        "timestamp": _now(),
    }
    _write_json(output_path, payload)
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    endpoints = payload.get("endpoints", [])
    lines = [
        f"# SLM-185 (FFE0-03): judge resolution audit fixture ({payload.get('run_id', '')})",
        "",
        f"Matrix set: `{payload.get('matrix_set', MATRIX_SET)}`",
        "",
        f"Version: `{payload.get('matrix_version', MATRIX_VERSION)}`",
        "",
        f"Status: **{payload.get('status', 'fixture')}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable weights "
        "were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        payload.get(
            "hypothesis",
            "Deterministic fixture judges can expose test-retest reliability, "
            "canonical-equivalence invariance, and semantic perturbation detection "
            "at a per-endpoint resolution floor.",
        ),
        "",
        "## Falsifier",
        "",
        payload.get(
            "falsifier",
            "The reliability metrics collapse (NaN/undefined) or the equivalence "
            "invariance error rate is non-zero for canonical-equivalent pairs.",
        ),
        "",
        "## Endpoints",
        "",
        "| endpoint | flip_rate | equiv_invariance_error | perturbation_detection | min_delta | margin |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for ep in endpoints:
        lines.append(
            f"| {ep.get('endpoint_label', '')} | "
            f"{ep.get('measured_flip_rate', None)} | "
            f"{ep.get('equivalence_invariance_error_rate', None)} | "
            f"{ep.get('perturbation_detection_rate', None)} | "
            f"{ep.get('minimum_resolvable_delta', '')} | "
            f"{ep.get('equivalence_margin', '')} |"
        )

    lines.extend(
        [
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The test-retest "
            "reliability, canonical-equivalence invariance, and semantic-resolution "
            "metrics are wired and exercised on deterministic synthetic judges, but no "
            "live external judge or production eval suite was used. The mechanism "
            "remains ``retain_diagnostic`` / ``blocked_pending_real_eval`` until it is "
            "validated with independent judges and real suite results.",
            "",
            "## Honest caveats",
            "",
            "- Fixture judges are deterministic and synthetic; real judges have sampling "
            "  variance, cost, latency, and provenance constraints not modeled here.",
            "- The `fixture_seeded_hash_scorer` endpoint deliberately varies by repeat "
            "  seed to exercise flip-rate computation; it is not a semantic judge.",
            "- Canonical-equivalent pairs are verified with `canonical_equal` using the "
            "  current D2 canonicalizer. Not every named surface transformation "
            "  (e.g., dead-binding prune) is normalized by the current canonicalizer; "
            "  only pairs that canonicalize equally are included.",
            "- No ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_judge_resolution_audit --mode describe",
            "python -m scripts.run_judge_resolution_audit --mode build-corpus",
            "python -m scripts.run_judge_resolution_audit --mode run --write-design-docs",
            "python -m scripts.run_judge_resolution_audit --mode analyze-history \\",
            "  --history <scoreboard-or-manifest.json>",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument(
        "--mode",
        choices={"describe", "build-corpus", "run", "analyze-history"},
        default="describe",
        help="Audit mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/judge-resolution-{_today_yyyymmdd()}"),
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to a corpus JSONL for --mode run.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Path to a scoreboard, manifest, or JSONL for --mode analyze-history.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional SemanticResolutionManifestV1 for --mode analyze-history.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="Output path for analyze-history (default: <output-dir>/judge_resolution_history_reclassification.json).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated judgments (default: 5).",
    )
    parser.add_argument(
        "--write-design-docs",
        action="store_true",
        help="Write design doc pair in run mode.",
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        default=None,
        help="Override path for the design JSON (run mode).",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown (run mode).",
    )

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.mode == "describe":
        print(_describe_schema())
        return 0

    if args.mode == "build-corpus":
        _run_build_corpus(args.output_dir, repeats=args.repeats)
        print(json.dumps({"summary": str(args.output_dir / "judge_resolution_corpus_summary.json")}, indent=2))
        return 0

    if args.mode == "run":
        _run_run(
            args.output_dir,
            args.corpus,
            repeats=args.repeats,
            write_design_docs=args.write_design_docs,
            design_json=args.design_json,
            design_md=args.design_md,
        )
        print(json.dumps({"report": str(args.output_dir / "judge_resolution_report.json")}, indent=2))
        return 0

    if args.mode == "analyze-history":
        if args.history is None:
            print("error: --history is required for analyze-history mode", file=sys.stderr)
            return 2
        output_path = args.analysis_output or (
            args.output_dir / "judge_resolution_history_reclassification.json"
        )
        _run_analyze_history(args.history, args.manifest, output_path)
        print(json.dumps({"reclassification": str(output_path)}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
