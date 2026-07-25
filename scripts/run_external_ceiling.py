#!/usr/bin/env python3
"""Run the SLM-108 external constrained-decoding semantic ceiling matrix.

Example (fixture, no model download):
  python -m scripts.run_external_ceiling --mode fixture --output-dir outputs/runs/slm108_fixture

Example (SLM-294 frontier arm chunk, resumable under the run cap):
  python -m scripts.run_external_ceiling --mode frontier --arm B \
      --chunk-size 5 --output-dir outputs/runs/slm294_external_ceiling

Example (score all arm raw outputs into scoreboard.json):
  python -m scripts.run_external_ceiling --mode score \
      --output-dir outputs/runs/slm294_external_ceiling
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.external_ceiling_matrix import (
    DEFAULT_FROZEN_REQUESTS_FILE,
    DEFAULT_FROZEN_REQUESTS_SHA256,
    FRONTIER_ARMS,
    build_external_ceiling_manifest,
    render_markdown,
    run_fixture_matrix,
    run_frontier_chunk,
    run_score_pass,
    run_tiny_baseline_chunk,
    validate_external_ceiling_manifest,
)


def _run_disposition(args: Any, expected_sha: str) -> int:
    """Score, apply locked thresholds, bind evidence, emit AgentV."""
    from slm_training.harness_core.evidence_bundle import (
        ArtifactRef,
        EvidenceBundleV1,
        verify_evidence_bundle,
    )
    from slm_training.harnesses.experiments.external_ceiling_matrix import (
        compute_disposition,
    )
    from slm_training.versioning import component_version, git_commit, git_dirty

    scoreboard = run_score_pass(
        output_dir=args.output_dir,
        requests_file=args.requests_file,
        expected_sha256=expected_sha,
    )
    not_run = tuple(args.not_run or ())
    disposition = compute_disposition(scoreboard, not_run=not_run)
    out_path = args.output_dir / "disposition.json"
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(
        json.dumps(disposition, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, out_path)

    prereg = Path("docs/design/slm294-external-ceiling-preregistration.json")
    raw_dir = args.output_dir / "raw"
    envelopes = tuple(
        ArtifactRef(
            name=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            uri=str(path),
        )
        for path in sorted(raw_dir.glob("*.jsonl"))
    )
    checkpoint = None
    if args.tiny_checkpoint and args.tiny_checkpoint.exists():
        checkpoint = ArtifactRef(
            name=args.tiny_checkpoint.name,
            sha256=hashlib.sha256(args.tiny_checkpoint.read_bytes()).hexdigest(),
            size_bytes=args.tiny_checkpoint.stat().st_size,
            uri=str(args.tiny_checkpoint),
        )
    bundle = EvidenceBundleV1(
        run_id="slm294_external_ceiling",
        claim_class="diagnostic",
        source_commit=git_commit() or "UNKNOWN",
        source_dirty=git_dirty(),
        config_sha256=hashlib.sha256(prereg.read_bytes()).hexdigest(),
        seeds=(0,),
        environment_digest=hashlib.sha256(
            Path("pyproject.toml").read_bytes()
        ).hexdigest(),
        corpus_sha256=expected_sha,
        suite_hashes={
            "slm294_frozen": ArtifactRef(
                name=str(args.requests_file),
                sha256=expected_sha,
                uri=str(args.requests_file),
            )
        },
        checkpoint=checkpoint,
        raw_envelopes=envelopes,
        evaluator_version=component_version("harness.experiments.external_ceiling"),
        gate_version=component_version("gates.ship"),
        metadata={"disposition": disposition["disposition"]},
    )
    bundle_failures = verify_evidence_bundle(
        bundle, artifact_root=Path("."), store=None
    )
    bundle_path = args.output_dir / "evidence_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    agentv = None
    try:
        from slm_training.evals.agentv import publish_agentv_evaluation
        from slm_training.versioning import build_version_stamp

        cases = [
            {
                "id": f"arm_{arm}",
                "criteria": (
                    "arm evidence complete within the run policy and locked "
                    "thresholds evaluable"
                ),
                "pass": arm not in disposition["incomplete_arms"],
                "failures": []
                if arm not in disposition["incomplete_arms"]
                else ["incomplete_or_not_run"],
                "result": scoreboard["arms"].get(arm, {"n": 0}),
                "metadata": {"disposition": disposition["disposition"]},
            }
            for arm in ("A", "B", "C")
        ]
        agentv = publish_agentv_evaluation(
            args.output_dir,
            name="slm294-external-ceiling",
            claim="diagnostic",
            cases=cases,
            version_stamp=build_version_stamp("harness.experiments.external_ceiling"),
        )
    except Exception as exc:  # AgentV publication must not sink the disposition
        agentv = {"status": "agentv_publication_failed", "error": str(exc)}

    print(
        json.dumps(
            {
                "disposition": disposition,
                "evidence_bundle_sha256": bundle.sha,
                "evidence_bundle_failures": bundle_failures,
                "agentv": agentv if isinstance(agentv, dict) else {"status": "ok"},
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-108 external constrained-decoding semantic ceiling matrix"
    )
    parser.add_argument(
        "--mode",
        choices=("fixture", "frontier", "score", "disposition"),
        default="fixture",
        help=(
            "fixture runs a torch-free wiring check; frontier generates raw "
            "outputs for one arm in resumable chunks; score (re)scores all "
            "arm raw JSONLs into scoreboard.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm108_external_ceiling"),
    )
    parser.add_argument(
        "--requests-file",
        type=Path,
        default=DEFAULT_FROZEN_REQUESTS_FILE,
        help="Frozen SLM-294 requests JSONL (sha256 verified before use)",
    )
    parser.add_argument(
        "--requests-sha256",
        default=None,
        help="Expected sha256 of --requests-file (defaults to the pinned freeze)",
    )
    parser.add_argument(
        "--arm",
        choices=("A",) + tuple(sorted(FRONTIER_ARMS)),
        default=None,
        help="Frontier arm to execute (A: tiny baseline via --tiny-checkpoint, B: SmolLM2-135M, C: Qwen2.5-7B)",
    )
    parser.add_argument(
        "--tiny-checkpoint",
        type=Path,
        default=None,
        help="Local checkpoint for arm A (tiny baseline), e.g. outputs/runs/slm294_tiny_baseline/checkpoints/last.pt",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/data/train/slm230_symbol_only_v1"),
        help="Train dir used to rebuild the arm-A plugin (config lineage)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/data/eval/v1"),
        help="Eval dir for the arm-A plugin config",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Requests processed per frontier invocation (run-cap chunking)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Greedy generation cap per request (preregistered at 256)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Model dtype for frontier arms (default: float32 for B, bfloat16 for C)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Alias for --mode score: (re)score existing raw JSONLs only",
    )
    parser.add_argument(
        "--checkpoint-reference-uri",
        default=None,
        help="Durable checkpoint reference for the tiny-SLM baseline arm A",
    )
    parser.add_argument(
        "--tiny-slm-run-id",
        default=None,
        help="Run id of the tiny-SLM baseline to compare against",
    )
    parser.add_argument(
        "--not-run",
        action="append",
        default=None,
        help="Arm declared not_run for the disposition (repeatable, e.g. --not-run B --not-run C)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print resolved manifest without running",
    )
    args = parser.parse_args(argv)

    expected_sha = args.requests_sha256 or DEFAULT_FROZEN_REQUESTS_SHA256

    if args.mode == "disposition":
        return _run_disposition(args, expected_sha)

    if args.mode == "score" or args.score_only:
        try:
            scoreboard = run_score_pass(
                output_dir=args.output_dir,
                requests_file=args.requests_file,
                expected_sha256=expected_sha,
            )
        except ValueError as exc:
            print(f"score error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(scoreboard["arms"], indent=2, sort_keys=True))
        return 0

    if args.mode == "frontier":
        if args.arm is None:
            print("frontier mode requires --arm {A,B,C}", file=sys.stderr)
            return 1
        try:
            if args.arm == "A":
                if args.tiny_checkpoint is None:
                    print(
                        "arm A requires --tiny-checkpoint <path to last.pt>",
                        file=sys.stderr,
                    )
                    return 1
                summary = run_tiny_baseline_chunk(
                    output_dir=args.output_dir,
                    checkpoint=args.tiny_checkpoint,
                    train_dir=args.train_dir,
                    test_dir=args.test_dir,
                    requests_file=args.requests_file,
                    expected_sha256=expected_sha,
                    chunk_size=args.chunk_size,
                )
            else:
                summary = run_frontier_chunk(
                    arm=args.arm,
                    output_dir=args.output_dir,
                    requests_file=args.requests_file,
                    expected_sha256=expected_sha,
                    chunk_size=args.chunk_size,
                    dtype=args.dtype,
                    max_new_tokens=args.max_new_tokens,
                )
        except ValueError as exc:
            print(f"frontier error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    manifest = build_external_ceiling_manifest(
        tiny_slm_run_id=args.tiny_slm_run_id,
        checkpoint_reference_uri=args.checkpoint_reference_uri,
    )
    errors = validate_external_ceiling_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    if args.describe:
        print(manifest.to_dict())
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_fixture_matrix(
        manifest,
        run_id="slm108_fixture",
        output_dir=args.output_dir,
    )

    markdown = render_markdown(report)
    (args.output_dir / "external_ceiling_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
