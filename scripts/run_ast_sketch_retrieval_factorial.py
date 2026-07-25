#!/usr/bin/env python3
"""Run AST-sketch retrieval fixtures and SLM-295's local patch diagnostic."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.retrieval.verified_patch_v1 import (
    RetrievalRequest,
    build_verified_patch_index,
    evaluate_local_baseline,
)
from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
    build_ast_sketch_retrieval_manifest,
    render_markdown,
    run_fixture_matrix,
    validate_manifest,
)
from slm_training.versioning import build_version_stamp


def _record(record_id: str, prompt: str, openui: str, split: str) -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        split=split,
        source="slm295_fixture",
        meta={"split_group_id": record_id, "lineage_id": record_id},
    )


def _fixtures() -> tuple[
    list[ExampleRecord], list[tuple[RetrievalRequest, ExampleRecord]]
]:
    train = [
        _record(
            "train-card",
            "show a title card",
            'root = Card([title])\ntitle = TextContent(":title")',
            "train",
        ),
        _record(
            "train-action",
            "show a save action",
            'root = Stack([save], "column")\nsave = Button(":save")',
            "train",
        ),
        _record(
            "train-form",
            "show an email form",
            'root = Form([email])\nemail = Input(":email")',
            "train",
        ),
    ]
    gold = _record(
        "held-out",
        "show a title and save action",
        'root = Stack([title, save], "column")\ntitle = TextContent(":title")\nsave = Button(":save")',
        "held_out",
    )
    request = RetrievalRequest(
        "held-out", gold.prompt, 'root = Stack([], "column")', "held-out", "held-out"
    )
    return train, [(request, gold)]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SLM-295 verified retrieval plus grammar-patch baseline",
        "",
        "Status: local diagnostic; non-promotable",
        "",
        "| Arm | n | Strict meaningful | Binder/ref F1 | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, values in report["arms"].items():
        lines.append(
            f"| `{arm}` | {values['n']} | {values['binding_aware_meaningful_v2_rate_strict']!s} | {values['binder_reference_f1']!s} | {values['latency_ms_p50']!s} | {values['latency_ms_p95']!s} |"
        )
    lines.extend(
        [
            "",
            "The index contains only verified training records and excludes exact, prompt, structure, split-group, and derivative overlaps. Oracle top-k is diagnostic-only. No checkpoint, human rating, remote replay, HF run, or ship claim is involved.",
        ]
    )
    return "\n".join(lines) + "\n"


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "output-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "frontier", "verified-patch-local"),
        default="plan-only",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/runs/slm295_verified_patch_v1")
    )
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--parent-checkpoint-uri", default=None)
    parser.add_argument("--checkpoint-bucket", default="hf://buckets/TKendrick/OpenUI")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "verified-patch-local":
        seeds = tuple(
            int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()
        )
        manifest = build_ast_sketch_retrieval_manifest(
            parent_checkpoint_uri=args.parent_checkpoint_uri,
            checkpoint_bucket=args.checkpoint_bucket,
            seeds=seeds,
            include_controls=args.include_controls,
        )
        errors = validate_manifest(manifest)
        if errors:
            for error in errors:
                print(f"manifest error: {error}", file=sys.stderr)
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(args.output_dir / "ast_sketch_retrieval_manifest.json")
        run_id = "slm133_plan" if args.mode == "plan-only" else "slm133_fixture"
        if args.mode == "frontier":
            print(
                "frontier mode requires GPU host and durable checkpoints; emitting fixture plan only",
                file=sys.stderr,
            )
            run_id = "slm133_frontier_partial"
        report = run_fixture_matrix(manifest, run_id=run_id, output_dir=args.output_dir)
        markdown = render_markdown(report)
        (args.output_dir / "ast_sketch_retrieval_report.md").write_text(
            markdown, encoding="utf-8"
        )
        report.to_json(args.output_dir / "ast_sketch_retrieval_report.json")
        print(markdown)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, cases = _fixtures()
    index = build_verified_patch_index(train)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report = evaluate_local_baseline(index, cases, peak_memory_bytes=peak)
    stamp = build_version_stamp(
        "harness.retrieval.verified_patch",
        "harness.experiments.ast_sketch_retrieval",
        "evals.meaningful_program",
    )
    report["version_stamp"] = stamp
    report["recipe"] = {
        "backend": "local_cpu",
        "fixture_n": len(cases),
        "index_n": len(train),
        "human_ratings": "not_used",
        "remote_replay": "not_used",
    }
    report["agentv"] = _portable(
        publish_agentv_evaluation(
            args.output_dir,
            name="slm295-local-retrieval",
            claim="local_diagnostic_non_ship",
            cases=[
                {
                    "id": "index-and-patch-verifier",
                    "criteria": "Every emitted result remains parser and verifier valid; oracle remains diagnostic-only.",
                    "pass": all(
                        row["result"]["verifier_ok"] for row in report["records"]
                    ),
                    "result": {"arms": report["arms"]},
                }
            ],
            version_stamp=stamp,
        ),
        args.output_dir,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown = _markdown(report)
    (args.output_dir / "slm295_verified_patch_v1.json").write_text(
        payload, encoding="utf-8"
    )
    (args.output_dir / "slm295_verified_patch_v1.md").write_text(
        markdown, encoding="utf-8"
    )
    if args.docs_dir:
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        (args.docs_dir / "iter-slm295-verified-patch-v1-20260725.json").write_text(
            payload, encoding="utf-8"
        )
        (args.docs_dir / "iter-slm295-verified-patch-v1-20260725.md").write_text(
            markdown, encoding="utf-8"
        )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
