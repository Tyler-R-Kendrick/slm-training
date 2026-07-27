#!/usr/bin/env python3
"""Publish AP-018 (SLM-306) bottleneck warm-up records from SemanticPlanV1.

Builds ``[prompt; privileged_plan; abstract_span; target]`` warm-up records
from the existing decontaminated ``openui_verified_v1`` corpus, excluding any
row already reserved by the ``abstract_planning_locked_v1`` promotion holdout,
and writes the committed data artifact plus the paired docs/design report
(Iron Law: no data build without a matching docs/design pair).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.data.abstract_cot.warmup import (
    AbstractCotWarmupConfig,
    build_abstract_cot_warmup_corpus,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.harness_core.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS_PATH = (
    ROOT
    / "src/slm_training/resources/data/train/openui_verified_v1/records.jsonl"
)
DEFAULT_OUT_PATH = ROOT / "src/slm_training/resources/data/abstract_cot/warmup_v1.jsonl"
DEFAULT_DESIGN_JSON = ROOT / "docs/design/ap-018-abstract-cot-warmup.json"
DEFAULT_DESIGN_MD = ROOT / "docs/design/ap-018-abstract-cot-warmup.md"


def _load_records(path: Path) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        records.append(
            ExampleRecord(
                id=str(row["id"]),
                prompt=str(row.get("prompt") or ""),
                openui=str(row.get("openui") or ""),
                placeholders=list(row.get("placeholders") or []),
                split=str(row.get("split") or "train"),
                source=str(row.get("source") or "fixture"),
                meta=dict(row.get("meta") or {}),
                design_md=row.get("design_md"),
            )
        )
    return records


def _markdown(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    reasons: dict[str, int] = {}
    for item in corpus["exclusions"]:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    lines = [
        "# AP-018 (SLM-306): bottleneck warm-up records from SemanticPlanV1 and verbal plans",
        "",
        f"Date: {report['date']}",
        "Scope: data build only -- no training run, no checkpoint, no ship claim.",
        "",
        "## Decision",
        "",
        "Build privileged-plan warm-up examples for the Abstract-CoT causal pilot",
        "(AP2) while forcing the eventual generation head to depend only on the",
        "prompt plus abstract codebook tokens, never directly on the privileged",
        "plan. This issue builds the data and its typed bottleneck contract",
        "(`SegmentLayoutV1`); AP-019 implements the model-side masked attention",
        "that enforces it.",
        "",
        "## Source corpus",
        "",
        f"- input: `{report['input_path']}`",
        f"- eligible rows: {corpus['eligible_count']}",
        f"- accepted rows (valid plan derived): {corpus['accepted_count']}",
        f"- acceptance rate: {corpus['acceptance_rate']:.4f}",
        f"- emitted rows (committed artifact, balanced by binder bucket): {corpus['emitted_count']}",
        "",
        "## Exclusion ledger (nothing dropped silently)",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    for reason in sorted(reasons):
        lines.append(f"| `{reason}` | {reasons[reason]} |")
    if not reasons:
        lines.append("| (none) | 0 |")
    lines.extend(
        [
            "",
            "## Binder/reference complexity balance",
            "",
            "| Bucket (binding count) | Accepted rows |",
            "| --- | ---: |",
        ]
    )
    for bucket, count in sorted(corpus["binder_bucket_counts"].items()):
        lines.append(f"| `{bucket}` | {count} |")
    lines.extend(
        [
            "",
            "## Acceptance criteria",
            "",
            f"- Every eligible row received a valid plan or a typed exclusion reason: "
            f"{corpus['eligible_count']} == {corpus['accepted_count']} + {len(corpus['exclusions'])} "
            f"({'yes' if corpus['eligible_count'] == corpus['accepted_count'] + len(corpus['exclusions']) else 'no'}).",
            f"- >=90% of eligible verified rows received a valid plan: "
            f"{'yes' if corpus['acceptance_rate'] >= 0.9 else 'no'} ({corpus['acceptance_rate']:.4f}).",
            "- Iteration-0 abstract spans are deterministic (record-id + fixed seed "
            "keyed RNG) and leak no plan/target content -- see "
            "`tests/test_data/test_abstract_cot_warmup.py::test_iteration0_abstract_span_is_deterministic_across_runs`.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m scripts.publish_abstract_cot_warmup",
            "pytest -q tests/test_data/test_abstract_cot_warmup.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--design-json", type=Path, default=DEFAULT_DESIGN_JSON)
    parser.add_argument("--design-md", type=Path, default=DEFAULT_DESIGN_MD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=200,
        help="Cap on emitted (committed) rows; coverage stats always cover the full input.",
    )
    parser.add_argument("--date", default="2026-07-25")
    args = parser.parse_args(argv)

    records = _load_records(args.records)
    config = AbstractCotWarmupConfig(seed=args.seed, max_records=args.max_records)
    corpus = build_abstract_cot_warmup_corpus(records, config)
    corpus.write_jsonl(args.out)

    stamp = build_version_stamp("data.abstract_cot_warmup")
    report = {
        "schema": "ap018_abstract_cot_warmup_report/v1",
        "date": args.date,
        "input_path": str(args.records.relative_to(ROOT)),
        "output_path": str(args.out.relative_to(ROOT)),
        "run": {
            "kind": "data_build",
            "device": "none",
            "steps": 0,
            "checkpoint": None,
            "ship_claim": False,
        },
        "corpus": corpus.to_dict(),
        "version_stamp": stamp,
    }
    args.design_json.parent.mkdir(parents=True, exist_ok=True)
    args.design_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.design_md.write_text(_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "eligible_count": corpus.eligible_count,
                "accepted_count": corpus.accepted_count,
                "emitted_count": corpus.emitted_count,
                "acceptance_rate": corpus.acceptance_rate,
                "out": str(args.out),
                "design_json": str(args.design_json),
                "design_md": str(args.design_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
