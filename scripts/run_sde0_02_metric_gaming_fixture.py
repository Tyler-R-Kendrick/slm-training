"""SDE0-02 fixture: build and score the metric-gaming stress suite.

This script emits:

* `outputs/runs/sde0-02-metric-gaming/iter-sde0-02-<date>/summary.json`
* `docs/design/iter-sde0-02-metric-gaming-<date>.json`
* the Markdown evidence memo (written by the caller or kept in docs/)

It uses only deterministic OpenUI transforms and the existing
`binding_aware_meaningful_v2` judge.  No model, tokenizer, or GPU is required.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from slm_training.evals.metric_gaming import (
    build_all_cases,
    evaluate_all_retry_cases,
    evaluate_metric_gaming,
    write_manifest,
)
from slm_training.versioning import build_version_stamp


OUTPUT_ROOT = Path("outputs/runs/sde0-02-metric-gaming")
DOCS_JSON = Path("docs/design")


def main() -> None:
    today = date.today().isoformat().replace("-", "")
    run_dir = OUTPUT_ROOT / f"iter-sde0-02-{today}"
    docs_json = DOCS_JSON / f"iter-sde0-02-metric-gaming-{today}.json"

    cases = build_all_cases(seed=0)
    report = evaluate_metric_gaming(cases)
    retry_results = evaluate_all_retry_cases(cases)

    payload = {
        "schema_version": report.schema_version,
        "metric_name": report.metric_name,
        "metric_version": report.metric_version,
        "n_cases": report.n_cases,
        "strict_rate": report.strict_rate,
        "coverage_conditioned_rate": report.coverage_conditioned_rate,
        "false_positive_count": report.false_positive_count,
        "false_negative_count": report.false_negative_count,
        "slices": {
            name: rep.to_dict() for name, rep in report.slices.items()
        },
        "retry_sensitive": {
            "n": len(retry_results),
            "first_selected_oracle_differ": sum(
                1
                for r in retry_results
                if r["first_attempt_pass"]
                != r["selected_attempt_pass"]
                or r["first_attempt_pass"] != r["oracle_best_pass"]
            ),
            "rows": retry_results,
        },
        "version_stamp": build_version_stamp(),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    docs_json.parent.mkdir(parents=True, exist_ok=True)
    docs_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    manifest_path = run_dir / "manifest.json"
    write_manifest(report, retry_results, manifest_path)

    print(f"SDE0-02 metric-gaming fixture wrote {summary_path}")
    print(f"SDE0-02 metric-gaming fixture wrote {docs_json}")
    print(f"SDE0-02 metric-gaming fixture wrote {manifest_path}")
    print(
        "cases:",
        report.n_cases,
        "strict_rate:",
        report.strict_rate,
        "fp:",
        report.false_positive_count,
        "fn:",
        report.false_negative_count,
    )


if __name__ == "__main__":
    main()
