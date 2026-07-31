"""Run the anti-E237 semantic-factor frontier fixture campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return sha, dirty
    except Exception:
        return "0" * 40, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/semantic_factor_frontier_fixture"),
    )
    parser.add_argument(
        "--docs-json",
        type=Path,
        default=Path("docs/design/semantic-factor-frontier-results.json"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override campaign seeds (default: campaign.v1.json seeds).",
    )
    args = parser.parse_args(argv)

    from slm_training.harnesses.experiments.anti_e237_semantic_factor_frontier import (
        run_semantic_factor_frontier,
    )
    from slm_training.harnesses.experiments.semantic_factor_config import load_sff_campaign
    from slm_training.harnesses.experiments.semantic_factor_metrics import (
        ScoreboardMetricsError,
        validate_scoreboard_metrics,
    )

    sha, dirty = _git_sha()
    seeds = tuple(args.seeds) if args.seeds is not None else load_sff_campaign().seeds
    payload = run_semantic_factor_frontier(
        seeds=seeds,
        out_dir=args.out_dir,
        source_commit=sha if len(sha) == 40 else "0" * 40,
        source_dirty=dirty,
    )
    # Double-check before durable write (harness already validates).
    try:
        validate_scoreboard_metrics(payload)
    except ScoreboardMetricsError as exc:
        print(str(exc), flush=True)
        return 2
    args.docs_json.parent.mkdir(parents=True, exist_ok=True)
    args.docs_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {
            "claim_class": payload["claim_class"],
            "mechanism_causally_useful": payload["mechanism_causally_useful"],
            "campaign_digest": payload["campaign_lock"]["manifest_sha256"],
            "docs_json": str(args.docs_json),
            "out_dir": str(args.out_dir),
            "runtime": payload.get("runtime"),
            "claims_summary": payload.get("claims_summary"),
            "arms": [
                {
                    "arm_id": a["arm_id"],
                    "ranking_accuracy": a["ranking_accuracy"],
                    "mrr": a.get("mrr"),
                    "applications": a["applications"],
                    "choice_changes": a["choice_changes"],
                    "wall_ms_mean": a.get("wall_ms_mean"),
                    "wall_ms_p50": a.get("wall_ms_p50"),
                    "wall_ms_p95": a.get("wall_ms_p95"),
                    "wall_ms_mean_vs_control": a.get("wall_ms_mean_vs_control"),
                    "quality_per_ms": a.get("quality_per_ms"),
                    "delta_accuracy_per_extra_ms": a.get("delta_accuracy_per_extra_ms"),
                    "kill": a["kill_criteria_hit"],
                }
                for a in payload["arms"]
            ],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
