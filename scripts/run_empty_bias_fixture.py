"""EFS1-03 wiring fixture: score-policy comparison for empty-length bias.

Builds synthetic candidate paths that mimic the A1 emptiness probe (short empty
program vs longer populated program), runs the five declared score policies,
and writes a version-stamped result bundle. This is evidence-only wiring: no
checkpoint is loaded and no real decode is run.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.score_policy import (
    CandidatePath,
    ContentFloorPolicy,
    GrammarAlignedMassPolicy,
    MinimumMassRemaskPolicy,
    RawCumulativePolicy,
    SemanticLengthNormPolicy,
    compare_policies,
)
from slm_training.versioning import build_version_stamp


def _make_fixture_paths() -> tuple[CandidatePath, CandidatePath]:
    """Synthetic empty and populated candidates.

    The empty candidate has fewer decisions but the same per-decision model
    log-probability as the populated candidate, so raw cumulative ranking
    prefers the empty candidate. This exposes the length bias.
    """
    empty = CandidatePath(
        candidate_id="empty",
        token_ids=(0, 1),
        log_probs=(-1.0, -1.0),
        removed_mass=(0.1, 0.1),
        semantic_mask=(0.0, 0.0),
    )
    populated = CandidatePath(
        candidate_id="populated",
        token_ids=(0, 1, 2, 3, 4),
        log_probs=(-1.0, -1.0, -1.0, -1.0, -1.0),
        removed_mass=(0.05, 0.05, 0.05, 0.05, 0.05),
        semantic_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    return empty, populated


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/efs1-03-empty-length-bias"),
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    empty, populated = _make_fixture_paths()
    policies = [
        RawCumulativePolicy(),
        SemanticLengthNormPolicy(alpha=1.0),
        GrammarAlignedMassPolicy(beta=1.0),
        MinimumMassRemaskPolicy(gamma=0.5),
        ContentFloorPolicy(min_semantic_decisions=1),
    ]
    comparison = compare_policies([empty, populated], policies)

    result: dict[str, Any] = {
        "version_stamp": build_version_stamp("evals.scoring", "evals.loss_suite"),
        "run_id": run_id,
        "schema": "efs1-03/empty_bias_fixture/v1",
        "claim_class": "diagnostic",
        "candidates": [empty.to_dict(), populated.to_dict()],
        "policies": [p.to_dict() for p in policies],
        "rankings": comparison["rankings"],
        "scores": comparison["scores"],
        "honest_caveats": [
            "Synthetic candidates only; no checkpoint or real compiler decode was run.",
            "Removed mass and semantic masks are hand-set for illustration.",
            "A production EFS1-03 run must load durable frontier checkpoints and use live legal action sets.",
        ],
    }

    json_path = out_dir / "empty_bias_fixture.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_safe_json(result), fh, indent=2, sort_keys=True)

    md_path = out_dir / "README.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# EFS1-03 empty-length-bias score-policy fixture ({run_id})\n\n")
        fh.write("Wiring-only diagnostic run. See `empty_bias_fixture.json` for full metrics.\n\n")
        fh.write("## Policy rankings\n\n")
        for name, ranking in comparison["rankings"].items():
            fh.write(f"- **{name}**: {' > '.join(ranking)}\n")
        fh.write("\n## Honest caveats\n\n")
        for note in result["honest_caveats"]:
            fh.write(f"- {note}\n")
        fh.write(f"\nArtifact: `{json_path}`\n")

    print(f"Wrote EFS1-03 fixture to {out_dir}")


if __name__ == "__main__":
    main()
