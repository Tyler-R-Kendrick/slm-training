"""EFS2-04 wiring fixture: cached cheap-to-expensive verifier cascade.

Runs a small set of synthetic OpenUI programs through a conservative cascade
built from the existing G0-G12 gate stack plus a mock expensive semantic stage.
Reports pruning, stage reach, and cache-hit savings. This is evidence-only
wiring: no checkpoint is loaded and no real model decode is run.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataclasses import replace

from slm_training.evals.verifier_cascade import (
    Verdict,
    VerifierCache,
    VerifierCascade,
    VerifierResultV1,
    VerifierStage,
    VerifierStageSpec,
    default_openui_cascade,
)
from slm_training.versioning import build_version_stamp


def _semantic_check(source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
    """Mock expensive whole-program semantic check.

    The inline ``# BAD`` marker is a stand-in for an expensive verifier that
    discovers a structured-contract violation the cheap gates cannot see.
    """
    if "BAD" in source:
        return VerifierResultV1(
            stage_id="semantic",
            status=Verdict.FAIL,
            sound=True,
            reason="structured-contract violation marker",
        )
    return VerifierResultV1(stage_id="semantic", status=Verdict.PASS)


def _make_cascade(cache: VerifierCache) -> VerifierCascade:
    base = default_openui_cascade(cache=cache)
    # If any cheap stage fails soundly, the expensive semantic stage must be
    # skipped just like the later gate stages.
    base_stages: list[VerifierStage] = []
    for stage in base.stages:
        new_spec = replace(
            stage.spec,
            skip_stages_on_fail=(*stage.spec.skip_stages_on_fail, "semantic"),
        )
        base_stages.append(VerifierStage(new_spec, stage.evaluate))

    semantic = VerifierStage(
        VerifierStageSpec(
            stage_id="semantic",
            version="1",
            name="expensive_semantic_check",
            sound_fail=True,
            cache_policy="exact",
            cost_hint=100.0,
            reason_schema="contract_marker",
        ),
        _semantic_check,
    )
    return VerifierCascade(
        [*base_stages, semantic],
        cache=cache,
        pack_version=base.pack_version,
    )


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
        default=Path("outputs/runs/efs2-04-verifier-cascade"),
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, str]] = [
        ("empty", ""),
        ("unclosed", "root = Stack(["),
        ("duplicate_binder", "x = Stack([])\nx = Stack([])"),
        ("valid_minimal", "root = Stack([])"),
        ("bad_contract_marker", "root = Stack([]) # BAD"),
        ("valid_minimal_repeat", "root = Stack([])"),  # exercises cache hit
    ]

    cache = VerifierCache()
    cascade = _make_cascade(cache)
    # Use a fresh cache for the flat baseline so the comparison shows the cost
    # of running every stage, not the cost after the cascade has warmed a cache.
    flat_cascade = _make_cascade(VerifierCache())

    per_record: list[dict[str, Any]] = []
    cascade_cost = 0.0
    flat_cost = 0.0
    expensive_calls = 0
    expensive_skipped = 0

    for candidate_id, source in candidates:
        cascade_result = cascade.evaluate(candidate_id, source)
        flat_result = flat_cascade.evaluate_flat(candidate_id, source)
        cascade_cost += cascade_result.total_cost
        flat_cost += flat_result.total_cost

        semantic_cascade = next(
            (r for r in cascade_result.results if r.stage_id == "semantic"), None
        )
        if semantic_cascade and not semantic_cascade.skipped:
            expensive_calls += 1
        else:
            expensive_skipped += 1

        per_record.append(
            {
                "candidate_id": candidate_id,
                "source": source,
                "cascade": cascade_result.to_dict(),
                "flat": flat_result.to_dict(),
                "prune_equivalent": cascade_result.pruned == flat_result.pruned,
            }
        )

    result: dict[str, Any] = {
        "version_stamp": build_version_stamp("evals.scoring", "evals.loss_suite"),
        "run_id": run_id,
        "schema": "efs2-04/verifier_cascade_fixture/v1",
        "claim_class": "diagnostic",
        "cascade_config": cascade.to_dict(),
        "per_record": per_record,
        "summary": {
            "n_candidates": len(candidates),
            "cascade_total_cost": cascade_cost,
            "flat_total_cost": flat_cost,
            "cost_ratio": cascade_cost / flat_cost if flat_cost > 0 else None,
            "expensive_calls": expensive_calls,
            "expensive_skipped": expensive_skipped,
            "cache_entries": len(cache.entries),
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
        },
        "honest_caveats": [
            "Synthetic programs only; no checkpoint or live compiler decode was run.",
            "The expensive 'semantic' stage is a marker-based mock, not a real contract verifier.",
            "Cost hints are abstract units, not measured wall time.",
            "A production EFS2-04 run must use durable checkpoints, real DSL packs, and matched flat/cascade arms.",
        ],
    }

    json_path = out_dir / "verifier_cascade_fixture.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_safe_json(result), fh, indent=2, sort_keys=True)

    md_path = out_dir / "README.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# EFS2-04 verifier-cascade wiring fixture ({run_id})\n\n")
        fh.write("Wiring-only diagnostic run. See `verifier_cascade_fixture.json` for full results.\n\n")
        fh.write("## Summary\n\n")
        summary = result["summary"]
        fh.write(f"- candidates: {summary['n_candidates']}\n")
        fh.write(f"- cascade cost: {summary['cascade_total_cost']}\n")
        fh.write(f"- flat cost: {summary['flat_total_cost']}\n")
        fh.write(f"- cost ratio: {summary['cost_ratio']}\n")
        fh.write(f"- expensive calls: {summary['expensive_calls']}\n")
        fh.write(f"- expensive skipped: {summary['expensive_skipped']}\n")
        fh.write(f"- cache entries: {summary['cache_entries']}\n")
        fh.write(f"- cache hits: {summary['cache_hits']}\n\n")
        fh.write("## Per-candidate cascade status\n\n")
        for rec in per_record:
            cascade = rec["cascade"]
            fh.write(
                f"- **{rec['candidate_id']}**: pruned={cascade['pruned']} "
                f"cost={cascade['total_cost']} final={cascade['final_status']}\n"
            )
        fh.write("\n## Honest caveats\n\n")
        for note in result["honest_caveats"]:
            fh.write(f"- {note}\n")
        fh.write(f"\nArtifact: `{json_path}`\n")

    print(f"Wrote EFS2-04 fixture to {out_dir}")


if __name__ == "__main__":
    main()
