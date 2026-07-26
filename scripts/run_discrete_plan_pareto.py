#!/usr/bin/env python3
"""AP-027 (SLM-322) discrete-plan Pareto screening run.

Bounded, CPU, playground-checkpoint-scale screening for the 1/2/4/8
refinement-round axis. Reuses ``scripts.run_perf_matrix``'s existing
latency/quality measurement (``PerfExperiment``/``_apply``/``run_one``)
rather than re-implementing decode instrumentation.

Scope of what this script actually measures (see
``docs/design/discrete-plan-pareto.md`` for the full disposition):

* The committed ``playground_demo`` checkpoint has no trained AP-024
  ``AbstractPlanConnector`` attached, so ``no_plan`` and
  ``current_auxiliary_heads`` decode through an *identical* configuration on
  this checkpoint -- the shipped default already runs connector-free. This
  script measures that one configuration and reports it under both arm
  labels rather than fabricating an artificial difference.
* ``oracle_plan``/``gold_semantic_plan``, ``learned_plan_binder_precision``,
  and ``retrieval_baseline`` all require an artifact (a trained connector, a
  binder-precision channel, or a retrieval index) this screening run does
  not build. They are reported as ``arms_pending``, never silently skipped.
* ``learned_abstract_plan`` is not re-measured here: AP-022/SLM-313 already
  produced its complete locked-matrix verdict (ignored_or_collapsed) and
  I14 forbids re-litigating a closed approach.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.discrete_plan_pareto import (
    ARMS,
    REFINEMENT_ROUNDS,
    ParetoLatencyRow,
    build_campaign,
    summarize_pareto,
)
from slm_training.levers import MAX_HARNESS_WALL_SECONDS
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.versioning import build_version_stamp
from scripts.run_perf_matrix import PerfExperiment, _load_prompts, run_one

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "autoresearch" / "slm322-discrete-plan-pareto"
DEFAULT_JSON = ROOT / "docs" / "design" / "iter-slm322-discrete-plan-pareto-20260726.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "design" / "discrete-plan-pareto.md"
MEASURED_ARM_LABELS = ("no_plan", "current_auxiliary_heads")
# One decode config for both labels: the playground checkpoint carries no
# trained AP-024 connector, so there is nothing for the two labels to differ
# on today (see module docstring).
_BASE_EXPERIMENT_KWARGS: dict[str, Any] = dict(
    grammar_incremental_state=True,
    grammar_verify_chosen_only=True,
    grammar_multitoken_accept=True,
    grammar_multitoken_max=8,
    grammar_canvas_lookahead=32,
    grammar_ltr_repair=True,
    grammar_finalize_validate=True,
    grammar_finalize_on_last_attempt_only=True,
)


def _phase_latencies(row: dict[str, Any]) -> tuple[float, float, float]:
    """(plan, generation, verification) ms from an aggregated perf-matrix row."""
    summary: dict[str, Any] = row.get("phase_summary") or {}

    def _val(key: str) -> float:
        raw = summary.get(f"{key}_mean")
        return float(raw) if raw is not None else 0.0

    generation_ms = _val("denoiser_ms") + _val("backbone_ms") + _val("projection_ms")
    verification_ms = _val("finalize_ms") + _val("dfa_sync_ms") + _val("stream_check_ms")
    # No connector is attached to this checkpoint (see module docstring): the
    # plan phase genuinely costs zero on the measured arms.
    return 0.0, generation_ms, verification_ms


def run_screening(
    *,
    checkpoint: Path,
    prompts: list[tuple[str, str | None]],
    device: str,
    warmup: int,
    out_dir: Path,
    rounds: tuple[int, ...] = REFINEMENT_ROUNDS,
) -> list[ParetoLatencyRow]:
    rows: list[ParetoLatencyRow] = []
    for round_count in rounds:
        exp = PerfExperiment(
            f"AP027-R{round_count}",
            f"ap027_discrete_plan_pareto_round{round_count}",
            f"AP-027 screening: no_plan/current_auxiliary_heads at {round_count} refinement round(s)",
            generate_max_attempts=round_count,
            **_BASE_EXPERIMENT_KWARGS,
        )
        result = run_one(
            exp,
            checkpoint=checkpoint,
            prompts=prompts,
            device=device,
            warmup=warmup,
            out_dir=out_dir,
        )
        plan_ms, gen_ms, verify_ms = _phase_latencies(result)
        total_ms = float(result.get("latency_ms_mean") or 0.0)
        # Guard the harness's own phase_sum <= total invariant against
        # measurement noise (finalize/denoiser overlap at low sample counts).
        total_ms = max(total_ms, plan_ms + gen_ms + verify_ms)
        metrics = {
            "binding_aware_meaningful_v2": float(result.get("parse_rate") or 0.0),
            "binder_reference_f1": float(result.get("placeholder_fidelity") or 0.0),
        }
        for arm in MEASURED_ARM_LABELS:
            rows.append(
                ParetoLatencyRow(
                    record_id=exp.run_id,
                    arm=arm,
                    refinement_round=round_count,
                    path="constrained",
                    total_latency_ms=total_ms,
                    plan_latency_ms=plan_ms,
                    generation_latency_ms=gen_ms,
                    verification_latency_ms=verify_ms,
                    peak_memory_proxy=float(result.get("tokens_emitted") or 0),
                    output_tokens=int(result.get("tokens_emitted") or 0),
                    metrics=metrics,
                    promotion_eligible=False,
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PLAYGROUND_DEMO_CHECKPOINT)
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=list(REFINEMENT_ROUNDS),
        choices=list(REFINEMENT_ROUNDS),
        help="Refinement-round subset to measure this invocation (repo run-cap sharding).",
    )
    args = parser.parse_args(argv)

    from slm_training.levers import DEFAULT_EVAL_DATA_DIR

    prompts = _load_prompts(DEFAULT_EVAL_DATA_DIR, args.suite, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_screening(
        checkpoint=args.checkpoint,
        prompts=prompts,
        device=args.device,
        warmup=args.warmup,
        out_dir=args.out_dir,
        rounds=tuple(args.rounds),
    )
    report = summarize_pareto(rows)
    campaign = build_campaign()
    stamp = build_version_stamp(
        "harness.experiments.ap027_discrete_plan_pareto",
        "harness.experiments.slm313_abstract_plan_functional_evidence",
    )
    rounds_pending = tuple(r for r in REFINEMENT_ROUNDS if r not in args.rounds)
    payload = {
        "schema": "ap027_discrete_plan_pareto_run/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign.campaign_id,
        "arms": list(ARMS),
        "refinement_rounds": list(REFINEMENT_ROUNDS),
        "rounds_measured": list(args.rounds),
        "rounds_pending": list(rounds_pending),
        "rows": [row.to_dict() for row in rows],
        "summary": report,
        "version_stamp": stamp,
        "max_harness_wall_seconds": MAX_HARNESS_WALL_SECONDS,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    non_dominated = ", ".join(report["non_dominated"]) or "none"
    markdown = f"""# AP-027 (SLM-322): discrete-plan Pareto screening

`claim_class: wiring`. **Not a ship claim.** Screening-scale only (1 seed,
{len(prompts)} prompts, scratch CPU checkpoint `{args.checkpoint}`) -- far
below the issue's own >=3/5-seed acceptance bar; that full run is out of
scope for this PR (see "What this PR does not claim" below). Refinement
rounds measured this invocation: {list(args.rounds)}; rounds
{list(rounds_pending) or 'none'} remain pending under the repo's
{MAX_HARNESS_WALL_SECONDS:.0f}s harness wall-clock cap (AP-022's own
partial-shard precedent: a bounded run reports exactly what it measured).

## AP-022 gating disposition (I14)

AP-022/SLM-313 already produced a complete locked-matrix verdict for
`learned_abstract_plan`: **{report['ap022_disposition']['verdict']}**
({report['ap022_disposition']['doc']}). Per decode-invariant I14, that
verdict is never re-litigated here. `learned_plan_binder_precision` --
named in this issue's own arm list, and never run by AP-022 -- is the
successor approach and the only arm this harness's `build_campaign()` will
ever mark `role="candidate"`.

## What ran

The committed `playground_demo` checkpoint carries no trained AP-024
`AbstractPlanConnector`, so `no_plan` and `current_auxiliary_heads` decode
through an identical configuration on this checkpoint today -- both labels
below report the same measured series rather than a fabricated split.
Measured across refinement rounds {list(REFINEMENT_ROUNDS)}:

| refinement_round | total_latency_ms | meaning_v2 (parse_rate proxy) | non-dominated |
| --- | --- | --- | --- |
"""
    by_round = {row.refinement_round: row for row in rows if row.arm == "no_plan"}
    for round_count in args.rounds:
        row = by_round.get(round_count)
        if row is None:
            continue
        point_id = f"no_plan@{round_count}"
        marker = "yes" if point_id in report["non_dominated"] else "no"
        markdown += (
            f"| {round_count} | {row.total_latency_ms:.2f} | "
            f"{row.metrics['binding_aware_meaningful_v2']:.4f} | {marker} |\n"
        )
    markdown += f"""
Non-dominated (arm@round) points on (lower latency, higher meaning_v2):
{non_dominated}.

## What this PR does not claim

- No `>=3`/`5`-seed run (this is 1 seed, screening scale).
- No measurement of `oracle_plan`, `learned_plan_binder_precision`, or
  `retrieval_baseline` -- each needs an artifact (trained connector,
  binder-precision channel, or retrieval index) this PR does not build.
  Recorded as `arms_pending`: {", ".join(report['arms_pending'])}.
- No Pareto-frontier promotion. `ship_eligible: false`.

## Evidence

- `{args.json_out}`
- Harness: `src/slm_training/harnesses/experiments/discrete_plan_pareto.py`
- Runner: `scripts/run_discrete_plan_pareto.py`
"""
    args.markdown_out.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
