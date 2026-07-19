"""EFS2-02 wiring fixture: observe-only trigger telemetry across decode regimes.

Synthesizes small decision-step trajectories for greedy, temperature-sampled,
and beam decode regimes, runs the observe-only ``TriggerObserver``, and writes
raw observations plus a firing-rate summary.

This is evidence-only wiring: no checkpoint is loaded, no model is run, and no
quality or ship claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.search_trigger_telemetry import (
    DecisionStep,
    TriggerRegime,
    TriggerThresholdManifest,
    compare_trigger_regimes,
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _synthetic_examples() -> list[tuple[str, list[DecisionStep], bool, bool]]:
    """Return a small deterministic set of trajectories with varied triggers."""
    examples: list[tuple[str, list[DecisionStep], bool, bool]] = []

    # Example 1: repeated state -> STAGNATION trigger.
    stagnation_steps = [
        DecisionStep(
            state_fingerprint="stag",
            decision_depth=1,
            live_action_scores=(0.9, 0.1),
            certified_reductions=0,
            value_score=0.1 * i,
            verifier_calls=0,
            model_forwards=i,
            wall_ms=float(i),
        )
        for i in range(5)
    ]
    examples.append(("stagnation_path", stagnation_steps, False, True))

    # Example 2: hard conflict at the final step -> BOTTOM retraction event.
    bottom_steps = [
        DecisionStep(
            state_fingerprint="ok",
            decision_depth=1,
            live_action_scores=(0.8, 0.2),
            certified_reductions=1,
            value_score=0.6,
            model_forwards=1,
            wall_ms=1.0,
        ),
        DecisionStep(
            state_fingerprint="conflict",
            decision_depth=2,
            live_action_scores=(0.0,),
            certified_reductions=0,
            value_score=0.0,
            model_forwards=2,
            wall_ms=2.0,
            is_bottom=True,
            pending_conflict_reason="unsatisfiable_binding",
        ),
    ]
    examples.append(("bottom_path", bottom_steps, False, False))

    # Example 3: low-margin, high-entropy steps -> UNCERTAINTY trigger.
    uncertainty_steps = [
        DecisionStep(
            state_fingerprint=f"u{i}",
            decision_depth=i + 1,
            live_action_scores=(0.51, 0.50),
            certified_reductions=0,
            value_score=0.55 - 0.02 * i,
            verifier_calls=i,
            model_forwards=i + 1,
            wall_ms=float(i + 1),
        )
        for i in range(4)
    ]
    examples.append(("uncertainty_path", uncertainty_steps, True, False))

    # Example 4: budget pressure.
    budget_steps = [
        DecisionStep(
            state_fingerprint="bp",
            decision_depth=i + 1,
            live_action_scores=(0.8, 0.2),
            certified_reductions=1,
            value_score=0.7,
            verifier_calls=0,
            model_forwards=i + 1,
            wall_ms=float(i + 1),
        )
        for i in range(8)
    ]
    examples.append(("budget_path", budget_steps, False, True))

    return examples


def _summarize(result: Any) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for run in result.runs:
        key = f"{run.regime.value}/{run.example_id}"
        total = len(run.observations)
        fired = sum(1 for o in run.observations if o.triggered)
        by_predicate: dict[str, int] = {}
        for o in run.observations:
            by_predicate[o.predicate.value] = by_predicate.get(o.predicate.value, 0) + 1
        rows[key] = {
            "regime": run.regime.value,
            "example_id": run.example_id,
            "observations": total,
            "fired": fired,
            "firing_rate": fired / total if total else 0.0,
            "by_predicate": by_predicate,
            "final_pass": (
                run.observations[-1].outcome_final_pass if run.observations else None
            ),
            "recoverable": (
                run.observations[-1].outcome_recoverable if run.observations else None
            ),
        }

    regime_rates: dict[str, list[float]] = {}
    for run in result.runs:
        regime_rates.setdefault(run.regime.value, []).append(run.firing_rate())

    return {
        "rows": rows,
        "regime_mean_firing_rate": {
            regime: sum(rates) / len(rates) if rates else 0.0
            for regime, rates in regime_rates.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/efs2-02-trigger-telemetry"),
    )
    parser.add_argument(
        "--repeat-window",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--no-progress-window",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--margin-quantile",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--entropy-quantile",
        type=float,
        default=0.75,
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = TriggerThresholdManifest(
        repeat_window=args.repeat_window,
        no_progress_window=args.no_progress_window,
        margin_quantile=args.margin_quantile,
        entropy_quantile=args.entropy_quantile,
        budget_pressure_forward_limit=5,
    )

    result = compare_trigger_regimes(
        _synthetic_examples(),
        thresholds=thresholds,
        regimes=(TriggerRegime.GREEDY, TriggerRegime.TEMPERATURE, TriggerRegime.BEAM),
        seed=2026,
        stamp_components=("evals.scoring",),
    )

    summary = {
        "run_id": run_id,
        "fixture": "efs2-02-trigger-telemetry",
        "thresholds": thresholds.to_dict(),
        "summary": _summarize(result),
        "version_stamp": _safe_json(result.version_stamp),
    }

    json_path = out_dir / "trigger_telemetry.json"
    json_path.write_text(
        json.dumps(_safe_json(result.to_dict()), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = "\n".join(
        f"| {key} | {d['observations']} | {d['fired']} | {d['firing_rate']:.2f} | {d['final_pass']} | {d['recoverable']} |"
        for key, d in sorted(summary["summary"]["rows"].items())
    )
    regime_rows = "\n".join(
        f"| {regime} | {rate:.3f} |"
        for regime, rate in sorted(summary["summary"]["regime_mean_firing_rate"].items())
    )

    readme = f"""# EFS2-02 Observe-Only Trigger Telemetry Fixture

Run ID: `{run_id}`

This fixture exercises the SLM-112 observe-only trigger telemetry harness over
four synthetic decision trajectories and three decode regimes (greedy,
temperature, beam).  The trigger observer records events but never branches,
remasks, or otherwise changes generation.

## Threshold manifest

```json
{json.dumps(thresholds.to_dict(), indent=2, sort_keys=True)}
```

## Per-run firing summary

| Run | Observations | Fired | Firing rate | Final pass | Recoverable |
| --- | ---: | ---: | ---: | --- | --- |
{rows}

## Mean firing rate by regime

| Regime | Mean firing rate |
| --- | ---: |
{regime_rows}

## Artifacts

* `trigger_telemetry.json` — full raw observations and version stamp.
* `summary.json` — aggregate firing rates by run/regime.

## Honest caveats

Wiring-only evidence with hand-designed synthetic decision sequences.  A real
EFS2-02 Phase A run requires durable checkpoints, the actual compiler-tree
decoder, validation-selected thresholds frozen before test analysis, and
binding-aware semantic outcome labels across standard suites.
"""
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    print(f"EFS2-02 fixture written to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
