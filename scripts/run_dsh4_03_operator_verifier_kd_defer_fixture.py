"""Fixture runner for SLM-388 (DSH4-03): accepted-set and verifier-backed
operator distillation with DEFER.

Wiring-only demonstration of
``slm_training.harnesses.distill.operator_verifier_kd_defer``: reuses
DSH4-02's exact fixture-generation logic (``_build_decision_state``, the
same fixed five-candidate operator and ``GOLD_WEIGHTS``) to build a
verifier-training pool, an evaluation pool, and three disjoint held-out
per-seed pools, then runs the full matched-arm comparison (hard accepted-set,
teacher argmax, offline conditional KD, verifier-ranking+KD, +DEFER) against
the nondistilled current-scorer baseline, and prints/writes the harness's
honest go/no-go/defer stop-rule verdict.

**No external teacher model is downloaded or scored.** This reuses DSH4-02's
``SyntheticDescriptorTeacherV1`` unchanged; whatever this script reports --
including a defer -- is the harness's honest finding on synthetic data,
never a production KD-readiness claim. Do not edit this fixture's gold-label
distribution or either script's arm construction to force a particular
stop-rule outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.run_dsh4_02_operator_teacher_ceiling_fixture import (
    GOLD_WEIGHTS,
    N_CANDIDATES,
    _build_decision_state,
)
from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill.operator_teacher_ceiling import (
    SyntheticDescriptorTeacherV1,
)
from slm_training.harnesses.distill.operator_verifier_kd_defer import (
    compare_operator_verifier_kd_defer,
    write_operator_verifier_kd_defer_report,
)
from slm_training.versioning import build_version_stamp

N_VERIFIER_TRAINING_STATES = 30
N_EVAL_STATES = 30
N_HELD_OUT_STATES_PER_SEED = 30
HELD_OUT_SEEDS = (0, 1, 2)


def _build_pool(prefix: str, n: int, rng: random.Random) -> list:
    return [
        _build_decision_state(
            f"trace-dsh4-03-{prefix}-{i:03d}",
            supervision=SupervisionSource.GOLD if i % 2 == 0 else SupervisionSource.ON_POLICY,
            rng=rng,
        )
        for i in range(n)
    ]


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/dsh4-03-operator-verifier-kd-defer"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--change-threshold", type=float, default=0.05)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    verifier_training_traces = _build_pool("train", N_VERIFIER_TRAINING_STATES, rng)
    eval_traces = _build_pool("eval", N_EVAL_STATES, rng)
    held_out_traces_by_seed = {
        seed: _build_pool(f"holdout{seed}", N_HELD_OUT_STATES_PER_SEED, rng)
        for seed in HELD_OUT_SEEDS
    }

    teacher = SyntheticDescriptorTeacherV1()
    report = compare_operator_verifier_kd_defer(
        eval_traces=eval_traces,
        verifier_training_traces=verifier_training_traces,
        held_out_traces_by_seed=held_out_traces_by_seed,
        teacher=teacher,
        change_threshold=args.change_threshold,
        n_bootstrap=args.n_bootstrap,
    )

    try:
        version_stamp = build_version_stamp("harness.distill")
    except Exception:
        version_stamp = {"stamp_schema": "version_stamp/v1", "note": "unavailable"}

    payload = report.to_dict()
    payload["version_stamp"] = version_stamp
    payload["run_id"] = run_id
    payload["fixture"] = "dsh4-03-operator-verifier-kd-defer"
    payload["gold_weights"] = list(GOLD_WEIGHTS)
    payload["n_candidates"] = N_CANDIDATES
    payload["n_verifier_training_states"] = N_VERIFIER_TRAINING_STATES
    payload["n_eval_states"] = N_EVAL_STATES
    payload["n_held_out_states_per_seed"] = N_HELD_OUT_STATES_PER_SEED
    payload["held_out_seeds"] = list(HELD_OUT_SEEDS)

    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps(_safe_json(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {json_path}")

    write_operator_verifier_kd_defer_report(out_dir / "report.json", report)
    print(f"wrote {out_dir / 'report.json'}")

    verdict = report.stop_rule
    print(f"stop_rule.go = {verdict.go} (recommendation: {'go' if verdict.go else 'no_go_defer'})")
    for reason in verdict.reasons:
        print(f"  reason: {reason}")

    print("\nper-arm summary (vs nondistilled baseline):")
    for arm_id, arm_report in report.arms.items():
        change = arm_report.decision_change_vs_baseline
        print(
            f"  {arm_id:24s} arm_go={arm_report.arm_go!s:5s} "
            f"changed={change.n_changed}/{change.n_eligible} "
            f"correct={change.n_correct_changes} wrong={change.n_wrong_changes} "
            f"identical_rejected={change.prediction_identical_rejected} "
            f"cap2_proxy_all_seeds={arm_report.cap2_proxy_improves_all_seeds}"
        )
        if arm_id == "verifier_ranking_kd_defer":
            print(
                f"    defer_rate={arm_report.structural.defer_rate} "
                f"reasons={dict(arm_report.structural.defer_reasons)}"
            )


if __name__ == "__main__":
    main()
