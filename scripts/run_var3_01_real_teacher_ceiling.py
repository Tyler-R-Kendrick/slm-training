"""VAR3-01 (SLM-429): re-run the DSH4-02 ceiling comparison against a real teacher.

Wires ``PromptedTeacherV1`` (a real prompted LLM behind the existing
``OperatorTeacherAdapter`` Protocol -- **no harness change**) into the exact
``compare_operator_teacher_ceiling`` harness, over the exact DSH4-02 fixture
trace pools (same ``GOLD_WEIGHTS`` latent distribution fixed *before* the
run, same seed, same train/eval split as
``run_dsh4_02_operator_teacher_ceiling_fixture.py``), with
``SyntheticDescriptorTeacherV1`` kept as a control arm.

Teacher exchanges are content-addressed under the run's ``teacher_cache/``
so the comparison re-runs without re-querying. The provider warm-up is
concurrent only to stay inside the repo hard run cap
(``levers.MAX_RUN_MINUTES``); cache keys make it identical to a sequential
warm-up.

Claim class: ``experiment``. Whatever ``evaluate_stop_rule`` reports over
the real teacher output -- including ``no_go_defer`` -- is the honest
result; nothing in this script tunes gold labels, the stop rule, the
decision families, or the perturbation set after seeing teacher output.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill import operator_teacher_ceiling as otc
from slm_training.harnesses.distill.prompted_teacher import (
    PromptedTeacherV1,
    prefetch_queries,
)
from slm_training.versioning import build_version_stamp

# Reuse -- never duplicate -- the DSH4-02 fixture's certified trace builders
# and its preregistered latent gold-label distribution.
from scripts.run_dsh4_02_operator_teacher_ceiling_fixture import (
    GOLD_WEIGHTS,
    N_EVAL_STATES,
    N_TRAIN_STATES,
    _build_decision_state,
    _safe_json,
)

PERTURBATION_FNS = {
    otc.PerturbationKind.CANDIDATE_ORDER: otc.perturb_candidate_order,
    otc.PerturbationKind.OPAQUE_ID_RELABEL: otc.perturb_opaque_id_labels,
    otc.PerturbationKind.DESCRIPTION_LENGTH: otc.perturb_description_length,
    otc.PerturbationKind.PROMPT_TEMPLATE: otc.perturb_prompt_template,
}


def _prefetch_queries(
    eval_traces, *, perturbation_seed: int = 0
) -> list[otc.TeacherQueryV1]:
    """Every query the unchanged harness will issue: the canonical ranking
    queries plus the perturbation-robustness queries (same seed arithmetic
    as ``run_perturbation_robustness``)."""
    queries: list[otc.TeacherQueryV1] = []
    seen: set[str] = set()

    def add(query: otc.TeacherQueryV1) -> None:
        key = json.dumps(
            {
                "order": list(query.candidate_order),
                "labels": dict(query.opaque_id_labels),
                "lengths": dict(query.description_lengths),
                "template": query.prompt_template_hash,
                "trace": query.trace.trace_id,
            },
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            queries.append(query)

    for i, trace in enumerate(eval_traces):
        canonical = otc.default_teacher_query(trace)
        add(canonical)
        if len(trace.legal_set.operator_actions) < 2:
            continue
        for fn in PERTURBATION_FNS.values():
            add(fn(canonical, perturbation_seed + i))
    return queries


def _verdict_block(report) -> dict[str, Any]:
    return report.stop_rule.to_dict()


def _paired_mrr_overall(report) -> list[dict[str, Any]]:
    return [
        c.to_dict()
        for c in report.paired_comparisons
        if c.family is None and c.metric_name == otc.PRIMARY_METRIC
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/var3-01-real-teacher-ceiling"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identical trace pools to the DSH4-02 fixture run (same builders, seed,
    # split) -- GOLD_WEIGHTS was fixed before this run and is not tuned here.
    rng = random.Random(args.seed)
    train_traces = [
        _build_decision_state(
            f"trace-dsh4-02-train-{i:03d}",
            supervision=SupervisionSource.GOLD if i % 2 == 0 else SupervisionSource.ON_POLICY,
            rng=rng,
        )
        for i in range(N_TRAIN_STATES)
    ]
    eval_traces = [
        _build_decision_state(
            f"trace-dsh4-02-eval-{i:03d}",
            supervision=SupervisionSource.GOLD if i % 2 == 0 else SupervisionSource.ON_POLICY,
            rng=rng,
        )
        for i in range(N_EVAL_STATES)
    ]

    teacher = PromptedTeacherV1(
        cache_dir=out_dir / "teacher_cache", model=args.model
    )

    started = time.monotonic()
    queries = _prefetch_queries(eval_traces)
    n_cached, n_failed = prefetch_queries(teacher, queries, workers=args.workers)
    print(
        f"teacher warm-up: {n_cached} exchanges cached, {n_failed} failed "
        f"(fail-closed -> no ranking) in {time.monotonic() - started:.1f}s",
        file=sys.stderr,
    )

    real_report = otc.compare_operator_teacher_ceiling(
        eval_traces=eval_traces,
        frequency_training_traces=train_traces,
        teacher=teacher,
        n_bootstrap=args.n_bootstrap,
    )
    control_report = otc.compare_operator_teacher_ceiling(
        eval_traces=eval_traces,
        frequency_training_traces=train_traces,
        teacher=otc.SyntheticDescriptorTeacherV1(),
        n_bootstrap=args.n_bootstrap,
    )

    try:
        version_stamp = build_version_stamp("harness.distill")
    except Exception:
        version_stamp = {"stamp_schema": "version_stamp/v1", "note": "unavailable"}

    payload = {
        "schema": "var3-01-real-teacher-ceiling/v1",
        "issue": "SLM-429",
        "claim_class": "experiment",
        "run_id": run_id,
        "teacher_model": args.model,
        "gold_weights": list(GOLD_WEIGHTS),
        "n_train_states": N_TRAIN_STATES,
        "n_eval_states": N_EVAL_STATES,
        "seed": args.seed,
        "teacher_warmup": {"cached": n_cached, "failed": n_failed},
        "real_teacher": real_report.to_dict(),
        "control_synthetic_teacher": control_report.to_dict(),
        "version_stamp": version_stamp,
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps(_safe_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {json_path}")

    for name, report in (("real", real_report), ("control", control_report)):
        verdict = report.stop_rule
        print(
            f"[{name}] stop_rule.go = {verdict.go} "
            f"(recommendation: {'go' if verdict.go else 'no_go_defer'})"
        )
        for reason in verdict.reasons:
            print(f"  reason: {reason}")
        print(f"[{name}] paired MRR vs required baselines (overall):")
        for comparison in _paired_mrr_overall(report):
            print(
                f"  {comparison['baseline_source_id']:35s} "
                f"mean_diff={comparison['mean_diff']:+.4f} "
                f"ci=[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}] "
                f"n={comparison['n_paired']} "
                f"significantly_better={comparison['teacher_significantly_better']}"
            )
        print(
            f"[{name}] verifier-regret correlation (teacher): "
            f"{report.verifier_regret_correlation.get(report.teacher_source_id)}"
        )
        print(f"[{name}] perturbation within bound: {report.perturbation.within_bound}")


if __name__ == "__main__":
    main()
