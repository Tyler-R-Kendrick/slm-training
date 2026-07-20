"""SPV2-04 wiring fixture: dense teacher distributions on the winning mixture.

Builds deterministic round snapshots that extend the EFS3-01 gold/on-policy
state-source comparison with SPV2-03 dense teacher labels.  No checkpoint is
loaded, no model is decoded, and no quality or ship claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.dense_teacher_mixture import (
    AcquisitionPolicy,
    compare_dense_teacher_mixtures,
)
from slm_training.evals.solver_state_supervision import (
    SupervisionSource,
    SolverStateTrainingExampleV1,
)
from slm_training.harnesses.distill.legal_set_kl import legal_set_teacher_distribution
from slm_training.harnesses.distill.legal_set_teacher_trace import (
    TeacherTraceManifest,
    build_teacher_trace_fixture,
)
from slm_training.versioning import build_version_stamp


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _make_rows_and_teacher_traces(
    problems: int,
    states_per_problem: int,
    vocab_size: int,
    seed: int,
) -> tuple[list[SolverStateTrainingExampleV1], TeacherTraceManifest, list[Any]]:
    rng = {"gold": 0, "on_policy": 0}
    held_out_fraction = 0.15
    rows: list[SolverStateTrainingExampleV1] = []
    state_index = 0

    for p in range(problems):
        problem_id = f"problem-{p:04d}"
        family_id = f"family-{p % 4}"
        group_id = f"group-{p}"
        split = "test" if (p / max(problems, 1)) < held_out_fraction else "train"
        for s in range(states_per_problem):
            source = SupervisionSource.GOLD if (p + s) % 2 == 0 else SupervisionSource.ON_POLICY
            verdict = "SUPPORTED" if (p * states_per_problem + s) % 5 != 0 else "UNKNOWN"
            legal_values = list(range(4))
            legal = [{"value": v, "family": f"family-{v % 2}"} for v in legal_values]
            acceptable = legal[:2] if verdict == "SUPPORTED" else []
            rows.append(
                SolverStateTrainingExampleV1(
                    problem_id=problem_id,
                    state_fingerprint=f"state-{state_index:04d}",
                    supervision_source=source,
                    legal_actions=tuple(legal),
                    acceptable_actions=tuple(acceptable),
                    support_verdict=verdict,
                    cost_to_go=float(rng[source.value]) if verdict == "SUPPORTED" else None,
                    cost_observed=verdict == "SUPPORTED",
                    split_group_id=group_id,
                    split=split,
                    lineage_id=f"lineage-{p}",
                    program_family_id=family_id,
                    replay_certified=source is SupervisionSource.GOLD,
                )
            )
            rng[source.value] += 1
            state_index += 1

    # Teacher traces cover only on-policy states (gold states are left unlabeled
    # here to keep the fixture explicit; a real run would label whichever states
    # the acquisition policy selects).
    on_policy_states = {r.state_fingerprint for r in rows if r.supervision_source is SupervisionSource.ON_POLICY}
    manifest, raw_traces = build_teacher_trace_fixture(
        n_states=len(on_policy_states), vocab_size=vocab_size, seed=seed
    )

    import torch

    teacher_traces = []
    for idx, trace in enumerate(raw_traces):
        state_id = sorted(on_policy_states)[idx % len(on_policy_states)]
        # Force legal_action_ids to the fixture row's legal set so alignment is
        # deterministic and complete.
        legal_action_ids = tuple(range(4))
        teacher_logits = torch.tensor(
            [float(((idx + 1) * (v + 1) + seed) % 7 - 3) for v in legal_action_ids],
            dtype=torch.float32,
        )
        teacher_probs = legal_set_teacher_distribution(
            teacher_logits, legal_action_ids, temperature=1.0
        )
        accepted_count = int((idx % 2) + 1)
        accepted = tuple(legal_action_ids[:accepted_count])
        teacher_traces.append(
            trace.__class__(  # LegalSetTeacherTrace
                trace_id=f"teacher-{state_id}",
                manifest_id=trace.manifest_id,
                state_id=state_id,
                prompt_hash=trace.prompt_hash,
                prefix_ids=trace.prefix_ids,
                legal_action_ids=legal_action_ids,
                teacher_logits=None,
                teacher_probs=tuple(teacher_probs.tolist()),
                accepted_action_ids=accepted,
                source="fixture",
                coverage="complete",
                approximate=False,
                provenance={"fixture_state_id": state_id, "seed": seed},
            )
        )

    return rows, manifest, teacher_traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/spv2-04-dense-teacher-mixture"),
    )
    parser.add_argument("--problems", type=int, default=40)
    parser.add_argument("--states-per-problem", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--decision-budget", type=int, default=64)
    parser.add_argument("--teacher-label-budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest, teacher_traces = _make_rows_and_teacher_traces(
        args.problems, args.states_per_problem, args.vocab_size, args.seed
    )

    try:
        version_stamp = build_version_stamp("evals.scoring")
    except Exception:
        version_stamp = {"stamp_schema": "version_stamp/v1", "note": "unavailable"}

    comparison = compare_dense_teacher_mixtures(
        rows,
        teacher_traces,
        seeds=(0, 1, 2),
        decision_budget=args.decision_budget,
        teacher_label_budget=args.teacher_label_budget,
        acquisition_policy=AcquisitionPolicy.UNIFORM,
        round_id=f"spv2-04-{run_id}",
        teacher_source="fixture",
        manifest=manifest.to_dict(),
    )

    # Stamp each snapshot.
    for snapshot in comparison["snapshots"]:
        snapshot["version_stamp"] = _safe_json(version_stamp)

    summary = {
        "run_id": run_id,
        "fixture": "spv2-04-dense-teacher-mixture",
        "synthetic_rows": len(rows),
        "teacher_traces": len(teacher_traces),
        "decision_budget": args.decision_budget,
        "teacher_label_budget": args.teacher_label_budget,
        "aggregate_arm_sizes": comparison["aggregate_arm_sizes"],
        "version_stamp": _safe_json(version_stamp),
    }

    json_path = out_dir / "dense_teacher_mixture.json"
    json_path.write_text(
        json.dumps(_safe_json(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_safe_json(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    arm_table = "\n".join(
        f"| {arm} | {stats['mean']:.1f} | {stats['min']} | {stats['max']} |"
        for arm, stats in sorted(comparison["aggregate_arm_sizes"].items())
    )

    readme = f"""# SPV2-04 Dense Teacher Mixture Fixture

Run ID: `{run_id}`

This fixture extends the EFS3-01 gold/on-policy state-source comparison with
SPV2-03 dense legal-set teacher distributions.  It builds immutable round
snapshots for the canonical arms requested by SLM-152.

## Counts per arm (aggregated over seeds)

| Arm | Mean size | Min | Max |
| --- | ---: | ---: | ---: |
{arm_table}

## Recipe

* Synthetic solver-state rows: {len(rows)}
* Teacher traces (fixture): {len(teacher_traces)}
* Decision budget: {args.decision_budget}
* Teacher-label budget: {args.teacher_label_budget}
* Acquisition policy: uniform
* Seeds: 0, 1, 2

## Artifacts

* `dense_teacher_mixture.json` — full round snapshots and per-arm corpora.
* `summary.json` — headline counts and version stamp.

## Honest caveats

This is wiring-only evidence.  Rows and teacher distributions are synthetic;
no solver trace replay, no external teacher model, no checkpoint train, and no
ship-grade evaluation were performed.  The acquisition scores are deterministic
placeholders, not real student-teacher divergence.
"""
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    print(f"SPV2-04 fixture written to {out_dir}")
    print(json.dumps(_safe_json(summary), indent=2))


if __name__ == "__main__":
    main()
