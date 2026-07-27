"""Fixture runner for SLM-387 (DSH4-02): one-teacher ceiling over exact legal sets.

Wiring-only demonstration of
``slm_training.harnesses.distill.operator_teacher_ceiling``: builds a
deterministic pool of DSH4-01 ``OperatorDecisionStateTraceV1`` decision
states over a small compiler operator with a fixed, reused five-candidate
vocabulary, splits them into a frequency-training pool and a disjoint
evaluation pool, runs the full comparison (frequency, deterministic compiler
order, current-scorer, descriptor-similarity, and oracle baselines against
``SyntheticDescriptorTeacherV1``), and prints/writes the harness's honest
go/no-go/defer stop-rule verdict.

**No external teacher model is downloaded or scored.**
``SyntheticDescriptorTeacherV1`` is a deterministic, non-learned stand-in
documented in ``operator_teacher_ceiling.py``; whatever this script reports
-- including a defer -- is the harness's honest finding on synthetic data,
never a production KD-readiness claim. Do not edit this fixture's gold-label
distribution to force a particular stop-rule outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    RefKind,
)
from slm_training.dsl.operators.references import ReferenceDescriptorV1, build_reference_table
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.pack import get_pack
from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill.legal_set_teacher_trace import TeacherTraceManifest
from slm_training.harnesses.distill.operator_decision_state import (
    capture_operator_decision_state,
)
from slm_training.harnesses.distill.operator_teacher_ceiling import (
    SyntheticDescriptorTeacherV1,
    compare_operator_teacher_ceiling,
    write_operator_teacher_ceiling_report,
)
from slm_training.versioning import build_version_stamp

OPERATOR_ID = "openui.dsh4_02_fixture"
N_CANDIDATES = 5
#: A deliberately skewed but arbitrary latent preference over the fixed
#: candidate vocabulary -- not derived from, or known to, any comparator.
#: Nothing in ``SyntheticDescriptorTeacherV1`` or the descriptor-similarity
#: baseline reads this; only ``FrequencyBaselineV1`` (trained on a disjoint
#: pool) and ``CurrentScorerBaselineV1`` (given noisy current_scores below)
#: have a legitimate path to it.
GOLD_WEIGHTS = (0.10, 0.15, 0.50, 0.15, 0.10)
N_TRAIN_STATES = 30
N_EVAL_STATES = 30


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"dsh4-02-fixture-candidate-{index}"),
        value_type="openui.level",
    )


def _declaration() -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.TOPOLOGY,),
        locality="node",
        cost=1.0,
    )


def _executor(semantic_by_ref: dict):
    def execute(state, arguments):
        ref = arguments[0].value
        index = semantic_by_ref[ref]
        return OperatorMutationV1(
            source=state.source.replace(":box.pending", f":box.level{index}"),
            effect=ActionEffectV1(
                topology_deltas=(
                    EffectDeltaV1(
                        EffectDeltaKind.TOPOLOGY, ref, "box.pending", f"box.level{index}"
                    ),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    return execute


def _build_decision_state(
    trace_id: str, *, supervision: SupervisionSource, rng: random.Random
):
    """Build one certified DSH4-01 decision-state trace over the fixed
    five-candidate vocabulary. The accepted action is drawn from
    ``GOLD_WEIGHTS`` -- structurally invisible to the teacher and the
    descriptor-similarity baseline, which never see acceptance labels."""
    base_pack = get_pack("openui")
    source = 'root = TextContent(":box.pending")'
    state0 = OperatorStateV1.from_source(base_pack, source)
    descriptors = tuple(_descriptor(i) for i in range(N_CANDIDATES))
    table = build_reference_table(
        request_id=f"request-{trace_id}",
        state_digest=state0.state_digest,
        branch_digest=_sha(f"branch-{trace_id}"),
        descriptors=descriptors,
        seed=1,
    )
    index_by_semantic = {
        _descriptor(i).semantic_fingerprint: i for i in range(N_CANDIDATES)
    }
    semantic_by_ref = {
        entry.ref: index_by_semantic[entry.descriptor.semantic_fingerprint]
        for entry in table.entries
    }
    library = OperatorLibraryV1(
        (RegisteredOperatorV1(_declaration(), _executor(semantic_by_ref)),)
    )
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.dsh4_02_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(source + trace_id),
        request_id=f"request-{trace_id}",
    )
    manifest = TeacherTraceManifest(
        manifest_id=f"manifest-{trace_id}",
        teacher_model_id="fixture/teacher",
        teacher_revision="fixture",
        prompt_template_hash=_sha("dsh4-02-fixture-prompt-template"),
        pack_id="openui",
        compiler_version="openui-fixture",
        state_schema_version="dsh4-01.v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        provenance={"source": "synthetic"},
    )
    ref_by_index = {
        index_by_semantic[entry.descriptor.semantic_fingerprint]: entry.ref
        for entry in table.entries
    }

    def _dry(index: int):
        result = library.dry_run(
            pack, state0, OPERATOR_ID, (BoundArgumentV1("value", ref_by_index[index]),), provenance
        )
        assert result.succeeded, result.rejection
        return result

    gold_index = rng.choices(range(N_CANDIDATES), weights=GOLD_WEIGHTS, k=1)[0]
    accepted_id = _dry(gold_index).application_id
    # A plausible "current scorer": already roughly tracks the latent prior,
    # with per-state noise -- current_scorer is meant to be a real contender,
    # not a strawman.
    current_scores = {
        _dry(i).application_id: GOLD_WEIGHTS[i] + rng.gauss(0.0, 0.05)
        for i in range(N_CANDIDATES)
    }

    trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table,
        provenance=provenance,
        manifest=manifest,
        supervision_source=supervision,
        trace_id=trace_id,
        accepted_application_ids=(accepted_id,),
        current_scores=current_scores,
    )
    return trace


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
        default=Path("outputs/runs/dsh4-02-operator-teacher-ceiling"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

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

    teacher = SyntheticDescriptorTeacherV1()
    report = compare_operator_teacher_ceiling(
        eval_traces=eval_traces,
        frequency_training_traces=train_traces,
        teacher=teacher,
        n_bootstrap=args.n_bootstrap,
    )

    try:
        version_stamp = build_version_stamp("harness.distill")
    except Exception:
        version_stamp = {"stamp_schema": "version_stamp/v1", "note": "unavailable"}

    payload = report.to_dict()
    payload["version_stamp"] = version_stamp
    payload["run_id"] = run_id
    payload["fixture"] = "dsh4-02-operator-teacher-ceiling"
    payload["gold_weights"] = list(GOLD_WEIGHTS)
    payload["n_train_states"] = N_TRAIN_STATES
    payload["n_eval_states"] = N_EVAL_STATES

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(_safe_json(payload), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {json_path}")

    write_operator_teacher_ceiling_report(out_dir / "report.json", report)
    print(f"wrote {out_dir / 'report.json'}")

    verdict = report.stop_rule
    print(f"stop_rule.go = {verdict.go} (recommendation: {'go' if verdict.go else 'no_go_defer'})")
    if verdict.reasons:
        for reason in verdict.reasons:
            print(f"  reason: {reason}")
    print("teacher vs required baselines (paired MRR, overall):")
    for comparison in report.paired_comparisons:
        if comparison.family is None and comparison.metric_name == "mrr":
            print(
                f"  {comparison.baseline_source_id:35s} mean_diff={comparison.mean_diff:+.4f} "
                f"ci=[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] "
                f"n={comparison.n_paired} significantly_better={comparison.teacher_significantly_better}"
            )
    print(f"verifier-regret correlation (teacher): {report.verifier_regret_correlation.get(teacher.source_id)}")
    print(f"perturbation robustness within bound: {report.perturbation.within_bound}")


if __name__ == "__main__":
    main()
