"""Fixture runner for SLM-386 (DSH4-01): operator decision-state export.

Wiring-only demonstration of
``slm_training.harnesses.distill.operator_decision_state``: it captures
DSH3-06 operator legal-set evidence into DSH4-01 decision-state traces for a
gold and an on-policy state, replays an accepted two-step trajectory to a
canonical verified result, and demonstrates the stop rule (an exporter must
block any teacher query when an accepted trajectory no longer replays).

No external teacher model is downloaded or scored, and no checkpoint is
loaded or trained; this only exercises the export contract and its fail-closed
guarantees over a small deterministic compiler fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from slm_training.dsl.operators.references import (
    ReferenceDescriptorV1,
    build_reference_table,
)
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.pack import get_pack
from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill.legal_set_teacher_trace import TeacherTraceManifest
from slm_training.harnesses.distill.operator_decision_state import (
    OperatorDecisionStateError,
    build_operator_trajectory,
    capture_operator_decision_state,
    export_for_teacher_query,
    replay_operator_trajectory,
    write_operator_decision_state_traces,
)
from slm_training.versioning import build_version_stamp

SOURCE0 = 'root = TextContent(":hero.title")'
OPERATOR_ID = "openui.dsh4_01_fixture"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"dsh4-01-fixture-value-{index}"),
        value_type="openui.string",
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


def _executor(semantic_by_ref: dict[Any, str], sem0: str, sem1: str, *, strict: bool):
    def execute(state: OperatorStateV1, arguments):
        ref = arguments[0].value
        semantic = semantic_by_ref[ref]
        if semantic == sem0:
            if ":hero.title" not in state.source:
                raise OperatorRejectedError("fixture.title_not_present")
            return OperatorMutationV1(
                source=state.source.replace(":hero.title", ":hero.body"),
                effect=ActionEffectV1(
                    topology_deltas=(
                        EffectDeltaV1(EffectDeltaKind.TOPOLOGY, ref, "hero.title", "hero.body"),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                ),
            )
        if semantic == sem1:
            if strict:
                raise OperatorRejectedError("fixture.registry_drift_rejected")
            if ":hero.body" not in state.source:
                raise OperatorRejectedError("fixture.body_not_present")
            return OperatorMutationV1(
                source=state.source.replace(":hero.body", ":hero.subtitle"),
                effect=ActionEffectV1(
                    topology_deltas=(
                        EffectDeltaV1(EffectDeltaKind.TOPOLOGY, ref, "hero.body", "hero.subtitle"),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                ),
            )
        raise OperatorRejectedError("fixture.unknown_value")

    return execute


def _build_library(state0: OperatorStateV1, table0, *, strict: bool):
    semantic_by_ref = {
        entry.ref: entry.descriptor.semantic_fingerprint for entry in table0.entries
    }
    sem0, sem1 = _descriptor(0).semantic_fingerprint, _descriptor(1).semantic_fingerprint
    declaration = _declaration()
    library = OperatorLibraryV1(
        (RegisteredOperatorV1(declaration, _executor(semantic_by_ref, sem0, sem1, strict=strict)),)
    )
    return library, semantic_by_ref, sem0, sem1


def _ref_for(table, semantic_target: str):
    return next(
        entry.ref for entry in table.entries if entry.descriptor.semantic_fingerprint == semantic_target
    )


def _manifest() -> TeacherTraceManifest:
    return TeacherTraceManifest(
        manifest_id="manifest-dsh4-01-fixture-0",
        teacher_model_id="fixture/teacher",
        teacher_revision="fixture",
        prompt_template_hash=_sha("dsh4-01-fixture-prompt-template"),
        pack_id="openui",
        compiler_version="openui-fixture",
        state_schema_version="dsh4-01.v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        provenance={"source": "synthetic"},
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        import math

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
        default=Path("outputs/runs/dsh4-01-operator-decision-state"),
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    base_pack = get_pack("openui")
    state0 = OperatorStateV1.from_source(base_pack, SOURCE0)
    table0 = build_reference_table(
        request_id="request-dsh4-01",
        state_digest=state0.state_digest,
        branch_digest=_sha("branch-dsh4-01"),
        descriptors=(_descriptor(0), _descriptor(1)),
        seed=1,
    )
    library, semantic_by_ref, sem0, sem1 = _build_library(state0, table0, strict=False)
    pack = replace(base_pack, operator_library=library)
    provenance0 = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.dsh4_01_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE0),
        request_id="request-dsh4-01",
    )
    manifest = _manifest()

    # -- gold-state capture at state0 --------------------------------------- #
    ref0 = _ref_for(table0, sem0)
    dry0 = library.dry_run(pack, state0, OPERATOR_ID, (BoundArgumentV1("value", ref0),), provenance0)
    gold_trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=manifest,
        supervision_source=SupervisionSource.GOLD,
        trace_id="trace-dsh4-01-gold-000",
        accepted_application_ids=(dry0.application_id,) if dry0.succeeded else (),
        current_scores={dry0.application_id: 0.91} if dry0.succeeded else None,
    )

    # -- accepted 2-step trajectory: state0 -> state1 -> state2 ------------- #
    result1 = library.apply(pack, state0, OPERATOR_ID, (BoundArgumentV1("value", ref0),), provenance0)
    assert result1.succeeded and result1.state is not None
    state1 = result1.state
    table1 = build_reference_table(
        request_id="request-dsh4-01",
        state_digest=state1.state_digest,
        branch_digest=_sha("branch-dsh4-01"),
        descriptors=(_descriptor(0), _descriptor(1)),
        seed=1,
    )
    provenance1 = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.dsh4_01_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state1.source),
        request_id="request-dsh4-01",
    )
    ref1 = _ref_for(table1, sem1)

    # -- on-policy capture at state1 ----------------------------------------- #
    dry1 = library.dry_run(pack, state1, OPERATOR_ID, (BoundArgumentV1("value", ref1),), provenance1)
    on_policy_trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state1,
        reference_table=table1,
        provenance=provenance1,
        manifest=manifest,
        supervision_source=SupervisionSource.ON_POLICY,
        trace_id="trace-dsh4-01-on-policy-000",
        accepted_application_ids=(dry1.application_id,) if dry1.succeeded else (),
        current_scores={dry1.application_id: 0.74} if dry1.succeeded else None,
    )

    result2 = library.apply(pack, state1, OPERATOR_ID, (BoundArgumentV1("value", ref1),), provenance1)
    assert result2.succeeded and result2.state is not None
    state2 = result2.state

    trajectory = build_operator_trajectory(
        "traj-dsh4-01-fixture-0",
        ((state0, result1.application), (state1, result2.application)),
    )

    # -- replay to a canonical verified result -------------------------------- #
    replay_result = replay_operator_trajectory(pack=pack, library=library, trajectory=trajectory)
    assert replay_result.verified
    assert replay_result.canonical_final_state is not None
    assert replay_result.canonical_final_state.state_digest == state2.state_digest

    exported = export_for_teacher_query(gold_trace, trajectory, pack=pack, library=library)

    # -- stop-rule demonstration: registry drift blocks the teacher query ---- #
    strict_library, _, _, _ = _build_library(state0, table0, strict=True)
    strict_pack = replace(base_pack, operator_library=strict_library)
    stop_rule_blocked = False
    stop_rule_message = None
    try:
        export_for_teacher_query(gold_trace, trajectory, pack=strict_pack, library=strict_library)
    except OperatorDecisionStateError as exc:
        stop_rule_blocked = True
        stop_rule_message = str(exc)
    diverged_replay = replay_operator_trajectory(
        pack=strict_pack, library=strict_library, trajectory=trajectory
    )

    write_operator_decision_state_traces(
        out_dir / "decision_state_traces.jsonl", [gold_trace, on_policy_trace]
    )

    try:
        version_stamp = build_version_stamp("harness.distill")
    except Exception:
        version_stamp = {"stamp_schema": "version_stamp/v1", "note": "unavailable"}

    summary = {
        "run_id": run_id,
        "fixture": "dsh4-01-operator-decision-state",
        "traces": {
            "gold": gold_trace.to_dict(),
            "on_policy": on_policy_trace.to_dict(),
        },
        "trajectory_replay": {
            "verified_against_original_registry": replay_result.to_dict(),
            "diverged_against_drifted_registry": diverged_replay.to_dict(),
        },
        "export_for_teacher_query": {
            "succeeded_with_verified_trajectory": exported is not None,
            "blocked_by_stop_rule_on_drifted_registry": stop_rule_blocked,
            "stop_rule_message": stop_rule_message,
        },
        "version_stamp": version_stamp,
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps(_safe_json(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {json_path}")
    print(f"wrote {out_dir / 'decision_state_traces.jsonl'}")


if __name__ == "__main__":
    main()
