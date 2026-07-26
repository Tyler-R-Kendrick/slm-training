"""SLM-429 (VAR3-01): re-run DSH4-02's teacher ceiling against `PromptedTeacherV1`.

Reuses the exact `compare_operator_teacher_ceiling` harness and the same
fixed five-candidate decision-state pool
`scripts/run_dsh4_02_operator_teacher_ceiling_fixture.py` builds -- only the
`teacher=` argument changes, matching the DSH4-02 module's own contract that
swapping in a real teacher requires no harness change. `GOLD_WEIGHTS` is
frozen before the run, exactly as it was for the synthetic-teacher fixture.

Honesty scoping (read first)
-------------------------------
This script tries to construct a real `openai.OpenAI()` client. If the
`openai` package is not installed, or no credentials are configured, it
degrades to `--allow-fake-client`, an explicitly-labelled deterministic
stand-in that lets the wiring run end-to-end without ever claiming a real
teacher was scored. **A report produced with the fake client satisfies
precondition (1) of the three DSH4-02 preconditions (`PromptedTeacherV1`
exists and is wired into the harness) and nothing else** -- preconditions
(2) (a real teacher's output) and (3) (an honest read of that real output)
are only met when `real_llm_used: true` in the published report.

Example:
  python -m scripts.run_var3_01_real_teacher_ceiling
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
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.distill.legal_set_teacher_trace import TeacherTraceManifest
from slm_training.harnesses.distill.operator_decision_state import (
    capture_operator_decision_state,
)
from slm_training.harnesses.distill.operator_teacher_ceiling import (
    compare_operator_teacher_ceiling,
)
from slm_training.harnesses.distill.prompted_teacher import PromptedTeacherV1

# Identical vocabulary/gold distribution to the DSH4-02 fixture -- SLM-429
# reruns the same decision question against a different teacher, not a new
# one. See scripts/run_dsh4_02_operator_teacher_ceiling_fixture.py.
OPERATOR_ID = "openui.var3_01_fixture"
N_CANDIDATES = 5
GOLD_WEIGHTS = (0.10, 0.15, 0.50, 0.15, 0.10)
N_TRAIN_STATES = 30
N_EVAL_STATES = 30

DEFAULT_JSON_OUT = Path("docs/design/var3-01-real-teacher-ceiling-20260726.json")
DEFAULT_MD_OUT = Path("docs/design/var3-01-real-teacher-ceiling-20260726.md")

COMPONENT = "harness.distill"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"var3-01-fixture-candidate-{index}"),
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


def _build_decision_state(trace_id: str, *, supervision: SupervisionSource, rng: random.Random):
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
    index_by_semantic = {_descriptor(i).semantic_fingerprint: i for i in range(N_CANDIDATES)}
    semantic_by_ref = {
        entry.ref: index_by_semantic[entry.descriptor.semantic_fingerprint]
        for entry in table.entries
    }
    library = OperatorLibraryV1((RegisteredOperatorV1(_declaration(), _executor(semantic_by_ref)),))
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.var3_01_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(source + trace_id),
        request_id=f"request-{trace_id}",
    )
    manifest = TeacherTraceManifest(
        manifest_id=f"manifest-{trace_id}",
        teacher_model_id="var3-01/prompted",
        teacher_revision="fixture",
        prompt_template_hash=_sha("var3-01-fixture-prompt-template"),
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
    current_scores = {
        _dry(i).application_id: GOLD_WEIGHTS[i] + rng.gauss(0.0, 0.05) for i in range(N_CANDIDATES)
    }

    return capture_operator_decision_state(
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


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


class _FakeWiringClient:
    """Deterministic stand-in used ONLY when no real client is available.

    Ranks candidates by the identity term already present in the prompt
    (never by anything a real acceptance label would leak) so the harness
    runs end-to-end -- this is wiring evidence, never a teacher-quality
    claim. Every report built with this client is stamped
    ``real_llm_used: false``.
    """

    class _Responses:
        def parse(self, *, model, input, text_format):  # noqa: A002
            import re

            labels = re.findall(r"label: ([\w-]+)", input)
            identities = re.findall(r"identity: ([0-9a-f]+)", input)
            order = [
                label
                for _, label in sorted(zip(identities, labels), key=lambda pair: pair[0])
            ]

            class _Resp:
                pass

            resp = _Resp()
            resp.output_parsed = text_format(ranked_candidate_labels=order)
            return resp

    def __init__(self) -> None:
        self.responses = self._Responses()


def _resolve_client() -> tuple[Any, bool]:
    try:
        from openai import OpenAI

        return OpenAI(), True
    except Exception:  # noqa: BLE001 - package missing or client construction failed
        return _FakeWiringClient(), False


def run(*, n_bootstrap: int, seed: int, allow_fake_client: bool) -> dict[str, Any]:
    client, real_llm_used = _resolve_client()
    if not real_llm_used and not allow_fake_client:
        raise RuntimeError(
            "no real openai client available (package missing or unconfigured); "
            "pass --allow-fake-client to run a wiring-only demonstration instead"
        )

    rng = random.Random(seed)
    train_traces = [
        _build_decision_state(
            f"trace-var3-01-train-{i:03d}",
            supervision=SupervisionSource.GOLD if i % 2 == 0 else SupervisionSource.ON_POLICY,
            rng=rng,
        )
        for i in range(N_TRAIN_STATES)
    ]
    eval_traces = [
        _build_decision_state(
            f"trace-var3-01-eval-{i:03d}",
            supervision=SupervisionSource.GOLD if i % 2 == 0 else SupervisionSource.ON_POLICY,
            rng=rng,
        )
        for i in range(N_EVAL_STATES)
    ]

    teacher = PromptedTeacherV1(client=client)
    report = compare_operator_teacher_ceiling(
        eval_traces=eval_traces,
        frequency_training_traces=train_traces,
        teacher=teacher,
        n_bootstrap=n_bootstrap,
    )

    try:
        version_stamp = build_version_stamp(COMPONENT)
    except Exception:
        version_stamp = {"stamp_schema": "version_stamp/v1", "note": "unavailable"}

    payload = report.to_dict()
    payload["version_stamp"] = version_stamp
    payload["fixture"] = "var3-01-real-teacher-ceiling"
    payload["gold_weights"] = list(GOLD_WEIGHTS)
    payload["n_train_states"] = N_TRAIN_STATES
    payload["n_eval_states"] = N_EVAL_STATES
    payload["real_llm_used"] = real_llm_used
    payload["teacher_model"] = teacher.model
    payload["preconditions_met"] = {
        "precondition_1_real_teacher_implemented": True,
        "precondition_2_real_output_re_run": real_llm_used,
        "precondition_3_honest_stop_rule_read": real_llm_used,
    }
    return payload


def _decision_sentence(payload: dict[str, Any]) -> str:
    if not payload["real_llm_used"]:
        return (
            "WIRING ONLY: no real LLM client was available in this environment "
            "(no openai package/credentials configured), so this run used the "
            "explicitly-labelled deterministic _FakeWiringClient stand-in. "
            "Precondition 1 (PromptedTeacherV1 implemented and harness-compatible) "
            "is met; preconditions 2 and 3 (a real teacher's output, and an honest "
            "stop-rule read of it) are NOT met by this run. Re-run with a "
            "provisioned OPENAI_API_KEY / installed openai package to satisfy "
            "the issue's acceptance criteria."
        )
    verdict = payload["stop_rule"]
    outcome = "go" if verdict["go"] else "no_go_defer"
    return (
        f"Real teacher run: stop_rule.go={verdict['go']} (recommendation: {outcome}). "
        "All three DSH4-02 preconditions are met by this run: a real teacher "
        "(PromptedTeacherV1 against a real openai client) ranked every eval "
        "trace, the unmodified compare_operator_teacher_ceiling harness scored "
        "it, and this is the honest verdict over that real output -- "
        "no_go_defer here is a legitimate, acceptable outcome (I14), not a "
        "reason to adjust the harness."
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VAR3-01 (SLM-429): real teacher behind OperatorTeacherAdapter",
        "",
        f"- **Status:** {'real teacher run' if payload['real_llm_used'] else 'wiring only (no real LLM available)'}",
        "- **Claim class:** `experiment`",
        f"- real_llm_used: `{payload['real_llm_used']}`",
        f"- teacher_model: `{payload['teacher_model']}`",
        "",
        "## Honesty scoping (read first)",
        "",
        "This script reuses `compare_operator_teacher_ceiling` unchanged from "
        "DSH4-02, swapping only the `teacher=` argument for "
        "`PromptedTeacherV1`. Whether that satisfies the issue's three "
        "preconditions depends entirely on `real_llm_used` above -- see "
        "`preconditions_met` in the JSON twin for the per-precondition "
        "breakdown.",
        "",
        "## Preconditions",
        "",
    ]
    for key, met in payload["preconditions_met"].items():
        lines.append(f"- `{key}`: {met}")
    lines += [
        "",
        "## Stop rule",
        "",
        f"- go: `{payload['stop_rule']['go']}`",
        f"- reasons: {payload['stop_rule']['reasons']}",
        "",
        "## Perturbation robustness",
        "",
        f"- within_bound: `{payload['perturbation']['within_bound']}`",
        f"- kind_distances: {payload['perturbation']['kind_distances']}",
        "",
        "## Decision this run licenses",
        "",
        _decision_sentence(payload),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-fake-client",
        action="store_true",
        help="Run a wiring-only demonstration if no real openai client is available.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    payload = run(
        n_bootstrap=args.n_bootstrap, seed=args.seed, allow_fake_client=args.allow_fake_client
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(_safe_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(_decision_sentence(payload))
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
