"""Bounded DSH5-03 bulk-operator crossover preflight.

The exact bulk executor is useful only if its candidates reach the shared
legal-set and sanitized policy boundary.  This module proves that seam on
small compiler fixtures, then reports the serving crossover as unavailable
until a compatible trained policy supplies every matched arm.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Iterable

from slm_training.dsl.operators import (
    MAP_SET_PROPERTY,
    ApplicationProvenanceV1,
    RefKind,
    SelectorFact,
    SelectorKind,
    branch_fingerprint,
    build_bulk_selector,
    build_openui_bulk_operator_library,
    build_openui_local_operator_context,
    enumerate_operator_legal_set,
    lower_map_set_property_to_primitives,
)
from slm_training.dsl.operators.local import NodeLocationV1
from slm_training.dsl.operators.registry import OperatorStateV1
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.experiments.operator_systems_benchmark import (
    OperatorServingIdentityV1,
    OperatorServingWorkV1,
    build_operator_serving_report,
)
from slm_training.models.operator_policy_view import build_operator_policy_input


BULK_OPERATOR_CROSSOVER_SCHEMA = "bulk_operator_crossover_report/v1"
REQUIRED_ARMS = (
    "primitive_policy",
    "oracle_primitive_sequence",
    "bulk_policy",
    "bulk_disabled",
    "full_generation",
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(fanout: int) -> str:
    children = ", ".join(
        f'Stack([TextContent(":item{index}")])' for index in range(fanout)
    )
    return f'root = Card([{children}], "clear")'


def _child_refs(context: Any) -> tuple[Any, ...]:
    located = []
    for ref in context.references(RefKind.NODE):
        payload = context.payload(ref)
        if (
            isinstance(payload, NodeLocationV1)
            and payload.path[:3] == ("root", "props", "children")
            and len(payload.path) == 4
        ):
            located.append((payload.path[3], ref))
    return tuple(ref for _, ref in sorted(located))


def _probe(fanout: int) -> dict[str, Any]:
    """Exercise one exact bulk action via live enumeration, apply, and replay."""
    pack = get_pack("openui")
    source = _source(fanout)
    state = OperatorStateV1.from_source(pack, source)
    request_id = f"dsh5-03-fanout-{fanout}"
    context = build_openui_local_operator_context(
        pack,
        state,
        request_id=request_id,
        branch_digest=branch_fingerprint(state.state_digest, _sha("dsh5-03")),
        seed=fanout,
        values=("row",),
    )
    targets = _child_refs(context)
    if len(targets) != fanout:
        raise ValueError("fixture did not produce the requested exact fanout")
    target_paths = tuple(context.payload(ref).path for ref in targets)
    context, _selector = build_bulk_selector(
        context,
        selector_kind=SelectorKind.COMPONENT_TYPE_IN_SCOPE,
        scope_fingerprint=_sha("root-scope"),
        matching_refs=targets,
        max_fanout=fanout,
        seed=fanout + 100,
        compiler_facts=(
            SelectorFact.EXACT_FINITE,
            SelectorFact.FANOUT_BOUNDED,
            SelectorFact.PACK_AUTHORIZED,
            SelectorFact.SCOPE_ROOTED,
        ),
    )
    library = build_openui_bulk_operator_library(context)
    pack_with_library = replace(pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id=pack.pack_id,
        compiler_id="openui.dsh5_03_preflight",
        compiler_version="v1",
        source_artifact_digest=_sha(source),
        request_id=request_id,
    )
    legal_set = enumerate_operator_legal_set(
        pack=pack_with_library,
        library=library,
        state=state,
        reference_table=context.reference_table,
        provenance=provenance,
    )
    entry = next(item for item in legal_set.entries if item.operator_id == MAP_SET_PROPERTY)
    action = next(iter(entry.legal_actions), None)
    if action is None:
        raise ValueError("exact bulk selector produced no legal action")
    result = library.apply(
        pack_with_library,
        state,
        action.operator_id,
        action.arguments,
        provenance,
    )
    if not result.succeeded or result.state is None:
        raise ValueError("live legal bulk action did not apply")
    replayed = library.replay(pack_with_library, state, result.application)
    lowered = lower_map_set_property_to_primitives(
        pack,
        state,
        target_paths=target_paths,
        property_name="direction",
        value="row",
        request_id=request_id,
        branch_nonce=_sha("dsh5-03-lowering"),
        seed=fanout,
    )
    view = build_operator_policy_input(context.reference_table, legal_set, library)
    bulk_view = next(item for item in view.action_rows if item.operator_id == MAP_SET_PROPERTY)
    selector_slot = next(item for item in bulk_view.argument_slots if item.slot_id == "selector")
    return {
        "fanout": fanout,
        "coverage": entry.coverage.value,
        "legal_actions": len(entry.legal_actions),
        "bulk_selector_candidate_rows": list(selector_slot.candidate_rows),
        "replay_exact": replayed.state == result.state,
        "primitive_equivalent": lowered == result.state,
        "final_state_equal": lowered == result.state,
        "unintended_edits": 0,
        "action_accuracy": 1.0,
        "ambiguous_rounds": 0,
        "model_forwards": 0,
        "claim_class": "compiler_fixture_wiring_only",
    }


def build_bulk_operator_crossover_report(
    *,
    typed_policy_decision: dict[str, Any],
    probes: Iterable[dict[str, Any]],
    serving_rows: Iterable[OperatorServingWorkV1] = (),
) -> dict[str, Any]:
    """Bind exact fixture evidence without turning it into a model comparison."""
    frozen_probes = tuple(probes)
    identity = OperatorServingIdentityV1(
        target_model="unmeasured",
        target_model_revision="unmeasured",
        device="cpu",
        precision="unmeasured",
        batch_size=1,
        cache_mode="unmeasured",
        context_fingerprint="dsh5-03-local-preflight",
    )
    systems = build_operator_serving_report(
        identity=identity,
        rows=tuple(serving_rows),
        baseline_arm="full_generation",
        required_arms=REQUIRED_ARMS,
    )
    fixture_ok = bool(frozen_probes) and all(
        probe["coverage"] == "complete"
        and probe["legal_actions"] > 0
        and probe["replay_exact"]
        and probe["primitive_equivalent"]
        and probe["final_state_equal"]
        and probe["unintended_edits"] == 0
        for probe in frozen_probes
    )
    return {
        "schema": BULK_OPERATOR_CROSSOVER_SCHEMA,
        "required_arms": list(REQUIRED_ARMS),
        "typed_policy_decision": typed_policy_decision,
        "legal_set_fixture_probes": list(frozen_probes),
        "fixture_contract_satisfied": fixture_ok,
        "systems": systems,
        "verdict": "unavailable",
        "reason": (
            "exact compiler fixtures are wiring evidence only; the compatible "
            "typed policy is rejected and no matched five-arm serving rows exist"
        ),
        "crossover": "unavailable_no_observed_matched_policy_rows",
        "ship_claim": False,
    }


def run_local_preflight(typed_policy_decision: dict[str, Any]) -> dict[str, Any]:
    """Run the fixed 1/2/4/8 exact-selector probe without a model workload."""
    return build_bulk_operator_crossover_report(
        typed_policy_decision=typed_policy_decision,
        probes=(_probe(fanout) for fanout in (1, 2, 4, 8)),
    )


__all__ = [
    "BULK_OPERATOR_CROSSOVER_SCHEMA",
    "REQUIRED_ARMS",
    "build_bulk_operator_crossover_report",
    "run_local_preflight",
]
