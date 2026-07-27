"""Deterministic ``CompilerReasoningTraceV1`` extraction (LOT0-02 / SLM-249).

Reuses the existing gold ``SemanticPlanV1`` extractor
(:class:`OpenUISemanticPlanExtractor`) and the production codec's canonical
statement order -- no new compiler, parser, or validity definition. Extraction
reads only ``program_spec``/``pack``; no evaluator or model output enters any
target field (test coverage: ``tests/test_data/test_semantic_plan_extraction/
test_compiler_reasoning_trace_extract.py``).
"""

from __future__ import annotations

import json

from slm_training.data.progspec.compiler_reasoning_trace import (
    CANONICAL_STAGE_ORDER,
    CompilerReasoningTraceV1,
    TraceStepV1,
)
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.semantic_plan.canonicalize import canonicalize_plan
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.pack import DslPack
from slm_training.dsl.production_codec import parse_statement_bindings, statement_binding_order


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def extract_compiler_reasoning_trace(
    program_spec: ProgramSpec,
    pack: DslPack,
    *,
    source: str = "programspec_extracted",
    split: str | None = None,
    plan: SemanticPlanV1 | None = None,
) -> CompilerReasoningTraceV1:
    """Extract the full (K=6-stage) saturated trace for ``program_spec``.

    ``plan``, when supplied, is used instead of re-running
    :class:`OpenUISemanticPlanExtractor` (callers that already extracted a plan for
    other purposes should pass it so the two never silently disagree).
    """
    semantic_plan = plan or OpenUISemanticPlanExtractor().extract(program_spec, pack)
    canonical_plan = canonicalize_plan(semantic_plan)
    fingerprint = canonical_fingerprint(program_spec.canonical_openui)
    record_split = split or program_spec.split

    steps: list[TraceStepV1] = []

    def add_step(
        step_kind: str,
        *,
        canonical_target: str,
        source_refs: tuple[str, ...],
        set_valued_fields: tuple[str, ...] = (),
        support: float | None = 1.0,
        confidence: float | None = 1.0,
        ambiguity_reason: str | None = None,
        partial_order: tuple[str, ...] = (),
    ) -> None:
        steps.append(
            TraceStepV1(
                record_id=program_spec.id,
                source=source,
                split=record_split,
                fingerprint=fingerprint,
                step_index=len(steps),
                step_kind=step_kind,  # type: ignore[arg-type]
                canonical_target=canonical_target,
                accepted_targets=(canonical_target,),
                set_valued_fields=set_valued_fields,
                partial_order=partial_order,
                support=support,
                confidence=confidence,
                ambiguity_reason=ambiguity_reason,
                source_refs=source_refs,
            )
        )

    # 1. intent_contract <- archetype.
    archetype_id = canonical_plan.archetype.id or "unknown_archetype"
    add_step(
        "intent_contract",
        canonical_target=archetype_id,
        source_refs=("archetype.id",),
        confidence=canonical_plan.archetype.confidence,
        ambiguity_reason=None if canonical_plan.archetype.id else "no root archetype resolved",
    )

    # 2. component_inventory <- role_slots (unordered: set-valued).
    inventory = sorted(
        (slot.role_id, slot.component_family or "", bool(slot.required))
        for slot in canonical_plan.role_slots
    )
    add_step(
        "component_inventory",
        canonical_target=_stable_json(inventory),
        source_refs=tuple(slot.role_id for slot in canonical_plan.role_slots),
        set_valued_fields=("role_slots",),
        partial_order=("intent_contract",),
        ambiguity_reason=None if inventory else "no role slots extracted",
    )

    # 3. topology_skeleton <- parent/child structure, no component lexemes.
    edges = canonical_plan.topology.parent_relation_candidates or ()
    skeleton = sorted(
        (str(edge.get("parent_role_id")), str(edge.get("child_role_id")))
        for edge in edges
    )
    add_step(
        "topology_skeleton",
        canonical_target=_stable_json(skeleton),
        source_refs=("topology.parent_relation_candidates",),
        set_valued_fields=("topology.parent_relation_candidates",),
        partial_order=("component_inventory",),
        ambiguity_reason=None if skeleton else "no topology edges extracted",
    )

    # 4. semantic_edges_roles <- ownership/relation + sibling occupancy.
    edge_roles = sorted(
        (
            str(edge.get("parent_role_id")),
            str(edge.get("child_role_id")),
            str(edge.get("relation") or "contains"),
        )
        for edge in edges
    )
    sibling_groups = sorted(
        tuple(sorted(group)) for group in (canonical_plan.topology.sibling_order_groups or ())
    )
    add_step(
        "semantic_edges_roles",
        canonical_target=_stable_json({"edges": edge_roles, "sibling_groups": sibling_groups}),
        source_refs=("topology.parent_relation_candidates", "topology.sibling_order_groups"),
        set_valued_fields=("edges", "sibling_groups"),
        partial_order=("topology_skeleton",),
        ambiguity_reason=None if edge_roles else "no semantic edges extracted",
    )

    # 5. binding_scope_references <- symbols + bindings.
    symbol_roles = sorted(
        (sym.symbol_id, sym.semantic_role or "") for sym in canonical_plan.symbols
    )
    binding_targets = sorted(
        (binding.role_slot_id, tuple(sorted(binding.candidate_symbols or ())))
        for binding in canonical_plan.bindings
    )
    add_step(
        "binding_scope_references",
        canonical_target=_stable_json({"symbols": symbol_roles, "bindings": binding_targets}),
        source_refs=("symbols", "bindings"),
        set_valued_fields=("symbols", "bindings"),
        partial_order=("semantic_edges_roles",),
        ambiguity_reason=None if symbol_roles else "no bound symbols extracted",
    )

    # 6. serialization_plan <- codec canonical statement order (deterministic, not
    #    set-valued: emission order is a total order by construction).
    bindings = parse_statement_bindings(program_spec.canonical_openui)
    order = statement_binding_order(bindings)
    add_step(
        "serialization_plan",
        canonical_target=_stable_json(order),
        source_refs=("production_codec.statement_binding_order",),
        partial_order=("binding_scope_references",),
    )

    return CompilerReasoningTraceV1(
        record_id=program_spec.id,
        source=source,
        split=record_split,
        fingerprint=fingerprint,
        stage_order=CANONICAL_STAGE_ORDER,
        steps=tuple(steps),
    )
