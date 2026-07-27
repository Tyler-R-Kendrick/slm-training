"""DSH2-02: deterministic SemanticFrameV1 from canonical AST + structured schema."""

from __future__ import annotations

from dataclasses import replace

import pytest

from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.semantic_frame import (
    CAP1SchemaV1,
    CounterfactualError,
    UnsupportedSemanticFactError,
    check_constraint,
    derive_semantic_frame,
    frame_conflicts,
    frame_to_constraint,
    mutate_one_fact,
    verify_single_fact_change,
)

CARD_SRC = (
    'a = TextContent(":ns.body", 14)\n'
    'root = Card([a], "filled", "column", 8, "start", "start", "nowrap")\n'
)
# Same layout: different binder name and statement order.
CARD_SRC_RENAMED = (
    'root = Card([b], "filled", "column", 8, "start", "start", "nowrap")\n'
    'b = TextContent(":ns.body", 14)\n'
)
CALLOUT_SRC = 'root = Callout("info", ":ns.title", ":ns.desc", true)'


@pytest.fixture(scope="module")
def schema() -> CAP1SchemaV1:
    return CAP1SchemaV1.from_pack("openui")


def test_schema_from_pack_reuses_declared_prop_order(schema: CAP1SchemaV1) -> None:
    assert schema.dsl == "openui"
    assert "Card" in schema.component_names
    assert schema.component_prop_order["Card"] == (
        "children",
        "variant",
        "direction",
        "gap",
        "align",
        "justify",
        "wrap",
    )
    assert "text" in schema.content_props


def test_every_required_and_forbidden_fact_has_exact_provenance(
    schema: CAP1SchemaV1,
) -> None:
    frame = derive_semantic_frame(CARD_SRC, schema)
    binding = {
        (f.category, f.predicate, f.ast_path): (f.schema_field, f.value)
        for f in frame.facts
    }
    assert binding[("required", "component_type", ("typeName",))] == (
        "component_registry",
        "Card",
    )
    assert binding[("required", "cardinality", ("props", "children"))] == (
        "cardinality",
        1,
    )
    assert binding[
        ("required", "component_type", ("props", "children", 0, "typeName"))
    ] == ("component_registry", "TextContent")
    assert binding[
        (
            "required",
            "content_placeholder_id",
            ("props", "children", 0, "props", "text"),
        )
    ] == ("content_props", ":ns.body")
    assert binding[
        ("unspecified", "content_text", ("props", "children", 0, "props", "text"))
    ] == ("content_props", None)

    for fact in frame.facts:
        if fact.category in {"required", "forbidden"}:
            assert isinstance(fact.ast_path, tuple)
            assert fact.schema_field


def test_closed_value_facts_declare_required_and_forbidden_alternatives(
    schema: CAP1SchemaV1,
) -> None:
    frame = derive_semantic_frame(CALLOUT_SRC, schema)
    required = [f for f in frame.facts if f.predicate == "closed_value"]
    forbidden = [f for f in frame.facts if f.predicate == "excluded_value"]
    assert len(required) == 1 and required[0].value is True
    assert len(forbidden) == 1 and forbidden[0].value is False
    assert required[0].ast_path == forbidden[0].ast_path == ("props", "visible")
    assert not frame_conflicts(frame)


def test_equivalent_canonical_asts_yield_equivalent_frames(
    schema: CAP1SchemaV1,
) -> None:
    frame_a = derive_semantic_frame(CARD_SRC, schema)
    frame_b = derive_semantic_frame(CARD_SRC_RENAMED, schema)
    assert frame_a.canonical_source == frame_b.canonical_source
    assert frame_a.equivalent_to(frame_b)
    assert frozenset(frame_a.facts) == frozenset(frame_b.facts)


def test_distinct_layouts_are_not_equivalent(schema: CAP1SchemaV1) -> None:
    frame_a = derive_semantic_frame(CARD_SRC, schema)
    frame_b = derive_semantic_frame(CALLOUT_SRC, schema)
    assert not frame_a.equivalent_to(frame_b)


def test_relations_and_effects_from_query_and_mutation_refs(
    schema: CAP1SchemaV1,
) -> None:
    query_frame = derive_semantic_frame(
        'items = Query("get_items", {})\nroot = Button(":actions.save", items)\n',
        schema,
    )
    assert any(r.kind == "reads_query" for r in query_frame.relations)
    assert not query_frame.effects  # a read has no side effect

    mutation_frame = derive_semantic_frame(
        'save = Mutation("save_item", {})\nroot = Button(":actions.save", save)\n',
        schema,
    )
    assert any(r.kind == "invokes_mutation" for r in mutation_frame.relations)
    assert any(e.effect_kind == "mutation" for e in mutation_frame.effects)


def test_state_ref_produces_reads_state_relation(schema: CAP1SchemaV1) -> None:
    frame = derive_semantic_frame(
        "$gap = 8\n"
        'root = Card([TextContent(":ns.body", 14)], "filled", "column", $gap, '
        '"start", "start", "nowrap")\n',
        schema,
    )
    assert any(r.kind == "reads_state" for r in frame.relations)


# -- fail-closed negative controls (acceptance: "unsupported facts fail closed") --


def test_unknown_component_fails_closed_instead_of_guessing(
    schema: CAP1SchemaV1,
) -> None:
    """`Text` parses fine syntactically but is not in the declared component
    vocabulary (only `TextContent` is) — a real, non-vacuous negative control."""
    # Fail-closed either way: with the official @openuidev bridge installed,
    # upstream validation raises ParseError first; the Lark backend reaches
    # derive_semantic_frame, which raises UnsupportedSemanticFactError.
    with pytest.raises((UnsupportedSemanticFactError, ParseError)):
        derive_semantic_frame('root = Text(":ns.body")', schema)


def test_excess_positional_args_fail_closed_instead_of_guessing(
    schema: CAP1SchemaV1,
) -> None:
    """More positional args than the schema declares names for become the Lark
    backend's `_args` catch-all, which has no schema-declared role."""
    # Same backend-dependent layer as the unknown-component control above.
    with pytest.raises((UnsupportedSemanticFactError, ParseError)):
        derive_semantic_frame('root = TextContent(":ns.body", 14, 99)', schema)


def test_value_outside_declared_closed_domain_fails_closed(
    schema: CAP1SchemaV1,
) -> None:
    domain_schema = replace(
        schema, closed_value_domains={"direction": ("row", "column")}
    )
    with pytest.raises(UnsupportedSemanticFactError):
        derive_semantic_frame(
            'root = Card([TextContent(":ns.body", 14)], "filled", "diagonal", 8, '
            '"start", "start", "nowrap")\n',
            domain_schema,
        )


def test_unrecognized_builtin_effect_call_fails_closed(schema: CAP1SchemaV1) -> None:
    """`@Run`/`@Set`/`@Count` are real OpenUI builtins (see
    ``slm_training.dsl.openui_tokens.STRUCTURAL_TOKENS``) that are only decode-time
    token priors, not a hard schema authority this module re-derives; the deriver
    fails closed on them rather than guessing an effect/role for each one."""
    v05_src = (
        "root = Stack([button, count])\n"
        '$filter = "all"\n'
        'items = Query("get_items", {filter: $filter}, {rows: []})\n'
        'save = Mutation("save_item", {filter: $filter})\n'
        'submit = Action([@Run(save), @Run(items), @Set($filter, "all")])\n'
        'button = Button(":actions.save", submit)\n'
        'count = TextContent("" + @Count(items.rows))'
    )
    with pytest.raises(UnsupportedSemanticFactError):
        derive_semantic_frame(v05_src, schema)


def test_default_value_downgrades_matching_fact_to_optional(
    schema: CAP1SchemaV1,
) -> None:
    default_schema = replace(schema, default_values={("Card", "wrap"): "nowrap"})
    frame = derive_semantic_frame(CARD_SRC, default_schema)
    wrap_facts = [f for f in frame.facts if f.ast_path == ("props", "wrap")]
    assert len(wrap_facts) == 1
    assert wrap_facts[0].category == "optional"


def test_frame_conflicts_detects_a_contradictory_fact_pair(
    schema: CAP1SchemaV1,
) -> None:
    frame = derive_semantic_frame(CALLOUT_SRC, schema)
    contradiction = replace(
        frame,
        facts=frame.facts
        + (
            replace(
                [f for f in frame.facts if f.predicate == "closed_value"][0],
                category="forbidden",
                predicate="excluded_value",
            ),
        ),
    )
    conflicts = frame_conflicts(contradiction)
    assert conflicts
    assert conflicts[0].ast_path == ("props", "visible")


# -- inverse frame-to-constraint (candidate/ambiguity evaluation) --


def test_constraint_is_satisfied_by_the_same_candidate_and_violated_by_a_divergent_one(
    schema: CAP1SchemaV1,
) -> None:
    frame = derive_semantic_frame(CALLOUT_SRC, schema)
    constraint = frame_to_constraint(frame)

    same = check_constraint(constraint, CALLOUT_SRC, schema)
    assert same.satisfied and not same.violations

    divergent_src = 'root = Callout("info", ":ns.title", ":ns.desc", false)'
    divergent = check_constraint(constraint, divergent_src, schema)
    assert not divergent.satisfied
    assert any(v.clause.ast_path == ("props", "visible") for v in divergent.violations)


def test_constraint_check_rejects_a_schema_fingerprint_mismatch(
    schema: CAP1SchemaV1,
) -> None:
    frame = derive_semantic_frame(CALLOUT_SRC, schema)
    constraint = frame_to_constraint(frame)
    other_schema = replace(
        schema, closed_value_domains={"direction": ("row", "column")}
    )
    with pytest.raises(UnsupportedSemanticFactError):
        check_constraint(constraint, CALLOUT_SRC, other_schema)


# -- one-fact counterfactual --


def test_mutate_one_fact_proves_exactly_one_fact_changed(schema: CAP1SchemaV1) -> None:
    frame = derive_semantic_frame(CALLOUT_SRC, schema)
    mutated, proof = mutate_one_fact(frame)

    assert proof.single_fact_change
    assert proof.old_value is True and proof.new_value is False
    assert proof.changed_ast_path == ("props", "visible")
    assert not frame.equivalent_to(mutated)
    assert not frame_conflicts(mutated)

    # Independently re-verify the one-fact-change proof.
    reproof = verify_single_fact_change(frame, mutated)
    assert reproof == proof

    # The mutated frame's constraint is violated by the original source.
    constraint = frame_to_constraint(mutated)
    result = check_constraint(constraint, CALLOUT_SRC, schema)
    assert not result.satisfied


def test_mutate_one_fact_refuses_to_guess_without_a_closed_domain(
    schema: CAP1SchemaV1,
) -> None:
    """No bool/closed-domain fact exists to flip; refuses rather than guessing
    a replacement for an open-domain literal (a real negative control)."""
    frame = derive_semantic_frame('root = TextContent(":ns.body", 14)', schema)
    with pytest.raises(CounterfactualError):
        mutate_one_fact(frame)
