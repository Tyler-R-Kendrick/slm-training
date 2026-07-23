"""B1 (SLM-42): pure grammar-choice stream property tests.

The choice codec lives in canonical space (see ``encode_choices``): laws are
the same shape as the B2 canonical-alignment invariants
(``tests/test_dsl/test_canonical_alignment.py``):

* the token stream (the loss space) is a fixed point of ``decode_choices``;
* decode output is a lang-core serializer fixed point;
* decode preserves the resolved AST of the canonical form, modulo the
  documented positional alpha-renaming.

The serializer's own collapses (statement reorder, dead-binding and trailing
``null`` pruning, redundant-paren removal) are part of the canonical space and
therefore of the codec's target — encode canonicalizes first so those
collapses never break the fixed-point laws.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from slm_training.data.contract import canonical_slot_contract
from slm_training.dsl.lang_core import ParseError, bridge_available
from slm_training.dsl.lang_core import parse as lang_core_parse
from slm_training.dsl.parser import validate
from slm_training.dsl.production_codec import (
    CHOICE_STMT_MARKERS,
    OP_PREFIX,
    PUNCT_PREFIX,
    STMT,
    decode_choices,
    encode_choices,
    encode_openui,
    roundtrip_choices,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.evals.semantic_bits import (
    categorize_choice,
    compare_representations,
    semantic_bits,
)

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

# Mirrors the canonical-alignment CORPUS plus choice-specific stressors
# (ternary + state-first ordering, redundant parens, structural dead binding).
CORPUS: dict[str, str] = {
    "hero": (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        'hero_body = TextContent(":hero.body")\n'
        "hero = Card([hero_title, hero_body])"
    ),
    "dead_binding_chain": (
        'root = Card([t])\n'
        't = TextContent(":x.t")\n'
        'orphan = Button(":cta.label", act)\n'
        'act = Action([@ToAssistant(":msg.hi")])'
    ),
    "modal_bool_null": (
        'root = Modal(":m.title", true, [b])\n'
        'b = Input("$0", ":m.email", "text", null)'
    ),
    "escaped_name": (
        'root = Form("quote \\"q\\" name", btns, [f])\n'
        'f = FormControl(":f.label", i)\n'
        'i = Input("$0", ":f.email")\n'
        "btns = Buttons([s])\n"
        's = Button(":f.submit")'
    ),
    "name_in_literal": (
        'root = Stack([title, hero], "column")\n'
        'title = TextContent(":smoke.hero.kicker")\n'
        'head = CardHeader(":smoke.hero.title", ":smoke.hero.subtitle")\n'
        "hero = Card([head])"
    ),
    "v05_state_query": (
        "root = Stack([button, count])\n"
        '$filter = "all"\n'
        'items = Query("get_items", {filter: $filter}, {rows: []})\n'
        'save = Mutation("save_item", {filter: $filter})\n'
        'submit = Action([@Run(save), @Run(items), @Set($filter, "all")])\n'
        'button = Button(":actions.save", submit)\n'
        'count = TextContent("" + @Count(items.rows))'
    ),
    "v05_state_first_ternary": (
        "$count = 0\n"
        'root = Stack([hello], "column")\n'
        'hello = TextContent(@Count($count) > 0 ? ":a" : ":b")'
    ),
    "v05_arithmetic_parens": 'root = TextContent("" + (1 + 2 * 3))',
    "dead_binding_structural": (
        'root = Card(":card.title")\norphan = Button(":cta.label")'
    ),
}

_BINDER_RE = re.compile(r"(?m)^\s*(\$?[A-Za-z_][A-Za-z0-9_]*)\s*=")


def _fixture_records() -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for path in (
        "src/slm_training/resources/train_seeds.jsonl",
        "src/slm_training/resources/test_seeds.jsonl",
    ):
        records.extend(load_jsonl(path))
    return records


def _all_sources() -> list[tuple[str, str, list[str]]]:
    cases = [(name, src, []) for name, src in CORPUS.items()]
    cases.extend(
        (record.id, record.openui, list(record.placeholders))
        for record in _fixture_records()
    )
    return cases


def _canonical(source: str) -> str:
    serialized = validate(source).serialized
    assert serialized, "bridge must produce a serialized canonical form"
    return serialized


def _alpha_normalized_ast(source: str) -> Any:
    """Resolved root AST with statement-bound names renamed positionally."""
    rename: dict[str, str] = {}
    for match in _BINDER_RE.finditer(source):
        name = match.group(1)
        if name not in rename:
            prefix = "$a" if name.startswith("$") else "a"
            rename[name] = f"{prefix}{len(rename)}"

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: (
                    rename.get(value, value)
                    if key == "n" and isinstance(value, str)
                    else walk(value)
                )
                for key, value in node.items()
                if key != "statementId"
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(lang_core_parse(source).root)


# --- B2-shaped canonical-alignment laws --------------------------------------


def test_choice_stream_is_fixed_point_of_decode() -> None:
    """choices → decode(serialize) → re-encode → identical choices."""
    for name, source, placeholders in _all_sources():
        contract = canonical_slot_contract(source, declared=placeholders)
        program = encode_choices(source, slot_contract=contract)
        decoded = decode_choices(program.tokens, program.slot_contract)
        reencoded = encode_choices(decoded, slot_contract=program.slot_contract)
        assert reencoded.tokens == program.tokens, name


def test_choice_decode_emits_langcore_canonical_form() -> None:
    """Decode output is itself a serializer fixed point (routed through it)."""
    for name, source, placeholders in _all_sources():
        contract = canonical_slot_contract(source, declared=placeholders)
        program = encode_choices(source, slot_contract=contract)
        decoded = decode_choices(program.tokens, program.slot_contract)
        assert _canonical(decoded) == decoded, name


def test_choice_roundtrip_is_canonical_equal_modulo_alpha() -> None:
    """AST equality vs the canonical form of the input, modulo alpha-renaming.

    Compared against the canonical form (not the raw source) because the
    codec's target space IS canonical space: the official serializer prunes
    dead bindings / trailing nulls and drops redundant parens, and those
    collapses are inherited deliberately (documented in the iter doc).
    """
    for name, source, placeholders in _all_sources():
        canonical = _canonical(source)
        contract = canonical_slot_contract(canonical, declared=placeholders)
        program = encode_choices(source, slot_contract=contract)
        decoded = decode_choices(program.tokens, program.slot_contract)
        assert _alpha_normalized_ast(decoded) == _alpha_normalized_ast(
            canonical
        ), name


def test_choice_encode_is_order_invariant() -> None:
    """Encoding the canonical form yields the identical choice stream."""
    for name, source, placeholders in _all_sources():
        canonical = _canonical(source)
        contract = canonical_slot_contract(canonical, declared=placeholders)
        program = encode_choices(source, slot_contract=contract)
        from_canonical = encode_choices(canonical, slot_contract=contract)
        assert from_canonical.tokens == program.tokens, name


# --- pure-choice stream shape -------------------------------------------------


def test_choice_stream_has_no_punct_or_stmt_markers_on_structural_path() -> None:
    program = encode_choices(CORPUS["hero"])
    assert all(not tok.startswith(PUNCT_PREFIX) for tok in program.tokens)
    assert STMT not in program.tokens
    assert all(tok not in CHOICE_STMT_MARKERS for tok in program.tokens)


def test_choice_stream_v05_keeps_operator_choices_only() -> None:
    program = encode_choices(CORPUS["v05_state_first_ternary"])
    assert all(not tok.startswith(PUNCT_PREFIX) for tok in program.tokens)
    assert any(tok.startswith(OP_PREFIX) for tok in program.tokens)
    # v0.5 keeps statement-production markers (state vs binder is a decision).
    assert program.tokens[0] in CHOICE_STMT_MARKERS


def test_choice_stream_is_smaller_than_production_stream() -> None:
    for name, source, placeholders in _all_sources():
        contract = canonical_slot_contract(source, declared=placeholders)
        choice = encode_choices(source, slot_contract=contract)
        production = encode_openui(source, slot_contract=contract)
        assert len(choice.tokens) <= len(production.tokens), name


# --- fail-closed behavior ------------------------------------------------------


def test_choice_forward_reference_to_root_is_pruned_in_canonical_space() -> None:
    """A dead binding forward-referencing root is pruned by the serializer.

    encode_choices canonicalizes first, so unlike ``encode_openui`` (which
    fails closed on the raw source) the canonical-space stream simply omits
    the dead statement; the surviving stream still round-trips.
    """
    source = 'root = Card(":card.title")\necho = Stack([root], "column")'
    program = encode_choices(source)
    decoded = decode_choices(program.tokens, program.slot_contract)
    assert "echo" not in decoded and "Stack" not in decoded
    assert (
        encode_choices(decoded, slot_contract=program.slot_contract).tokens
        == program.tokens
    )


def test_choice_reference_cycle_fails_closed() -> None:
    source = 'root = Card([echo])\necho = Stack([root], "column")'
    with pytest.raises(ParseError):
        encode_choices(source)


def test_choice_missing_placeholder_fails_closed() -> None:
    # A declared-only contract that lacks the used placeholder must fail once
    # merged extraction is bypassed at decode time (out-of-range pointer).
    program = encode_choices(CORPUS["hero"])
    with pytest.raises(ParseError, match="slot pointer out of range"):
        decode_choices(program.tokens, program.slot_contract[:1])


def test_choice_out_of_scope_ref_fails_closed() -> None:
    with pytest.raises(ParseError, match="statement ref out of range"):
        decode_choices(("+Card", "&0", "-"), ())


def test_choice_unknown_component_fails_closed() -> None:
    with pytest.raises(ParseError):
        encode_choices('root = FancyWidget(":x")')


def test_choice_invalid_reconstruction_fails_closed() -> None:
    # Truncated stream: unterminated component call.
    with pytest.raises(ParseError):
        decode_choices(("+Card", "@0"), (":card.title",))


def test_choice_empty_stream_fails_closed() -> None:
    with pytest.raises(ParseError, match="empty choice stream"):
        decode_choices((), ())


# --- E2 semantic density --------------------------------------------------------


def test_choice_categories_collapse_surface_residue() -> None:
    """punct == structural == 0 on the choice stream; name is the documented
    irreducible remainder (object keys / member names are genuine key choices
    and appear only on the v0.5 path)."""
    records = _fixture_records()
    report = semantic_bits(records, stream="choice")
    cats = report["by_category"]
    assert cats.get("punct", 0) == 0
    assert cats.get("structural", 0) == 0
    # Fixture corpus is structural-only: no name/member/operator residue.
    assert cats.get("name", 0) == 0
    assert cats.get("member", 0) == 0

    # v0.5 programs keep name (object keys) and member tokens — a real choice.
    v05 = semantic_bits([CORPUS["v05_state_query"]], stream="choice")
    assert v05["by_category"].get("punct", 0) == 0
    assert v05["by_category"].get("structural", 0) == 0
    assert v05["by_category"].get("name", 0) > 0  # documented remainder


def test_choice_semantic_density_improves_on_production() -> None:
    report = compare_representations(_fixture_records())
    assert report["surface_to_choice_bit_ratio"] is not None
    assert (
        report["surface_to_choice_bit_ratio"]
        >= report["surface_to_production_bit_ratio"]
    )
    assert report["production_to_choice_bit_ratio"] >= 1.0
    assert (
        report["choice"]["n_decisions"] <= report["production"]["n_decisions"]
    )


def test_categorize_choice_token_classes() -> None:
    assert categorize_choice("o:+") == "operator"
    assert categorize_choice("o:?:") == "operator"
    assert categorize_choice(".rows") == "member"
    assert categorize_choice("[") == "arity"
    assert categorize_choice("-") == "arity"
    assert categorize_choice("{") == "arity"
    assert categorize_choice("r=") == "statement"
    assert categorize_choice("$=") == "statement"
    assert categorize_choice("+Card") == "production"
    assert categorize_choice("@0") == "slot"
    assert categorize_choice("$@0") == "state_ref"
    assert categorize_choice("&1") == "reference"
    assert categorize_choice('#"x"') == "literal"
    assert categorize_choice("n:filter") == "name"


# --- convenience entry ----------------------------------------------------------


def test_roundtrip_choices_helper() -> None:
    program, decoded = roundtrip_choices(
        CORPUS["hero"], slot_contract=[":hero.title", ":hero.body"]
    )
    assert program.slot_contract[:2] == (":hero.title", ":hero.body")
    assert _canonical(decoded) == decoded
