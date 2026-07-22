"""B2 (SLM-22): canonical-space train/decode alignment through the lang-core bridge.

Every codec that produces training targets or decodes model output must agree
with the official ``@openuidev/lang-core`` serializer — the canonical form per
``docs/design/task-equivalence-eval.md``. These property tests pin, per codec:

* **production codec** (``dsl/production_codec.py``) — decode of a canonical
  input is itself a serializer fixed point, and the token stream (the loss
  space) is a fixed point of decode (regression for #266, extended to v0.5
  state/query programs).
* **DSLNativeTokenizer** (``models/dsl_tokenizer.py``) — ``canonicalize`` of a
  canonical input is a serializer fixed point (same whitespace, quote style,
  prop order, statement order; alpha-renaming is the documented collapse and
  is itself preserved by the serializer), ids are decode fixed points, and
  encode→decode preserves the lang-core resolved structure.
* **training-target seam** (``data.contract.normalize_example_record``) — the
  record shape both output tokenizers (lexer-native and compositional
  ``models/tokenizer.py``) encode targets from is a serializer fixed point,
  so loss is computed in canonical space for both paths.

All checks require the Node bridge: without it ``validate(x).serialized`` is a
Lark input echo, not a canonical form.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from slm_training.data.contract import (
    canonical_slot_contract,
    normalize_example_record,
)
from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.lang_core import parse as lang_core_parse
from slm_training.dsl.parser import validate
from slm_training.dsl.production_codec import decode_productions, encode_openui
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable
from slm_training.models.tokenizer import OpenUITokenizer

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

# Representative programs: multi-statement with unreferenced bindings, mixed
# prop shapes (bool / null padding), escaped strings, name-aliasing literals,
# and v0.5 state/query/mutation/action.
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
        'b = Input("email", ":m.email", "text", null)'
    ),
    "escaped_name": (
        'root = Form("quote \\"q\\" name", btns, [f])\n'
        'f = FormControl(":f.label", i)\n'
        'i = Input("email", ":f.email")\n'
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


def _canonical(source: str) -> str:
    """Official lang-core serialized form — the canonical reference."""
    serialized = validate(source).serialized
    assert serialized, "bridge must produce a serialized canonical form"
    return serialized


def _alpha_normalized_ast(source: str) -> Any:
    """Resolved root AST with statement-bound names renamed positionally.

    The deterministic decoders alpha-rename binders and state names; the
    official serializer preserves them. Canonical equality is therefore
    resolved-structure equality modulo statement-bound names (statementId
    erased). Names not bound by a statement (member access, object keys,
    string literals) must survive exactly.
    """
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


def _all_sources() -> list[tuple[str, str, list[str]]]:
    cases = [(name, src, []) for name, src in CORPUS.items()]
    cases.extend(
        (record.id, record.openui, list(record.placeholders))
        for record in _fixture_records()
    )
    return cases


# --- (a) production codec: regression for #266, extended to v0.5 ------------


def test_production_decode_emits_langcore_canonical_form() -> None:
    """For canonical input, decode output is a serializer fixed point."""
    for name, source, placeholders in _all_sources():
        canonical = _canonical(source)
        contract = canonical_slot_contract(canonical, declared=placeholders)
        program = encode_openui(canonical, slot_contract=contract)
        decoded = decode_productions(program.tokens, program.slot_contract)
        assert _canonical(decoded) == decoded, name


def test_production_token_stream_is_fixed_point_of_decode() -> None:
    """The loss space (token stream) is a fixed point of the decode collapse."""
    for name, source, placeholders in _all_sources():
        contract = canonical_slot_contract(source, declared=placeholders)
        program = encode_openui(source, slot_contract=contract)
        decoded = decode_productions(program.tokens, program.slot_contract)
        reencoded = encode_openui(decoded, slot_contract=program.slot_contract)
        assert reencoded.tokens == program.tokens, name


def test_production_roundtrip_is_canonical_equal_modulo_alpha() -> None:
    for name, source, placeholders in _all_sources():
        contract = canonical_slot_contract(source, declared=placeholders)
        program = encode_openui(source, slot_contract=contract)
        decoded = decode_productions(program.tokens, program.slot_contract)
        assert _alpha_normalized_ast(decoded) == _alpha_normalized_ast(source), name


# --- (b) DSLNativeTokenizer (lexer-native output path) ----------------------


@pytest.fixture(scope="module")
def dsl_tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def _encodable_dsl_sources(
    tokenizer: DSLNativeTokenizer,
) -> list[tuple[str, str, list[str]]]:
    """Return only symbol/enum targets accepted by the hard output contract."""
    accepted: list[tuple[str, str, list[str]]] = []
    for case in _all_sources():
        _name, source, placeholders = case
        table = SymbolTable.from_placeholders(
            placeholders, max_slots=tokenizer.sym_slots
        )
        try:
            tokenizer.encode(source, add_special=False, table=table)
        except ValueError as exc:
            assert "free-form output string is forbidden" in str(exc)
        else:
            accepted.append(case)
    return accepted


def test_dsl_native_rejects_free_form_targets(dsl_tok: DSLNativeTokenizer) -> None:
    accepted = {name for name, _source, _slots in _encodable_dsl_sources(dsl_tok)}
    rejected = {name for name, _source, _slots in _all_sources()} - accepted

    assert "escaped_name" in rejected
    assert rejected


def test_dsl_native_canonicalize_agrees_with_langcore_serializer(
    dsl_tok: DSLNativeTokenizer,
) -> None:
    """canonicalize of a canonical input is itself a serializer fixed point.

    This pins whitespace, quote style, positional prop order, and statement
    order to the official serializer. The only remaining collapse is the
    documented deterministic alpha-renaming (root, b1, …, $s0, …), which the
    serializer preserves — so any spacing/order drift in ``_pretty_print``
    breaks the fixed point and fails here.
    """
    for name, source, placeholders in _encodable_dsl_sources(dsl_tok):
        canonical = _canonical(source)
        table = SymbolTable.from_placeholders(
            placeholders, max_slots=dsl_tok.sym_slots
        )
        collapsed = dsl_tok.canonicalize(canonical, table=table)
        assert _canonical(collapsed) == collapsed, name


def test_dsl_native_ids_are_fixed_point_of_decode(
    dsl_tok: DSLNativeTokenizer,
) -> None:
    """Training-target ids re-encode from their own decode (loss space fixed)."""
    for name, source, placeholders in _encodable_dsl_sources(dsl_tok):
        table = SymbolTable.from_placeholders(
            placeholders, max_slots=dsl_tok.sym_slots
        )
        ids = dsl_tok.encode(source, add_special=False, table=table)
        decoded = dsl_tok.decode(ids, table=table)
        fresh = SymbolTable.from_placeholders(
            placeholders, max_slots=dsl_tok.sym_slots
        )
        assert dsl_tok.encode(decoded, add_special=False, table=fresh) == ids, name


def test_dsl_native_roundtrip_is_canonical_equal_modulo_alpha(
    dsl_tok: DSLNativeTokenizer,
) -> None:
    for name, source, placeholders in _encodable_dsl_sources(dsl_tok):
        table = SymbolTable.from_placeholders(
            placeholders, max_slots=dsl_tok.sym_slots
        )
        ids = dsl_tok.encode(source, add_special=False, table=table)
        decoded = dsl_tok.decode(ids, table=table)
        assert _alpha_normalized_ast(decoded) == _alpha_normalized_ast(source), name


# --- (c) training-target seam: both output tokenizers -----------------------


def test_normalized_records_are_langcore_canonical_fixed_points() -> None:
    """The target-construction seam emits official-serializer canonical text.

    ``normalize_example_record`` is the owner that canonicalizes document
    records before training targets are encoded (train_data pipeline →
    records.jsonl → ``TwoTowerModel._encode_openui``). Raw seeds are allowed
    to be non-canonical — the seam must fix them.
    """
    raw_divergent = 0
    for record in _fixture_records():
        if record.openui != _canonical(record.openui):
            raw_divergent += 1
        normalized = normalize_example_record(record)
        assert normalized.openui == _canonical(normalized.openui), record.id
        again = normalize_example_record(normalized)
        assert again.openui == normalized.openui, record.id
    # The seam does real work: some committed seeds are not serializer
    # fixed points (statement order) and must be collapsed before training.
    assert raw_divergent > 0


def test_compositional_targets_from_normalized_records_are_decode_fixed() -> None:
    """models/tokenizer.py is an identity codec; targets must enter canonical."""
    records = [normalize_example_record(r) for r in _fixture_records()]
    tokenizer = OpenUITokenizer.build([r.openui for r in records])
    for record in records:
        decoded = tokenizer.decode(tokenizer.encode(record.openui))
        assert decoded == record.openui, record.id
        assert decoded == _canonical(decoded), record.id


def test_lexer_targets_from_normalized_records_are_decode_fixed(
    dsl_tok: DSLNativeTokenizer,
) -> None:
    for record in _fixture_records():
        normalized = normalize_example_record(record)
        table = SymbolTable.from_placeholders(
            normalized.placeholders, max_slots=dsl_tok.sym_slots
        )
        try:
            ids = dsl_tok.encode(
                normalized.openui, add_special=False, table=table
            )
        except ValueError as exc:
            assert "free-form output string is forbidden" in str(exc), record.id
            continue
        decoded = dsl_tok.decode(ids, table=table)
        fresh = SymbolTable.from_placeholders(
            normalized.placeholders, max_slots=dsl_tok.sym_slots
        )
        assert (
            dsl_tok.encode(decoded, add_special=False, table=fresh) == ids
        ), record.id
        assert _canonical(decoded) == decoded, record.id
