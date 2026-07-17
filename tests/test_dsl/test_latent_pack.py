"""G3 (SLM-47): the latent-DSL generator — task → grammar → instantiated pack.

Mirrors ``test_pack.py::test_toy_layout_filled_slots_work`` (NOT the full openui
e2e): synthesize a pack from a trivial task spec and prove the FILLED slots work
end-to-end while the HONEST-None slots fail closed. Every test registers/pops the
synthesized pack + backend in try/finally so the global registries stay clean.
"""

from __future__ import annotations

import contextlib

import pytest

import slm_training.dsl.grammar.backends as backends_mod
import slm_training.dsl.pack as pack_mod
from slm_training.dsl.grammar.backends import available_backends
from slm_training.dsl.latent import (
    LatentComponent,
    LatentTaskSpec,
    synthesize_grammar,
    synthesize_pack,
)
from slm_training.dsl.pack import PackSlotUnavailable, get_pack, list_packs

SPEC = LatentTaskSpec(
    task_id="kv-form",
    description="trivial key/value form task",
    components=(
        LatentComponent("row", ("children", "gap")),
        LatentComponent("field", ("label", "value")),
        LatentComponent("text", ("text", "size")),
    ),
)
PROGRAM = (
    'root = row([title, body])\n'
    'title = field(":hero.title")\n'
    'body = text(":cta.label")'
)


@contextlib.contextmanager
def _synthesized(spec: LatentTaskSpec, tmp_path):
    """Synthesize a pack into a temp grammars dir and pop it afterwards."""
    pack = synthesize_pack(spec, grammars_dir=tmp_path)
    try:
        yield pack
    finally:
        pack_mod._PACKS.pop(spec.dsl_id, None)
        backends_mod._REGISTRY.pop(spec.dsl_id, None)


def test_synthesize_grammar_is_deterministic() -> None:
    text_a, order_a = synthesize_grammar(SPEC)
    text_b, order_b = synthesize_grammar(SPEC)
    assert text_a == text_b
    assert order_a == order_b == {
        "row": ["children", "gap"],
        "field": ["label", "value"],
        "text": ["text", "size"],
    }
    # Skeleton conformance: the toy_layout rules must be present.
    for rule in ("start: statement*", "call: NAME", "STRING:", "%ignore WS_INLINE"):
        assert rule in text_a
    # Grammar-Prompting step is honestly labeled as stubbed.
    assert "STUBBED" in text_a


def test_globals_untouched_when_popped(tmp_path) -> None:
    before_packs = set(list_packs())
    before_backends = set(available_backends())
    with _synthesized(SPEC, tmp_path) as pack:
        assert pack.pack_id == "latent-kv-form"
        assert pack.pack_id in set(list_packs())
        assert pack.pack_id in set(available_backends())
    assert set(list_packs()) == before_packs
    assert set(available_backends()) == before_backends


def test_synthesized_pack_filled_slots_work(tmp_path) -> None:
    with _synthesized(SPEC, tmp_path) as pack:
        # Resolves through the registry just like a builtin pack.
        assert get_pack("latent-kv-form") is pack

        # Backend round-trip: parse -> serialize -> re-parse equal.
        parsed = pack.backend.parse(PROGRAM)
        assert parsed.root is not None
        serialized = pack.backend.serialize(parsed)
        reparsed = pack.backend.parse(serialized)
        assert reparsed.root == parsed.root

        # Scope extractor yields the four scope kinds.
        slices = pack.scope_extractor(PROGRAM)
        assert {s.scope for s in slices} == {
            "document",
            "statement",
            "expression",
            "lexical",
        }

        # Prop order comes from the task spec.
        assert pack.prop_order()["field"] == ["label", "value"]

        # Placeholder policy: slot contract covers the extracted placeholders.
        contract = pack.placeholder_policy.slot_contract(PROGRAM)
        assert contract == (":hero.title", ":cta.label")
        assert set(pack.placeholder_policy.extract(PROGRAM)) == set(contract)

        # Incremental engine: hole admissibility + next-terminal set.
        engine = pack.incremental_engine()
        assert engine.can_complete_with_holes("root = row(")
        assert engine.set_prefix("root = row")
        assert engine.next_terminals()  # non-empty accept set at the frontier

        assert pack.reward_label == "parse_only"


def test_synthesized_pack_honest_none_slots_fail_closed(tmp_path) -> None:
    with _synthesized(SPEC, tmp_path) as pack:
        assert set(pack.filled_slots()) == {
            "pack_id",
            "backend",
            "placeholder_policy",
            "reward_label",
            "scope_extractor",
            "prop_order",
            "incremental_engine",
        }
        for slot in ("canonicalize", "oracle", "corpus_generator"):
            with pytest.raises(PackSlotUnavailable, match=f"'latent-kv-form'.*{slot!r}"):
                pack.require(slot)


def test_idempotent_re_synthesis(tmp_path) -> None:
    pack_a = synthesize_pack(SPEC, grammars_dir=tmp_path)
    try:
        pack_b = synthesize_pack(SPEC, grammars_dir=tmp_path)
        assert pack_b.pack_id == pack_a.pack_id
        assert pack_b.backend.parse(PROGRAM).root == pack_a.backend.parse(PROGRAM).root
    finally:
        pack_mod._PACKS.pop(SPEC.dsl_id, None)
        backends_mod._REGISTRY.pop(SPEC.dsl_id, None)


def test_spec_from_dict_round_trip() -> None:
    spec = LatentTaskSpec.from_dict(
        {
            "task_id": "kv form",
            "description": "d",
            "components": [{"name": "row", "props": ["a", "b"]}, {"name": "leaf"}],
        }
    )
    assert spec.dsl_id == "latent-kv-form"  # slugified
    assert spec.components[0] == LatentComponent("row", ("a", "b"))
    assert spec.components[1] == LatentComponent("leaf", ())
