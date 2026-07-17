"""F1 (SLM-34): DSL-pack contract — registry, slots, and the e2e fixture run."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.pack import (
    DslPack,
    PackSlotUnavailable,
    get_pack,
    list_packs,
)

HERO = 'root = Stack([title], "column")\ntitle = TextContent(":hero.title")'
TOY = (
    'root = row(title, action)\n'
    'title = text(":hero.title")\n'
    'action = button(":cta.label")'
)


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_builtin_packs_registered() -> None:
    assert {"openui", "toy-layout"} <= set(list_packs())


def test_get_pack_default_is_openui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_GRAMMAR_DSL", raising=False)
    assert get_pack().pack_id == "openui"
    assert get_pack("default").pack_id == "openui"
    assert get_pack("auto").pack_id == "openui"


def test_get_pack_backend_aliases_resolve_to_openui() -> None:
    for alias in ("openui-lark", "openui-langcore", "lark-openui"):
        assert get_pack(alias).pack_id == "openui"


def test_get_pack_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_GRAMMAR_DSL", "toy-layout")
    # models.grammar.set_active_dsl state must not shadow the env default.
    import slm_training.models.grammar as grammar

    monkeypatch.setattr(grammar, "_ACTIVE_DSL", None)
    assert get_pack().pack_id == "toy-layout"


def test_get_pack_unknown_fails_closed() -> None:
    with pytest.raises(KeyError, match="unknown DSL pack"):
        get_pack("graphql-not-yet")


# ---------------------------------------------------------------------------
# OpenUI pack: every contract slot is filled and real
# ---------------------------------------------------------------------------


def test_openui_pack_fills_every_slot() -> None:
    pack = get_pack("openui")
    for slot in (
        "backend",
        "canonicalize",
        "oracle",
        "corpus_generator",
        "scope_extractor",
        "placeholder_policy",
        "prop_order",
        "incremental_engine",
    ):
        assert pack.require(slot) is not None


def test_openui_reward_label_is_honest() -> None:
    # F3 requirement: oracle-derived rewards must carry this label.
    assert get_pack("openui").reward_label == "well_formed_not_behavioral"


def test_openui_oracle_accepts_source_and_fails_closed() -> None:
    pack = get_pack("openui")
    report = pack.oracle(HERO)
    assert report.ok
    bad = pack.oracle("root = Broken(")
    assert not bad.ok


def test_openui_placeholder_policy() -> None:
    policy = get_pack("openui").placeholder_policy
    assert policy.is_placeholder(":hero.title")
    assert not policy.is_placeholder("plain text")
    assert "text" in policy.content_props
    assert policy.slot_contract(HERO) == (":hero.title",)
    assert policy.slot_contract(HERO, declared=[":extra"]) == (
        ":extra",
        ":hero.title",
    )


def test_openui_incremental_engine_slot() -> None:
    engine = get_pack("openui").incremental_engine()
    assert engine.can_complete_with_holes("root = Stack(")


# ---------------------------------------------------------------------------
# Default-parameter byte compatibility (the seams must not drift)
# ---------------------------------------------------------------------------


def test_prop_order_default_matches_openui() -> None:
    from slm_training.dsl.production_codec import _prop_order

    assert _prop_order() == _prop_order("openui")
    assert _prop_order("openui-lark") == _prop_order("openui")


def test_encode_openui_dsl_param_is_byte_identical_for_default() -> None:
    from slm_training.dsl.production_codec import encode_openui

    assert encode_openui(HERO) == encode_openui(HERO, dsl="openui")


def test_canonicalize_dsl_param_is_byte_identical_for_default() -> None:
    from slm_training.dsl.canonicalize import canonicalize

    assert canonicalize(HERO) == canonicalize(HERO, dsl="openui")


def test_verify_stack_verdicts_unchanged() -> None:
    from slm_training.data.verify import Tier, verify_record
    from slm_training.dsl.schema import ExampleRecord

    good = ExampleRecord(
        id="pack-good",
        prompt="hero",
        openui=HERO,
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    report = verify_record(good)
    assert report.ok and report.tier is Tier.SILVER
    bad = ExampleRecord(
        id="pack-bad",
        prompt="broken",
        openui="root = Broken(",
        split="train",
        source="fixture",
    )
    bad_report = verify_record(bad)
    assert not bad_report.ok and bad_report.tier is Tier.QUARANTINE
    assert bad_report.failing_gate is not None


# ---------------------------------------------------------------------------
# Toy-layout: a second pack resolves; filled slots work; gaps fail closed
# ---------------------------------------------------------------------------


def test_toy_layout_filled_slots_work() -> None:
    pack = get_pack("toy-layout")
    program = pack.backend.parse(TOY)
    assert program.root is not None
    slices = pack.scope_extractor(TOY)
    assert {s.scope for s in slices} == {
        "document",
        "statement",
        "expression",
        "lexical",
    }
    assert pack.prop_order()["text"] == ["text", "size"]
    assert pack.placeholder_policy.slot_contract(TOY) == (
        ":hero.title",
        ":cta.label",
    )
    engine = pack.incremental_engine()
    assert engine.can_complete_with_holes("root = row(")
    assert pack.reward_label == "parse_only"


def test_toy_layout_missing_slots_fail_closed() -> None:
    pack = get_pack("toy-layout")
    for slot in ("canonicalize", "oracle", "corpus_generator"):
        with pytest.raises(PackSlotUnavailable, match=f"'toy-layout'.*{slot!r}"):
            pack.require(slot)
    with pytest.raises(AttributeError, match="unknown pack slot"):
        pack.require("e_graph")


def test_engine_for_dsl_consults_pack_registry() -> None:
    from slm_training.dsl.grammar.fastpath import engine_for_dsl

    assert engine_for_dsl("toy-layout") is not None
    assert engine_for_dsl("openui") is not None
    assert engine_for_dsl("no-such-dsl") is None


# ---------------------------------------------------------------------------
# End-to-end fixture run through the pack interface:
# generate -> verify -> canonicalize -> scope -> train scratch (stub) -> eval
# ---------------------------------------------------------------------------


def test_end_to_end_fixture_run_through_pack_interface() -> None:
    from slm_training.data.progspec.schema import emit_record
    from slm_training.data.verify import Tier
    from slm_training.dsl.canonicalize import canonical_equal
    from slm_training.harnesses.model_build import ModelBuildConfig, build_model

    pack = get_pack("openui")

    # 1. Generate a typed-AST corpus (N=2, seeded).
    generator = pack.corpus_generator(seed=0)
    result = generator.generate(2)
    assert len(result.programs) == 2

    records = []
    for index, spec in enumerate(result.programs):
        source = spec.canonical_openui

        # 2. Oracle: every generated program passes the gate stack.
        report = pack.oracle(source)
        assert report.ok and report.tier is Tier.SILVER

        # 3. Canonicalizer: idempotent normal form.
        canonical = pack.canonicalize(source)
        assert pack.canonicalize(canonical) == canonical

        # 4. Scope rules: document/statement scopes always present.
        scopes = {s.scope for s in pack.scope_extractor(source)}
        assert {"document", "statement"} <= scopes

        # 5. Placeholder policy: slot contract covers extracted placeholders.
        contract = pack.placeholder_policy.slot_contract(source)
        assert set(pack.placeholder_policy.extract(source)) == set(contract)

        records.append(
            emit_record(
                spec, prompt=f"fixture prompt {index}", task="generation"
            )
        )

    # 6. Train scratch: stub model build threads the pack id through
    #    ModelBuildConfig.grammar_dsl (factory calls set_active_dsl with it).
    config = ModelBuildConfig(
        train_dir=Path("."), model_name="stub", grammar_dsl=pack.pack_id
    )
    assert config.grammar_dsl == pack.pack_id
    model = build_model(config, records)
    loss = model.forward(records)
    assert loss > 0.0

    # 7. Eval: the trained stub reproduces the gold program for a seen
    #    prompt, and the output canonical-matches gold under the pack.
    output = model.generate(records[0].prompt)
    assert canonical_equal(output, records[0].openui)
    assert pack.oracle(output).ok


def test_register_pack_roundtrip() -> None:
    import slm_training.dsl.pack as pack_mod

    base = get_pack("toy-layout")
    custom = DslPack(
        pack_id="toy-layout-test-clone",
        backend=base.backend,
        placeholder_policy=base.placeholder_policy,
        reward_label="parse_only",
    )
    pack_mod.register_pack(custom)
    try:
        assert get_pack("toy-layout-test-clone") is custom
    finally:
        pack_mod._PACKS.pop("toy-layout-test-clone", None)
