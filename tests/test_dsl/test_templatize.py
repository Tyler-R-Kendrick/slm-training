"""Content-literal templatizer tests (dsl.analysis.templatize)."""

from __future__ import annotations

from slm_training.dsl import validate
from slm_training.dsl.placeholders import PLACEHOLDER_RE
from slm_training.dsl.production_codec import encode_openui
from slm_training.dsl.analysis.templatize import templatize, templatize_fragment

LOGIN = (
    "hero = Card([hdr, msg])\n"
    'hdr = CardHeader("Welcome back", "Good to see you")\n'
    'msg = TextContent("Sign in to continue")\n'
    'root = Stack([hero, actions])\n'
    "actions = Buttons([b1])\n"
    'b1 = Button("Log in", "submit")'
)
LOGIN_RENAMED = (
    "zz = Buttons([qq])\n"
    'qq = Button("Log in", "submit")\n'
    "big = Card([h2, m1])\n"
    'h2 = CardHeader("Welcome back", "Good to see you")\n'
    'm1 = TextContent("Sign in to continue")\n'
    "root = Stack([big, zz])"
)


def test_content_literals_become_positional_placeholders() -> None:
    result = templatize(LOGIN)
    assert result.changed
    assert '"Welcome back"' not in result.source
    assert result.replacements[":v0.title"] == "Welcome back"
    assert result.replacements[":v0.subtitle"] == "Good to see you"
    assert result.replacements[":v1.text"] == "Sign in to continue"
    assert result.replacements[":v3.label"] == "Log in"
    assert '"submit"' not in result.source
    assert result.replacements[":v3.action"] == "submit"
    validate(result.source)


def test_generated_tokens_match_placeholder_re_and_slot_contract() -> None:
    result = templatize(LOGIN)
    for token in result.placeholders:
        assert PLACEHOLDER_RE.fullmatch(token), token
    # Slot-pointer round trip: every placeholder resolves in the codec.
    program = encode_openui(result.source, slot_contract=result.placeholders)
    assert set(result.placeholders) == set(program.slot_contract)


def test_alpha_equivalent_inputs_templatize_to_identical_bytes() -> None:
    assert templatize(LOGIN).source == templatize(LOGIN_RENAMED).source


def test_existing_placeholders_preserved_and_idempotent() -> None:
    src = 'root = Stack([t])\nt = TextContent(":acct.body")'
    result = templatize(src)
    assert not result.changed
    assert ":acct.body" in result.source
    once = templatize(LOGIN)
    twice = templatize(once.source)
    assert twice.source == once.source
    assert not twice.changed


def test_collision_bumps_ordinal_deterministically() -> None:
    # ``a`` is canonically v0 and its literal would claim :v0.text — but that
    # token already exists elsewhere in the program, so the ordinal bumps.
    src = 'root = Stack([a, b])\na = TextContent("Hello there")\nb = TextContent(":v0.text")'
    result = templatize(src)
    assert result.replacements == {":v0.text_2": "Hello there"}
    assert ":v0.text" in result.source


def test_repeated_binder_prop_pairs_get_ordinals() -> None:
    # Two inline Buttons inside one statement share (binder=v0, prop=label).
    src = 'root = Stack([w])\nw = Buttons([Button("Save"), Button("Cancel")])'
    result = templatize(src)
    assert result.replacements == {":v0.label": "Save", ":v0.label_2": "Cancel"}

    multi = 'root = Card([x, y])\nx = TextContent("One")\ny = TextContent("Two")'
    r2 = templatize(multi)
    assert r2.replacements == {":v0.text": "One", ":v1.text": "Two"}


def test_enum_is_preserved_and_array_strings_are_templatized() -> None:
    src = (
        "root = Stack([c, tags])\n"
        'c = Callout("info", ":c.title", "Longer description text")\n'
        'tags = TagBlock(["New Item Alpha"])'
    )
    result = templatize(src)
    assert result.skipped["enum_value"] == 1
    assert result.skipped["array_string"] == 0
    assert '"info"' in result.source
    assert '"New Item Alpha"' not in result.source
    assert result.replacements[":v1.tags"] == "New Item Alpha"
    assert result.replacements[":v0.description"] == "Longer description text"


def test_fragment_strings_are_templatized_without_touching_grammar_enums() -> None:
    result = templatize_fragment('Button("Save", "primary")')
    assert result.source == 'Button(":fragment.string_1", "primary")'
    assert result.replacements == {":fragment.string_1": "Save"}
