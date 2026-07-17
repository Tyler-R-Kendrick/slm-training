"""D2 canonicalizer tests."""

from __future__ import annotations

from slm_training.dsl import validate
from slm_training.dsl.canonicalize import (
    canonical_equal,
    canonical_fingerprint,
    canonicalize,
)

HERO = 'root = Stack([hero], "column")\nhero = Card([t])\nt = TextContent(":x")'
# Same layout, binders renamed (alpha-equivalent).
HERO_RENAMED = (
    'root = Stack([box], "column")\nbox = Card([label])\nlabel = TextContent(":x")'
)
CTA = 'root = Stack([cta])\ncta = Button(":c")'


def test_canonical_form_validates_and_is_idempotent() -> None:
    canonical = canonicalize(HERO)
    validate(canonical)  # must parse
    assert canonicalize(canonical) == canonical


def test_alpha_equivalent_programs_canonicalize_equal() -> None:
    assert canonicalize(HERO) == canonicalize(HERO_RENAMED)
    assert canonical_equal(HERO, HERO_RENAMED)
    assert canonical_fingerprint(HERO) == canonical_fingerprint(HERO_RENAMED)


def test_distinct_layouts_are_not_equal() -> None:
    assert not canonical_equal(HERO, CTA)
    assert canonical_fingerprint(HERO) != canonical_fingerprint(CTA)


def test_canonical_equal_is_false_on_unparseable() -> None:
    assert canonical_equal(HERO, "this is not (valid openui") is False


def test_canonicalize_raises_validate_flag_off_skips_check() -> None:
    # validate=False must not call the parser (hot-path escape hatch).
    out = canonicalize(HERO, validate=False)
    assert out == canonicalize(HERO)


def test_alpha_invariance_when_placeholders_alias_binder_names() -> None:
    """Statement order must not read binder names out of string literals.

    Regression: placeholder text like ":form.title" aliased the binders
    `form`/`title`, so canonical statement order (and thus the whole
    canonical form) depended on the original names.
    """
    src = (
        'root = Stack([title, form], "column")\n'
        'title = TextContent(":form.title")\n'
        'submit = Button(":form.submit")\n'
        'actions = Buttons([submit])\n'
        'form = Form("contact", actions, [])'
    )
    renamed = (
        'root = Stack([b1, b4], "column")\n'
        'b1 = TextContent(":form.title")\n'
        'b2 = Button(":form.submit")\n'
        'b3 = Buttons([b2])\n'
        'b4 = Form("contact", b3, [])'
    )
    assert canonicalize(src) == canonicalize(renamed)
