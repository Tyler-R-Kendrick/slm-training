"""Structure-only scrubbing and style-agnostic eval."""

from __future__ import annotations

from slm_training.data.leakage import normalize_openui_structure
from slm_training.data.structure import strip_style_literals
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.eval_runner import structural_similarity
from slm_training.preference import composite_reward


def test_strip_style_removes_gap_and_typography() -> None:
    styled = (
        'root = Stack([a, b], "column", "2xl")\n'
        'a = TextContent(":a.title", "large-heavy")\n'
        'b = Button(":b.cta", "primary")'
    )
    clean = strip_style_literals(styled)
    assert '"2xl"' not in clean
    assert "large-heavy" not in clean
    assert "primary" not in clean
    assert '"column"' in clean
    assert ":a.title" in clean


def test_structural_similarity_ignores_style_args() -> None:
    gold = 'root = Stack([a], "column")\na = TextContent(":t")'
    pred = 'root = Stack([a], "column", "m")\na = TextContent(":t", "large-heavy")'
    assert structural_similarity(pred, gold) == 1.0


def test_structure_fingerprint_ignores_gap() -> None:
    a = 'root = Stack([x], "column", "m")\nx = Button(":x")'
    b = 'root = Stack([x], "column")\nx = Button(":x")'
    assert normalize_openui_structure(a) == normalize_openui_structure(b)


def test_eval_reward_ignores_design_md_style() -> None:
    openui = 'root = Stack([cta], "column")\ncta = Button(":cta")'
    fancy = (
        "---\ncolors:\n  primary: \"#FF0000\"\n---\n"
        "## Overview\nStyled theme that must not change eval reward.\n"
    )
    gold = ExampleRecord(
        id="t",
        prompt="button",
        openui=openui,
        placeholders=[":cta"],
        design_md=fancy,
    )
    bare = composite_reward(openui, gold=gold, design_md=None)
    # Explicit None path is structure-only; passing fancy must not be used by eval.
    assert bare == composite_reward(openui, gold=gold)
    assert bare > 0.5
    # Even if someone passes design_md, style warnings should not collapse score.
    with_md = composite_reward(openui, gold=gold, design_md=fancy)
    assert with_md > 0.4
