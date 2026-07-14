"""Tests for DESIGN.md bridge and preference rewards."""

from __future__ import annotations

import pytest

from slm_training.dsl.design_md import bridge_available, lint, load_default_design_md
from slm_training.harnesses.preference import build_pairs_from_candidates, composite_reward


@pytest.mark.skipif(not bridge_available(), reason="design_md bridge missing")
def test_default_design_md_lints_clean() -> None:
    report = lint(load_default_design_md())
    assert report["ok"] is True
    assert report["summary"]["errors"] == 0
    assert float(report["score"]) >= 0.7


def test_composite_reward_prefers_valid_openui() -> None:
    good = (
        'root = Stack([cta], "column")\n'
        'cta = Button(":cta.label")'
    )
    bad = "root = Broken()"
    assert composite_reward(good) > composite_reward(bad)


def test_preference_pair_builder() -> None:
    good = 'root = Stack([cta])\ncta = Button(":cta.label")'
    bad = "root = Broken()"
    pair = build_pairs_from_candidates("Make a CTA", [good, bad])
    assert pair is not None
    assert pair.chosen == good
    assert pair.rejected == bad
