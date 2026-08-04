"""Tests for DESIGN.md bridge and preference rewards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slm_training.dsl import design_md
from slm_training.dsl.design_md import bridge_available, lint, load_default_design_md
from slm_training.harnesses.preference import build_pairs_from_candidates, composite_reward


@pytest.mark.skipif(not bridge_available(), reason="design_md bridge missing")
def test_default_design_md_lints_clean() -> None:
    report = lint(load_default_design_md())
    assert report["ok"] is True
    assert report["summary"]["errors"] == 0
    assert float(report["score"]) >= 0.7


def test_invoke_once_clears_host_node_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(design_md, "bridge_available", lambda: True)
    monkeypatch.setenv("NODE_OPTIONS", "--import tsx")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout=b'{"ok": true}', stderr=b"")

    monkeypatch.setattr(design_md.subprocess, "run", fake_run)

    design_md._invoke_once({"op": "lint", "source": "root = TextContent(':x')"})

    assert captured["env"]["NODE_OPTIONS"] == ""


def test_ensure_repl_clears_host_node_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(design_md, "bridge_available", lambda: True)
    monkeypatch.setattr(design_md, "_REPL_PROC", None)
    monkeypatch.setenv("NODE_OPTIONS", "--import tsx")
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(design_md.subprocess, "Popen", fake_popen)

    try:
        design_md._ensure_repl()
        assert captured["env"]["NODE_OPTIONS"] == ""
    finally:
        design_md._REPL_PROC = None


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
