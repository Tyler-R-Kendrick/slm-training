"""A bridge failure on an unsupported Node must name the version mismatch.

Running the bridge on Node 18 surfaces an ``ERR_REQUIRE_ESM`` from a transitive
dependency, which reads as a packaging bug and has cost real debugging time
twice. The hint turns that into the one fact that fixes it.
"""

from __future__ import annotations

import json
import subprocess

from slm_training.dsl import lang_core


def _hint(monkeypatch, *, running: str, declared: str, bridge_dir) -> str:
    (bridge_dir / "package.json").write_text(json.dumps({"engines": {"node": declared}}))
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=running, stderr="")
    monkeypatch.setattr(lang_core, "_node_bin", lambda: "/usr/bin/node")
    monkeypatch.setattr(lang_core, "_bridge_dir", lambda: bridge_dir)
    monkeypatch.setattr(lang_core.subprocess, "run", lambda *a, **k: completed)
    lang_core._node_engine_mismatch.cache_clear()
    try:
        return lang_core._node_engine_mismatch()
    finally:
        lang_core._node_engine_mismatch.cache_clear()


def test_unsupported_node_is_named(monkeypatch, tmp_path) -> None:
    hint = _hint(monkeypatch, running="v18.19.1", declared=">=20 <23", bridge_dir=tmp_path)
    assert "v18.19.1" in hint and ">=20 <23" in hint


def test_supported_node_adds_nothing(monkeypatch, tmp_path) -> None:
    assert _hint(monkeypatch, running="v22.23.1", declared=">=20 <23", bridge_dir=tmp_path) == ""


def test_too_new_node_is_named(monkeypatch, tmp_path) -> None:
    hint = _hint(monkeypatch, running="v23.0.0", declared=">=20 <23", bridge_dir=tmp_path)
    assert "v23.0.0" in hint


def test_hint_never_masks_the_real_failure(monkeypatch, tmp_path) -> None:
    # No package.json to read: the hint degrades to silence rather than
    # raising over the top of the bridge error it was meant to explain.
    monkeypatch.setattr(lang_core, "_node_bin", lambda: "/usr/bin/node")
    monkeypatch.setattr(lang_core, "_bridge_dir", lambda: tmp_path / "missing")
    lang_core._node_engine_mismatch.cache_clear()
    try:
        assert lang_core._node_engine_mismatch() == ""
    finally:
        lang_core._node_engine_mismatch.cache_clear()
