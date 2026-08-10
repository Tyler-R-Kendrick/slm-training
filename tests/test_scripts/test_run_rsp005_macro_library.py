"""CLI smoke for scripts.run_rsp005_macro_library."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_rsp005_macro_library import main


def _require_openui_bridge() -> None:
    from slm_training.dsl import lang_core
    from slm_training.dsl.canonicalize import canonicalize

    if not lang_core.bridge_available():
        pytest.skip("OpenUI bridge dependencies are unavailable")
    try:
        canonicalize(
            'root = Stack([x], "column")\nx = TextContent(":x")',
            validate=False,
        )
    except RuntimeError:
        pytest.skip("OpenUI bridge is not usable in this environment")


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["manifest"]["experiment_id"] == "exp-sr-9"
    assert payload["claim_class_execution"] == "fixture"
    assert payload["promotion"] is False


def test_fixture_writes_docs(tmp_path: Path, capsys) -> None:
    _require_openui_bridge()
    docs_out = tmp_path / "iter-slm480-rsp-005.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--max-macros",
                "8",
                "--out-dir",
                str(out_dir),
                "--docs-out",
                str(docs_out),
            ]
        )
        == 0
    )
    payload = json.loads(docs_out.read_text(encoding="utf-8"))
    assert payload["kind"] == "rsp005_macro_library_fixture/v1"
    assert payload["claim_class"] == "fixture"
    assert payload["promotion"] is False
    assert payload["catalogue_id"] == "exp-sr-9"
    assert payload["semantics_preserved"] is True
    assert "version_stamp" in payload
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert docs_out.with_suffix(".md").is_file()
    captured = capsys.readouterr().out
    assert "macro_library_size_reduction_rate=" in captured
    assert "semantics_preserved=True" in captured
