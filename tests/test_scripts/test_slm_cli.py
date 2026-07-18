"""The slm dispatcher forwards phase commands verbatim to the script mains."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

from scripts.slm import COMMANDS, GROUP_SUMMARIES, Command, main

REPO = Path(__file__).resolve().parents[2]


def _fake_command(monkeypatch, module_name: str, fake_main) -> tuple[str, str]:
    module = types.ModuleType(module_name)
    module.main = fake_main
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(
        COMMANDS, ("fakegroup", "go"), Command(module_name, "fake", "sft")
    )
    return "fakegroup", "go"


def test_every_registry_module_resolves() -> None:
    for key, command in COMMANDS.items():
        assert 1 <= len(key) <= 2
        for token in key:
            assert re.fullmatch(r"[a-z][a-z-]*", token), key
        assert command.summary
        assert command.guide
        assert importlib.util.find_spec(command.module) is not None, command.module


def test_group_summaries_cover_exactly_the_registry_groups() -> None:
    assert {key[0] for key in COMMANDS} == set(GROUP_SUMMARIES)


def test_dispatch_forwards_argv_verbatim_and_returns_code(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_main(argv=None):
        captured.append(list(argv))
        return 3

    group, action = _fake_command(monkeypatch, "fake_slm_target", fake_main)
    assert main([group, action, "--flag", "value"]) == 3
    assert captured == [["--flag", "value"]]


def test_dispatch_patches_sys_argv_for_argvless_main(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_main():
        seen.append(list(sys.argv))
        return 0

    group, action = _fake_command(monkeypatch, "fake_slm_argvless", fake_main)
    before = list(sys.argv)
    assert main([group, action, "--x", "1"]) == 0
    assert seen == [["fake_slm_argvless", "--x", "1"]]
    assert sys.argv == before


def test_dispatch_normalizes_none_and_systemexit(monkeypatch) -> None:
    group, action = _fake_command(monkeypatch, "fake_slm_none", lambda argv=None: None)
    assert main([group, action]) == 0

    def exit_two(argv=None):
        raise SystemExit(2)

    group, action = _fake_command(monkeypatch, "fake_slm_exit2", exit_two)
    assert main([group, action]) == 2

    def exit_none(argv=None):
        raise SystemExit(None)

    group, action = _fake_command(monkeypatch, "fake_slm_exitnone", exit_none)
    assert main([group, action]) == 0


def test_list_and_help_exit_zero_and_cover_registry(capsys) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for key in COMMANDS:
        assert f"slm {' '.join(key)}" in out
    assert main(["--help"]) == 0
    assert main([]) == 0


def test_guide_prints_reference_and_unknown_slug_fails(capsys) -> None:
    assert main(["guide", "sft"]) == 0
    assert "SFT (model build) phase" in capsys.readouterr().out
    assert main(["guide"]) == 0
    assert "available:" in capsys.readouterr().out
    assert main(["guide", "not-a-slug"]) == 2
    assert "unknown guide" in capsys.readouterr().err


def test_unknown_group_and_action_exit_two(capsys) -> None:
    assert main(["bogus"]) == 2
    assert "slm list" in capsys.readouterr().err
    assert main(["data", "bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown action" in err
    assert main(["data"]) == 2
    assert main(["data", "-h"]) == 0


def test_end_to_end_dispatch_builds_fixture_corpus(tmp_path: Path) -> None:
    assert (
        main(
            [
                "data",
                "build-train",
                "--source",
                "programspec",
                "--programspec-path",
                str(tmp_path / "missing-programs.jsonl"),
                "--programspec-count",
                "1",
                "--synthesizer",
                "none",
                "--no-frontier-artifacts",
                "--no-governance-artifacts",
                "--version",
                "vtest",
                "--output-root",
                str(tmp_path / "out" / "train"),
                "--publish-root",
                str(tmp_path / "published" / "train"),
            ]
        )
        == 0
    )
    manifest = tmp_path / "published" / "train" / "vtest" / "manifest.json"
    assert manifest.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["records"]
