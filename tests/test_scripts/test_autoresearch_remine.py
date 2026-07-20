"""Regression test for the autoresearch remine subcommand (SLM-132)."""

from __future__ import annotations

from pathlib import Path

from scripts.autoresearch import main as autoresearch_main


def test_autoresearch_remine_describe() -> None:
    rc = autoresearch_main(["remine", "--describe"])
    assert rc == 0


def test_autoresearch_remine_smoke(tmp_path: Path) -> None:
    rc = autoresearch_main(
        ["--root", str(tmp_path), "remine", "--smoke"]
    )
    assert rc == 0
