"""Build CLI publishes the built version into the committed store by default."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_train_data import main as build_main


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--source",
        "programspec",
        "--programspec-path",
        str(tmp_path / "missing-programs.jsonl"),  # fall back to generation
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
        *extra,
    ]


def test_build_publishes_by_default_and_reruns_are_noops(tmp_path: Path) -> None:
    assert build_main(_args(tmp_path)) == 0
    published = tmp_path / "published" / "train" / "vtest"
    assert (published / "records.jsonl").is_file()
    assert (published / "manifest.json").is_file()
    assert (published / "synthesis_telemetry.jsonl").is_file()
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"] == (published / "records.jsonl").as_posix()
    assert manifest["synthesis_telemetry"] == (
        published / "synthesis_telemetry.jsonl"
    ).as_posix()
    # An identical rebuild republishes as a no-op instead of failing.
    assert build_main(_args(tmp_path)) == 0
    assert (published / "records.jsonl").is_file()


def test_no_publish_opt_out(tmp_path: Path) -> None:
    assert build_main(_args(tmp_path, "--no-publish")) == 0
    assert not (tmp_path / "published" / "train" / "vtest").exists()


def test_changed_rebuild_of_same_version_fails_loudly(tmp_path: Path) -> None:
    assert build_main(_args(tmp_path)) == 0
    # Rebuild the same version with different content into the local store.
    assert (
        build_main(
            [
                *_args(tmp_path, "--no-publish"),
                "--programspec-seed",
                "7",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="differs"):
        build_main(_args(tmp_path, "--programspec-seed", "7"))


def test_sanitize_mode_flag_reflected_in_stats(tmp_path: Path) -> None:
    assert build_main(_args(tmp_path, "--no-publish", "--sanitize-mode", "audit")) == 0
    stats = json.loads(
        (tmp_path / "out" / "train" / "vtest" / "stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert stats["sanitize_mode"] == "audit"
    assert stats["sanitize"]["mode"] == "audit"


def test_sanitize_mode_defaults_to_profile(tmp_path: Path) -> None:
    # strict (default profile) resolves the unset flag to enforce.
    assert build_main(_args(tmp_path, "--no-publish")) == 0
    stats = json.loads(
        (tmp_path / "out" / "train" / "vtest" / "stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert stats["sanitize_mode"] == "enforce"


def test_programspec_natural_prompts_opt_in(tmp_path: Path) -> None:
    assert build_main(_args(tmp_path, "--no-publish", "--programspec-natural-prompts")) == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train" / "vtest" / "records.jsonl").read_text().splitlines()
    ]
    generated = next(row for row in rows if row["meta"]["source_family"] == "programspec_generated")
    assert generated["prompt"].startswith("Create a ")
