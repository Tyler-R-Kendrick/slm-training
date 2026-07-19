"""Tests for scripts/run_causal_peft_ftpo.py (SLM-121)."""

from __future__ import annotations

from pathlib import Path

from scripts import run_causal_peft_ftpo


def test_plan_only_mode(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = run_causal_peft_ftpo.main(
        ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
    )
    assert rc == 0
    assert (out / "causal_peft_ftpo_manifest.json").exists()
    assert (out / "causal_peft_ftpo_report.json").exists()
    assert (out / "causal_peft_ftpo_report.md").exists()


def test_fixture_mode_with_parent(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    rc = run_causal_peft_ftpo.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out),
            "--seeds",
            "0,1",
            "--objectives",
            "ftpo_single,ftpo_set",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    report_text = (out / "causal_peft_ftpo_report.json").read_text()
    assert "fixture" in report_text
    assert "slm121_fixture" in report_text


def test_invalid_objectives_return_error(tmp_path: Path) -> None:
    out = tmp_path / "bad"
    rc = run_causal_peft_ftpo.main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(out),
            "--objectives",
            "ftpo_single,bad_obj",
            "--seeds",
            "0",
        ]
    )
    assert rc == 1


def test_frontier_mode_emits_partial_fixture(tmp_path: Path) -> None:
    out = tmp_path / "frontier"
    rc = run_causal_peft_ftpo.main(
        [
            "--mode",
            "frontier",
            "--output-dir",
            str(out),
            "--seeds",
            "0",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    assert (out / "causal_peft_ftpo_report.json").exists()
