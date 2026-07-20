"""Tests for scripts/synthesize_efs_campaign.py (SLM-140)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import synthesize_efs_campaign


def test_missing_manifest_without_flag_returns_one(capsys, tmp_path: Path) -> None:
    rc = synthesize_efs_campaign.main(
        ["--manifest", str(tmp_path / "missing.json"), "--docs-design", str(tmp_path)]
    )
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_write_default_manifest_and_validate(capsys, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    rc = synthesize_efs_campaign.main(
        [
            "--manifest",
            str(manifest),
            "--docs-design",
            str(tmp_path),
            "--write-default-manifest",
        ]
    )
    assert rc == 0
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["campaign_id"] == "evidence-first-semantic-slm-campaign"
    rc = synthesize_efs_campaign.main(
        ["--manifest", str(manifest), "--docs-design", str(tmp_path), "--validate-only"]
    )
    assert rc == 0
    assert "validates" in capsys.readouterr().out.lower()


def test_describe_lists_hypotheses_and_matches(capsys, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    synthesize_efs_campaign.main(["--manifest", str(manifest), "--docs-design", str(tmp_path), "--write-default-manifest"])
    rc = synthesize_efs_campaign.main(
        ["--manifest", str(manifest), "--docs-design", str(tmp_path), "--describe"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Campaign:" in out
    assert "Hypotheses:" in out
    assert "Matched hypotheses:" in out


def test_synthesize_writes_json_and_markdown(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    synthesize_efs_campaign.main(["--manifest", str(manifest), "--docs-design", str(tmp_path), "--write-default-manifest"])
    out_json = tmp_path / "synthesis.json"
    out_md = tmp_path / "synthesis.md"
    rc = synthesize_efs_campaign.main(
        [
            "--manifest",
            str(manifest),
            "--docs-design",
            str(tmp_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["schema_version"] == "evidence_first_semantic_synthesis/v1"
    assert data["campaign_id"] == "evidence-first-semantic-slm-campaign"
    md = out_md.read_text(encoding="utf-8")
    assert "# EFS4-04" in md


def test_synthesize_writes_graph_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    synthesize_efs_campaign.main(["--manifest", str(manifest), "--docs-design", str(tmp_path), "--write-default-manifest"])
    out_json = tmp_path / "synthesis.json"
    out_md = tmp_path / "synthesis.md"
    graph_stem = tmp_path / "graph"
    rc = synthesize_efs_campaign.main(
        [
            "--manifest",
            str(manifest),
            "--docs-design",
            str(tmp_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--graph-output",
            str(graph_stem),
        ]
    )
    assert rc == 0
    assert (tmp_path / "graph.mmd").exists()
    assert (tmp_path / "graph.dot").exists()
    assert "digraph evidence_graph" in (tmp_path / "graph.dot").read_text(encoding="utf-8")


def test_validate_only_with_existing_manifest_returns_zero(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    synthesize_efs_campaign.main(["--manifest", str(manifest), "--docs-design", str(tmp_path), "--write-default-manifest"])
    rc = synthesize_efs_campaign.main(
        ["--manifest", str(manifest), "--docs-design", str(tmp_path), "--validate-only"]
    )
    assert rc == 0
