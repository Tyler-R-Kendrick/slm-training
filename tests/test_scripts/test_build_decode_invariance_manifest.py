"""CLI for the decode-invariance compatibility manifest (EFS0-02)."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.model_build.checkpoint_reference import CheckpointReferenceV1
from scripts.build_decode_invariance_manifest import main


def test_list_describes_paths_without_checkpoint(capsys) -> None:
    assert main(["--list"]) == 0
    out = json.loads(capsys.readouterr().out)
    ids = [p["path_id"] for p in out["decode_paths"]]
    assert ids == ["checkpoint_declared", "current_native", "current_exact_or_compiler"]


def test_default_build_is_deferred(tmp_path: Path, capsys) -> None:
    out = tmp_path / "manifest.json"
    assert main(["--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["checkpoint_count"] == 0
    assert summary["usable_for_audit"] is False
    manifest = json.loads(out.read_text())
    assert "deferred" in manifest["note"]


def test_reference_dir_builds_cells(tmp_path: Path, capsys) -> None:
    ref = CheckpointReferenceV1(
        run_id="demo",
        claim_class="diagnostic",
        checkpoint_role="last",
        checkpoint_filename="last.pt",
        sha256="a" * 64,
    )
    ref.write_json(tmp_path / "last.pt.ref.json")
    (tmp_path / "last.pt.meta.json").write_text(
        json.dumps({"kind": "twotower", "config": {"output_tokenizer": "choice"}, "output_contract_version": 1}),
        encoding="utf-8",
    )
    assert main(["--reference-dir", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["checkpoint_count"] == 1
    assert summary["complete_blocks"] == 1  # choice twotower supports all three paths


def test_validation_failure_exits_nonzero(tmp_path: Path, capsys) -> None:
    # frontier reference with no durable remote_uri -> validation error -> exit 1.
    ref = CheckpointReferenceV1(
        run_id="bad",
        claim_class="frontier",
        checkpoint_role="last",
        checkpoint_filename="last.pt",
        sha256="a" * 64,
    )
    ref.write_json(tmp_path / "last.pt.ref.json")
    (tmp_path / "last.pt.meta.json").write_text(
        json.dumps({"kind": "twotower", "config": {"output_tokenizer": "choice"}}),
        encoding="utf-8",
    )
    assert main(["--reference-dir", str(tmp_path)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["validation_errors"]
