"""Tests for ``scripts/run_gpu_reference.py``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.run_gpu_reference import main
from slm_training.harnesses.experiments.slm262_gpu_reference import (
    AcceleratorRunManifestV1,
    build_default_manifest,
)


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    manifest = build_default_manifest(
        "slm262_cli_test",
        data_snapshot_sha="b" * 64,
        eval_snapshot_sha="c" * 64,
    )
    manifest = manifest.__class__.from_dict(
        {**manifest.to_dict(), "dirty_tree_ok": True}
    )
    path = tmp_path / "manifest.json"
    manifest.write_json(path)
    return path


def test_init_command(tmp_path: Path) -> None:
    out = tmp_path / "new.json"
    rc = main(
        [
            "init",
            "--run-id",
            "slm262_init",
            "--data-snapshot-sha",
            "b" * 64,
            "--eval-snapshot-sha",
            "c" * 64,
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    manifest = AcceleratorRunManifestV1.load_json(out)
    assert manifest.run_id == "slm262_init"


def test_validate_command(manifest_path: Path) -> None:
    rc = main(["validate", "--manifest", str(manifest_path)])
    assert rc == 0


def test_describe_command(manifest_path: Path, capsys: Any) -> None:
    rc = main(["describe", "--manifest", str(manifest_path)])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["run_id"] == "slm262_cli_test"


def test_submit_dry_run_command(manifest_path: Path, monkeypatch: Any, tmp_path: Path) -> None:
    plan = {"command": ["hf", "jobs", "run"], "run_id": "slm262_cli_test"}

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps(plan), stderr=""
        )

    monkeypatch.setattr("subprocess.run", fake_run)
    out = tmp_path / "updated.json"
    rc = main(
        [
            "submit",
            "--manifest",
            str(manifest_path),
            "--provider",
            "hf_jobs",
            "--dry-run",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    updated = AcceleratorRunManifestV1.load_json(out)
    assert updated.provider == "hf_jobs"
    assert updated.provider_request_id == "dry-run"


def test_dry_run_adapter_submit_command(manifest_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "dry.json"
    rc = main(
        [
            "submit",
            "--manifest",
            str(manifest_path),
            "--provider",
            "dry_run",
            "--dry-run",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    updated = AcceleratorRunManifestV1.load_json(out)
    assert updated.provider == "dry_run"


def test_local_smoke_command(
    manifest_path: Path, monkeypatch: Any, tmp_path: Path
) -> None:
    run_dir = tmp_path / "slm262_cli_test_cpu_smoke"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "last.pt").write_bytes(b"serving")
    (ckpt_dir / "last_full_state.pt").write_bytes(b"full")
    (run_dir / "train_summary.json").write_text("{}", encoding="utf-8")

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    out = tmp_path / "smoke.json"
    rc = main(
        [
            "local-smoke",
            "--manifest",
            str(manifest_path),
            "--steps",
            "2",
            "--resume-steps",
            "1",
            "--run-root",
            str(tmp_path),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    updated = AcceleratorRunManifestV1.load_json(out)
    assert "last.pt" in updated.checkpoint_inventory
    assert "last_full_state.pt" in updated.full_state_inventory


def test_reconcile_command(manifest_path: Path, monkeypatch: Any, tmp_path: Path) -> None:
    manifest = AcceleratorRunManifestV1.load_json(manifest_path)
    manifest = manifest.__class__.from_dict(
        {**manifest.to_dict(), "provider": "hf_jobs", "provider_job_id": "job-123"}
    )
    manifest.write_json(manifest_path)
    payload = {"status": "completed"}

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps([payload]),
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_run)
    out = tmp_path / "reconciled.json"
    rc = main(
        [
            "reconcile",
            "--manifest",
            str(manifest_path),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    updated = AcceleratorRunManifestV1.load_json(out)
    assert "completed" in updated.notes[-1]


def test_evaluate_command_no_checkpoint(manifest_path: Path) -> None:
    rc = main(["evaluate", "--manifest", str(manifest_path)])
    assert rc == 7
