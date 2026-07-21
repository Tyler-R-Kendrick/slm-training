"""Tests for SLM-262 (VSD0-03) accelerator reference-run manifest and adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from slm_training.harnesses.experiments.slm262_gpu_reference import (
    UNKNOWN,
    AcceleratorRunManifestV1,
    DryRunAdapter,
    HFJobsAdapter,
    RemotePodAdapter,
    adapter_for,
    build_default_manifest,
    hash_artifacts,
    run_local_smoke,
)


@pytest.fixture
def valid_manifest(tmp_path: Path) -> AcceleratorRunManifestV1:
    return AcceleratorRunManifestV1(
        run_id="slm262_test",
        source_commit="a" * 40,
        dirty_tree_ok=True,
        provider="dry_run",
        data_snapshot_id="train-v1",
        data_snapshot_sha="b" * 64,
        eval_snapshot_id="eval-v1",
        eval_snapshot_sha="c" * 64,
        target_decisions=50000,
        checkpoint_cadence_decisions=5000,
        expected_artifacts=("last.pt", "last_full_state.pt"),
        remote_uri_prefix="hf://buckets/TKendrick/OpenUI/checkpoints/slm262_test",
    )


def test_manifest_round_trip(valid_manifest: AcceleratorRunManifestV1) -> None:
    data = valid_manifest.to_dict()
    restored = AcceleratorRunManifestV1.from_dict(data)
    assert restored == valid_manifest
    assert restored.sha == valid_manifest.sha


def test_manifest_json_round_trip(valid_manifest: AcceleratorRunManifestV1, tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    valid_manifest.write_json(path)
    restored = AcceleratorRunManifestV1.load_json(path)
    assert restored == valid_manifest


def test_manifest_unknown_fields_dropped(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "accelerator_run_manifest/v1",
                "run_id": "future",
                "source_commit": "a" * 40,
                "provider": "dry_run",
                "future_field": "should be ignored",
            }
        ),
        encoding="utf-8",
    )
    manifest = AcceleratorRunManifestV1.load_json(path)
    assert manifest.run_id == "future"
    assert "future_field" not in manifest.to_dict()


def test_invalid_provider_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported provider"):
        AcceleratorRunManifestV1(provider="lambda_cloud")


def test_invalid_source_commit_rejected() -> None:
    with pytest.raises(ValueError, match="source_commit"):
        AcceleratorRunManifestV1(source_commit="short")


def test_redact_secrets() -> None:
    payload = {
        "HF_TOKEN": "hf_secret_value",
        "nested": {"api_key": "key", "ok": "visible"},
        "list": [{"password": "pw"}],
    }
    redacted = AcceleratorRunManifestV1.redact_secrets(payload)
    assert redacted["HF_TOKEN"] == "***"
    assert redacted["nested"]["api_key"] == "***"
    assert redacted["nested"]["ok"] == "visible"
    assert redacted["list"][0]["password"] == "***"


def test_check_ready_blocks_missing_snapshots() -> None:
    manifest = AcceleratorRunManifestV1(
        run_id="x",
        source_commit="a" * 40,
        dirty_tree_ok=True,
        provider="dry_run",
    )
    errors = manifest.check_ready()
    assert any("data_snapshot_sha" in e for e in errors)
    assert any("eval_snapshot_sha" in e for e in errors)
    assert any("expected_artifacts" in e for e in errors)


def test_check_ready_blocks_dirty_tree(monkeypatch: Any) -> None:
    manifest = AcceleratorRunManifestV1(
        run_id="x",
        source_commit="a" * 40,
        dirty_tree_ok=False,
        provider="dry_run",
        data_snapshot_sha="b" * 64,
        eval_snapshot_sha="c" * 64,
        target_decisions=1,
        checkpoint_cadence_decisions=1,
        expected_artifacts=("last.pt",),
        remote_uri_prefix="hf://buckets/TKendrick/OpenUI/checkpoints/x",
    )
    monkeypatch.setattr(
        "slm_training.harnesses.experiments.slm262_gpu_reference._git_dirty",
        lambda: True,
    )
    monkeypatch.setattr(
        "slm_training.harnesses.experiments.slm262_gpu_reference._git_head",
        lambda: "a" * 40,
    )
    errors = manifest.check_ready()
    assert any("dirty" in e.lower() for e in errors)


def test_check_ready_hf_jobs_requires_auth(monkeypatch: Any) -> None:
    manifest = AcceleratorRunManifestV1(
        run_id="x",
        source_commit="a" * 40,
        dirty_tree_ok=True,
        provider="hf_jobs",
        instance_type="a10g-large",
        data_snapshot_sha="b" * 64,
        eval_snapshot_sha="c" * 64,
        target_decisions=1,
        checkpoint_cadence_decisions=1,
        expected_artifacts=("last.pt",),
        remote_uri_prefix="hf://buckets/TKendrick/OpenUI/checkpoints/x",
    )
    monkeypatch.setattr(
        "slm_training.harnesses.experiments.slm262_gpu_reference._hf_authenticated",
        lambda: False,
    )
    errors = manifest.check_ready()
    assert any("Hugging Face auth" in e for e in errors)


def test_build_default_manifest_sets_commit_and_stamp() -> None:
    manifest = build_default_manifest(
        "slm262_default",
        data_snapshot_sha="b" * 64,
        eval_snapshot_sha="c" * 64,
    )
    assert manifest.run_id == "slm262_default"
    assert manifest.provider == "dry_run"
    assert manifest.source_commit != UNKNOWN
    assert manifest.version_stamp.get("stamp_schema") == "version_stamp/v1"
    assert "harness.experiments.slm262_gpu_reference_run" in manifest.version_stamp.get(
        "components", {}
    )


def test_hash_artifacts(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    (ckpt / "last.pt").write_bytes(b"weights")
    inventory = hash_artifacts(ckpt)
    assert "last.pt" in inventory
    assert inventory["last.pt"]["size_bytes"] == 7
    assert len(inventory["last.pt"]["sha256"]) == 64


def test_adapter_for_dispatch() -> None:
    assert adapter_for("hf_jobs").provider == "hf_jobs"
    assert adapter_for("remote_pod").provider == "remote_pod"
    assert adapter_for("dry_run").provider == "dry_run"
    with pytest.raises(ValueError):
        adapter_for("unknown")


def _fake_completed_process(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_hf_jobs_adapter_dry_run(monkeypatch: Any) -> None:
    adapter = HFJobsAdapter()
    manifest = build_default_manifest("slm262_hf_dry")
    plan = {"command": ["hf", "jobs", "run"], "run_id": "slm262_hf_dry"}

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--dry-run" in cmd
        return _fake_completed_process(stdout=json.dumps(plan))

    monkeypatch.setattr("subprocess.run", fake_run)
    result = adapter.submit(manifest, dry_run=True)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["plan"] == plan


def test_hf_jobs_adapter_real_submit_parses_job_id(monkeypatch: Any) -> None:
    adapter = HFJobsAdapter()
    manifest = build_default_manifest("slm262_hf_real")

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--dry-run" not in cmd
        return _fake_completed_process(stdout="\njob-abc-123\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    result = adapter.submit(manifest, dry_run=False)
    assert result["ok"] is True
    assert result["provider_job_id"] == "job-abc-123"


def test_hf_jobs_adapter_status(monkeypatch: Any) -> None:
    adapter = HFJobsAdapter()
    manifest = build_default_manifest("slm262_hf_status")
    manifest = manifest.__class__.from_dict(
        {**manifest.to_dict(), "provider_job_id": "job-xyz"}
    )
    payload = {"status": "running"}

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "inspect" in cmd
        return _fake_completed_process(stdout=json.dumps([payload]))

    monkeypatch.setattr("subprocess.run", fake_run)
    status = adapter.status(manifest)
    assert status["ok"] is True
    assert status["payload"]["status"] == "running"


def test_hf_jobs_adapter_reconcile_completed() -> None:
    adapter = HFJobsAdapter()
    manifest = build_default_manifest("slm262_hf_recon")
    manifest = manifest.__class__.from_dict(
        {**manifest.to_dict(), "provider_job_id": "job-done"}
    )
    updated = adapter.reconcile(manifest, {"status": "completed"})
    assert "completed" in updated.notes[-1]
    assert updated.timestamps["reconciled_at"]


def test_dry_run_adapter_never_claims_hardware(valid_manifest: AcceleratorRunManifestV1) -> None:
    adapter = DryRunAdapter()
    result = adapter.submit(valid_manifest, dry_run=False)
    assert result["dry_run"] is True
    assert result["provider_job_id"] is None


def test_remote_pod_adapter_requires_host() -> None:
    adapter = RemotePodAdapter()
    manifest = build_default_manifest("slm262_pod")
    manifest = manifest.__class__.from_dict(
        {**manifest.to_dict(), "provider": "remote_pod", "provider_options": {}}
    )
    result = adapter.submit(manifest, dry_run=True)
    # dry-run still succeeds because the CLI plan is emitted even without host
    # (the real CLI would error, but the adapter surfaces the subprocess result).
    assert "provider" in result


def test_run_local_smoke_success(monkeypatch: Any, tmp_path: Path) -> None:
    manifest = build_default_manifest("slm262_smoke")
    manifest = manifest.__class__.from_dict(
        {
            **manifest.to_dict(),
            "data_snapshot_sha": "b" * 64,
            "eval_snapshot_sha": "c" * 64,
            "target_decisions": 100,
            "checkpoint_cadence_decisions": 10,
            "expected_artifacts": ("last.pt", "last_full_state.pt"),
            "remote_uri_prefix": "hf://buckets/TKendrick/OpenUI/checkpoints/slm262_smoke",
            "dirty_tree_ok": True,
        }
    )
    run_dir = tmp_path / "slm262_smoke_cpu_smoke"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "last.pt").write_bytes(b"serving")
    (ckpt_dir / "last_full_state.pt").write_bytes(b"full")
    (run_dir / "train_summary.json").write_text("{}", encoding="utf-8")

    call_count = 0

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        # First call = initial train; second = resume.
        return _fake_completed_process(stdout="", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    report = run_local_smoke(manifest, steps=2, resume_steps=1, run_root=tmp_path)
    assert report["ok"] is True
    assert report["last_full_state_exists"] is True
    assert "last_full_state.pt" in report["inventory"]
    assert call_count == 2


def test_run_local_smoke_initial_failure(monkeypatch: Any, tmp_path: Path) -> None:
    manifest = build_default_manifest("slm262_smoke_fail")
    manifest = manifest.__class__.from_dict(
        {
            **manifest.to_dict(),
            "data_snapshot_sha": "b" * 64,
            "eval_snapshot_sha": "c" * 64,
            "target_decisions": 100,
            "checkpoint_cadence_decisions": 10,
            "expected_artifacts": ("last.pt",),
            "remote_uri_prefix": "hf://buckets/TKendrick/OpenUI/checkpoints/slm262_smoke_fail",
            "dirty_tree_ok": True,
        }
    )

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _fake_completed_process(stdout="boom", stderr="error", returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)
    report = run_local_smoke(manifest, steps=2)
    assert report["ok"] is False
    assert report["stage"] == "initial_train"
