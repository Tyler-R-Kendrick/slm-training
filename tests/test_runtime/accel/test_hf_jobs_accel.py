"""Tests for CUDA/Jobs accel helpers and HF Jobs launcher dry-run."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest
import torch

from scripts import hf_jobs_train, remote_train
from slm_training.runtime.accel import (
    configure_cuda_training,
    detect_device,
    is_zerogpu_environment,
    maybe_compile,
    prefer_fast_train_env,
)


def test_prefer_fast_train_env_markers(monkeypatch) -> None:
    monkeypatch.delenv("SLM_FAST_TRAIN", raising=False)
    monkeypatch.delenv("HF_JOBS_FAST_TRAIN", raising=False)
    monkeypatch.delenv("HF_JOB_ID", raising=False)
    monkeypatch.delenv("JOB_ID", raising=False)
    monkeypatch.delenv("SPACE_HARDWARE", raising=False)
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert prefer_fast_train_env() is False

    monkeypatch.setenv("SLM_FAST_TRAIN", "1")
    assert prefer_fast_train_env() is True

    monkeypatch.setenv("SLM_FAST_TRAIN", "0")
    monkeypatch.setenv("HF_JOB_ID", "job-abc")
    assert prefer_fast_train_env() is False  # explicit disable wins

    monkeypatch.delenv("SLM_FAST_TRAIN", raising=False)
    assert prefer_fast_train_env() is True


def test_zerogpu_blocks_fast_train_and_compile(monkeypatch) -> None:
    monkeypatch.setenv("SPACE_HARDWARE", "zerogpu")
    monkeypatch.setenv("SLM_FAST_TRAIN", "1")
    assert is_zerogpu_environment() is True
    assert prefer_fast_train_env() is False
    mod = torch.nn.Linear(4, 4)
    assert maybe_compile(mod, enabled=True) is mod


def test_configure_cuda_training_smoke() -> None:
    applied = configure_cuda_training()
    assert "cuda" in applied
    if not torch.cuda.is_available():
        assert applied["cuda"] is False


def test_hf_jobs_train_dry_run() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = hf_jobs_train.main(
            [
                "--dry-run",
                "--run-id",
                "unit_jobs",
                "--steps",
                "10",
                "--branch",
                "main",
                "--skip-eval",
            ]
        )
    assert rc == 0
    plan = json.loads(buf.getvalue())
    assert plan["flavor"] == "a10g-large"
    assert plan["run_id"] == "unit_jobs"
    assert plan["timeout"] == "3m"
    assert "--fast-train" in plan["entrypoint"]
    assert "reduce-overhead" in plan["entrypoint"]
    assert "SLM_FAST_TRAIN=1" in plan["entrypoint"]
    cmd = plan["command"]
    assert cmd[0:3] == ["hf", "jobs", "run"]
    assert "--flavor" in cmd
    assert "--secrets" in cmd
    assert "HF_TOKEN" in cmd
    assert any("buckets/TKendrick/OpenUI" in c for c in cmd)


def test_hf_jobs_build_command_no_mount() -> None:
    cmd = hf_jobs_train.build_hf_jobs_command(
        flavor="a100-large",
        timeout="3m",
        image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        entrypoint="echo hi",
        checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
        mount_bucket=False,
    )
    assert "--volume" not in cmd
    assert cmd[-3:] == ["bash", "-lc", "echo hi"]


def test_hf_jobs_rejects_overlong_timeout() -> None:
    with pytest.raises(ValueError, match="must be 3m"):
        hf_jobs_train.build_hf_jobs_command(
            flavor="a100-large",
            timeout="4m",
            image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
            entrypoint="echo hi",
            checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
            mount_bucket=False,
        )


def test_remote_train_includes_fast_train() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = remote_train.main(
            [
                "--host",
                "example",
                "--run-id",
                "remote_fast",
                "--dry-run",
                "--steps",
                "5",
            ]
        )
    assert rc == 0
    plan = json.loads(buf.getvalue())
    script = plan["remote_script"]
    assert "--fast-train" in script
    assert "--compile-mode reduce-overhead" in script
    assert "--device auto" in script
    assert "SLM_FAST_TRAIN=1" in script
    assert "timeout --signal=INT --kill-after=10s 170s" in script


def test_detect_device_cpu_fallback() -> None:
    info = detect_device("cpu")
    assert info.backend == "cpu"
