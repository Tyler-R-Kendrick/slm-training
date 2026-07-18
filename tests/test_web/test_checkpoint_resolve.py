"""Latest-checkpoint resolution + playground hot-reload tests."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from slm_training.models.checkpoint_resolve import (
    ENV_CHECKPOINT,
    checkpoint_is_loadable,
    resolve_serving_checkpoint,
)
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT

DEMO_DIR = PLAYGROUND_DEMO_CHECKPOINT.parent.resolve()


def _install_checkpoint(target_dir: Path, *, mtime: float | None = None) -> Path:
    """Copy the committed demo checkpoint bundle to target_dir/last.pt."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for artifact in DEMO_DIR.iterdir():
        shutil.copy2(artifact, target_dir / artifact.name)
    checkpoint = target_dir / "last.pt"
    if mtime is not None:
        os.utime(checkpoint, (mtime, mtime))
    return checkpoint


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.config = SimpleNamespace(
        generate_max_attempts=3,
        grammar_finalize_on_last_attempt_only=False,
        grammar_finalize_validate=False,
        grammar_ltr_max_tokens=192,
    )
    return model


def test_loadability_requires_sidecars(tmp_path: Path) -> None:
    checkpoint = _install_checkpoint(tmp_path / "checkpoints")
    assert checkpoint_is_loadable(checkpoint)
    assert checkpoint_is_loadable(checkpoint, require_onnx=True)

    (tmp_path / "checkpoints" / "last.context.onnx").unlink()
    assert not checkpoint_is_loadable(checkpoint, require_onnx=True)
    assert checkpoint_is_loadable(checkpoint)

    meta = checkpoint.with_suffix(".meta.json")
    meta.write_text(json.dumps({"kind": "causal_lm"}), encoding="utf-8")
    assert not checkpoint_is_loadable(checkpoint)

    meta.unlink()
    assert not checkpoint_is_loadable(checkpoint)


def test_resolution_falls_back_to_demo_fixture(tmp_path: Path) -> None:
    resolved = resolve_serving_checkpoint(root=tmp_path)
    assert resolved.provenance == "demo-fixture"
    assert resolved.run_id == "playground_demo"


def test_resolution_prefers_newest_run(tmp_path: Path) -> None:
    _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-old" / "checkpoints", mtime=1_000
    )
    newest = _install_checkpoint(
        tmp_path / "outputs" / "autoresearch" / "exp" / "runs" / "run-new" / "checkpoints",
        mtime=2_000,
    )
    resolved = resolve_serving_checkpoint(root=tmp_path)
    assert resolved.provenance == "latest-run"
    assert resolved.path == newest
    assert resolved.run_id == "run-new"


def test_resolution_skips_unloadable_newest(tmp_path: Path) -> None:
    loadable = _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-good" / "checkpoints", mtime=1_000
    )
    broken_dir = tmp_path / "outputs" / "runs" / "run-broken" / "checkpoints"
    _install_checkpoint(broken_dir, mtime=2_000)
    (broken_dir / "last.tokenizer.json").unlink()

    resolved = resolve_serving_checkpoint(root=tmp_path)
    assert resolved.path == loadable
    assert resolved.run_id == "run-good"


def test_resolution_prefers_deployment_pointer(tmp_path: Path) -> None:
    _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-latest" / "checkpoints", mtime=2_000
    )
    deployed = _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-deployed" / "checkpoints", mtime=1_000
    )
    deployments = tmp_path / "outputs" / "lineage" / "deployments"
    record_dir = deployments / "twotower" / "history"
    record_dir.mkdir(parents=True)
    record = {
        "pointer_id": "ptr-1",
        "track": "twotower",
        "run_id": "run-deployed",
        "artifact_uri": str(deployed.relative_to(tmp_path)),
    }
    (record_dir / "ptr-1.json").write_text(json.dumps(record), encoding="utf-8")
    (deployments / "twotower" / "current.json").write_text(
        json.dumps({"record": "history/ptr-1.json"}), encoding="utf-8"
    )
    (deployments / "selected.json").write_text(
        json.dumps({"track": "twotower", "record": "deployments/twotower/history/ptr-1.json"}),
        encoding="utf-8",
    )

    resolved = resolve_serving_checkpoint(root=tmp_path)
    assert resolved.provenance == "deployment"
    assert resolved.path == deployed
    assert resolved.run_id == "run-deployed"


def test_resolution_skips_remote_pointer(tmp_path: Path) -> None:
    latest = _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-local" / "checkpoints"
    )
    deployments = tmp_path / "outputs" / "lineage" / "deployments"
    record_dir = deployments / "twotower" / "history"
    record_dir.mkdir(parents=True)
    record = {
        "pointer_id": "ptr-remote",
        "track": "twotower",
        "run_id": "run-remote",
        "artifact_uri": "hf://buckets/TKendrick/OpenUI/checkpoints/run-remote/last.pt",
    }
    (record_dir / "ptr-remote.json").write_text(json.dumps(record), encoding="utf-8")
    (deployments / "twotower" / "current.json").write_text(
        json.dumps({"record": "history/ptr-remote.json"}), encoding="utf-8"
    )

    resolved = resolve_serving_checkpoint(root=tmp_path)
    assert resolved.provenance == "latest-run"
    assert resolved.path == latest


def test_env_pin_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_checkpoint(tmp_path / "outputs" / "runs" / "run-any" / "checkpoints")
    pinned = _install_checkpoint(tmp_path / "pinned" / "checkpoints")
    monkeypatch.setenv(ENV_CHECKPOINT, str(pinned))
    resolved = resolve_serving_checkpoint(root=tmp_path)
    assert resolved.provenance == "env"
    assert resolved.path == pinned


def test_service_hot_reloads_newer_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.web import service as service_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service_mod, "_RESOLVE_INTERVAL_SECONDS", 0.0)
    _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-a" / "checkpoints", mtime=1_000
    )

    loads: list[Path] = []

    def factory(path: Path, device: str) -> MagicMock:
        loads.append(Path(path))
        return _fake_model()

    service = service_mod.PlaygroundService(
        checkpoint=None,
        model_factory=factory,
        annotations_path=tmp_path / "feedback.jsonl",
        human_train_path=tmp_path / "human_train.jsonl",
        human_pairs_path=tmp_path / "human_pairs.jsonl",
        bad_outputs_path=tmp_path / "bad_outputs.jsonl",
        generation_attempts_path=tmp_path / "attempts.jsonl",
        require_onnx=False,
    )
    assert service.info()["checkpoint_resolution"]["run_id"] == "run-a"

    first = service.load()
    assert loads[-1].parts[-3] == "run-a"
    assert service.load() is first  # unchanged target: no reload

    # A newer run appears: the service must move to it without a restart.
    _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-b" / "checkpoints", mtime=2_000
    )
    second = service.load()
    assert second is not first
    assert loads[-1].parts[-3] == "run-b"
    assert service.info()["checkpoint_resolution"]["run_id"] == "run-b"

    # The tracked file is overwritten in place (training rewrote last.pt).
    os.utime(service.checkpoint, (3_000, 3_000))
    third = service.load()
    assert third is not second
    assert loads[-1].parts[-3] == "run-b"


def test_service_keeps_old_model_when_reload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.web import service as service_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service_mod, "_RESOLVE_INTERVAL_SECONDS", 0.0)
    _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-a" / "checkpoints", mtime=1_000
    )

    calls: list[Path] = []

    def factory(path: Path, device: str) -> MagicMock:
        calls.append(Path(path))
        if len(calls) > 1:
            raise RuntimeError("torn checkpoint write")
        return _fake_model()

    service = service_mod.PlaygroundService(
        checkpoint=None,
        model_factory=factory,
        annotations_path=tmp_path / "feedback.jsonl",
        human_train_path=tmp_path / "human_train.jsonl",
        human_pairs_path=tmp_path / "human_pairs.jsonl",
        bad_outputs_path=tmp_path / "bad_outputs.jsonl",
        generation_attempts_path=tmp_path / "attempts.jsonl",
        require_onnx=False,
    )
    first = service.load()
    _install_checkpoint(
        tmp_path / "outputs" / "runs" / "run-b" / "checkpoints", mtime=2_000
    )
    assert service.load() is first  # reload failed; old model still serves
    assert service.info()["checkpoint_resolution"]["run_id"] == "run-a"


def test_injected_model_is_never_displaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.web import service as service_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service_mod, "_RESOLVE_INTERVAL_SECONDS", 0.0)
    service = service_mod.PlaygroundService(
        checkpoint=Path("/nonexistent.pt"),
        generation_attempts_path=tmp_path / "attempts.jsonl",
        require_onnx=False,
    )
    injected = _fake_model()
    service._model = injected  # noqa: SLF001 — the established test seam
    assert service.load() is injected
