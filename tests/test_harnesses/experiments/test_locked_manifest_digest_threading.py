"""SLM-430: opt-in threading of the canonical locked-manifest digest check.

SLM-306 (`docs/design/slm306-locked-manifest-digest-verification.md`) added
``locked_manifest_path`` to ``load_campaign_governance`` / ``evaluate_promotion``
/ ``register_promoted_checkpoint`` so a campaign's self-reported
``locked_eval_manifest_sha256`` can be independently re-derived from real
manifest bytes on disk. That parameter defaulted to ``None`` (skip the disk
check) at *every* call site, including the three real promotion entrypoints
(``scripts/resume_climb.py``, ``scripts/run_scaling_ladder.py``, and
``ModelBuildConfig.register_promoted`` in ``train_loop.py``) -- there was no
way to request the stronger check from any of them at all.

This module covers the new opt-in wiring: a
``ModelBuildConfig.verify_locked_manifest_digest`` flag and matching
``--verify-locked-manifest-digest`` CLI flags that, only when explicitly set,
supply ``locked_eval_manifest.canonical_manifest_path()`` as
``locked_manifest_path`` at those three entrypoints. Default behavior
(self-consistency-only, no disk check) is unchanged when the flag is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.data.locked_eval_manifest import canonical_manifest_path
from slm_training.harnesses.model_build import ModelBuildConfig, train


class _StopAtGovernanceLoad(Exception):
    """Raised by a patched ``load_campaign_governance`` to abort before any
    real training work happens, once the call's kwargs have been captured."""


def _patch_governance_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    import slm_training.harnesses.experiments.promotion as promotion_module

    captured: dict[str, object] = {}

    def _fake_load_campaign_governance(**kwargs: object):
        captured.update(kwargs)
        raise _StopAtGovernanceLoad

    monkeypatch.setattr(
        promotion_module, "load_campaign_governance", _fake_load_campaign_governance
    )
    return captured


def _campaign_config(tmp_path: Path, **updates: object) -> ModelBuildConfig:
    defaults: dict[str, object] = dict(
        train_dir=tmp_path / "missing_train",
        run_root=tmp_path,
        run_id="locked_manifest_digest_threading",
        register_promoted=True,
        campaign_manifest=tmp_path / "manifest.json",
        campaign_result=tmp_path / "result.json",
        campaign_store_root=tmp_path / "store",
        campaign_artifact_root=tmp_path / "artifacts",
    )
    defaults.update(updates)
    return ModelBuildConfig(**defaults)


def test_verify_locked_manifest_digest_defaults_to_false() -> None:
    config = ModelBuildConfig(train_dir=Path("unused"))
    assert config.verify_locked_manifest_digest is False


def test_train_loop_omits_disk_check_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_governance_capture(monkeypatch)
    config = _campaign_config(tmp_path)

    with pytest.raises(_StopAtGovernanceLoad):
        train(config)

    assert captured["locked_manifest_path"] is None


def test_train_loop_threads_canonical_path_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_governance_capture(monkeypatch)
    config = _campaign_config(tmp_path, verify_locked_manifest_digest=True)

    with pytest.raises(_StopAtGovernanceLoad):
        train(config)

    assert captured["locked_manifest_path"] == canonical_manifest_path()


def test_resume_climb_cli_omits_disk_check_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.resume_climb as resume_climb

    captured = _patch_governance_capture(monkeypatch)
    argv = [
        "--checkpoint",
        str(tmp_path / "ckpt.pt"),
        "--test-dir",
        str(tmp_path / "test"),
        "--campaign-manifest",
        str(tmp_path / "manifest.json"),
        "--campaign-result",
        str(tmp_path / "result.json"),
        "--campaign-store-root",
        str(tmp_path / "store"),
        "--campaign-artifact-root",
        str(tmp_path / "artifacts"),
    ]

    with pytest.raises(_StopAtGovernanceLoad):
        resume_climb.main(argv)

    assert captured["locked_manifest_path"] is None


def test_resume_climb_cli_threads_canonical_path_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.resume_climb as resume_climb

    captured = _patch_governance_capture(monkeypatch)
    argv = [
        "--checkpoint",
        str(tmp_path / "ckpt.pt"),
        "--test-dir",
        str(tmp_path / "test"),
        "--campaign-manifest",
        str(tmp_path / "manifest.json"),
        "--campaign-result",
        str(tmp_path / "result.json"),
        "--campaign-store-root",
        str(tmp_path / "store"),
        "--campaign-artifact-root",
        str(tmp_path / "artifacts"),
        "--verify-locked-manifest-digest",
    ]

    with pytest.raises(_StopAtGovernanceLoad):
        resume_climb.main(argv)

    assert captured["locked_manifest_path"] == canonical_manifest_path()


def test_run_scaling_ladder_cli_threads_canonical_path_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_scaling_ladder as run_scaling_ladder

    captured = _patch_governance_capture(monkeypatch)
    argv = [
        "--train-dir",
        str(tmp_path / "train"),
        "--test-dir",
        str(tmp_path / "test"),
        "--campaign-manifest",
        str(tmp_path / "manifest.json"),
        "--campaign-result",
        str(tmp_path / "result.json"),
        "--campaign-store-root",
        str(tmp_path / "store"),
        "--campaign-artifact-root",
        str(tmp_path / "artifacts"),
        "--verify-locked-manifest-digest",
    ]

    with pytest.raises(_StopAtGovernanceLoad):
        run_scaling_ladder.main(argv)

    assert captured["locked_manifest_path"] == canonical_manifest_path()


def test_run_scaling_ladder_cli_omits_disk_check_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_scaling_ladder as run_scaling_ladder

    captured = _patch_governance_capture(monkeypatch)
    argv = [
        "--train-dir",
        str(tmp_path / "train"),
        "--test-dir",
        str(tmp_path / "test"),
        "--campaign-manifest",
        str(tmp_path / "manifest.json"),
        "--campaign-result",
        str(tmp_path / "result.json"),
        "--campaign-store-root",
        str(tmp_path / "store"),
        "--campaign-artifact-root",
        str(tmp_path / "artifacts"),
    ]

    with pytest.raises(_StopAtGovernanceLoad):
        run_scaling_ladder.main(argv)

    assert captured["locked_manifest_path"] is None
