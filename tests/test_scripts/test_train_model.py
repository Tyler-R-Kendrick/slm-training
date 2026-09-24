from __future__ import annotations

import json

import pytest

import scripts.train_model as train_model
from scripts.train_model import resolve_published_train_version


def test_published_train_version_resolves_canonical_mixture(tmp_path) -> None:
    version_dir = tmp_path / "v2"
    version_dir.mkdir()
    mixture = version_dir / "mixture.json"
    mixture.write_text(json.dumps({"mixture_id": "v2"}), encoding="utf-8")

    train_dir, resolved_mixture = resolve_published_train_version(
        "v2", root=tmp_path
    )

    assert train_dir == version_dir
    assert resolved_mixture == mixture


def test_published_train_version_allows_corpus_without_mixture(tmp_path) -> None:
    version_dir = tmp_path / "v2"
    version_dir.mkdir()

    train_dir, resolved_mixture = resolve_published_train_version(
        "v2", root=tmp_path
    )

    assert train_dir == version_dir
    assert resolved_mixture is None


def test_train_cli_wires_honest_slot_contract(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {"run_id": config.run_id}

    monkeypatch.setattr(train_model, "train", fake_train)

    assert (
        train_model.main(
            [
                "--train-dir",
                str(tmp_path),
                "--run-root",
                str(tmp_path / "runs"),
                "--run-id",
                "honest-slot-contract",
                "--model",
                "stub",
                "--steps",
                "1",
                "--slot-contract-in-context",
                "--honest-slot-contract",
                "--no-sync-checkpoints",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.slot_contract_in_context is True
    assert config.honest_slot_contract is True


def test_train_cli_wires_action_alias_args(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {"run_id": config.run_id}

    monkeypatch.setattr(train_model, "train", fake_train)

    assert (
        train_model.main(
            [
                "--train-dir",
                str(tmp_path),
                "--run-root",
                str(tmp_path / "runs"),
                "--run-id",
                "alias-test",
                "--model",
                "stub",
                "--steps",
                "1",
                "--action-embedding-init",
                "alias_aware_description",
                "--action-alias-mode",
                "fixed",
                "--action-description-name-mode",
                "alias_aware_description",
                "--no-sync-checkpoints",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.action_embedding_init == "alias_aware_description"
    assert config.action_alias_mode == "fixed"
    assert config.action_description_name_mode == "alias_aware_description"


@pytest.mark.parametrize("backend", ["scratch", "hf"])
@pytest.mark.parametrize("sync_flag", [None, "--sync-checkpoints", "--no-sync-checkpoints"])
@pytest.mark.parametrize("bucket", [None, "hf://buckets/example/checkpoints"])
def test_checkpoint_sync_resolution(tmp_path, backend, sync_flag, bucket) -> None:
    from slm_training.harnesses.model_build.checkpoint_bucket import (
        DEFAULT_CHECKPOINT_BUCKET_URI,
    )

    argv = ["--train-dir", str(tmp_path), "--device", "cpu", "--context-backend", backend]
    if sync_flag:
        argv.append(sync_flag)
    if bucket:
        argv.extend(["--checkpoint-bucket", bucket])
    config = train_model.resolve_config(argv)
    expected_sync = sync_flag != "--no-sync-checkpoints" and (
        sync_flag == "--sync-checkpoints" or backend == "hf"
    )
    assert config.sync_checkpoints is expected_sync
    assert config.checkpoint_bucket == (
        bucket if bucket is not None else DEFAULT_CHECKPOINT_BUCKET_URI if expected_sync else None
    )


def test_checkpoint_sync_conflicting_flags_still_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        train_model.resolve_config([
            "--train-dir", str(tmp_path), "--sync-checkpoints", "--no-sync-checkpoints",
        ])
    assert error.value.code == 2
