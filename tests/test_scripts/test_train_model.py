from __future__ import annotations

import json

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
