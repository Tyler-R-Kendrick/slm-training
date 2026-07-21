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


def test_train_cli_wires_semantic_plan_and_schema_decode_weights(
    monkeypatch, tmp_path
) -> None:
    """E623: scripts/train_model.py previously never exposed the
    semantic_plan_*/schema_*/schema_role_slot decode-weight levers that
    scripts/evaluate_model.py already exposed (E622 finding), so these
    biases could never be exercised during real training, only in
    standalone evaluate_model.py replay. This locks the CLI-to-config
    wiring in place.
    """
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
                "semantic-plan-decode-weights",
                "--model",
                "stub",
                "--steps",
                "1",
                "--honest-slot-contract",
                "--slot-contract-constrained-decode",
                "--semantic-role-contract-in-context",
                "--semantic-role-decode-weight",
                "8.0",
                "--schema-role-slot-decode-weight",
                "8.0",
                "--slot-coverage-close-decode-weight",
                "2.0",
                "--schema-value-decode-weight",
                "4.0",
                "--schema-opaque-close-decode-weight",
                "4.0",
                "--semantic-plan-decode-weight",
                "4.0",
                "--semantic-plan-margin-decode-weight",
                "2.0",
                "--semantic-plan-binding-decode-weight",
                "1.0",
                "--semantic-plan-root-decode-weight",
                "8.0",
                "--semantic-plan-root-margin-decode-weight",
                "2.0",
                "--semantic-plan-repeated-array-close-margin-decode-weight",
                "2.0",
                "--semantic-plan-repeated-slot-margin-decode-weight",
                "2.0",
                "--semantic-plan-typed-array-nonempty-margin-decode-weight",
                "2.0",
                "--semantic-plan-typed-array-item-margin-decode-weight",
                "2.0",
                "--no-sync-checkpoints",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.semantic_role_contract_in_context is True
    assert config.semantic_role_decode_weight == 8.0
    assert config.schema_role_slot_decode_weight == 8.0
    assert config.slot_coverage_close_decode_weight == 2.0
    assert config.schema_value_decode_weight == 4.0
    assert config.schema_opaque_close_decode_weight == 4.0
    assert config.semantic_plan_decode_weight == 4.0
    assert config.semantic_plan_margin_decode_weight == 2.0
    assert config.semantic_plan_binding_decode_weight == 1.0
    assert config.semantic_plan_root_decode_weight == 8.0
    assert config.semantic_plan_root_margin_decode_weight == 2.0
    assert config.semantic_plan_repeated_array_close_margin_decode_weight == 2.0
    assert config.semantic_plan_repeated_slot_margin_decode_weight == 2.0
    assert config.semantic_plan_typed_array_nonempty_margin_decode_weight == 2.0
    assert config.semantic_plan_typed_array_item_margin_decode_weight == 2.0
    # Untouched levers stay at the default-off value.
    assert config.schema_opaque_decode_weight == 0.0
    assert config.schema_enum_close_decode_weight == 0.0
    assert config.semantic_plan_seed_decode_weight == 0.0
    assert config.semantic_plan_inline_decode_weight == 0.0
    assert config.visible_reference_decode_weight == 0.0
    assert config.semantic_role_schema_candidates is False
