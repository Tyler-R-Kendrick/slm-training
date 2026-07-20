"""E621: scripts.train_model must expose the same decode-weight CLI surface
as scripts.evaluate_model, or these levers stay silently None/inert during
live training and its periodic in-training eval (see E622's finding that
train_model.py had zero occurrences of `semantic_plan` in its CLI)."""

from pathlib import Path

import scripts.train_model as train_model
from slm_training.harnesses.model_build.config import ModelBuildConfig

NEWLY_WIRED_DECODE_WEIGHT_FIELDS = (
    "semantic_role_decode_weight",
    "slot_coverage_close_decode_weight",
    "schema_value_decode_weight",
    "schema_enum_close_decode_weight",
    "schema_opaque_decode_weight",
    "schema_opaque_close_decode_weight",
    "schema_role_slot_decode_weight",
    "semantic_plan_decode_weight",
    "semantic_plan_margin_decode_weight",
    "semantic_plan_seed_decode_weight",
    "semantic_plan_inline_decode_weight",
    "semantic_plan_binding_decode_weight",
    "semantic_plan_root_decode_weight",
    "semantic_plan_root_margin_decode_weight",
    "semantic_plan_repeated_array_close_margin_decode_weight",
    "semantic_plan_repeated_slot_margin_decode_weight",
    "semantic_plan_typed_array_nonempty_margin_decode_weight",
    "semantic_plan_typed_array_item_margin_decode_weight",
    "visible_reference_decode_weight",
)


def test_every_newly_wired_field_is_declared_on_model_build_config() -> None:
    declared = {f.name for f in __import__("dataclasses").fields(ModelBuildConfig)}
    missing = set(NEWLY_WIRED_DECODE_WEIGHT_FIELDS) - declared
    assert not missing, f"ModelBuildConfig is missing fields: {missing}"


def test_train_model_cli_threads_decode_weights_into_model_build_config(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, ModelBuildConfig] = {}

    def _fake_train(config: ModelBuildConfig) -> dict:
        captured["config"] = config
        return {}

    monkeypatch.setattr(train_model, "train", _fake_train)

    argv = [
        "--train-dir",
        str(tmp_path),
        "--run-root",
        str(tmp_path / "runs"),
        "--run-id",
        "e621-cli-wiring-test",
        "--steps",
        "0",
        "--semantic-role-decode-weight",
        "1.0",
        "--semantic-role-schema-candidates",
        "--slot-coverage-close-decode-weight",
        "2.0",
        "--schema-value-decode-weight",
        "3.0",
        "--schema-enum-close-decode-weight",
        "4.0",
        "--schema-opaque-decode-weight",
        "5.0",
        "--schema-opaque-close-decode-weight",
        "6.0",
        "--schema-role-slot-decode-weight",
        "7.0",
        "--semantic-plan-decode-weight",
        "8.0",
        "--semantic-plan-margin-decode-weight",
        "9.0",
        "--semantic-plan-seed-decode-weight",
        "10.0",
        "--semantic-plan-inline-decode-weight",
        "11.0",
        "--semantic-plan-binding-decode-weight",
        "12.0",
        "--semantic-plan-root-decode-weight",
        "13.0",
        "--semantic-plan-root-margin-decode-weight",
        "14.0",
        "--semantic-plan-repeated-array-close-margin-decode-weight",
        "15.0",
        "--semantic-plan-repeated-slot-margin-decode-weight",
        "16.0",
        "--semantic-plan-typed-array-nonempty-margin-decode-weight",
        "17.0",
        "--semantic-plan-typed-array-item-margin-decode-weight",
        "18.0",
        "--visible-reference-decode-weight",
        "19.0",
    ]

    exit_code = train_model.main(argv)

    assert exit_code == 0
    config = captured["config"]
    assert config.semantic_role_decode_weight == 1.0
    assert config.semantic_role_schema_candidates is True
    assert config.slot_coverage_close_decode_weight == 2.0
    assert config.schema_value_decode_weight == 3.0
    assert config.schema_enum_close_decode_weight == 4.0
    assert config.schema_opaque_decode_weight == 5.0
    assert config.schema_opaque_close_decode_weight == 6.0
    assert config.schema_role_slot_decode_weight == 7.0
    assert config.semantic_plan_decode_weight == 8.0
    assert config.semantic_plan_margin_decode_weight == 9.0
    assert config.semantic_plan_seed_decode_weight == 10.0
    assert config.semantic_plan_inline_decode_weight == 11.0
    assert config.semantic_plan_binding_decode_weight == 12.0
    assert config.semantic_plan_root_decode_weight == 13.0
    assert config.semantic_plan_root_margin_decode_weight == 14.0
    assert config.semantic_plan_repeated_array_close_margin_decode_weight == 15.0
    assert config.semantic_plan_repeated_slot_margin_decode_weight == 16.0
    assert config.semantic_plan_typed_array_nonempty_margin_decode_weight == 17.0
    assert config.semantic_plan_typed_array_item_margin_decode_weight == 18.0
    assert config.visible_reference_decode_weight == 19.0


def test_train_model_cli_decode_weights_default_to_zero_not_none(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, ModelBuildConfig] = {}

    def _fake_train(config: ModelBuildConfig) -> dict:
        captured["config"] = config
        return {}

    monkeypatch.setattr(train_model, "train", _fake_train)

    argv = [
        "--train-dir",
        str(tmp_path),
        "--run-root",
        str(tmp_path / "runs"),
        "--run-id",
        "e621-cli-wiring-default-test",
        "--steps",
        "0",
    ]

    exit_code = train_model.main(argv)

    assert exit_code == 0
    config = captured["config"]
    for field in NEWLY_WIRED_DECODE_WEIGHT_FIELDS:
        assert getattr(config, field) == 0.0, field
