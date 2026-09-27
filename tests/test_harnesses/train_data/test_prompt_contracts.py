"""Training-prompt contracts for component and placeholder visibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

from ._pipeline_fixtures import seed_file


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_prompt_contracts_expose_component_counts_and_slots(tmp_path: Path) -> None:
    baseline = build_train_data(
        TrainDataConfig(
            seed_path=seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="baseline",
            synthesizer="none",
        )
    )
    result = build_train_data(
        TrainDataConfig(
            seed_path=seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="contracts",
            synthesizer="none",
            prompt_component_contract=True,
            prompt_slot_contract=True,
        )
    )
    rows = {row.id: row for row in load_jsonl(Path(result["output_dir"]) / "records.jsonl")}
    assert "Components: Card x1, Stack x1, TextContent x2" in rows["t1"].prompt
    assert "Placeholders: :slot_0, :slot_1" in rows["t1"].prompt
    assert result["stats"]["prompt_component_contract"] is True
    assert result["stats"]["prompt_slot_contract"] is True
    assert result["manifest"]["ids"] == baseline["manifest"]["ids"]


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_component_contract_can_expose_types_without_counts(tmp_path: Path) -> None:
    result = build_train_data(
        TrainDataConfig(
            seed_path=seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="types",
            synthesizer="none",
            prompt_component_contract=True,
            prompt_component_contract_mode="types",
        )
    )
    rows = {row.id: row for row in load_jsonl(Path(result["output_dir"]) / "records.jsonl")}
    assert "Components: Card, Stack, TextContent" in rows["t1"].prompt
    assert " x" not in rows["t1"].prompt
    assert result["stats"]["prompt_component_contract_mode"] == "types"


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_semantic_role_contract_uses_only_visible_slots_and_types(
    tmp_path: Path,
) -> None:
    result = build_train_data(
        TrainDataConfig(
            seed_path=seed_file(tmp_path),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="roles",
            synthesizer="none",
            prompt_slot_contract=True,
            prompt_component_contract=True,
            prompt_component_contract_mode="types",
            prompt_semantic_role_contract=True,
        )
    )
    rows = {row.id: row for row in load_jsonl(Path(result["output_dir"]) / "records.jsonl")}
    assert "Semantic roles: slot_0(value); slot_1(value)" in rows["t1"].prompt
    assert "Semantic roles: slot_0(value)" in rows["t2"].prompt
    assert " x" not in rows["t2"].prompt
    assert result["stats"]["prompt_semantic_role_contract"] is True


def test_semantic_role_contract_requires_visible_authority(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires visible component and slot"):
        build_train_data(
            TrainDataConfig(
                seed_path=seed_file(tmp_path),
                rico_path=None,
                source="fixture",
                output_root=tmp_path / "train_data",
                version="invalid-roles",
                synthesizer="none",
                prompt_semantic_role_contract=True,
            )
        )
