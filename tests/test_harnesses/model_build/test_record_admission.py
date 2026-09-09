"""Loader admission must reject the same persisted records as the trainer."""

from dataclasses import replace

import pytest

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build.data import load_train_records
from slm_training.harnesses.test_data.certified import _assert_certified_role_safe
from slm_training.models.twotower import TwoTowerModel


def _record():
    return ExampleRecord(
        id="admission-case", prompt="Show a card",
        openui='root = Stack([c1])\nc1 = TextContent([":slot_0"])',
        placeholders=[":slot_0"],
    )


@pytest.mark.parametrize("fault", ["role", "prompt_label", "design_label", "named_marker"])
def test_loader_rejects_records_refused_by_actual_trainer(tmp_path, fault):
    record = _record()
    if fault == "role":
        record.openui = 'root = Stack([c1])\nc1 = RadialChart([":slot_0"], [1])'
    elif fault == "prompt_label":
        record.prompt = "Slot roles: :slot_0 = title"
    elif fault == "design_label":
        record.design_md = "Slot roles: :slot_0 = title"
    else:
        record.prompt = "Show :customer_name"
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [record])
    before = path.read_bytes()
    for consumer in (_assert_certified_role_safe,
                     lambda row: TwoTowerModel.from_records([row])):
        with pytest.raises(ValueError):
            consumer(record)
    with pytest.raises(ValueError, match="admission-case"):
        load_train_records(tmp_path)
    assert path.read_bytes() == before  # Admission never sanitizes frozen inputs.


@pytest.mark.parametrize("kind", ["document", "lexical"])
def test_valid_record_keeps_declared_output_kind(tmp_path, kind):
    record = _record()
    if kind == "lexical":
        record = replace(record, openui='":slot_0"', target_kind="lexical")
    write_jsonl(tmp_path / "records.jsonl", [record])
    _assert_certified_role_safe(record)
    assert load_train_records(tmp_path) == [record]
