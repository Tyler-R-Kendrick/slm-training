from dataclasses import replace

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.learner_corrections import (
    CopyProgramOracle,
    LearnerState,
    corrective_example,
    corrective_mixture,
)


def fixture():
    oracle = CopyProgramOracle('root = TextContent(":slot_0")', "root-a")
    record = ExampleRecord(
        id="a",
        prompt=oracle.prompt,
        openui=oracle.program,
        split="train",
        placeholders=[":slot_0"],
        meta={"root_family_id": "root-a"},
    )
    state = LearnerState((1, 2, 3), (False, True, False), "a" * 64, "b" * 64, 7)
    return record, state, oracle


def test_legal_wrong_state_gets_task_label_and_preserves_nonprefix_canvas():
    record, state, oracle = fixture()
    result = corrective_example(record, state, 'root = Button(":slot_0")', oracle)
    assert result.openui == record.openui
    assert result.meta["learner_correction"]["state"]["unknown"] == [False, True, False]
    assert result.meta["learner_correction"]["label_status"] == "verified_task"
    assert state.checkpoint_sha256 not in result.prompt
    assert state.tokenizer_sha256 not in result.prompt


def test_parse_success_cannot_label_arbitrary_intent():
    record, state, oracle = fixture()
    with pytest.raises(ValueError, match="abstains"):
        corrective_example(
            replace(record, prompt="Build a nice app"),
            state,
            'root = Button(":slot_0")',
            oracle,
        )


def test_evaluation_input_cannot_be_corrective_training():
    record, state, oracle = fixture()
    with pytest.raises(ValueError, match="training-only"):
        corrective_example(
            replace(record, split="held_out"), state, 'root = Button(":slot_0")', oracle
        )


def test_shared_root_rejected_even_with_renamed_case():
    record, state, oracle = fixture()
    held = replace(record, id="renamed", split="held_out", prompt="a different prompt")
    with pytest.raises(ValueError, match="leakage"):
        corrective_mixture(
            [record],
            [(record, state, 'root = Button(":slot_0")', oracle)],
            scored_suites={"held_out": [held]},
            training_ancestors=[],
        )


def test_invalid_unknown_mask_rejected():
    with pytest.raises(ValueError, match="booleans"):
        LearnerState((1,), (1,), "a" * 64, "b" * 64, 7)
