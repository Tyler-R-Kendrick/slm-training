"""Tests for ProgramSpec + lineage + split-before-derive (F1 / SLM-2)."""

from __future__ import annotations

import pytest

from slm_training.data.leakage import find_split_group_leakage, split_group_fingerprint
from slm_training.data.progspec.schema import (
    TASKS,
    TASK_GENERATE,
    ProgramSpec,
    assign_split_groups,
    resolve_split_group_id,
    structural_family_id,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.catalog import classify_source_family

# Two isomorphic layouts (renamed binders + placeholders) + one distinct layout.
HERO_A = 'root = Stack([hero], "column")\nhero_t = TextContent(":hero.title")\nhero = Card([hero_t])'
HERO_B = 'root = Stack([card], "column")\ncard_x = TextContent(":foo.bar")\ncard = Card([card_x])'
ROW_C = 'root = Stack([a, b], "row")\na = Button(":x.y")\nb = Button(":z.w")'


def test_structural_family_id_is_isomorphism_invariant() -> None:
    assert structural_family_id(HERO_A) == structural_family_id(HERO_B)
    assert structural_family_id(HERO_A) != structural_family_id(ROW_C)


def test_program_spec_round_trips_through_a_record() -> None:
    record = ExampleRecord(
        id="p1",
        prompt="Create a vertical hero card.",
        openui=HERO_A,
        placeholders=[":hero.title"],
        split="train",
        source="programspec_generated",
    )
    spec = ProgramSpec.from_record(record)
    out = spec.emit_record(id="p1_gen", task=TASK_GENERATE, prompt=record.prompt)
    assert out.openui == record.openui
    assert out.prompt == record.prompt
    assert out.meta["program_family_id"] == structural_family_id(HERO_A)


def test_emit_record_stamps_meta_for_every_task() -> None:
    spec = ProgramSpec(openui=HERO_A, prompt="Create a hero card.")
    for task in TASKS:
        rec = spec.emit_record(id=f"t_{task}", task=task, prompt="do the thing")
        assert rec.meta["task"] == task
        assert rec.meta["contract_id"]  # contract_id stamped
        assert rec.meta["split_group_id"] == spec.split_group_id
        assert rec.meta["tier"] == "silver"
        assert rec.split == "train"


def test_emit_record_rejects_unknown_task() -> None:
    spec = ProgramSpec(openui=HERO_A, prompt="p")
    with pytest.raises(ValueError):
        spec.emit_record(id="bad", task="NOT_A_TASK", prompt="p")


def test_split_before_derive_inherits_root_group() -> None:
    # child has a *different* structure (ROW_C) but the same root parent, so it
    # must inherit the parent's split group — not get its own.
    parent = ExampleRecord(
        id="root1",
        prompt="p",
        openui=HERO_A,
        placeholders=[":hero.title"],
        split="train",
        source="programspec_generated",
        meta={"root_parent_id": "root1"},
    )
    child = ExampleRecord(
        id="root1_aug",
        prompt="p2",
        openui=ROW_C,
        placeholders=[":x.y", ":z.w"],
        split="train",
        source="programspec_generated",
        meta={"root_parent_id": "root1"},
    )
    assign_split_groups([parent, child])
    assert parent.meta["split_group_id"] == child.meta["split_group_id"]
    assert parent.meta["split_group_id"] == structural_family_id(HERO_A)


def test_split_group_gate_blocks_cross_split_derivative() -> None:
    parent = ExampleRecord(
        id="root1",
        prompt="p",
        openui=HERO_A,
        placeholders=[":hero.title"],
        split="train",
        source="programspec_generated",
        meta={"root_parent_id": "root1"},
    )
    child = ExampleRecord(
        id="root1_aug",
        prompt="p2",
        openui=ROW_C,
        placeholders=[":x.y"],
        split="train",
        source="programspec_generated",
        meta={"root_parent_id": "root1"},
    )
    assign_split_groups([parent, child])
    reserved = {split_group_fingerprint(HERO_A)}  # the train side reserves this group
    # the derivative must be caught even though its own structure differs.
    assert find_split_group_leakage(child, reserved) is True
    # an unrelated program is not blocked.
    unrelated = ExampleRecord(
        id="u", prompt="p", openui=ROW_C, placeholders=[":x.y"], split="held_out"
    )
    assert find_split_group_leakage(unrelated, reserved) is False


def test_resolve_split_group_id_honors_override() -> None:
    assert resolve_split_group_id(HERO_A) == structural_family_id(HERO_A)
    assert resolve_split_group_id(HERO_A, override="page:acme") == "page:acme"


def test_new_families_classify() -> None:
    gen = ExampleRecord(
        id="g", prompt="p", openui=HERO_A, split="train", source="programspec_generated"
    )
    assert classify_source_family(gen) == "programspec_generated"
    edit = ExampleRecord(
        id="e",
        prompt="p",
        openui=HERO_A,
        split="train",
        source="rico+edit",
        meta={"synth": "edit"},
    )
    assert classify_source_family(edit) == "edit_patch"
    web = ExampleRecord(
        id="w", prompt="p", openui=HERO_A, split="train", source="deconstruct"
    )
    assert classify_source_family(web) == "web_projection"
