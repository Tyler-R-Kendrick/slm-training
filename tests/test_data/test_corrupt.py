from __future__ import annotations

import pytest

from slm_training.data.corrupt import (
    CorruptionOperator,
    build_corruption,
    generate_corruptions,
)
from slm_training.data.progspec import ProgramSpec
from slm_training.data.verify import Tier, verify_record
from slm_training.dsl.lang_core import validate
from slm_training.dsl.schema import ExampleRecord


RICH = '''root = Stack([panel, dialog], "column")
email = Input("email", ":form.email", "email")
control = FormControl(":form.label", email)
button = Button(":form.submit")
buttons = Buttons([button])
form = Form("signup", buttons, [control])
panel = Card([form])
copy = TextContent(":modal.body")
dialog = Modal(":modal.title", true, [copy])'''
SIMPLE = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


@pytest.mark.parametrize("operator", list(CorruptionOperator))
def test_every_operator_is_rejected_and_has_a_verified_repair(
    operator: CorruptionOperator,
) -> None:
    case = build_corruption(RICH, operator)
    assert case.operator is operator
    assert case.family is operator.family
    assert case.broken_openui != case.clean_openui
    assert case.location
    assert case.diagnostics
    assert case.edit_distance > 0
    assert len(case.preserved_nodes) >= 8
    assert case.minimal_repair == case.clean_openui
    assert case.acceptable_repairs == (case.clean_openui,)
    assert case.exact_repair

    validate(case.minimal_repair)
    report = verify_record(
        ExampleRecord(id="broken", prompt="Repair", openui=case.broken_openui)
    )
    assert not report.ok
    assert report.failing_gate is not None


def test_generation_is_complete_and_deterministic() -> None:
    first = generate_corruptions(RICH)
    second = generate_corruptions(RICH)
    assert tuple(case.operator for case in first) == tuple(CorruptionOperator)
    assert first == second


def test_simple_program_still_covers_every_operator_family() -> None:
    cases = generate_corruptions(SIMPLE)
    assert len(cases) >= 35
    assert {case.family for case in cases} == {
        operator.family for operator in CorruptionOperator
    }


def test_nonminimal_patch_records_more_edits_than_single_local_corruption() -> None:
    single = build_corruption(RICH, CorruptionOperator.WRONG_ARG_COUNT)
    nonminimal = build_corruption(RICH, CorruptionOperator.PATCH_NONMINIMAL)
    assert nonminimal.edit_distance > single.edit_distance


def test_repair_row_keeps_clean_target_and_split_lineage() -> None:
    spec = ProgramSpec.from_openui(
        id="program-1",
        openui=RICH,
        facts={"component": "Form"},
        program_family_id="forms",
        lineage_id="lineage-1",
        split_group_id="group-1",
        split="held_out",
    )
    case = build_corruption(RICH, CorruptionOperator.MISSING_QUOTE)
    row = case.to_record(spec)

    assert row.openui == case.clean_openui
    assert row.prompt.startswith("Repair this OpenUI program.\n---BROKEN---\n")
    assert row.prompt.endswith(case.broken_openui)
    assert row.split == "held_out"
    assert row.source == "repair_taxonomy"
    assert row.meta["task"] == "repair"
    assert row.meta["split_group_id"] == "group-1"
    repair = row.meta["repair"]
    assert repair["family"] == "repair_taxonomy"
    assert repair["operator"] == "missing_quote"
    assert repair["minimal_repair"] == case.clean_openui
    assert repair["exact_repair"] is True
    validate(row.openui)
    assert verify_record(row).tier is Tier.SILVER


def test_multiple_acceptable_repairs_are_not_exact() -> None:
    alternative = RICH.replace('"column"', '"row"')
    case = build_corruption(
        RICH,
        CorruptionOperator.MISSING_QUOTE,
        acceptable_repairs=(RICH, alternative),
    )
    assert not case.exact_repair
    assert len(case.acceptable_repairs) == 2
