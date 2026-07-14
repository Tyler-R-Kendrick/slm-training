from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.generalization import generalization_report


def _record(
    rid: str,
    prompt: str,
    openui: str,
    *,
    group: str,
    contract: str,
    domain: str,
    operators: list[str],
) -> ExampleRecord:
    return ExampleRecord(
        id=rid,
        prompt=prompt,
        openui=openui,
        split="train" if rid.startswith("train") else "held_out",
        meta={
            "split_group_id": group,
            "contract_id": contract,
            "provenance": {"domain": domain},
            "edit": {"operators": operators},
        },
    )


def test_generalization_slices_and_decontamination_are_deterministic() -> None:
    train_program = 'root = Stack([cta])\ncta = Button(":cta.label")'
    novel_program = (
        "root = Stack([card, field])\n"
        "card = Card([title])\n"
        'title = TextContent(":title.text")\n'
        'field = Input(":field.label")'
    )
    train = [
        _record(
            "train-a",
            "train prompt",
            train_program,
            group="train-group",
            contract="contract-a",
            domain="train.example",
            operators=["replace"],
        )
    ]
    held = [
        _record(
            "held-novel",
            "novel held prompt",
            novel_program,
            group="held-group",
            contract="contract-b",
            domain="held.example",
            operators=["wrap", "move"],
        ),
        _record(
            "held-leak",
            "different prompt",
            train_program,
            group="train-group",
            contract="contract-a",
            domain="train.example",
            operators=["replace"],
        ),
    ]

    first = generalization_report(train, held)
    second = generalization_report(train, reversed(held))
    assert first == second
    assert first["decontaminated"] is False
    assert first["contaminated"][0]["id"] == "held-leak"
    assert "split_group_id" in first["contaminated"][0]["reasons"]
    slices = set(first["records"][0]["slices"])
    assert {
        "unseen_component_pair",
        "unseen_component_triple",
        "deeper_tree",
        "longer_program",
        "new_edit_composition",
        "new_domain_or_site",
        "new_contract_version",
    } <= slices
