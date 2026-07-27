from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.retrieval.verified_patch_v1 import (
    RetrievalRequest,
    build_verified_patch_index,
    run_verified_patch_baseline,
)


def _record(
    record_id: str, prompt: str, openui: str, *, group: str | None = None
) -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        split="train",
        source="fixture",
        meta={"split_group_id": group or record_id, "lineage_id": group or record_id},
    )


def test_retrieval_excludes_same_example_derivatives_and_never_reads_gold_target() -> (
    None
):
    train = [
        _record(
            "same",
            "same prompt",
            'root = Card([x])\nx = TextContent(":title")',
            group="root",
        ),
        _record(
            "same-prompt",
            "save action",
            'root = Stack([x], "column")\nx = Button(":submit")',
        ),
        _record(
            "same-structure",
            "different request",
            'root = Card([x])\nx = TextContent(":body")',
        ),
        _record(
            "eligible",
            "save primary button",
            'root = Stack([x], "column")\nx = Button(":save")',
        ),
    ]
    index = build_verified_patch_index(train)
    request = RetrievalRequest(
        "held",
        "save action",
        'root = Card([x])\nx = TextContent(":title")',
        "root",
        "root",
    )
    first = run_verified_patch_baseline(index, request)
    changed_gold = ExampleRecord(
        id="held",
        prompt="unrelated",
        openui='root = Form([x])\nx = Input(":email")',
        split="held_out",
        source="gold",
    )
    del changed_gold  # Selection has no gold argument by contract.
    second = run_verified_patch_baseline(index, request)
    assert first.selected_id == second.selected_id == "eligible"
    assert first.excluded_reasons["same_example_derivative"] == 1
    assert first.excluded_reasons.get("prompt", 0) >= 1
    assert first.excluded_reasons.get("structure", 0) >= 1
    assert first.verifier_ok is True
    assert first.patch is not None
