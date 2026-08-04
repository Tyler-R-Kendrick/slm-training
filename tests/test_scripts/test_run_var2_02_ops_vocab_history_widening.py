from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from scripts.run_var2_02_ops_vocab_history_widening import (
    _decision_sentence,
    render_markdown,
    run_audit,
)

pytestmark = pytest.mark.skipif(
    not bridge_available(), reason="OpenUI bridge deps missing"
)


def _payload() -> dict:
    return run_audit(generated_at="2026-07-28T00:00:00Z")


def test_widened_trace_realizes_all_four_targeted_history_ops() -> None:
    payload = _payload()
    audit = payload["adequacy_audit"]
    assert audit["targeted_history_ops"] == [
        "conversation.checkout_state",
        "conversation.copy_state",
        "conversation.redo",
        "conversation.undo",
    ]
    assert audit["newly_covered_ops"] == audit["targeted_history_ops"]
    assert audit["still_missing_targeted_ops"] == []


def test_op_id_sequence_contains_every_targeted_history_op_and_no_ast_edit_or_transaction_commit() -> (
    None
):
    payload = _payload()
    op_ids = payload["op_id_sequence"]
    for op_id in (
        "conversation.copy_state",
        "conversation.undo",
        "conversation.redo",
        "conversation.checkout_state",
        "conversation.fork",
    ):
        assert op_id in op_ids
    # Structurally unreachable per VAR2-01 -- never attempted here.
    assert "conversation.ast_edit" not in op_ids
    assert "conversation.transaction_commit" not in op_ids


def test_topology_ops_are_explicitly_left_out_of_scope() -> None:
    payload = _payload()
    audit = payload["adequacy_audit"]
    topology = audit["topology_ops_left_out_of_scope"]
    assert len(topology) == 10
    assert all(op.startswith("openui.") for op in topology)
    # Never silently claimed as covered by this narrower slice.
    assert set(topology).isdisjoint(set(audit["newly_covered_ops"]))


def test_unused_after_widening_is_union_with_var2_01_not_this_trace_alone() -> None:
    payload = _payload()
    audit = payload["adequacy_audit"]
    unused_after = set(audit["unused_ops_after_widening_union_with_var2_01"])
    # The two structurally unreachable ids always remain.
    assert {"conversation.ast_edit", "conversation.transaction_commit"} <= unused_after
    # Every targeted history op was closed, so none of them remain unused.
    assert unused_after.isdisjoint(set(audit["targeted_history_ops"]))
    # This computation must never silently drop an id that VAR2-01 already
    # covered in its own corpus just because this narrower trace alone
    # doesn't happen to touch it too -- the union set is always a subset of
    # VAR2-01's originally published 16 unused ids.
    assert unused_after <= set(
        audit["unused_ops_before_widening_var2_01_published"]
    )


def test_off_arm_is_byte_identical_with_the_lever_disabled() -> None:
    payload = _payload()
    probe = payload["wiring_probe"]
    assert probe["off_arm_unchanged"] is True
    assert probe["production_conditioning_changed"] is False
    assert probe["hypothetical"] is True


def test_on_arm_layers_reserved_tokens_without_collision() -> None:
    payload = _payload()
    probe = payload["wiring_probe"]
    assert probe["representable"] is True
    assert probe["reserved_tokens_all_layered"] is True
    assert probe["on_arm_adds_history_section"] is True


def test_decision_sentence_reports_full_closure_for_this_fixture() -> None:
    payload = _payload()
    sentence = _decision_sentence(payload)
    assert "4/4 targeted history ops" in sentence
    assert "structurally unreachable" in sentence


def test_render_markdown_includes_every_required_section() -> None:
    payload = _payload()
    markdown = render_markdown(payload)
    for heading in (
        "# VAR2-02:",
        "## Honesty scoping",
        "## Op-id sequence realized by the widened trace",
        "## Adequacy delta vs VAR2-01",
        "## Wiring probe (off vs on, widened trace)",
        "## Decision this audit licenses",
    ):
        assert heading in markdown


def test_main_writes_both_output_files(tmp_path: Path) -> None:
    from scripts.run_var2_02_ops_vocab_history_widening import main

    json_out = tmp_path / "var2-02.json"
    md_out = tmp_path / "var2-02.md"
    exit_code = main(["--json-out", str(json_out), "--md-out", str(md_out)])
    assert exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "var2_02_ops_vocab_history_widening/v1"
    assert md_out.read_text(encoding="utf-8").startswith(
        "# VAR2-02: OPS_VOCAB history-op corpus widening"
    )
