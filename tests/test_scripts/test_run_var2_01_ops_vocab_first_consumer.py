from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from scripts.run_var2_01_ops_vocab_first_consumer import (
    _decision_sentence,
    render_markdown,
    run_audit,
)

pytestmark = pytest.mark.skipif(
    not bridge_available(), reason="OpenUI bridge deps missing"
)


def _payload() -> dict:
    return run_audit(generated_at="2026-07-26T00:00:00Z")


def test_off_arm_is_byte_identical_with_the_lever_disabled() -> None:
    payload = _payload()
    probe = payload["wiring_probe"]
    assert probe["all_off_arms_unchanged"] is True
    assert probe["production_conditioning_changed"] is False
    assert probe["hypothetical"] is True


def test_on_arm_layers_reserved_tokens_without_collision() -> None:
    payload = _payload()
    probe = payload["wiring_probe"]
    assert probe["representable_fraction"] == 1.0
    assert probe["all_reserved_tokens_layered"] is True
    for row in probe["rows"]:
        assert row["representable"] is True
        assert row["on_arm_adds_history_section"] is True


def test_adequacy_audit_reports_all_four_required_tables() -> None:
    payload = _payload()
    audit = payload["adequacy_audit"]
    assert set(audit) >= {
        "unrepresentable_intents",
        "unused_ops",
        "family_confusions",
        "arity_mismatches",
    }
    assert audit["claim_class"] == "measurement"
    # This fixture corpus only exercises the local operator library -- a
    # real, non-fabricated finding, not a bug: most of OPS_VOCAB's 20 ops go
    # unused at this scale.
    assert len(audit["unused_ops"]) > 0
    assert audit["unrepresentable_intents"] == []
    assert audit["arity_mismatches"] == []
    assert audit["family_confusions"]["status"] == "UNRUN_CONDITIONAL"


def test_ast_edit_and_transaction_commit_history_ids_are_structurally_unreachable() -> (
    None
):
    # conversation.ast_edit / conversation.transaction_commit are real
    # OPS_VOCAB history-family ids (ConversationOperation enumerates them),
    # but resolve_turn_op_ids always resolves those turn kinds to their
    # specific underlying operator id instead -- so these two ids can never
    # appear in used_op_ids regardless of corpus size.
    payload = _payload()
    audit = payload["adequacy_audit"]
    assert audit["unused_ops_structurally_unreachable_by_this_consumer"] == [
        "conversation.ast_edit",
        "conversation.transaction_commit",
    ]
    assert set(
        audit["unused_ops_structurally_unreachable_by_this_consumer"]
    ) <= set(audit["unused_ops"])


def test_campaign_binds_both_fingerprints_and_stays_wiring_claim_class() -> None:
    payload = _payload()
    campaign = payload["experiment_campaign"]
    assert campaign["claim_class"] == "wiring"
    assert payload["ops_vocab_fingerprint"][:12] in campaign["hypothesis"]
    assert payload["corpus_content_fingerprint"][:12] in campaign["hypothesis"]
    arm_hashes = {arm["config_sha256"] for arm in campaign["arms"]}
    assert len(arm_hashes) == len(campaign["arms"])  # every arm hash is unique
    assert any(arm["role"] == "control" for arm in campaign["arms"])
    assert any(arm["role"] == "candidate" for arm in campaign["arms"])


def test_decision_sentence_is_deterministic_for_this_fixture_corpus() -> None:
    payload = _payload()
    sentence = _decision_sentence(payload)
    assert "Wiring is reachable and non-regressing" in sentence
    assert "structurally unreachable by this consumer's own design" in sentence


def test_render_markdown_includes_every_required_section() -> None:
    payload = _payload()
    markdown = render_markdown(payload)
    for heading in (
        "# VAR2-01 (SLM-428)",
        "## Honesty scoping",
        "## Adequacy audit (the deliverable)",
        "## Wiring probe (off vs on)",
        "## Decision this audit licenses",
    ):
        assert heading in markdown


def test_main_writes_both_output_files(tmp_path: Path) -> None:
    from scripts.run_var2_01_ops_vocab_first_consumer import main

    json_out = tmp_path / "var2-01.json"
    md_out = tmp_path / "var2-01.md"
    exit_code = main(["--json-out", str(json_out), "--md-out", str(md_out)])
    assert exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "var2_01_ops_vocab_first_consumer/v1"
    assert md_out.read_text(encoding="utf-8").startswith(
        "# VAR2-01 (SLM-428): OPS_VOCAB first consumer + adequacy audit"
    )
