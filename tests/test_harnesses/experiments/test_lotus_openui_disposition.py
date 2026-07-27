"""Tests for slm_training.harnesses.experiments.lotus_openui_disposition (SLM-258)."""

from __future__ import annotations

from slm_training.harnesses.experiments.lotus_openui_disposition import (
    ADOPTION_VERDICTS,
    INCONCLUSIVE,
    build_disposition,
    load_project_evidence,
    render_markdown,
)


def test_real_artifacts_yield_inconclusive_missing_evidence() -> None:
    """Against the currently-committed upstream contracts, adoption is
    fail-closed: SLM-248 is needs_target_trace_contract, SLM-249 is
    inconclusive, and every LOT1+ gate is not_authorized.
    """
    disposition = build_disposition(load_project_evidence())

    assert disposition.adoption_verdict == INCONCLUSIVE
    assert disposition.adoption_verdict not in ADOPTION_VERDICTS
    assert disposition.upstream_verdicts["fidelity_contract"] == "needs_target_trace_contract"
    assert disposition.upstream_verdicts["trace_gate"] == "inconclusive"
    assert disposition.upstream_verdicts["slm250_activation"] == "not_authorized"
    assert any("fidelity_contract" in b for b in disposition.adoption_blockers)
    assert any("trace_gate" in b for b in disposition.adoption_blockers)


def test_missing_artifact_blocks_adoption() -> None:
    """Rule: disposition schema rejects missing mandatory gate refs."""
    evidence = load_project_evidence(
        {"fidelity_contract": "docs/design/does-not-exist.json"}
    )
    disposition = build_disposition(evidence)

    assert disposition.adoption_verdict == INCONCLUSIVE
    assert "missing_or_invalid_artifact:fidelity_contract" in disposition.adoption_blockers


def test_interpretability_language_blocked_without_causal_gate() -> None:
    disposition = build_disposition(load_project_evidence())

    assert "reasoning" not in disposition.allowed_language
    assert "planning" not in disposition.allowed_language
    assert any("interpretable-reasoning" in b for b in disposition.blocked_claims)


def test_no_production_default_change_and_no_adopt_default() -> None:
    disposition = build_disposition(load_project_evidence())

    assert disposition.allowed_code_config_default_changes == [
        "none: production defaults, decode paths, and serving configs are unchanged"
    ]
    assert "adopt_default" not in {disposition.adoption_verdict}


def test_trace_contract_kept_as_diagnostic_asset() -> None:
    """Useful trace/eval assets survive the negative model disposition."""
    disposition = build_disposition(load_project_evidence())

    assert (
        disposition.mechanism_dispositions["compiler_reasoning_trace_contract"]
        == "keep_diagnostic"
    )
    assert disposition.mechanism_dispositions["kxc_causal_workspace"] == "not_identifiable"


def test_render_markdown_links_verdicts_and_blockers() -> None:
    disposition = build_disposition(load_project_evidence())
    markdown = render_markdown(disposition)

    assert "SLM-258" in markdown
    assert INCONCLUSIVE in markdown
    assert "not_authorized" in markdown
    assert "Production defaults" in markdown
