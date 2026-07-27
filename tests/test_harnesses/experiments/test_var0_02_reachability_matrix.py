"""VAR0-02 (SLM-423): per-variant reachability matrix.

Reachability is a function of (action alphabet x seed x target set), never a
program-wide scalar -- these tests pin that the optional ``variant``
parameter changes nothing when omitted (the regression guard on the
refactor), that a non-tree-edit variant is honestly refused rather than
silently analyzed with the wrong engine, and that the matrix builder never
infers one variant's row from another's.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.variants import VARIANTS
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    DEFAULT_SEED_SOURCE,
    ReachabilityNotMeasuredError,
    TREE_EDIT_VARIANT_ID,
    analyze_reachability,
    tree_edit_variant,
)
from scripts.run_var0_02_reachability_matrix import (
    NOT_MEASURED_REASONS,
    build_matrix,
    render_markdown,
)

SEED = DEFAULT_SEED_SOURCE


def test_tree_edit_variant_resolves_the_registered_contract() -> None:
    variant = tree_edit_variant()
    assert variant.variant_id == TREE_EDIT_VARIANT_ID
    assert variant in VARIANTS


def test_omitting_variant_reproduces_explicit_tree_edit_variant_exactly() -> None:
    target = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'
    implicit = analyze_reachability(SEED, target, slot_inventory=[":cta.label"])
    explicit = analyze_reachability(
        SEED, target, slot_inventory=[":cta.label"], variant=tree_edit_variant()
    )
    assert implicit.to_dict() == explicit.to_dict()


def test_non_tree_edit_variant_is_refused_not_silently_analyzed() -> None:
    for variant in VARIANTS:
        if variant.variant_id == TREE_EDIT_VARIANT_ID:
            continue
        with pytest.raises(ReachabilityNotMeasuredError, match=variant.variant_id):
            analyze_reachability(SEED, SEED, slot_inventory=[], variant=variant)


def _mini_corpora() -> dict[str, list[dict[str, object]]]:
    return {
        "smoke": [
            {
                "id": "hit",
                "openui": 'root = Stack([b], "column")\nb = Button(":x")',
                "placeholders": [":x"],
            },
        ],
        "ood": [],
    }


def test_build_matrix_measures_only_tree_edit_and_never_infers_others(
    monkeypatch,
) -> None:
    import scripts.run_var0_02_reachability_matrix as module

    monkeypatch.setattr(module, "load_corpora", _mini_corpora)
    payload = build_matrix(max_edits=8, node_budget=40, generated_at="2026-07-26T00:00:00Z")

    tree_edit_rows = payload["matrix"]["tree_edit_diffusion"]
    assert tree_edit_rows["smoke"]["status"] == "measured"
    assert tree_edit_rows["smoke"]["reachable_fraction"] == 1.0
    assert tree_edit_rows["ood"]["status"] == "corpus_unavailable"
    assert tree_edit_rows["ood"]["reachable_fraction"] is None

    for variant_id in ("repl_operators", "twotower_prompt_ast"):
        rows = payload["matrix"][variant_id]
        expected_status, expected_reason = NOT_MEASURED_REASONS[variant_id]
        for suite in ("smoke", "ood"):
            assert rows[suite]["status"] == expected_status
            assert rows[suite]["reason"] == expected_reason
            # Never inferred from the tree-edit row: no reachable_fraction,
            # regardless of what the tree-edit row measured for this suite.
            assert rows[suite]["reachable_fraction"] is None


def test_not_measured_reasons_distinguish_deferred_from_inapplicable() -> None:
    deferred_status, _ = NOT_MEASURED_REASONS["repl_operators"]
    inapplicable_status, _ = NOT_MEASURED_REASONS["twotower_prompt_ast"]
    assert deferred_status == "not_measured_deferred"
    assert inapplicable_status == "not_applicable"
    assert deferred_status != inapplicable_status


def test_render_markdown_is_deterministic(monkeypatch) -> None:
    import scripts.run_var0_02_reachability_matrix as module

    monkeypatch.setattr(module, "load_corpora", _mini_corpora)
    payload = build_matrix(max_edits=8, node_budget=40, generated_at="2026-07-26T00:00:00Z")
    first = render_markdown(payload)
    second = render_markdown(payload)
    assert first == second
    assert "tree_edit_diffusion" in first
    assert "not_measured_deferred" in first
    assert "not_applicable" in first
