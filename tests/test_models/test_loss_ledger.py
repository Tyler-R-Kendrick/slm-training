"""Tests for SLM-261 LossLedgerV1 reconciliation."""

from __future__ import annotations

import math

import pytest

from slm_training.models.loss_ledger import LOSS_LEDGER_SCHEMA_VERSION, LossLedgerV1


def _base_metrics() -> dict:
    return {
        "primary_final_reconstruction_loss": 1.2,
        "primary_final_reconstruction_loss_weight": 1.0,
        "primary_final_reconstruction_loss_contribution": 1.2,
        "recursive_depth_supervision_unweighted_loss": 0.5,
        "recursive_depth_supervision_loss_weight": 0.8,
        "recursive_depth_supervision_loss_contribution": 0.4,
        "reported_total_loss": 1.6,
        "detached_auxiliary_loss": 0.0,
    }


def test_ledger_reconstructs_principal_only() -> None:
    metrics = {
        "primary_final_reconstruction_loss": 1.2,
        "primary_final_reconstruction_loss_weight": 1.0,
        "primary_final_reconstruction_loss_contribution": 1.2,
        "reported_total_loss": 1.2,
        "detached_auxiliary_loss": 0.0,
    }
    ledger = LossLedgerV1.from_metrics(
        metrics,
        vocab_size=100,
        active_example_count=4,
        active_token_count=16,
    )
    assert ledger.schema_version == LOSS_LEDGER_SCHEMA_VERSION
    assert ledger.vocab_size == 100
    assert math.isclose(ledger.full_vocab_uniform_floor, math.log(100), rel_tol=1e-6)
    assert ledger.absolute_reconciliation_error == 0.0
    assert ledger.total_reconstructed_from_components == 1.2
    assert ledger.reported_total == 1.2


def test_ledger_includes_inactive_terms_with_zero_weight() -> None:
    ledger = LossLedgerV1.from_metrics(
        _base_metrics(),
        vocab_size=100,
        active_example_count=2,
        active_token_count=8,
    )
    names = {t.name for t in ledger.terms}
    expected = {
        "principal_mask_ce",
        "recursive_depth_supervision",
        "diffusion_length",
        "fidelity",
        "symbol_boundary",
        "ltr",
        "compiler_alignment",
        "component_inventory",
        "component_plan",
        "slot_component",
        "component_edge",
        "binder_arity",
        "root_reference_arity",
        "root_reference_identity",
        "component_edge_alignment",
        "binder_component_plan",
        "binder_topology",
        "fastpath_aux",
        "detached_auxiliary",
    }
    assert names == expected
    inactive_aux = [t for t in ledger.terms if t.name not in {"principal_mask_ce", "recursive_depth_supervision", "detached_auxiliary"}]
    assert all(t.raw == 0.0 and t.weight == 0.0 and t.contribution == 0.0 for t in inactive_aux)
    detached = ledger.term("detached_auxiliary")
    assert detached is not None
    assert detached.raw == 0.0 and detached.weight == 1.0 and detached.contribution == 0.0


def test_ledger_computes_auxiliary_sum() -> None:
    ledger = LossLedgerV1.from_metrics(
        _base_metrics(),
        vocab_size=100,
        active_example_count=2,
        active_token_count=8,
    )
    assert ledger.total_auxiliary_contribution == 0.4
    assert ledger.total_reconstructed_from_components == 1.6
    assert ledger.reported_total == 1.6
    assert ledger.absolute_reconciliation_error == 0.0


def test_ledger_includes_detached_auxiliary() -> None:
    metrics = _base_metrics()
    metrics["detached_auxiliary_loss"] = 0.3
    ledger = LossLedgerV1.from_metrics(
        metrics,
        vocab_size=100,
        active_example_count=2,
        active_token_count=8,
    )
    assert ledger.reported_total == 1.9
    assert ledger.absolute_reconciliation_error == 0.0
    detached = ledger.term("detached_auxiliary")
    assert detached is not None
    assert detached.raw == 0.3
    assert detached.contribution == 0.3


def test_ledger_rejects_mismatched_total() -> None:
    metrics = _base_metrics()
    metrics["reported_total_loss"] = 999.0
    with pytest.raises(ValueError, match="reconciliation failed"):
        LossLedgerV1.from_metrics(
            metrics,
            vocab_size=100,
            active_example_count=2,
            active_token_count=8,
        )


def test_ledger_roundtrips_through_dict() -> None:
    ledger = LossLedgerV1.from_metrics(
        _base_metrics(),
        vocab_size=256,
        active_example_count=2,
        active_token_count=8,
        candidate_set_size_mean=16.0,
        trainable_parameter_count=1234,
        total_gradient_norm=0.42,
        per_component_gradient_norm={"denoiser": 0.3},
    )
    rebuilt = LossLedgerV1.from_dict(ledger.to_dict())
    assert rebuilt == ledger


def test_ledger_candidate_floor() -> None:
    ledger = LossLedgerV1.from_metrics(
        _base_metrics(),
        vocab_size=100,
        active_example_count=2,
        active_token_count=8,
        candidate_set_size_mean=20.0,
    )
    assert ledger.candidate_set_size_mean == 20.0
    assert ledger.candidate_uniform_floor == pytest.approx(math.log(20.0))
