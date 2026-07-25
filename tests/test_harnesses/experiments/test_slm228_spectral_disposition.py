"""Tests for the SLM-228 spectral disposition owner."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.experiments.slm228_spectral_disposition import (
    Disposition,
    build_report,
    require_spectral_disposition,
    validate_spectral_recipe,
    validate_report,
)
from slm_training.harnesses.model_build.config import ModelBuildConfig


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_complete_report_has_no_training_adoption() -> None:
    report = build_report(_root())
    assert report.schema == "SpectralDispositionV1"
    assert validate_report(report, _root()) == []
    assert not any(
        entry.disposition in {Disposition.ADOPT_OPTIONAL, Disposition.ADOPT_PRIMARY}
        for entry in report.entries
    )


def test_raw_alpha_and_projection_are_rejected() -> None:
    report = build_report(_root())
    by_id = {entry.mechanism_id: entry for entry in report.entries}
    assert (
        by_id["raw_alpha_as_quality_or_criticality_signal"].disposition
        == Disposition.REJECT
    )
    assert by_id["ww_pgd_trace_log_projection"].disposition == Disposition.BLOCKED


def test_rejected_mechanism_cannot_enter_training_or_production() -> None:
    report = build_report(_root())
    for use in ("training", "promotion", "production"):
        with pytest.raises(RuntimeError, match="cannot be used"):
            require_spectral_disposition(
                report,
                mechanism_id="ww_pgd_trace_log_projection",
                requested_use=use,
            )


def test_adopted_diagnostic_is_not_a_production_mechanism() -> None:
    report = build_report(_root())
    require_spectral_disposition(
        report,
        mechanism_id="native_spectral_snapshot_and_null_cache",
        requested_use="diagnostic",
    )
    with pytest.raises(RuntimeError, match="cannot be used"):
        require_spectral_disposition(
            report,
            mechanism_id="native_spectral_snapshot_and_null_cache",
            requested_use="production",
        )


def test_recipe_guard_blocks_absolute_fields_and_muon_promotion() -> None:
    with pytest.raises(ValueError, match="not authorized"):
        validate_spectral_recipe(
            {"alpha_target": 2.0},
            requested_use="manifest",
        )
    validate_spectral_recipe(
        {"optimizer_name": "muon_hybrid"},
        requested_use="scratch",
    )
    with pytest.raises(ValueError, match="fixture/research-only"):
        validate_spectral_recipe(
            {"optimizer_name": "muon_hybrid"},
            requested_use="promotion",
        )


def test_model_build_rejects_unknown_optimizer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="optimizer_name must be one of"):
        ModelBuildConfig(train_dir=tmp_path, optimizer_name="spectral_magic")
