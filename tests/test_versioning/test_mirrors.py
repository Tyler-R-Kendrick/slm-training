"""The registry mirrors the in-code version constants (single source per pair).

In-code constants stay canonical for runtime behavior; the registry is the
normalized ledger the bump checker diffs. These tests pin the two views
together so neither can drift silently.
"""

from __future__ import annotations

from slm_training.evals.denoising_nll import DenoisingNLLConfig
from slm_training.evals.emptiness_probe import EmptinessProbeConfig
from slm_training.evals.loss_suites import LOSS_SUITE_VERSION
from slm_training.evals.meaningful_program import METRIC_VERSION
from slm_training.versioning import component_version


def test_loss_suite_version_is_mirrored() -> None:
    assert component_version("evals.loss_suite") == LOSS_SUITE_VERSION


def test_loss_suite_configs_share_the_canonical_version() -> None:
    # Regression: these defaults were once hardcoded "v1" literals, so a bump
    # of LOSS_SUITE_VERSION would not have propagated to the probes.
    assert EmptinessProbeConfig().suite_version == LOSS_SUITE_VERSION
    assert DenoisingNLLConfig().suite_version == LOSS_SUITE_VERSION


def test_meaningful_program_version_is_mirrored() -> None:
    from slm_training.harnesses.model_build.ship_gates import MEANINGFUL_METRIC_POLICY

    assert component_version("evals.meaningful_program") == METRIC_VERSION
    assert (
        MEANINGFUL_METRIC_POLICY["binding_aware_meaningful_v2"]["version"]
        == METRIC_VERSION
    )


def test_ship_gate_threshold_version_is_mirrored() -> None:
    from slm_training.harnesses.model_build.ship_gates import MEANINGFUL_METRIC_POLICY

    assert component_version("gates.ship") == MEANINGFUL_METRIC_POLICY["threshold_version"]


def test_verified_solver_matrix_version_is_mirrored() -> None:
    from slm_training.harnesses.experiments.verified_solver_matrix import MATRIX_VERSION

    assert component_version("matrix.verified_solver") == MATRIX_VERSION
