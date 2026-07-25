from __future__ import annotations

from copy import deepcopy

import pytest

from slm_training.harnesses.experiments.slm287_power_protocol import (
    BACKENDS,
    METRICS,
    SEEDS,
    VARIANTS,
    LockedPowerProtocol,
    summarize_cells,
    validate_cells,
)


@pytest.fixture
def protocol() -> LockedPowerProtocol:
    return LockedPowerProtocol("a" * 64, ("locked-1",), {"model_name": "twotower"})


@pytest.fixture
def cells(protocol: LockedPowerProtocol) -> list[dict]:
    return [
        {
            "seed": seed,
            "backend": backend,
            "locked_eval_manifest_sha256": protocol.manifest_sha256,
            "initial_tensor_sha256": f"seed-{seed}".ljust(64, "0"),
            "repeat_initial_tensor_sha256": f"seed-{seed}".ljust(64, "0"),
            "records": {
                "locked-1": {
                    variant: {
                        "binding_aware_meaningful_v2": float(backend == BACKENDS[1]),
                        "binder_reference_f1": 0.5,
                        "latency_ms": 1.0,
                        "compute_proxy_forwards": 1.0,
                        "peak_rss_bytes": 1.0,
                    }
                    for variant in VARIANTS
                }
            },
        }
        for seed in SEEDS
        for backend in BACKENDS
    ]


def test_complete_grid_has_paired_mde_and_no_best_seed(protocol, cells) -> None:
    result = summarize_cells(protocol, cells)
    assert result["status"] == "complete_local_grid"
    assert result["paired"]["primary_metric"] == "binding_aware_meaningful_v2"
    assert "best_seed" not in result
    assert set(result["aggregate"]) == set(VARIANTS)


def test_rejects_missing_cell(protocol, cells) -> None:
    with pytest.raises(ValueError, match="five seeds"):
        validate_cells(protocol, cells[:-1])


def test_rejects_nonidentical_repeat_initialization(protocol, cells) -> None:
    broken = deepcopy(cells)
    broken[0]["repeat_initial_tensor_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="bit exact"):
        validate_cells(protocol, broken)


def test_rejects_backend_pair_with_different_initialization(protocol, cells) -> None:
    broken = deepcopy(cells)
    broken[1]["initial_tensor_sha256"] = "b" * 64
    broken[1]["repeat_initial_tensor_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="paired backends"):
        validate_cells(protocol, broken)


def test_rejects_missing_metric(protocol, cells) -> None:
    broken = deepcopy(cells)
    del broken[0]["records"]["locked-1"]["raw"][METRICS[0]]
    with pytest.raises(ValueError, match="missing"):
        validate_cells(protocol, broken)


def test_mde_uses_observed_seed_and_target_effect_variance(protocol, cells) -> None:
    varied = deepcopy(cells)
    for cell in varied:
        for record in cell["records"].values():
            record["repaired"]["binding_aware_meaningful_v2"] = float(
                cell["seed"] % 2 == 0 and cell["backend"] == BACKENDS[1]
            )
    result = summarize_cells(protocol, varied)
    assert result["mde"]["sigma_seed"] > 0
    assert result["mde"]["sigma_target"] == 0
    assert result["paired"]["pairing"] == "record_id paired within seed; target-cluster bootstrap"
