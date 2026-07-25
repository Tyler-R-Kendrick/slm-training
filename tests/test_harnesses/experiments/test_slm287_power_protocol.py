from __future__ import annotations

from copy import deepcopy

import pytest

from slm_training.autoresearch.schemas import CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.slm287_power_protocol import (
    BACKENDS,
    METRICS,
    SEEDS,
    VARIANTS,
    LockedPowerProtocol,
    build_locked_campaign,
    locked_shard_ids,
    merge_locked_shards,
    summarize_cells,
    validate_cells,
)


@pytest.fixture
def protocol() -> LockedPowerProtocol:
    return LockedPowerProtocol(
        "a" * 64,
        ("locked-1",),
        {
            "model_name": "twotower",
            "backends": {backend: {"name": backend} for backend in BACKENDS},
        },
    )


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
                    }
                    for variant in VARIANTS
                }
            },
            "cell_metrics": {
                variant: {
                    "compute_proxy_forwards": 1.0,
                    "peak_rss_bytes": 1.0,
                }
                for variant in VARIANTS
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


def test_locked_campaign_binds_the_manifest_and_all_seeds(protocol, tmp_path) -> None:
    campaign = build_locked_campaign(protocol)
    assert campaign.locked_eval_manifest_sha256 == protocol.manifest_sha256
    assert campaign.seeds == SEEDS
    store = CampaignStore(campaign.campaign_id, tmp_path / "campaigns")
    spec = CampaignSpec(
        campaign_id=campaign.campaign_id,
        objective="Measure the five-seed two-backend locked local baseline.",
        primary_metric="binding_aware_meaningful_v2",
        researcher_mode="diagnostic",
        budget=campaign.budget,
        created_at=campaign.created_at,
    )
    store.initialize(spec)
    assert store.lock_experiment_campaign(campaign) == store.lock_experiment_campaign(campaign)


def test_locked_shards_are_balanced_and_nonempty(protocol) -> None:
    sharded_protocol = LockedPowerProtocol(
        protocol.manifest_sha256,
        tuple(f"locked-{index}" for index in range(5)),
        protocol.recipe,
    )
    shards = [
        locked_shard_ids(sharded_protocol, shard_index=index, shard_count=5)
        for index in range(5)
    ]
    assert all(len(ids) == 1 for ids in shards)
    assert {record_id for ids in shards for record_id in ids} == set(sharded_protocol.record_ids)


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


def test_merges_exact_shards_and_sums_cell_compute(protocol, cells) -> None:
    sharded_protocol = LockedPowerProtocol(
        protocol.manifest_sha256,
        ("locked-1", "locked-2"),
        protocol.recipe,
    )
    source_cells = []
    for cell in cells:
        copied = deepcopy(cell)
        copied["records"]["locked-2"] = deepcopy(copied["records"]["locked-1"])
        source_cells.append(copied)
    shards = []
    for cell in source_cells:
        for shard_index in range(2):
            ids = locked_shard_ids(
                sharded_protocol, shard_index=shard_index, shard_count=2
            )
            shards.append(
                {
                    "protocol_sha256": sharded_protocol.to_dict()["protocol_sha256"],
                    "campaign_manifest_sha256": "c" * 64,
                    "shard_count": 2,
                    "shard_index": shard_index,
                    "record_ids": list(ids),
                    **{key: cell[key] for key in (
                        "seed", "backend", "locked_eval_manifest_sha256",
                        "initial_tensor_sha256", "repeat_initial_tensor_sha256",
                        "cell_metrics",
                    )},
                    "records": {record_id: cell["records"][record_id] for record_id in ids},
                }
            )
    merged = merge_locked_shards(sharded_protocol, shards, shard_count=2)
    assert len(merged) == 10
    assert set(merged[0]["records"]) == {"locked-1", "locked-2"}
    assert merged[0]["cell_metrics"]["raw"]["compute_proxy_forwards"] == 2.0


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
