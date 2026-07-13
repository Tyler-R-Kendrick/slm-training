"""Tests for GPU multi-farm MCP (offline / mock)."""

from __future__ import annotations

import os
from typing import Any

import pytest

from gpu_multi_farm.config import Settings
from gpu_multi_farm.cost import project_costs
from gpu_multi_farm.farms.base import filter_offers
from gpu_multi_farm.farms.mock import MockClient
from gpu_multi_farm.models import FarmListResult, Offer
from gpu_multi_farm.registry import resolve_farms


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("GPU_MULTI_FARM_MODE", "mock")
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.setenv("CACTUS_OVERHEAD", "1.08")
    return Settings.from_env()


def test_cheapest_offer_picks_min_price() -> None:
    from gpu_multi_farm.cost import cheapest_offer

    results = {
        "vast": FarmListResult(
            farm="vast",
            offers=[
                Offer(farm="vast", offer_id="exp", gpu_type="RTX 4090", price_per_hr=0.9),
                Offer(farm="vast", offer_id="cheap", gpu_type="RTX 4090", price_per_hr=0.2),
            ],
        )
    }
    best = cheapest_offer(results, gpu_type="4090")
    assert best["vast"] is not None
    assert best["vast"].offer_id == "cheap"


def test_cactus_overhead_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACTUS_OVERHEAD", "not-a-float")
    settings = Settings.from_env()
    assert settings.cactus_overhead == 1.08


def test_filter_offers_by_price_and_type() -> None:
    offers = [
        Offer(farm="vast", offer_id="1", gpu_type="RTX 4090", price_per_hr=0.5),
        Offer(farm="vast", offer_id="2", gpu_type="A100", price_per_hr=0.2),
        Offer(farm="vast", offer_id="3", gpu_type="RTX 4090", price_per_hr=0.9),
    ]
    filtered = filter_offers(offers, gpu_type="4090", max_price_per_hr=0.6)
    assert [o.offer_id for o in filtered] == ["1"]


@pytest.mark.asyncio
async def test_mock_list_and_launch_gate(mock_settings: Settings) -> None:
    client = MockClient("vast")
    listed = await client.list_offers(gpu_type="4090", max_price_per_hr=1.0)
    assert listed.error is None
    assert listed.offers
    assert all("4090" in o.gpu_type for o in listed.offers)

    refused = await client.launch({"gpu_type": "RTX 4090"})
    assert refused.status == "refused"
    assert refused.error

    ok = await client.launch({"gpu_type": "RTX 4090", "allow_mock_launch": True})
    assert ok.status == "running"
    assert ok.pod_id.startswith("mock-vast-")
    assert ok.estimated_cost_per_hr > 0


@pytest.mark.asyncio
async def test_resolve_farms_auto_uses_mock_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_MULTI_FARM_MODE", "auto")
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    settings = Settings.from_env()
    clients = resolve_farms("all", settings)
    assert set(clients) == {"vast", "runpod", "lambda"}
    assert all(isinstance(c, MockClient) for c in clients.values())


@pytest.mark.asyncio
async def test_project_training_cost_recommends_cheapest(
    mock_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    # Isolate from workspace outputs/cactus/bench.json overhead override.
    monkeypatch.chdir(tmp_path)
    clients = resolve_farms("all", mock_settings)
    results: dict[str, FarmListResult] = {}
    for name, client in clients.items():
        results[name] = await client.list_offers(gpu_type="4090")
    # Ensure at least vast has 4090; others may fall back to all offers in projector
    projection = project_costs(
        results,
        hours=10,
        gpu_type="4090",
        model_size_gb=2.0,
        overhead=mock_settings.cactus_overhead,
    )
    assert projection["cactus_overhead"] == 1.08
    assert projection["recommended"] in {"vast", "runpod", "lambda"}
    assert projection["farms"]["vast"]["available"] is True
    # Vast mock 4090 at 0.29 should beat RunPod 0.44 for 4090-ish
    assert projection["recommended"] == "vast"


@pytest.mark.asyncio
async def test_tool_list_available_gpus_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_MULTI_FARM_MODE", "mock")
    from gpu_multi_farm.server import list_available_gpus

    payload = await list_available_gpus(farm="vast", gpu_type="4090", max_price_per_hr=1.0)
    assert payload["mode"] == "mock"
    assert payload["farms"]["vast"]["offers"]
    assert "runpod" not in payload["farms"]


@pytest.mark.asyncio
async def test_tool_launch_and_cost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("GPU_MULTI_FARM_MODE", "mock")
    monkeypatch.setenv("CACTUS_OVERHEAD", "1.08")
    monkeypatch.chdir(tmp_path)
    from gpu_multi_farm.server import launch_training_pod, project_training_cost

    bad = await launch_training_pod("all", {"gpu_type": "4090"})
    assert bad.get("status") == "error"

    launched = await launch_training_pod(
        "vast",
        {"gpu_type": "4090", "allow_mock_launch": True, "name": "unit"},
    )
    assert launched["status"] == "running"

    cost = await project_training_cost(hours=5, gpu_type="4090", model_size_gb=1.5)
    assert cost["recommended"] == "vast"
    assert cost["farms"]["vast"]["total_cost"] == pytest.approx(0.29 * 5 * 1.08)


@pytest.mark.asyncio
async def test_live_missing_key_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_MULTI_FARM_MODE", "live")
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    settings = Settings.from_env()
    clients = resolve_farms("vast", settings)
    result = await clients["vast"].list_offers()
    assert result.error == "missing_api_key"
    assert result.offers == []
