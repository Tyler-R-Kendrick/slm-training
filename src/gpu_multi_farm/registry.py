"""Resolve farm clients from settings / mode."""

from __future__ import annotations

from typing import Any

from gpu_multi_farm.config import Settings
from gpu_multi_farm.farms.lambda_labs import LambdaClient
from gpu_multi_farm.farms.mock import MockClient
from gpu_multi_farm.farms.runpod import RunPodClient
from gpu_multi_farm.farms.vast import VastClient
from gpu_multi_farm.models import FarmListResult

FARM_NAMES = ("vast", "runpod", "lambda")


def _client_for_farm(farm: str, settings: Settings) -> Any:
    if settings.mode == "mock":
        return MockClient(farm)

    key_map = {
        "vast": settings.vast_api_key,
        "runpod": settings.runpod_api_key,
        "lambda": settings.lambda_api_key,
    }
    key = key_map[farm]
    if settings.mode == "auto" and not key:
        return MockClient(farm)
    if farm == "vast":
        return VastClient(key, timeout_s=settings.http_timeout_s)
    if farm == "runpod":
        return RunPodClient(key, timeout_s=settings.http_timeout_s)
    if farm == "lambda":
        return LambdaClient(key, timeout_s=settings.http_timeout_s)
    raise ValueError(f"unknown farm {farm!r}")


def resolve_farms(farm: str, settings: Settings) -> dict[str, Any]:
    farm = farm.lower().strip()
    if farm == "all":
        names = list(FARM_NAMES)
    elif farm in FARM_NAMES:
        names = [farm]
    else:
        raise ValueError(f"farm must be one of all|{ '|'.join(FARM_NAMES) }, got {farm!r}")
    return {name: _client_for_farm(name, settings) for name in names}


async def list_across_farms(
    clients: dict[str, Any],
    gpu_type: str | None,
    max_price_per_hr: float | None,
) -> dict[str, FarmListResult]:
    results: dict[str, FarmListResult] = {}
    for name, client in clients.items():
        results[name] = await client.list_offers(
            gpu_type=gpu_type,
            max_price_per_hr=max_price_per_hr,
        )
    return results
