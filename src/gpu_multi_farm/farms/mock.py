"""Deterministic mock farm for offline CI and missing API keys."""

from __future__ import annotations

from typing import Any

from gpu_multi_farm.config import DEFAULT_DISK_GB, DEFAULT_TRAINING_IMAGE
from gpu_multi_farm.farms.base import filter_offers, require_gpu_type
from gpu_multi_farm.models import FarmListResult, LaunchResult, Offer

MOCK_OFFERS: dict[str, list[Offer]] = {
    "vast": [
        Offer(
            farm="vast",
            offer_id="vast-mock-4090-spot",
            gpu_type="RTX 4090",
            price_per_hr=0.29,
            spot=True,
            vram_gb=24,
            availability="available",
            raw_ref={"source": "mock"},
        ),
        Offer(
            farm="vast",
            offer_id="vast-mock-a100",
            gpu_type="A100",
            price_per_hr=0.85,
            spot=False,
            vram_gb=40,
            availability="available",
            raw_ref={"source": "mock"},
        ),
    ],
    "runpod": [
        Offer(
            farm="runpod",
            offer_id="runpod-mock-4090",
            gpu_type="NVIDIA GeForce RTX 4090",
            price_per_hr=0.44,
            spot=False,
            vram_gb=24,
            availability="available",
            raw_ref={"source": "mock"},
        ),
        Offer(
            farm="runpod",
            offer_id="runpod-mock-a6000",
            gpu_type="NVIDIA RTX A6000",
            price_per_hr=0.49,
            spot=False,
            vram_gb=48,
            availability="available",
            raw_ref={"source": "mock"},
        ),
    ],
    "lambda": [
        Offer(
            farm="lambda",
            offer_id="gpu_1x_a10",
            gpu_type="gpu_1x_a10",
            price_per_hr=0.75,
            spot=False,
            vram_gb=24,
            availability="available",
            raw_ref={"source": "mock"},
        ),
        Offer(
            farm="lambda",
            offer_id="gpu_1x_a100",
            gpu_type="gpu_1x_a100",
            price_per_hr=1.29,
            spot=False,
            vram_gb=40,
            availability="available",
            raw_ref={"source": "mock"},
        ),
    ],
}


class MockClient:
    def __init__(self, farm: str = "vast") -> None:
        if farm not in MOCK_OFFERS:
            raise ValueError(f"unknown mock farm {farm!r}")
        self.name = farm

    async def list_offers(
        self,
        gpu_type: str | None = None,
        max_price_per_hr: float | None = None,
    ) -> FarmListResult:
        offers = filter_offers(
            list(MOCK_OFFERS[self.name]),
            gpu_type=gpu_type,
            max_price_per_hr=max_price_per_hr,
        )
        return FarmListResult(farm=self.name, offers=offers)

    async def launch(self, config: dict[str, Any]) -> LaunchResult:
        if not config.get("allow_mock_launch"):
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=0.0,
                status="refused",
                error="mock launch refused; set config.allow_mock_launch=true",
            )
        gpu_type = require_gpu_type(config)
        listed = await self.list_offers(gpu_type=gpu_type)
        offer = None
        offer_id = config.get("offer_id")
        if offer_id:
            offer = next((o for o in listed.offers if o.offer_id == offer_id), None)
        if offer is None and listed.offers:
            offer = listed.offers[0]
        if offer is None:
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=0.0,
                status="error",
                error=f"no mock offers for gpu_type={gpu_type!r}",
            )
        image = config.get("image") or DEFAULT_TRAINING_IMAGE
        disk_gb = int(config.get("disk_gb") or DEFAULT_DISK_GB)
        name = config.get("name") or f"slm-train-{self.name}"
        pod_id = f"mock-{self.name}-{offer.offer_id}"
        return LaunchResult(
            pod_id=pod_id,
            farm=self.name,
            estimated_cost_per_hr=offer.price_per_hr,
            status="running",
            ssh_command=f"ssh mock@{pod_id}.example.local",
            connect_url=f"https://mock.local/pods/{pod_id}",
            raw={
                "image": image,
                "disk_gb": disk_gb,
                "name": name,
                "offer": offer.to_dict(),
            },
        )
