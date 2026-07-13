"""Vast.ai REST adapter."""

from __future__ import annotations

from typing import Any

import httpx

from gpu_multi_farm.config import DEFAULT_DISK_GB, DEFAULT_TRAINING_IMAGE
from gpu_multi_farm.farms.base import filter_offers, require_gpu_type
from gpu_multi_farm.models import FarmListResult, LaunchResult, Offer

VAST_BASE = "https://console.vast.ai/api/v0"


class VastClient:
    name = "vast"

    def __init__(self, api_key: str | None, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def list_offers(
        self,
        gpu_type: str | None = None,
        max_price_per_hr: float | None = None,
    ) -> FarmListResult:
        if not self.api_key:
            return FarmListResult(farm=self.name, offers=[], error="missing_api_key")

        query: dict[str, Any] = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "num_gpus": {"eq": 1},
        }
        if gpu_type:
            query["gpu_name"] = {"eq": gpu_type}
        if max_price_per_hr is not None:
            query["dph_total"] = {"lte": max_price_per_hr}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                # Vast search uses PUT on bundles with JSON query body.
                resp = await client.put(
                    f"{VAST_BASE}/bundles/",
                    headers=self._headers(),
                    json={"q": query, "order": [["dph_total", "asc"]], "type": "ask"},
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001 — surface as farm error
            return FarmListResult(farm=self.name, offers=[], error=str(exc))

        offers: list[Offer] = []
        try:
            for item in payload.get("offers") or payload.get("bundles") or []:
                if not isinstance(item, dict):
                    continue
                search = item.get("search")
                search_price = None
                if isinstance(search, dict):
                    search_price = search.get("dph_total") or search.get("dph_base")
                raw_price = item.get("dph_total") or item.get("dph_base") or search_price or 0.0
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    price = 0.0
                gpu_name = str(item.get("gpu_name") or item.get("gpu_name_full") or "unknown")
                vram = item.get("gpu_ram")
                if vram is not None:
                    try:
                        vram = float(vram) / 1024.0 if float(vram) > 256 else float(vram)
                    except (TypeError, ValueError):
                        vram = None
                offers.append(
                    Offer(
                        farm=self.name,
                        offer_id=str(item.get("id") or item.get("ask_id")),
                        gpu_type=gpu_name,
                        price_per_hr=price,
                        spot=bool(
                            item.get("is_bid")
                            or (item.get("external") is False and item.get("min_bid"))
                        ),
                        vram_gb=vram,
                        availability="available" if item.get("rentable", True) else "unavailable",
                        raw_ref={"id": item.get("id"), "machine_id": item.get("machine_id")},
                    )
                )
        except Exception as exc:  # noqa: BLE001
            return FarmListResult(farm=self.name, offers=[], error=str(exc))

        # Client-side filter if Vast ignored substring match
        offers = filter_offers(offers, gpu_type=gpu_type, max_price_per_hr=max_price_per_hr)
        return FarmListResult(farm=self.name, offers=offers)

    async def launch(self, config: dict[str, Any]) -> LaunchResult:
        if not self.api_key:
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=0.0,
                status="error",
                error="missing_api_key",
            )
        gpu_type = require_gpu_type(config)
        listed = await self.list_offers(gpu_type=gpu_type)
        if listed.error:
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=0.0,
                status="error",
                error=listed.error,
            )
        offer = None
        offer_id = config.get("offer_id")
        if offer_id:
            offer = next((o for o in listed.offers if o.offer_id == str(offer_id)), None)
            if offer is None:
                return LaunchResult(
                    pod_id="",
                    farm=self.name,
                    estimated_cost_per_hr=0.0,
                    status="error",
                    error=f"offer_id={offer_id!r} not found for gpu_type={gpu_type!r}",
                )
        elif listed.offers:
            offer = min(listed.offers, key=lambda o: o.price_per_hr)
        if offer is None:
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=0.0,
                status="error",
                error=f"no vast offers for gpu_type={gpu_type!r}",
            )

        image = config.get("image") or DEFAULT_TRAINING_IMAGE
        disk_gb = int(config.get("disk_gb") or DEFAULT_DISK_GB)
        body = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "runtype": "ssh",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.put(
                    f"{VAST_BASE}/asks/{offer.offer_id}/",
                    headers=self._headers(),
                    json=body,
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=offer.price_per_hr,
                status="error",
                error=str(exc),
            )

        pod_id = str(payload.get("new_contract") or payload.get("id") or "")
        return LaunchResult(
            pod_id=pod_id,
            farm=self.name,
            estimated_cost_per_hr=offer.price_per_hr,
            status="creating",
            ssh_command=None,
            connect_url=f"https://cloud.vast.ai/instances/{pod_id}" if pod_id else None,
            raw=payload if isinstance(payload, dict) else {"response": payload},
        )
