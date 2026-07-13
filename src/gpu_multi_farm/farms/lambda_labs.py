"""Lambda Labs Cloud API adapter."""

from __future__ import annotations

from typing import Any

import httpx

from gpu_multi_farm.config import DEFAULT_DISK_GB
from gpu_multi_farm.farms.base import filter_offers, require_gpu_type
from gpu_multi_farm.models import FarmListResult, LaunchResult, Offer

LAMBDA_BASE = "https://cloud.lambda.ai/api/v1"


class LambdaClient:
    name = "lambda"

    def __init__(self, api_key: str | None, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def list_offers(
        self,
        gpu_type: str | None = None,
        max_price_per_hr: float | None = None,
    ) -> FarmListResult:
        if not self.api_key:
            return FarmListResult(farm=self.name, offers=[], error="missing_api_key")

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.get(
                    f"{LAMBDA_BASE}/instance-types",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return FarmListResult(farm=self.name, offers=[], error=str(exc))

        data = payload.get("data") if isinstance(payload, dict) else payload
        offers: list[Offer] = []
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = [(str(i.get("instance_type") or i.get("name")), i) for i in data]
        else:
            items = []

        for type_name, info in items:
            if not isinstance(info, dict):
                continue
            # Nested shape: {"instance_type": {...}, "regions_with_capacity_available": [...]}
            itype = info.get("instance_type") if "instance_type" in info else info
            if not isinstance(itype, dict):
                itype = info
            name = str(
                itype.get("name")
                or itype.get("description")
                or type_name
            )
            price_cents = itype.get("price_cents_per_hour")
            if price_cents is not None:
                price = float(price_cents) / 100.0
            else:
                price = float(itype.get("price_per_hour") or info.get("price_per_hour") or 0.0)
            regions = info.get("regions_with_capacity_available") or []
            available = "available" if regions else "unavailable"
            gpus = itype.get("gpus") or itype.get("specs", {}).get("gpus")
            vram = None
            if isinstance(itype.get("gpu_description"), str) and "GB" in itype["gpu_description"]:
                # best-effort parse omitted; leave None
                vram = None
            offers.append(
                Offer(
                    farm=self.name,
                    offer_id=str(itype.get("name") or type_name),
                    gpu_type=name,
                    price_per_hr=price,
                    spot=False,
                    vram_gb=float(gpus) if isinstance(gpus, (int, float)) else vram,
                    availability=available,
                    raw_ref={"regions": regions, "name": itype.get("name") or type_name},
                )
            )

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
        # Prefer available regions
        candidates = [o for o in listed.offers if o.availability == "available"] or listed.offers
        offer = None
        offer_id = config.get("offer_id")
        if offer_id:
            offer = next((o for o in candidates if o.offer_id == offer_id), None)
        if offer is None and candidates:
            offer = candidates[0]
        if offer is None:
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=0.0,
                status="error",
                error=f"no lambda offers for gpu_type={gpu_type!r}",
            )

        regions = list(offer.raw_ref.get("regions") or [])
        region_name = None
        if regions:
            first = regions[0]
            region_name = first.get("name") if isinstance(first, dict) else str(first)
        if not region_name:
            region_name = str(config.get("region") or "us-west-1")

        ssh_keys = config.get("ssh_keys") or []
        if isinstance(ssh_keys, str):
            ssh_keys = [ssh_keys]
        body = {
            "region_name": region_name,
            "instance_type_name": offer.offer_id,
            "ssh_key_names": ssh_keys,
            "file_system_names": [],
            "quantity": 1,
            "name": config.get("name") or "slm-train-lambda",
        }
        # disk_gb not used by Lambda launch the same way; keep for metadata
        _ = int(config.get("disk_gb") or DEFAULT_DISK_GB)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{LAMBDA_BASE}/instance-operations/launch",
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

        data = payload.get("data") if isinstance(payload, dict) else payload
        instance_ids = []
        if isinstance(data, dict):
            instance_ids = data.get("instance_ids") or data.get("instances") or []
        pod_id = ""
        if instance_ids:
            first = instance_ids[0]
            pod_id = str(first.get("id") if isinstance(first, dict) else first)
        return LaunchResult(
            pod_id=pod_id,
            farm=self.name,
            estimated_cost_per_hr=offer.price_per_hr,
            status="creating",
            connect_url="https://cloud.lambda.ai/instances",
            raw=payload if isinstance(payload, dict) else {"response": payload},
        )
