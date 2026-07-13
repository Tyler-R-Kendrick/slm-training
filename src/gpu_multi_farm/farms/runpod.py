"""RunPod GraphQL adapter."""

from __future__ import annotations

from typing import Any

import httpx

from gpu_multi_farm.config import DEFAULT_DISK_GB, DEFAULT_TRAINING_IMAGE
from gpu_multi_farm.farms.base import filter_offers, require_gpu_type
from gpu_multi_farm.models import FarmListResult, LaunchResult, Offer

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


class RunPodClient:
    name = "runpod"

    def __init__(self, api_key: str | None, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _gql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                RUNPOD_GRAPHQL,
                headers=self._headers(),
                json={"query": query, "variables": variables or {}},
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("errors"):
                raise RuntimeError(str(payload["errors"]))
            return payload.get("data") or {}

    async def list_offers(
        self,
        gpu_type: str | None = None,
        max_price_per_hr: float | None = None,
    ) -> FarmListResult:
        if not self.api_key:
            return FarmListResult(farm=self.name, offers=[], error="missing_api_key")

        query = """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            lowestPrice(input: { gpuCount: 1 }) {
              uninterruptablePrice
              minimumBidPrice
              stockStatus
            }
          }
        }
        """
        try:
            data = await self._gql(query)
        except Exception as exc:  # noqa: BLE001
            return FarmListResult(farm=self.name, offers=[], error=str(exc))

        offers: list[Offer] = []
        for item in data.get("gpuTypes") or []:
            lowest = item.get("lowestPrice") or {}
            on_demand = lowest.get("uninterruptablePrice")
            spot = lowest.get("minimumBidPrice")
            price = None
            is_spot = False
            if on_demand is not None:
                price = float(on_demand)
            elif spot is not None:
                price = float(spot)
                is_spot = True
            if price is None:
                continue
            gpu_name = str(item.get("displayName") or item.get("id") or "unknown")
            offers.append(
                Offer(
                    farm=self.name,
                    offer_id=str(item.get("id")),
                    gpu_type=gpu_name,
                    price_per_hr=price,
                    spot=is_spot,
                    vram_gb=float(item["memoryInGb"]) if item.get("memoryInGb") is not None else None,
                    availability=str(lowest.get("stockStatus") or "unknown"),
                    raw_ref={"id": item.get("id")},
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
                error=f"no runpod offers for gpu_type={gpu_type!r}",
            )

        image = config.get("image") or DEFAULT_TRAINING_IMAGE
        disk_gb = int(config.get("disk_gb") or DEFAULT_DISK_GB)
        name = config.get("name") or "slm-train-runpod"
        mutation = """
        mutation Deploy($input: PodFindAndDeployOnDemandInput) {
          podFindAndDeployOnDemand(input: $input) {
            id
            imageName
            machineId
            desiredStatus
          }
        }
        """
        variables = {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "volumeInGb": disk_gb,
                "containerDiskInGb": disk_gb,
                "gpuTypeId": offer.offer_id,
                "name": name,
                "imageName": image,
                "ports": "22/tcp",
                "volumeMountPath": "/workspace",
            }
        }
        try:
            data = await self._gql(mutation, variables)
        except Exception as exc:  # noqa: BLE001
            # No string-built GraphQL fallback: interpolating name/image is injectable.
            return LaunchResult(
                pod_id="",
                farm=self.name,
                estimated_cost_per_hr=offer.price_per_hr,
                status="error",
                error=str(exc),
            )

        pod = data.get("podFindAndDeployOnDemand") or {}
        pod_id = str(pod.get("id") or "")
        return LaunchResult(
            pod_id=pod_id,
            farm=self.name,
            estimated_cost_per_hr=offer.price_per_hr,
            status=str(pod.get("desiredStatus") or "creating"),
            connect_url=f"https://www.runpod.io/console/pods/{pod_id}" if pod_id else None,
            raw=pod,
        )
