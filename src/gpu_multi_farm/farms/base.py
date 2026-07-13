"""Farm client protocol and shared helpers."""

from __future__ import annotations

from typing import Any, Protocol

from gpu_multi_farm.models import FarmListResult, LaunchResult, Offer


class FarmClient(Protocol):
    name: str

    async def list_offers(
        self,
        gpu_type: str | None = None,
        max_price_per_hr: float | None = None,
    ) -> FarmListResult:
        ...

    async def launch(self, config: dict[str, Any]) -> LaunchResult:
        ...


def filter_offers(
    offers: list[Offer],
    gpu_type: str | None = None,
    max_price_per_hr: float | None = None,
) -> list[Offer]:
    out = offers
    if gpu_type:
        needle = gpu_type.lower()
        out = [o for o in out if needle in o.gpu_type.lower()]
    if max_price_per_hr is not None:
        out = [o for o in out if o.price_per_hr <= max_price_per_hr]
    return sorted(out, key=lambda o: o.price_per_hr)


def require_gpu_type(config: dict[str, Any]) -> str:
    gpu_type = config.get("gpu_type")
    if not gpu_type or not str(gpu_type).strip():
        raise ValueError("config.gpu_type is required")
    return str(gpu_type).strip()
