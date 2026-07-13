"""FastMCP server exposing multi-farm GPU tools."""

from __future__ import annotations

from typing import Any, Literal

from gpu_multi_farm.config import get_settings
from gpu_multi_farm.cost import project_costs
from gpu_multi_farm.registry import list_across_farms, resolve_farms

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fastmcp is required. Install with: pip install -e '.[mcp]'"
    ) from exc

mcp = FastMCP(name="multi-gpu-farm")

FarmArg = Literal["all", "vast", "runpod", "lambda"]


async def list_available_gpus(
    farm: FarmArg = "all",
    gpu_type: str | None = None,
    max_price_per_hr: float | None = None,
) -> dict[str, Any]:
    """Query real-time availability and pricing across GPU farms.

    Args:
        farm: Which farm to query (`all`, `vast`, `runpod`, or `lambda`).
        gpu_type: Optional substring filter (e.g. `4090`, `A100`).
        max_price_per_hr: Optional maximum $/hr filter.
    """
    settings = get_settings()
    clients = resolve_farms(farm, settings)
    results = await list_across_farms(clients, gpu_type, max_price_per_hr)
    return {
        "mode": settings.mode,
        "farms": {name: result.to_dict() for name, result in results.items()},
    }


async def launch_training_pod(farm: str, config: dict[str, Any]) -> dict[str, Any]:
    """Launch a pod optimized for training (e.g. TwoTower denoiser workloads).

    Args:
        farm: Target farm (`vast`, `runpod`, or `lambda`).
        config: Launch config. Required: `gpu_type`. Optional: `image`, `disk_gb`,
            `name`, `offer_id`, `hours_hint`, `ssh_keys`, `region`,
            `allow_mock_launch` (required true for mock launches).
    """
    farm = farm.lower().strip()
    if farm == "all":
        return {
            "error": "launch_training_pod requires a concrete farm (vast|runpod|lambda)",
            "status": "error",
        }
    settings = get_settings()
    clients = resolve_farms(farm, settings)
    client = clients[farm]
    result = await client.launch(config)
    return result.to_dict()


async def project_training_cost(
    hours: int,
    gpu_type: str,
    model_size_gb: float,
) -> dict[str, Any]:
    """Cross-farm cost projection including spot pricing + Cactus runtime overhead.

    Args:
        hours: Expected training hours.
        gpu_type: Desired GPU type substring (e.g. `4090`).
        model_size_gb: Approximate model/checkpoint size in GB (reserved for sizing).
    """
    if hours < 1:
        raise ValueError("hours must be >= 1")
    settings = get_settings()
    clients = resolve_farms("all", settings)
    results = await list_across_farms(clients, gpu_type=gpu_type, max_price_per_hr=None)
    return project_costs(
        results,
        hours=hours,
        gpu_type=gpu_type,
        model_size_gb=model_size_gb,
        overhead=settings.cactus_overhead,
    )


mcp.tool(list_available_gpus)
mcp.tool(launch_training_pod)
mcp.tool(project_training_cost)


def create_server() -> FastMCP:
    return mcp
