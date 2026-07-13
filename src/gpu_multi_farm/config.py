"""Configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

FarmName = Literal["vast", "runpod", "lambda"]
ModeName = Literal["auto", "live", "mock"]

DEFAULT_TRAINING_IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
DEFAULT_DISK_GB = 40
DEFAULT_CACTUS_OVERHEAD = 1.08


@dataclass(frozen=True)
class Settings:
    vast_api_key: str | None
    runpod_api_key: str | None
    lambda_api_key: str | None
    mode: ModeName
    cactus_overhead: float
    http_timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        mode_raw = (os.getenv("GPU_MULTI_FARM_MODE") or "auto").strip().lower()
        if mode_raw not in {"auto", "live", "mock"}:
            mode_raw = "auto"
        overhead = float(os.getenv("CACTUS_OVERHEAD") or DEFAULT_CACTUS_OVERHEAD)
        return cls(
            vast_api_key=_nonempty(os.getenv("VAST_API_KEY")),
            runpod_api_key=_nonempty(os.getenv("RUNPOD_API_KEY")),
            lambda_api_key=_nonempty(os.getenv("LAMBDA_API_KEY")),
            mode=mode_raw,  # type: ignore[arg-type]
            cactus_overhead=overhead,
        )


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_settings() -> Settings:
    return Settings.from_env()
