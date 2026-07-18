"""Repository-owned removable model adapters (LDI2-01).

``TwoTowerAdapterSpec`` is torch-free (config round-trip); import ``LowRankAdapter``
from :mod:`slm_training.models.adapters.low_rank` where torch is available.
"""

from __future__ import annotations

from slm_training.models.adapters.spec import ADAPTER_SCHEMA_VERSION, TwoTowerAdapterSpec

__all__ = ["ADAPTER_SCHEMA_VERSION", "TwoTowerAdapterSpec"]
