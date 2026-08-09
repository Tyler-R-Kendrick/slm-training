"""Evidence-store test isolation: ambient Supabase credentials must never leak in."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_supabase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
        monkeypatch.delenv(var, raising=False)
