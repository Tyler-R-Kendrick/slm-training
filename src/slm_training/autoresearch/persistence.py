"""Explicit remote mirroring for complete local autoresearch bundles."""

from __future__ import annotations

import subprocess
from pathlib import Path


def bucket_uri(campaign_id: str) -> str:
    return f"hf://buckets/TKendrick/OpenUI/autoresearch/{campaign_id}"


def sync_campaign(
    campaign_root: Path | str,
    campaign_id: str,
    *,
    push: bool = False,
) -> dict[str, object]:
    local = Path(campaign_root) / campaign_id
    if not (local / "campaign.json").is_file():
        raise FileNotFoundError(f"campaign bundle not found: {local}")
    command = [
        "hf",
        "buckets",
        "sync",
        str(local),
        bucket_uri(campaign_id),
        "--no-delete",
    ]
    if not push:
        return {"push": False, "command": command, "remote_uri": bucket_uri(campaign_id)}
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "push": True,
        "command": command,
        "remote_uri": bucket_uri(campaign_id),
        "stdout": completed.stdout.strip(),
    }
