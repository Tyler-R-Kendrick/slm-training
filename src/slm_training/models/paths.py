"""Canonical checkpoint paths (fixtures are committed; outputs/ is local-only)."""

from __future__ import annotations

from pathlib import Path

# Demo checkpoint for the web playground — committed under src/slm_training/resources/checkpoints/.
PLAYGROUND_DEMO_CHECKPOINT = Path("src/slm_training/resources/checkpoints/playground_demo/last.pt")
